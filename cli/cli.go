// Package cli is agentflow's command-line front end. The agentflow command
// is cli.Main with no entrypoints: it inspects traces. Call cli.Main from
// your own program with your agents and workflows as Entrypoints, and the
// program also gets replay and fork, which need your code:
//
//	func main() {
//		cli.Main(cli.Config{Entrypoints: []agentflow.Entrypoint{newTriage()}})
//	}
//
//	$ mytool show <id>        # the run tree
//	$ mytool blame <id>       # root causes of a failure
//	$ mytool replay <id>      # replay against the current code
//	$ mytool fork <id> -path /metrics#0/sql#0 -effect llm#0 -system "…" -samples 5 -baseline
package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

// Config configures the command line.
type Config struct {
	// Entrypoints are the agents and workflows the program can replay and
	// fork, matched to a trace by its root run's agent name. Without any,
	// the replay and fork commands are absent.
	Entrypoints []agentflow.Entrypoint
	// Store is the trace store. Default: a file store at the -store flag,
	// $AGENTFLOW_STORE_DIR, or ".agentflow".
	Store trace.Store
	// Name is the program's name in usage messages; default: the base
	// name of os.Args[0].
	Name string
	// Version is printed by the version command.
	Version string
	// Stdin is where the hook command reads its input; default os.Stdin.
	Stdin io.Reader
}

// Exit codes.
const (
	ExitOK       = 0
	ExitError    = 1
	ExitUsage    = 2
	ExitDiverged = 3 // a replay diverged, or a fork's treatment had failing samples
)

// DefaultStoreDir is the default file store directory.
const DefaultStoreDir = ".agentflow"

// Main runs the command line on os.Args and exits.
func Main(cfg Config) {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	code := Run(ctx, os.Args[1:], os.Stdout, os.Stderr, cfg)
	cancel()
	os.Exit(code)
}

// errUsage marks errors that are the caller's misuse.
type errUsage struct{ msg string }

func (e errUsage) Error() string { return e.msg }

func usageErr(format string, a ...any) error { return errUsage{fmt.Sprintf(format, a...)} }

// errDiverged carries exit code 3 with a message already printed.
var errDiverged = errors.New("diverged")

// Run runs the command line on args (without the program name) and returns
// the exit code.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer, cfg Config) int {
	if cfg.Name == "" {
		cfg.Name = filepath.Base(os.Args[0])
	}
	global := flag.NewFlagSet(cfg.Name, flag.ContinueOnError)
	global.SetOutput(stderr)
	dir := global.String("store", "", "trace store directory (default $AGENTFLOW_STORE_DIR or "+DefaultStoreDir+")")
	global.Usage = func() { usage(stderr, cfg) }
	if err := global.Parse(args); err != nil {
		return ExitUsage
	}
	args = global.Args()
	if len(args) == 0 {
		usage(stderr, cfg)
		return ExitUsage
	}
	c := &env{cfg: cfg, out: stdout, errOut: stderr, dir: *dir}
	var err error
	switch cmd, rest := args[0], args[1:]; cmd {
	case "runs":
		err = c.runs(ctx, rest)
	case "show":
		err = c.show(ctx, rest)
	case "diff":
		err = c.diff(ctx, rest)
	case "blame":
		err = c.blame(ctx, rest)
	case "export":
		err = c.export(ctx, rest)
	case "replay":
		err = c.replay(ctx, rest)
	case "fork":
		err = c.fork(ctx, rest)
	case "import":
		err = c.importCmd(ctx, rest)
	case "hook":
		err = c.hook(ctx, rest)
	case "version":
		v := cfg.Version
		if v == "" {
			v = "dev"
		}
		_, err = fmt.Fprintf(stdout, "%s %s\n", cfg.Name, v)
	case "help", "-h", "--help":
		usage(stdout, cfg)
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command %q\n\n", cmd)
		usage(stderr, cfg)
		return ExitUsage
	}
	if c.closer != nil {
		_ = c.closer.Close()
	}
	var ue errUsage
	switch {
	case err == nil:
		return ExitOK
	case errors.Is(err, errDiverged):
		return ExitDiverged
	case errors.As(err, &ue), errors.Is(err, flag.ErrHelp):
		_, _ = fmt.Fprintf(stderr, "%v\n", err)
		return ExitUsage
	}
	_, _ = fmt.Fprintf(stderr, "error: %v\n", err)
	return ExitError
}

func usage(w io.Writer, cfg Config) {
	run := ""
	if len(cfg.Entrypoints) > 0 {
		var names []string
		for _, e := range cfg.Entrypoints {
			names = append(names, e.Name())
		}
		run = fmt.Sprintf(`
  %[1]s replay <id> [flags]           Replay a trace against the current code
  %[1]s fork <id> [flags]             Fork a trace: change it at a point, continue live

Replay and fork run: %[2]s.
`, cfg.Name, strings.Join(names, ", "))
	}
	_, _ = fmt.Fprintf(w, `%[1]s inspects recorded agent traces.

Usage:
  %[1]s [-store dir] <command> [arguments]

  %[1]s runs [-all] [-n N]            List traces (newest first)
  %[1]s show <id> [-events] [-run p]  Show a trace's run tree (or its events)
  %[1]s diff <a> <b> [-exact]         Compare two traces run by run
  %[1]s blame <id> [-at path[@seq]]   Find the root causes of a failure
  %[1]s export <id> [-html] [-o file] Write a trace as a bundle, or as one HTML page
  %[1]s import claude-code <path>…    Record Claude Code sessions (transcripts or a project dir)
  %[1]s hook claude-code [-settings]  Record the session a Claude Code hook fires for
  %[1]s version
%[2]s
<id> is a trace or run ID, or a unique prefix or suffix of one (at least 6
characters). IDs start with a timestamp, so a suffix is usually shortest.
Run "%[1]s <command> -h" for a command's flags.
`, cfg.Name, run)
}

// env is one invocation's state.
type env struct {
	cfg    Config
	out    io.Writer
	errOut io.Writer
	dir    string
	store  trace.Store
	closer io.Closer
}

func (c *env) storeDir() string {
	dir := c.dir
	if dir == "" {
		dir = os.Getenv("AGENTFLOW_STORE_DIR")
	}
	if dir == "" {
		dir = DefaultStoreDir
	}
	return dir
}

func (c *env) open() (trace.Store, error) {
	return c.openWith(filestore.Options{})
}

func (c *env) openWith(opts filestore.Options) (trace.Store, error) {
	if c.store != nil {
		return c.store, nil
	}
	if c.cfg.Store != nil {
		c.store = c.cfg.Store
		return c.store, nil
	}
	fs, err := filestore.Open(c.storeDir(), opts)
	if err != nil {
		return nil, err
	}
	c.store, c.closer = fs, fs
	return fs, nil
}

// parse parses flags and positional arguments in any order, and checks
// the number of positional arguments.
func parse(fs *flag.FlagSet, args []string, positional int, synopsis string) ([]string, error) {
	fs.SetOutput(io.Discard)
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil, usageErr("usage: %s\n%s", synopsis, flagHelp(fs))
			}
			return nil, usageErr("%v\nusage: %s", err, synopsis)
		}
		if fs.NArg() == 0 {
			break
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
	if len(pos) != positional {
		return nil, usageErr("usage: %s", synopsis)
	}
	return pos, nil
}

func flagHelp(fs *flag.FlagSet) string {
	var sb strings.Builder
	fs.SetOutput(&sb)
	fs.PrintDefaults()
	return sb.String()
}

// resolve finds the trace that a trace ID, run ID, or a unique prefix or
// suffix of one names.
func (c *env) resolve(ctx context.Context, id string) (trace.Store, string, error) {
	s, err := c.open()
	if err != nil {
		return nil, "", err
	}
	if len(id) < 6 {
		return nil, "", usageErr("id %q is too short (use at least 6 characters)", id)
	}
	matches := map[string]bool{}
	for r, err := range s.Runs(ctx, trace.RunFilter{}) {
		if err != nil {
			return nil, "", err
		}
		if strings.HasPrefix(r.TraceID, id) || strings.HasPrefix(r.RunID, id) ||
			strings.HasSuffix(r.TraceID, id) || strings.HasSuffix(r.RunID, id) {
			matches[r.TraceID] = true
		}
	}
	switch len(matches) {
	case 0:
		return nil, "", fmt.Errorf("no trace or run matches %q", id)
	case 1:
		for t := range matches {
			return s, t, nil
		}
	}
	ids := make([]string, 0, len(matches))
	for t := range matches {
		ids = append(ids, t)
	}
	sort.Strings(ids)
	return nil, "", fmt.Errorf("%q is ambiguous; it matches traces:\n  %s", id, strings.Join(ids, "\n  "))
}

// printer writes formatted output and remembers the first error.
type printer struct {
	w   io.Writer
	err error
}

func (p *printer) printf(format string, a ...any) {
	if p.err == nil {
		_, p.err = fmt.Fprintf(p.w, format, a...)
	}
}
