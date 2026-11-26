package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/kumarlokesh/agentflow/external/claudecode"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

// openForImport opens the store that imports write to: redacting
// credentials unless told not to. It returns the directory for lock files
// ("" for a store given in Config).
func (c *env) openForImport(noRedact bool) (trace.Store, string, error) {
	opts := filestore.Options{Redact: trace.SecretPatterns}
	if noRedact {
		opts.Redact = nil
	}
	s, err := c.openWith(opts)
	if err != nil || c.cfg.Store != nil {
		return s, "", err
	}
	return s, filepath.Join(c.storeDir(), "locks"), nil
}

const importSynopsis = "import claude-code [-final] [-no-redact] <transcript.jsonl | project dir> …"

func (c *env) importCmd(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] != "claude-code" {
		return usageErr("usage: %s", importSynopsis)
	}
	fs := flag.NewFlagSet("import", flag.ContinueOnError)
	final := fs.Bool("final", false, "the sessions are over: finish their root runs")
	noRedact := fs.Bool("no-redact", false, "store credentials found in transcripts instead of redacting them")
	paths, err := parseAny(fs, args[1:], importSynopsis)
	if err != nil {
		return err
	}
	var files []string
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			return err
		}
		if !st.IsDir() {
			files = append(files, p)
			continue
		}
		sessions, err := claudecode.Sessions(p)
		if err != nil {
			return err
		}
		if len(sessions) == 0 {
			return fmt.Errorf("%s holds no session transcripts (*.jsonl)", p)
		}
		files = append(files, sessions...)
	}
	s, lockDir, err := c.openForImport(*noRedact)
	if err != nil {
		return err
	}
	p := &printer{w: c.out}
	for _, f := range files {
		id, err := claudecode.Import(ctx, s, f, claudecode.ImportOptions{Options: claudecode.Options{Final: *final}, LockDir: lockDir})
		switch {
		case err != nil:
			return fmt.Errorf("%s: %w", f, err)
		case id == "":
			p.printf("%s: nothing to record yet\n", f)
		default:
			p.printf("%s  %s\n", id, f)
		}
	}
	return p.err
}

// hookInput is the part of Claude Code's hook input the hook reads.
type hookInput struct {
	SessionID      string `json:"session_id"`
	TranscriptPath string `json:"transcript_path"`
	HookEventName  string `json:"hook_event_name"`
}

// hookSettings installs the hook for the events that end a turn, a
// sub-agent, and the session.
const hookSettings = `{
  "hooks": {
    "Stop":         [{"hooks": [{"type": "command", "command": "agentflow hook claude-code"}]}],
    "SubagentStop": [{"hooks": [{"type": "command", "command": "agentflow hook claude-code"}]}],
    "SessionEnd":   [{"hooks": [{"type": "command", "command": "agentflow hook claude-code"}]}]
  }
}
`

func (c *env) hook(ctx context.Context, args []string) error {
	const synopsis = "hook claude-code [-settings] [-no-redact]"
	if len(args) == 0 || args[0] != "claude-code" {
		return usageErr("usage: %s", synopsis)
	}
	fs := flag.NewFlagSet("hook", flag.ContinueOnError)
	settings := fs.Bool("settings", false, "print the Claude Code settings that install this hook, and exit")
	noRedact := fs.Bool("no-redact", false, "store credentials found in transcripts instead of redacting them")
	if _, err := parse(fs, args[1:], 0, synopsis); err != nil {
		return err
	}
	if *settings {
		_, err := io.WriteString(c.out, hookSettings)
		return err
	}
	// A hook must never get in Claude Code's way: problems are reported,
	// and the exit code is 0.
	if err := c.runHook(ctx, *noRedact); err != nil {
		_, _ = fmt.Fprintf(c.errOut, "agentflow hook: %v\n", err)
	}
	return nil
}

func (c *env) runHook(ctx context.Context, noRedact bool) error {
	in := c.cfg.Stdin
	if in == nil {
		in = os.Stdin
	}
	var h hookInput
	if err := json.NewDecoder(in).Decode(&h); err != nil {
		return fmt.Errorf("reading the hook input: %w", err)
	}
	if h.TranscriptPath == "" {
		return errors.New("the hook input has no transcript_path")
	}
	s, lockDir, err := c.openForImport(noRedact)
	if err != nil {
		return err
	}
	_, err = claudecode.Import(ctx, s, h.TranscriptPath, claudecode.ImportOptions{
		Options: claudecode.Options{Final: h.HookEventName == "SessionEnd"}, LockDir: lockDir})
	return err
}

// parseAny is parse for commands taking one or more positional arguments.
func parseAny(fs *flag.FlagSet, args []string, synopsis string) ([]string, error) {
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
	if len(pos) == 0 {
		return nil, usageErr("usage: %s", synopsis)
	}
	return pos, nil
}
