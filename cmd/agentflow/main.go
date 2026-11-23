// Command agentflow inspects recorded traces.
//
// Usage:
//
//	agentflow <command> [flags]
//
// Commands:
//
//	runs     List recorded traces
//	show     Show a trace's runs and events
//	diff     Compare two traces effect by effect
//	export   Write a trace as a portable bundle
//	version  Print version information
//
// Replay needs your agent's code, so it isn't a CLI command: use
// replay.Run in your program, or agentflowtest.Cassette in tests.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

// Build-time variables set via ldflags.
var (
	version = "dev"
	commit  = "none"
)

const defaultStoreDir = ".agentflow"

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	dir := os.Getenv("AGENTFLOW_STORE_DIR")
	if dir == "" {
		dir = defaultStoreDir
	}
	var err error
	switch cmd, args := os.Args[1], os.Args[2:]; cmd {
	case "runs":
		err = cmdRuns(ctx, dir, args, os.Stdout)
	case "show":
		err = cmdShow(ctx, dir, args, os.Stdout)
	case "diff":
		err = cmdDiff(ctx, dir, args, os.Stdout)
	case "export":
		err = cmdExport(ctx, dir, args, os.Stdout)
	case "version":
		fmt.Printf("agentflow %s (%s)\n", version, commit)
	case "help", "-h", "--help":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", cmd)
		usage(os.Stderr)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage(w io.Writer) {
	_, _ = fmt.Fprintf(w, `agentflow inspects recorded agent traces.

Usage:
  agentflow runs [-all] [-n N]        List traces (newest first)
  agentflow show <id>                 Show a trace's runs and events
  agentflow diff <id-a> <id-b>        Compare two traces effect by effect
  agentflow export <id> [-o file]     Write a trace as a portable bundle
  agentflow version

<id> is a trace or run ID, or a unique prefix or suffix of one (at least 6
characters). IDs start with a timestamp, so a suffix is usually shortest.

Replay needs your agent's code: use replay.Run in your program, or
agentflowtest.Cassette in tests.

Environment:
  AGENTFLOW_STORE_DIR   Trace store directory (default: %s)
`, defaultStoreDir)
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

func open(dir string) (*filestore.Store, error) {
	if matches, _ := filepath.Glob(filepath.Join(dir, "runs", "*.jsonl")); len(matches) > 0 {
		fmt.Fprintf(os.Stderr, "note: %s holds prototype traces, which this version can't read; they are ignored.\n", filepath.Join(dir, "runs"))
	}
	return filestore.Open(dir, filestore.Options{})
}

func allRuns(ctx context.Context, s trace.Store, f trace.RunFilter) ([]trace.RunInfo, error) {
	var out []trace.RunInfo
	for info, err := range s.Runs(ctx, f) {
		if err != nil {
			return nil, err
		}
		out = append(out, info)
	}
	return out, nil
}

// resolve finds the trace that a trace ID, run ID, or a unique prefix or
// suffix of one names. IDs are UUIDv7, whose leading characters are a
// timestamp, so traces created close together share long prefixes; the
// random tail makes suffixes the practical short form.
func resolve(ctx context.Context, s trace.Store, id string) (string, error) {
	if len(id) < 6 {
		return "", fmt.Errorf("id %q is too short (use at least 6 characters)", id)
	}
	runs, err := allRuns(ctx, s, trace.RunFilter{})
	if err != nil {
		return "", err
	}
	matches := map[string]bool{}
	for _, r := range runs {
		if strings.HasPrefix(r.TraceID, id) || strings.HasPrefix(r.RunID, id) ||
			strings.HasSuffix(r.TraceID, id) || strings.HasSuffix(r.RunID, id) {
			matches[r.TraceID] = true
		}
	}
	switch len(matches) {
	case 0:
		return "", fmt.Errorf("no trace or run matches %q", id)
	case 1:
		for t := range matches {
			return t, nil
		}
	}
	ids := make([]string, 0, len(matches))
	for t := range matches {
		ids = append(ids, t)
	}
	sort.Strings(ids)
	return "", fmt.Errorf("%q is ambiguous; it matches traces:\n  %s", id, strings.Join(ids, "\n  "))
}

func cmdRuns(ctx context.Context, dir string, args []string, w io.Writer) error {
	fs := flag.NewFlagSet("runs", flag.ContinueOnError)
	all := fs.Bool("all", false, "include replay and fork traces")
	n := fs.Int("n", 20, "maximum traces to list")
	if err := fs.Parse(args); err != nil {
		return err
	}
	s, err := open(dir)
	if err != nil {
		return err
	}
	f := trace.RunFilter{RootsOnly: true, Limit: *n}
	if !*all {
		f.Origins = []trace.OriginKind{trace.OriginLive}
	}
	runs, err := allRuns(ctx, s, f)
	if err != nil {
		return err
	}
	if len(runs) == 0 {
		_, err := fmt.Fprintln(w, "No traces found.")
		return err
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	p := &printer{w: tw}
	p.printf("TRACE\tAGENT\tSTATUS\tORIGIN\tSTARTED\n")
	for _, r := range runs {
		origin := string(trace.OriginLive)
		if r.Origin != nil {
			origin = string(r.Origin.Kind)
			if r.Origin.SourceTraceID != "" {
				origin += " of " + shortID(r.Origin.SourceTraceID)
			}
		}
		p.printf("%s\t%s\t%s\t%s\t%s\n", r.TraceID, r.Agent, r.Status, origin, r.Started.Local().Format(time.DateTime))
	}
	if p.err != nil {
		return p.err
	}
	return tw.Flush()
}

func cmdShow(ctx context.Context, dir string, args []string, w io.Writer) error {
	if len(args) != 1 {
		return errors.New("usage: agentflow show <id>")
	}
	s, err := open(dir)
	if err != nil {
		return err
	}
	traceID, err := resolve(ctx, s, args[0])
	if err != nil {
		return err
	}
	runs, err := allRuns(ctx, s, trace.RunFilter{TraceID: traceID})
	if err != nil {
		return err
	}
	sort.SliceStable(runs, func(i, j int) bool { return runs[i].Path < runs[j].Path })
	p := &printer{w: w}
	p.printf("trace %s\n", traceID)
	for _, r := range runs {
		p.printf("\nrun %s  %s  agent=%s  status=%s\n", r.Path, r.RunID, r.Agent, r.Status)
		for e, err := range s.Events(ctx, r.RunID, 0) {
			if err != nil {
				return err
			}
			p.printf("  %4d  %-21s %s\n", e.Seq, e.Kind(), describe(e))
		}
	}
	return p.err
}

// describe summarizes an event on one line.
func describe(e trace.Event) string {
	switch b := e.Body.(type) {
	case *trace.RunStarted:
		s := fmt.Sprintf("task=%q", preview(llm.TextOf(b.Task)))
		if b.Origin != nil && b.Origin.Kind != trace.OriginLive {
			s += fmt.Sprintf(" origin=%s of %s", b.Origin.Kind, shortID(b.Origin.SourceTraceID))
		}
		return s
	case *trace.RunFinished:
		s := fmt.Sprintf("status=%s steps=%d tokens=%d/%d", b.Status, b.Steps, b.Usage.InputTokens, b.Usage.OutputTokens)
		if b.Error != nil {
			s += " error=" + preview(b.Error.Message)
		}
		return s
	case *trace.MessageAppended:
		if b.SourceSeq != nil {
			return fmt.Sprintf("assistant message from seq %d", *b.SourceSeq)
		}
		return fmt.Sprintf("%s %q", b.Message.Role, preview(blocksText(b.Message.Blocks)))
	case *trace.LLMStarted:
		return fmt.Sprintf("step=%d messages=%d ephemeral=%d", b.Step, b.Context.Messages, len(b.Ephemeral))
	case *trace.LLMCompleted:
		return fmt.Sprintf("stop=%s tokens=%d/%d %q", b.StopReason, b.Usage.InputTokens, b.Usage.OutputTokens, preview(blocksText(b.Message.Blocks)))
	case *trace.LLMFailed:
		return fmt.Sprintf("%s: %s", b.Error.Code, preview(b.Error.Message))
	case *trace.ToolStarted:
		return fmt.Sprintf("%s(%s) %s", b.Tool, b.CallID, preview(string(b.Input)))
	case *trace.ToolCompleted:
		s := fmt.Sprintf("%s %q", b.CallID, preview(llm.TextOf(b.Content)))
		if b.IsError {
			s += " (error result)"
		}
		return s
	case *trace.ToolFailed:
		return fmt.Sprintf("%s %s: %s", b.CallID, b.Error.Code, preview(b.Error.Message))
	case *trace.EffectStarted:
		return fmt.Sprintf("%s %s", b.Name, preview(string(b.Input)))
	case *trace.EffectCompleted:
		return fmt.Sprintf("%s %s", b.Name, preview(string(b.Output)))
	case *trace.EffectFailed:
		return fmt.Sprintf("%s %s: %s", b.Name, b.Error.Code, preview(b.Error.Message))
	case *trace.RunModeChanged:
		return fmt.Sprintf("now %s (%s)", b.Mode, b.Reason)
	case *trace.Annotation:
		return fmt.Sprintf("%s=%s", b.Key, preview(string(b.Value)))
	}
	raw, _ := json.Marshal(e.Body)
	return preview(string(raw))
}

func blocksText(bs llm.Blocks) string {
	var parts []string
	for _, b := range bs {
		switch v := b.(type) {
		case llm.Text:
			parts = append(parts, v.Text)
		case llm.ToolUse:
			parts = append(parts, fmt.Sprintf("[call %s]", v.Name))
		case llm.ToolResult:
			parts = append(parts, fmt.Sprintf("[result %s: %s]", v.CallID, llm.TextOf(v.Content)))
		case llm.Reasoning:
			parts = append(parts, "[reasoning]")
		default:
			parts = append(parts, "["+string(b.BlockType())+"]")
		}
	}
	return strings.Join(parts, " ")
}

func preview(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 80 {
		return s[:77] + "..."
	}
	return s
}

func shortID(id string) string {
	if len(id) > 13 {
		return id[:13]
	}
	return id
}

func cmdDiff(ctx context.Context, dir string, args []string, w io.Writer) error {
	if len(args) != 2 {
		return errors.New("usage: agentflow diff <id-a> <id-b>")
	}
	s, err := open(dir)
	if err != nil {
		return err
	}
	a, err := resolve(ctx, s, args[0])
	if err != nil {
		return err
	}
	b, err := resolve(ctx, s, args[1])
	if err != nil {
		return err
	}
	d, err := replay.DiffTraces(ctx, s, a, b)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(w, d.String())
	return err
}

func cmdExport(ctx context.Context, dir string, args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	out := fs.String("o", "", "output file (default: stdout)")
	if len(args) == 0 {
		return errors.New("usage: agentflow export <id> [-o file]")
	}
	id := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	s, err := open(dir)
	if err != nil {
		return err
	}
	traceID, err := resolve(ctx, s, id)
	if err != nil {
		return err
	}
	w := stdout
	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
		if err != nil {
			return err
		}
		defer func() { _ = f.Close() }()
		w = f
	}
	return trace.Export(ctx, w, s, traceID, trace.BundleOptions{})
}
