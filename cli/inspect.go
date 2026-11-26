package cli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/kumarlokesh/agentflow/inspect"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

func (c *env) runs(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("runs", flag.ContinueOnError)
	all := fs.Bool("all", false, "include replay and fork traces")
	n := fs.Int("n", 20, "maximum traces to list")
	if _, err := parse(fs, args, 0, "runs [-all] [-n N]"); err != nil {
		return err
	}
	s, err := c.open()
	if err != nil {
		return err
	}
	f := trace.RunFilter{RootsOnly: true, Limit: *n}
	if !*all {
		f.Origins = []trace.OriginKind{trace.OriginLive}
	}
	var runs []trace.RunInfo
	for info, err := range s.Runs(ctx, f) {
		if err != nil {
			return err
		}
		runs = append(runs, info)
	}
	if len(runs) == 0 {
		_, err := fmt.Fprintln(c.out, "No traces found.")
		return err
	}
	tw := tabwriter.NewWriter(c.out, 0, 4, 2, ' ', 0)
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

func (c *env) show(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("show", flag.ContinueOnError)
	events := fs.Bool("events", false, "list every event instead of the run tree")
	only := fs.String("run", "", "with -events, only the run at this path")
	pos, err := parse(fs, args, 1, "show <id> [-events] [-run path]")
	if err != nil {
		return err
	}
	s, traceID, err := c.resolve(ctx, pos[0])
	if err != nil {
		return err
	}
	t, err := inspect.Load(ctx, s, traceID)
	if err != nil {
		return err
	}
	if !*events {
		return inspect.RenderTree(c.out, t)
	}
	p := &printer{w: c.out}
	p.printf("trace %s\n", traceID)
	found := false
	for _, r := range t.Runs {
		if *only != "" && r.Path != *only {
			continue
		}
		found = true
		p.printf("\nrun %s  %s  agent=%s  status=%s\n", r.Path, r.RunID, r.Agent, r.Status)
		for _, e := range r.Events {
			p.printf("  %4d  %-21s %s\n", e.Seq, e.Kind(), describe(e))
		}
	}
	if !found {
		return fmt.Errorf("no run at %s", *only)
	}
	return p.err
}

func (c *env) diff(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("diff", flag.ContinueOnError)
	exact := fs.Bool("exact", false, "compare call IDs, reasoning, provider extensions and usage too")
	pos, err := parse(fs, args, 2, "diff <id-a> <id-b> [-exact]")
	if err != nil {
		return err
	}
	s, a, err := c.resolve(ctx, pos[0])
	if err != nil {
		return err
	}
	_, b, err := c.resolve(ctx, pos[1])
	if err != nil {
		return err
	}
	d, err := inspect.Diff(ctx, s, a, b, inspect.DiffOptions{Exact: *exact})
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(c.out, d.String())
	return err
}

func (c *env) blame(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("blame", flag.ContinueOnError)
	at := fs.String("at", "", "the symptom: a run path, or <path>@<seq> (default: the root's failure)")
	pos, err := parse(fs, args, 1, "blame <id> [-at path[@seq]]")
	if err != nil {
		return err
	}
	s, traceID, err := c.resolve(ctx, pos[0])
	if err != nil {
		return err
	}
	t, err := inspect.Load(ctx, s, traceID)
	if err != nil {
		return err
	}
	b, err := inspect.Blame(t, *at)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(c.out, b.String())
	return err
}

func (c *env) export(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	out := fs.String("o", "", "output file (default: stdout)")
	html := fs.Bool("html", false, "write one self-contained HTML page (with the blame result) instead of a bundle")
	pos, err := parse(fs, args, 1, "export <id> [-html] [-o file]")
	if err != nil {
		return err
	}
	s, traceID, err := c.resolve(ctx, pos[0])
	if err != nil {
		return err
	}
	w := c.out
	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
		if err != nil {
			return err
		}
		defer func() { _ = f.Close() }()
		w = f
	}
	if !*html {
		return trace.Export(ctx, w, s, traceID, trace.BundleOptions{})
	}
	t, err := inspect.Load(ctx, s, traceID)
	if err != nil {
		return err
	}
	b, err := inspect.Blame(t, "")
	if err != nil {
		return err
	}
	return inspect.WriteHTML(w, t, b)
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
	case *trace.RunSpawned:
		return fmt.Sprintf("%s %s task=%q", b.Path, b.Agent, preview(llm.TextOf(b.Task)))
	case *trace.RunJoined:
		s := fmt.Sprintf("status=%s tokens=%d/%d %q", b.Status, b.Usage.InputTokens, b.Usage.OutputTokens, preview(llm.TextOf(b.Output)))
		if b.Error != nil {
			s += fmt.Sprintf(" error=%s: %s", b.Error.Code, preview(b.Error.Message))
		}
		if b.Intercepted {
			s += " (intercepted)"
		}
		return s
	case *trace.RunModeChanged:
		return fmt.Sprintf("now %s (%s)", b.Mode, b.Reason)
	case *trace.ConfigChanged:
		raw, _ := json.Marshal(b)
		return preview(string(raw))
	case *trace.ConversationReplaced:
		return fmt.Sprintf("%d messages", len(b.Messages))
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
