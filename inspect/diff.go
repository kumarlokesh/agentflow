package inspect

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
)

// DiffOptions configure Diff.
type DiffOptions struct {
	// Exact compares everything recorded except IDs, times and durations:
	// tool call IDs, reasoning, provider extensions and usage included.
	Exact bool
}

// TreeDiff compares two traces run by run.
type TreeDiff struct {
	A, B string
	Runs []RunDiff // in path order; only runs that differ
}

// RunDiff is how one run differs between two traces.
type RunDiff struct {
	Path string
	// OnlyIn is "a" or "b" for a run present in one trace only.
	OnlyIn  string
	Entries []EntryDiff
}

// EntryDiff is one difference within a run.
type EntryDiff struct {
	// Effect is the normalized label ("tool@0.1" for the second tool call
	// requested at step 0), or "run" for the run's status and output.
	Effect string
	// What differs: "request", "outcome", "only_in_a", "only_in_b", "output".
	What  string
	Diffs []replay.FieldDiff
}

// Identical reports whether no differences were found.
func (d *TreeDiff) Identical() bool { return len(d.Runs) == 0 }

// Diff compares the traces a and b in s. Runs are matched by path, effects
// by label, with tool calls identified by their model step and position in
// the response rather than by call ID, so two independent live runs that
// behaved the same compare identical. Unless opts.Exact, provider noise
// (reasoning, signatures and other extensions, usage, model versions) is
// ignored too.
func Diff(ctx context.Context, s trace.Store, a, b string, opts DiffOptions) (*TreeDiff, error) {
	ta, err := Load(ctx, s, a)
	if err != nil {
		return nil, err
	}
	tb, err := Load(ctx, s, b)
	if err != nil {
		return nil, err
	}
	out := &TreeDiff{A: a, B: b}
	paths := map[string]bool{}
	for _, r := range ta.Runs {
		paths[r.Path] = true
	}
	for _, r := range tb.Runs {
		paths[r.Path] = true
	}
	sorted := make([]string, 0, len(paths))
	for p := range paths {
		sorted = append(sorted, p)
	}
	sort.Strings(sorted)
	for _, p := range sorted {
		ra, rb := ta.Find(p), tb.Find(p)
		switch {
		case rb == nil:
			out.Runs = append(out.Runs, RunDiff{Path: p, OnlyIn: "a"})
		case ra == nil:
			out.Runs = append(out.Runs, RunDiff{Path: p, OnlyIn: "b"})
		default:
			d, err := diffRun(ra, rb, opts)
			if err != nil {
				return nil, err
			}
			if len(d.Entries) > 0 {
				out.Runs = append(out.Runs, d)
			}
		}
	}
	return out, nil
}

// entryView is an effect reduced to what Diff compares.
type entryView struct {
	label            string
	request, outcome []byte
}

func diffRun(a, b *Run, opts DiffOptions) (RunDiff, error) {
	d := RunDiff{Path: a.Path}
	va, err := views(a, opts)
	if err != nil {
		return d, err
	}
	vb, err := views(b, opts)
	if err != nil {
		return d, err
	}
	byLabel := map[string]entryView{}
	for _, e := range vb {
		byLabel[e.label] = e
	}
	seen := map[string]bool{}
	for _, ea := range va {
		seen[ea.label] = true
		eb, ok := byLabel[ea.label]
		if !ok {
			d.Entries = append(d.Entries, EntryDiff{Effect: ea.label, What: "only_in_a"})
			continue
		}
		for _, part := range []struct {
			what string
			x, y []byte
		}{{"request", ea.request, eb.request}, {"outcome", ea.outcome, eb.outcome}} {
			if string(part.x) == string(part.y) {
				continue
			}
			diffs, err := replay.DiffJSON(part.x, part.y)
			if err != nil {
				return d, err
			}
			if len(diffs) > 0 {
				d.Entries = append(d.Entries, EntryDiff{Effect: ea.label, What: part.what, Diffs: diffs})
			}
		}
	}
	for _, eb := range vb {
		if !seen[eb.label] {
			d.Entries = append(d.Entries, EntryDiff{Effect: eb.label, What: "only_in_b"})
		}
	}
	ra, _ := json.Marshal(map[string]any{"status": a.Status, "output": a.Output})
	rb, _ := json.Marshal(map[string]any{"status": b.Status, "output": b.Output})
	if string(ra) != string(rb) {
		diffs, err := replay.DiffJSON(ra, rb)
		if err != nil {
			return d, err
		}
		d.Entries = append(d.Entries, EntryDiff{Effect: "run", What: "output", Diffs: diffs})
	}
	return d, nil
}

// views reduces a run's effects to normalized requests and outcomes.
func views(r *Run, opts DiffOptions) ([]entryView, error) {
	// Tool calls are named by model step and position, not call ID.
	callNames := map[string]string{}  // call ID → call@<step>.<pos>
	labelNames := map[string]string{} // tool label → tool@<step>.<pos>
	for _, e := range r.Effects {
		if e.Kind == "tool" && e.Parent == "" {
			n := fmt.Sprintf("%d.%d", e.Step, e.Position)
			callNames[strings.TrimPrefix(strings.SplitN(e.Label, "#", 2)[0], "tool:")] = "call@" + n
			labelNames[e.Label] = "tool@" + n
		}
	}
	normLabel := func(l string) string {
		if opts.Exact {
			return l
		}
		head, rest, nested := strings.Cut(l, "/")
		if n, ok := labelNames[head]; ok {
			if nested {
				return n + "/" + rest
			}
			return n
		}
		return l
	}
	norm := func(v any) ([]byte, error) {
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		var x any
		if err := json.Unmarshal(raw, &x); err != nil {
			return nil, err
		}
		if !opts.Exact {
			x = normalize(x, callNames)
		}
		return json.Marshal(x)
	}
	// A model call's view is its configuration and the messages added to
	// the conversation since the previous call: a difference shows once, at
	// the call where it entered the conversation, and long sessions diff in
	// linear time.
	f := trace.NewFolder()
	reqs := map[uint64]any{}
	shown := 0
	for _, e := range r.Events {
		switch b := e.Body.(type) {
		case *trace.LLMStarted:
			req, err := f.Request(b)
			if err != nil {
				return nil, err
			}
			conv := f.Conversation()
			n := min(b.Context.Messages, len(conv))
			if n < shown {
				shown = 0
			}
			added := conv[shown:n]
			if added == nil {
				added = []llm.Message{}
			}
			reqs[e.Seq] = map[string]any{"model": req.Model, "system": req.System, "tools": req.Tools, "tool_choice": req.ToolChoice,
				"params": req.Params, "ephemeral": b.Ephemeral, "new_messages": added}
			shown = n
		case *trace.ConversationReplaced:
			shown = 0
		}
		if err := f.Apply(e); err != nil {
			return nil, err
		}
	}
	out := make([]entryView, 0, len(r.Effects))
	for _, e := range r.Effects {
		var request, outcome any
		switch e.Kind {
		case "llm":
			request = reqs[e.Seq]
		case "tool":
			request = map[string]any{"tool": e.Name, "input": e.Input}
		case "effect":
			request = map[string]any{"name": e.Name, "input": e.Input}
		case "spawn":
			request = map[string]any{"agent": e.Name, "child": e.Child, "task": e.Input}
		}
		switch {
		case !e.Done:
			outcome = nil
		case e.Failure != nil:
			outcome = map[string]any{"error": map[string]any{"code": e.Failure.Code, "message": e.Failure.Message}}
		case e.Kind == "tool":
			outcome = map[string]any{"content": e.Output, "is_error": e.IsError}
		default:
			outcome = map[string]any{"output": e.Output}
		}
		if opts.Exact && e.Kind == "llm" {
			outcome = map[string]any{"output": e.Output, "usage": e.Usage, "model": e.Name}
		}
		rq, err := norm(request)
		if err != nil {
			return nil, err
		}
		oc, err := norm(outcome)
		if err != nil {
			return nil, err
		}
		out = append(out, entryView{label: normLabel(e.Label), request: rq, outcome: oc})
	}
	return out, nil
}

// normalize replaces tool call IDs by their step and position, and drops
// what differs between runs that behaved the same: reasoning blocks and
// provider extensions.
func normalize(v any, calls map[string]string) any {
	switch x := v.(type) {
	case map[string]any:
		if t, _ := x["type"].(string); t == string(llm.TypeReasoning) {
			return nil
		}
		out := map[string]any{}
		for k, val := range x {
			if k == "ext" {
				continue
			}
			if s, ok := val.(string); ok && k == "call_id" {
				if n, ok := calls[s]; ok {
					val = n
				}
			}
			out[k] = normalize(val, calls)
		}
		return out
	case []any:
		out := make([]any, 0, len(x))
		for _, e := range x {
			if n := normalize(e, calls); n != nil || e == nil {
				out = append(out, n)
			}
		}
		return out
	}
	return v
}

// String renders the diff for terminals.
func (d *TreeDiff) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "diff %s %s: ", short(d.A), short(d.B))
	if d.Identical() {
		sb.WriteString("identical")
		return sb.String()
	}
	n := 0
	for _, r := range d.Runs {
		n += max(len(r.Entries), 1)
	}
	fmt.Fprintf(&sb, "%d difference(s) in %d run(s)", n, len(d.Runs))
	for _, r := range d.Runs {
		switch r.OnlyIn {
		case "a":
			fmt.Fprintf(&sb, "\n\nrun %s only in A", r.Path)
			continue
		case "b":
			fmt.Fprintf(&sb, "\n\nrun %s only in B", r.Path)
			continue
		}
		fmt.Fprintf(&sb, "\n\nrun %s", r.Path)
		for _, e := range r.Entries {
			fmt.Fprintf(&sb, "\n  %s — %s", e.Effect, strings.ReplaceAll(e.What, "_", " "))
			writeDiffs(&sb, e.Diffs, "    ")
		}
	}
	return sb.String()
}

func writeDiffs(sb *strings.Builder, diffs []replay.FieldDiff, pad string) {
	const maxDiffs = 10
	for i, f := range diffs {
		if i == maxDiffs {
			fmt.Fprintf(sb, "\n%s… and %d more", pad, len(diffs)-maxDiffs)
			return
		}
		switch {
		case f.TextDiff != "":
			fmt.Fprintf(sb, "\n%s%s %s:\n%s%s", pad, f.Path, f.Op, pad+"  ", strings.ReplaceAll(f.TextDiff, "\n", "\n"+pad+"  "))
		case f.Op == replay.OpAdded:
			fmt.Fprintf(sb, "\n%s%s only in B: %s", pad, f.Path, clipRaw(f.Current))
		case f.Op == replay.OpRemoved:
			fmt.Fprintf(sb, "\n%s%s only in A: %s", pad, f.Path, clipRaw(f.Recorded))
		default:
			fmt.Fprintf(sb, "\n%s%s:\n%s  A %s\n%s  B %s", pad, f.Path, pad, clipRaw(f.Recorded), pad, clipRaw(f.Current))
		}
	}
}

func clipRaw(raw json.RawMessage) string { return clip(string(raw), 200) }

func short(id string) string {
	if len(id) > 13 {
		return id[:13]
	}
	return id
}
