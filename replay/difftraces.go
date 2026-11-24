package replay

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/trace"
)

// EntryDiff is one difference between two traces' effects.
type EntryDiff struct {
	// Effect is the effect's replay label, e.g. "llm#1" or "tool:calc(t1)".
	Effect string
	// What differs: "request", "outcome", "only_in_a", "only_in_b", or "output".
	What  string
	Diffs []FieldDiff
}

// TraceDiff compares the root runs of two traces effect by effect.
type TraceDiff struct {
	A, B    string
	Entries []EntryDiff
}

// Identical reports whether no differences were found.
func (d *TraceDiff) Identical() bool { return len(d.Entries) == 0 }

// DiffTraces compares the root runs of traces a and b, aligning effects by
// replay key (llm#n, tool call, effect name#n). Tool calls are aligned by
// call ID, so two independent live runs (whose providers assign different
// call IDs) show their tool calls as only_in_a / only_in_b. Call IDs are
// not normalized yet.
func DiffTraces(ctx context.Context, s trace.Store, a, b string) (*TraceDiff, error) {
	ta, err := rootTape(ctx, s, a)
	if err != nil {
		return nil, err
	}
	tb, err := rootTape(ctx, s, b)
	if err != nil {
		return nil, err
	}
	out := &TraceDiff{A: a, B: b}
	seen := map[string]bool{}
	for _, ea := range ta.entries {
		seen[ea.key] = true
		eb, ok := tb.byKey[ea.key]
		if !ok {
			out.Entries = append(out.Entries, EntryDiff{Effect: ea.label, What: "only_in_a"})
			continue
		}
		if ea.fingerprint != eb.fingerprint {
			da, _ := json.Marshal(ea.view)
			db, _ := json.Marshal(eb.view)
			diffs, err := diffJSON(da, db)
			if err != nil {
				return nil, err
			}
			out.Entries = append(out.Entries, EntryDiff{Effect: ea.label, What: "request", Diffs: diffs})
		}
		oa, _ := json.Marshal(outcomeView(ea))
		ob, _ := json.Marshal(outcomeView(eb))
		if string(oa) != string(ob) {
			diffs, err := diffJSON(oa, ob)
			if err != nil {
				return nil, err
			}
			out.Entries = append(out.Entries, EntryDiff{Effect: ea.label, What: "outcome", Diffs: diffs})
		}
	}
	for _, eb := range tb.entries {
		if !seen[eb.key] {
			out.Entries = append(out.Entries, EntryDiff{Effect: eb.label, What: "only_in_b"})
		}
	}
	oa, _ := json.Marshal(map[string]any{"status": ta.status, "output": ta.output})
	ob, _ := json.Marshal(map[string]any{"status": tb.status, "output": tb.output})
	if string(oa) != string(ob) {
		diffs, err := diffJSON(oa, ob)
		if err != nil {
			return nil, err
		}
		out.Entries = append(out.Entries, EntryDiff{Effect: "run", What: "output", Diffs: diffs})
	}
	return out, nil
}

// outcomeView is the comparable part of an outcome: without usage, model
// version, and provider metadata, which differ between runs that behaved
// the same way.
func outcomeView(e *entry) any {
	if e.outcome == nil {
		return nil
	}
	if e.outcome.Failure != nil {
		return map[string]any{"failure": e.outcome.Failure.Code, "message": e.outcome.Failure.Message}
	}
	switch r := e.outcome.Response.(type) {
	case json.RawMessage:
		return map[string]any{"output": r}
	case *agentflow.RunResult:
		return map[string]any{"status": r.Status, "output": r.Output}
	default:
		raw, _ := json.Marshal(r)
		var m map[string]any
		_ = json.Unmarshal(raw, &m)
		delete(m, "usage")
		delete(m, "model")
		delete(m, "ext")
		return m
	}
}

func rootTape(ctx context.Context, s trace.Store, traceID string) (*tape, error) {
	var root *trace.RunInfo
	for info, err := range s.Runs(ctx, trace.RunFilter{TraceID: traceID, RootsOnly: true}) {
		if err != nil {
			return nil, err
		}
		info := info
		root = &info
	}
	if root == nil {
		return nil, fmt.Errorf("replay: trace %s: %w", traceID, trace.ErrRunNotFound)
	}
	events, err := trace.Collect(s.Events(ctx, root.RunID, 0))
	if err != nil {
		return nil, err
	}
	return buildTape(events, s)
}

// String renders the diff for terminals.
func (d *TraceDiff) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "diff %s %s: ", short(d.A), short(d.B))
	if d.Identical() {
		sb.WriteString("identical")
		return sb.String()
	}
	fmt.Fprintf(&sb, "%d difference(s)", len(d.Entries))
	for _, e := range d.Entries {
		fmt.Fprintf(&sb, "\n\n%s — %s", e.Effect, strings.ReplaceAll(e.What, "_", " "))
		for _, f := range e.Diffs {
			switch {
			case f.TextDiff != "":
				fmt.Fprintf(&sb, "\n  %s %s:\n%s", f.Path, f.Op, indent(f.TextDiff, "    "))
			case f.Op == OpAdded:
				fmt.Fprintf(&sb, "\n  %s only in B: %s", f.Path, clip(f.Current))
			case f.Op == OpRemoved:
				fmt.Fprintf(&sb, "\n  %s only in A: %s", f.Path, clip(f.Recorded))
			default:
				fmt.Fprintf(&sb, "\n  %s:\n    A %s\n    B %s", f.Path, clip(f.Recorded), clip(f.Current))
			}
		}
	}
	return sb.String()
}

func short(id string) string {
	if len(id) > 13 {
		return id[:13]
	}
	return id
}
