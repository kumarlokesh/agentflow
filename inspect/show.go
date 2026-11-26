package inspect

import (
	"fmt"
	"io"
	"strings"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// RenderTree writes the trace's run tree: one line per run with its kind,
// agent, status, steps and effects, then its failed expectations and other
// anomalies. Effect markers: ✗ failed or denied, ! error result, ⇢ override;
// ↯ marks where a run went live in a replay or fork, with the reason.
func RenderTree(w io.Writer, t *Trace) error {
	var total llm.Usage
	for _, r := range t.Runs {
		total = total.Add(r.Usage)
	}
	origin := string(trace.OriginLive)
	if o := t.Origin; o != nil {
		origin = string(o.Kind)
		if o.SourceTraceID != "" {
			origin += " of " + short(o.SourceTraceID)
		}
		if o.At != nil {
			origin += " at " + o.At.Path + " " + o.At.Effect
		}
		if o.Arm != "" && o.Sample != nil {
			origin += fmt.Sprintf(" (%s sample %d)", o.Arm, *o.Sample)
		}
	}
	type row struct {
		name string
		r    *Run
	}
	var rows []row
	var walk func(r *Run, prefix, branch string)
	walk = func(r *Run, prefix, branch string) {
		rows = append(rows, row{prefix + branch + r.Path, r})
		next := prefix
		switch branch {
		case "├─ ":
			next += "│  "
		case "└─ ":
			next += "   "
		}
		for i, c := range r.Children {
			b := "├─ "
			if i == len(r.Children)-1 {
				b = "└─ "
			}
			walk(c, next, b)
		}
	}
	walk(t.Root, "", "")
	// Runs not reachable from the root (a crashed parent) are listed too.
	reached := map[*Run]bool{}
	for _, rw := range rows {
		reached[rw.r] = true
	}
	for _, r := range t.Runs {
		if !reached[r] {
			rows = append(rows, row{r.Path + " (unlinked)", r})
		}
	}
	nameW, agentW := 0, 0
	for _, rw := range rows {
		nameW = max(nameW, runeLen(rw.name))
		agentW = max(agentW, runeLen(rw.r.Agent))
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "trace %s  %s  %s  tokens %d/%d\n", t.ID, origin, t.Root.Status, total.InputTokens, total.OutputTokens)
	for _, rw := range rows {
		r := rw.r
		steps := ""
		if r.Kind == trace.RunKindAgent {
			steps = fmt.Sprintf("%d step", r.Steps)
			if r.Steps != 1 {
				steps += "s"
			}
		}
		fmt.Fprintf(&sb, "%s  %-8s  %s  %-9s  %-8s  %s\n", pad(rw.name, nameW), r.Kind, pad(r.Agent, agentW), r.Status, steps, effectsLine(r))
		for _, a := range r.Anomalies {
			if a.Kind == Expectation || a.Kind == RunFailed || a.Kind == Divergence {
				fmt.Fprintf(&sb, "%s  ✗ %s: %s\n", strings.Repeat(" ", nameW), a.Kind, a.Summary)
			}
		}
	}
	_, err := io.WriteString(w, sb.String())
	return err
}

// maxTokens bounds a run's effects line; real sessions have thousands.
const maxTokens = 40

func effectsLine(r *Run) string {
	var parts []string
	placed := make([]bool, len(r.Modes))
	delegated := map[string]string{} // tool label → its child's segment
	for _, e := range r.Effects {
		if e.Kind == "spawn" && e.Parent != "" {
			delegated[e.Parent] = e.Child[strings.LastIndex(e.Child, "/")+1:]
		}
	}
	for _, e := range r.Effects {
		for i, m := range r.Modes {
			if !placed[i] && ((m.Cause != nil && *m.Cause == e.Seq) || (m.Cause == nil && m.Seq < e.Seq)) {
				parts, placed[i] = append(parts, "↯"+m.Reason), true
			}
		}
		if e.Parent != "" {
			continue
		}
		var tok string
		switch e.Kind {
		case "tool":
			tok = "tool:" + e.Name
			if c, ok := delegated[e.Label]; ok {
				tok += "→" + c
			}
		case "spawn":
			tok = "spawn:" + e.Child[strings.LastIndex(e.Child, "/")+1:]
		default:
			tok = e.Label
		}
		switch {
		case e.Failure != nil && e.Failure.Code != trace.ErrCodeCancelled:
			tok += "✗"
		case e.IsError:
			tok += "!"
		}
		if e.Override {
			tok += "⇢"
		}
		parts = append(parts, tok)
	}
	for i, m := range r.Modes {
		if !placed[i] {
			parts = append(parts, "↯"+m.Reason)
		}
	}
	if len(parts) > maxTokens {
		n := len(parts) - maxTokens
		parts = append(parts[:maxTokens:maxTokens], fmt.Sprintf("… %d more (show -events)", n))
	}
	return strings.Join(parts, " ")
}

func runeLen(s string) int { return len([]rune(s)) }

func pad(s string, n int) string {
	if d := n - runeLen(s); d > 0 {
		return s + strings.Repeat(" ", d)
	}
	return s
}
