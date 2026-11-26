package trace

import (
	"encoding/json"
	"fmt"
)

// EffectRecord is one effect of a run: its start event, its answer, and its
// label (spec §10.1).
type EffectRecord struct {
	// Label is the effect's label, which is also its replay key: "llm#0",
	// "tool:<call_id>" (with "#<k>" for a repeated call ID),
	// "effect:<name>#<n>", "spawn:<path>", or a nested label such as
	// "tool:<call_id>/effect:<name>#<n>".
	Label string
	// Kind is "llm", "tool", "effect" or "spawn".
	Kind string
	// Parent is the label of the enclosing tool call of a nested effect.
	Parent string
	// Step is the model step of an llm or tool effect.
	Step int
	// Position is a tool call's position in the response that requested
	// it, from 0.
	Position int
	Start    Event
	// End is the start's answer, or nil if it has none (an incomplete run).
	End *Event
}

// Effects returns a run's effects (events in seq order, from seq 0), in the
// order they started.
func Effects(events []Event) []EffectRecord {
	var out []EffectRecord
	bySeq := map[uint64]int{} // start seq → index in out
	toolLabels := map[uint64]string{}
	positions := map[uint64]int{} // llm.completed seq → tool calls so far
	llmN, toolN, effectN := 0, map[string]int{}, map[string]int{}
	add := func(r EffectRecord) {
		bySeq[r.Start.Seq] = len(out)
		out = append(out, r)
	}
	parentOf := func(e Event) string {
		if e.Cause != nil {
			return toolLabels[*e.Cause]
		}
		return ""
	}
	nested := func(parent, label string) string {
		if parent == "" {
			return label
		}
		return parent + "/" + label
	}
	for _, e := range events {
		switch b := e.Body.(type) {
		case *LLMStarted:
			add(EffectRecord{Label: fmt.Sprintf("llm#%d", llmN), Kind: "llm", Step: b.Step, Start: e})
			llmN++
		case *ToolStarted:
			label := "tool:" + b.CallID
			if k := toolN[b.CallID]; k > 0 {
				label = fmt.Sprintf("tool:%s#%d", b.CallID, k)
			}
			toolN[b.CallID]++
			toolLabels[e.Seq] = label
			pos := 0
			if e.Cause != nil {
				pos = positions[*e.Cause]
				positions[*e.Cause]++
			}
			add(EffectRecord{Label: label, Kind: "tool", Step: b.Step, Position: pos, Start: e})
		case *EffectStarted:
			parent := parentOf(e)
			scoped := nested(parent, b.Name)
			add(EffectRecord{Label: nested(parent, fmt.Sprintf("effect:%s#%d", b.Name, effectN[scoped])), Kind: "effect", Parent: parent, Start: e})
			effectN[scoped]++
		case *RunSpawned:
			parent := parentOf(e)
			add(EffectRecord{Label: nested(parent, "spawn:"+b.Path), Kind: "spawn", Parent: parent, Start: e})
		case *LLMCompleted, *LLMFailed, *ToolCompleted, *ToolFailed, *EffectCompleted, *EffectFailed, *RunJoined:
			if e.Cause == nil {
				continue
			}
			if i, ok := bySeq[*e.Cause]; ok && out[i].End == nil {
				ev := e
				out[i].End = &ev
			}
		}
	}
	return out
}

// AnnotationExpectation is the annotation key of an expectation: a
// pass/fail check recorded by the application (agentflow.Expect).
const AnnotationExpectation = "agentflow.expectation"

// Expectation is the value of an expectation annotation.
type Expectation struct {
	Name   string `json:"name"`
	Pass   bool   `json:"pass"`
	Detail string `json:"detail,omitempty"`
}

// ExpectationOf returns the expectation an annotation records, if it is one.
func ExpectationOf(a *Annotation) (Expectation, bool) {
	if a == nil || a.Key != AnnotationExpectation {
		return Expectation{}, false
	}
	var x Expectation
	if err := json.Unmarshal(a.Value, &x); err != nil {
		return Expectation{}, false
	}
	return x, true
}
