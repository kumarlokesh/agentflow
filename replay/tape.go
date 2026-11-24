package replay

import (
	"encoding/json"
	"fmt"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// entry is one recorded effect.
type entry struct {
	key, label  string
	kind        agentflow.EffectKind
	step        int
	startedSeq  uint64
	fingerprint string
	view        any // the recorded request, as fingerprinted (for diffs)
	outcome     *agentflow.Outcome
	intercepted bool // an interceptor produced the outcome; not a world result
	consumed    bool
	// parent is the key of the enclosing tool call for a nested effect.
	// Nested effects are matched only while their tool call is verified.
	parent string
}

// tape is the recorded effects of one run, indexed by replay key (the
// effect labels of trace spec §10.1).
type tape struct {
	path    string
	entries []*entry
	byKey   map[string]*entry
	task    llm.Blocks
	status  trace.RunStatus
	output  llm.Blocks
	hasEnd  bool
}

// buildTape indexes a run's events. Effects nested in tool calls are indexed
// by their scoped keys but matched only when their tool call is verified;
// otherwise the tool call's outcome covers them.
func buildTape(events []trace.Event) (*tape, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("replay: empty run")
	}
	rs, ok := events[0].Body.(*trace.RunStarted)
	if !ok {
		return nil, fmt.Errorf("replay: run does not start with run.started")
	}
	t := &tape{path: rs.Path, byKey: map[string]*entry{}, task: rs.Task}
	bySeq := map[uint64]*entry{}
	toolKeys := map[uint64]string{} // tool.started seq → key
	llmN := 0
	effectN := map[string]int{}
	toolN := map[string]int{}
	f := trace.NewFolder()
	add := func(e *entry, seq uint64) error {
		if _, dup := t.byKey[e.key]; dup {
			return fmt.Errorf("replay: duplicate effect key %s in recording", e.key)
		}
		e.startedSeq = seq
		t.entries = append(t.entries, e)
		t.byKey[e.key] = e
		bySeq[seq] = e
		return nil
	}
	for _, ev := range events {
		switch b := ev.Body.(type) {
		case *trace.LLMStarted:
			req, err := f.Request(b)
			if err != nil {
				return nil, fmt.Errorf("replay: rebuild recorded request at seq %d: %w", ev.Seq, err)
			}
			key := fmt.Sprintf("llm#%d", llmN)
			llmN++
			if err := add(&entry{key: key, label: key, kind: agentflow.EffectLLM, step: b.Step, fingerprint: b.Fingerprint, view: trace.FingerprintView(req)}, ev.Seq); err != nil {
				return nil, err
			}
		case *trace.ToolStarted:
			key := agentflow.ToolKey(b.CallID, toolN[b.CallID])
			toolKeys[ev.Seq] = key
			toolN[b.CallID]++
			view := map[string]any{"tool": b.Tool, "input": b.Input}
			if err := add(&entry{key: key, label: fmt.Sprintf("tool:%s(%s)", b.Tool, b.CallID), kind: agentflow.EffectTool, step: b.Step, fingerprint: b.Fingerprint, view: view}, ev.Seq); err != nil {
				return nil, err
			}
		case *trace.EffectStarted:
			var parent string
			if ev.Cause != nil {
				parent = toolKeys[*ev.Cause]
			}
			prefix := ""
			if parent != "" {
				prefix = parent + "/"
			}
			key := fmt.Sprintf("%seffect:%s#%d", prefix, b.Name, effectN[prefix+b.Name])
			effectN[prefix+b.Name]++
			view := map[string]any{"name": b.Name, "input": withConversation(b.Input, f.Conversation())}
			if err := add(&entry{key: key, label: key, kind: agentflow.EffectCustom, fingerprint: b.Fingerprint, view: view, parent: parent}, ev.Seq); err != nil {
				return nil, err
			}
		case *trace.LLMCompleted:
			setOutcome(bySeq, ev, &agentflow.Outcome{Response: &llm.Response{Message: b.Message, StopReason: b.StopReason, Usage: b.Usage, Model: b.Model, Ext: b.Ext}}, b.Intercepted)
		case *trace.LLMFailed:
			setOutcome(bySeq, ev, &agentflow.Outcome{Failure: &b.Error}, b.Intercepted)
		case *trace.ToolCompleted:
			setOutcome(bySeq, ev, &agentflow.Outcome{Response: agentflow.ToolOutput{Content: b.Content, IsError: b.IsError}}, b.Intercepted)
		case *trace.ToolFailed:
			setOutcome(bySeq, ev, &agentflow.Outcome{Failure: &b.Error}, b.Intercepted)
		case *trace.EffectCompleted:
			setOutcome(bySeq, ev, &agentflow.Outcome{Response: append(json.RawMessage(nil), b.Output...)}, b.Intercepted)
		case *trace.EffectFailed:
			setOutcome(bySeq, ev, &agentflow.Outcome{Failure: &b.Error}, b.Intercepted)
		case *trace.RunFinished:
			t.status, t.output, t.hasEnd = b.Status, b.Output, true
		}
		if err := f.Apply(ev); err != nil {
			return nil, fmt.Errorf("replay: %w", err)
		}
	}
	return t, nil
}

func setOutcome(bySeq map[uint64]*entry, ev trace.Event, o *agentflow.Outcome, intercepted bool) {
	if ev.Cause == nil {
		return
	}
	if e, ok := bySeq[*ev.Cause]; ok {
		e.outcome, e.intercepted = o, intercepted
	}
}

// withConversation returns a context provider's input with the conversation
// fingerprint replaced by the conversation itself, so a mismatch can be
// shown message by message. Other inputs are returned unchanged.
func withConversation(input json.RawMessage, conv []llm.Message) any {
	var m map[string]json.RawMessage
	if json.Unmarshal(input, &m) != nil {
		return input
	}
	var fp string
	if raw, ok := m["conversation"]; !ok || json.Unmarshal(raw, &fp) != nil {
		return input
	}
	if want, err := trace.Fingerprint(conv); err != nil || want != fp {
		return input // not a fingerprint of this conversation
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	if conv == nil {
		conv = []llm.Message{}
	}
	out["conversation"] = conv
	return out
}
