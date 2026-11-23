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
}

// tape is the recorded effects of one run, indexed by replay key
// (RFC-0002 §4).
type tape struct {
	path    string
	entries []*entry
	byKey   map[string]*entry
	task    llm.Blocks
	status  trace.RunStatus
	output  llm.Blocks
	hasEnd  bool
}

// buildTape indexes a run's events. Effects nested in tool calls are left
// out: they are covered by their tool call (RFC-0002 amendment, 2026-11-23).
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
	toolStarts := map[uint64]bool{}
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
			toolStarts[ev.Seq] = true
			key := agentflow.ToolKey(b.CallID, toolN[b.CallID])
			toolN[b.CallID]++
			view := map[string]any{"tool": b.Tool, "input": b.Input}
			if err := add(&entry{key: key, label: fmt.Sprintf("tool:%s(%s)", b.Tool, b.CallID), kind: agentflow.EffectTool, step: b.Step, fingerprint: b.Fingerprint, view: view}, ev.Seq); err != nil {
				return nil, err
			}
		case *trace.EffectStarted:
			if ev.Cause != nil && toolStarts[*ev.Cause] {
				break // nested in a tool call: covered by it
			}
			key := fmt.Sprintf("effect:%s#%d", b.Name, effectN[b.Name])
			effectN[b.Name]++
			view := map[string]any{"name": b.Name, "input": b.Input}
			if err := add(&entry{key: key, label: key, kind: agentflow.EffectCustom, fingerprint: b.Fingerprint, view: view}, ev.Seq); err != nil {
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
