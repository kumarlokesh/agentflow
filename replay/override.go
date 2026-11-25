package replay

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// ErrInvalidOverride is returned by Fork, before anything runs, for an
// override that doesn't fit the effect at the fork point or that contains
// reasoning blocks.
var ErrInvalidOverride = errors.New("replay: invalid fork override")

// Override is a change a fork applies at its fork point: SetConfig,
// InjectMessage, SetTask, ReplaceOutcome or StripReasoning. Overrides are
// plain data; they are recorded in each fork sample's origin, so a fork is
// reproducible and its traces replayable.
type Override interface {
	overrideType() string
}

// SetConfig changes the target run's model configuration from the fork
// point (a model call, "llm#n") on. Zero fields are unchanged. Tools can
// only be removed: their implementations are code.
type SetConfig struct {
	System       llm.Blocks  `json:"system,omitempty"`
	Model        string      `json:"model,omitempty"`
	Params       *llm.Params `json:"params,omitempty"`
	DisableTools []string    `json:"disable_tools,omitempty"`
}

// InjectMessage adds a user message just before the model call at the
// fork point, e.g. a human correction or a hint.
type InjectMessage struct {
	Message llm.Message `json:"message"`
}

// SetTask starts the child run at the fork point (a spawn,
// "spawn:<path>") with a different task.
type SetTask struct {
	Task llm.Blocks `json:"task"`
}

// ReplaceOutcome substitutes the outcome of the effect at the fork point,
// which then isn't executed. Set the field that fits the effect: Message (a
// model call), ToolOutput (a tool call), Output (a custom effect),
// ChildOutput (a spawn: the child doesn't run); or Error, for a failure of
// any kind.
type ReplaceOutcome struct {
	Message     *llm.Message          `json:"message,omitempty"`
	ToolOutput  *agentflow.ToolOutput `json:"tool_output,omitempty"`
	Output      json.RawMessage       `json:"output,omitempty"`
	ChildOutput llm.Blocks            `json:"child_output,omitempty"`
	Error       *trace.Error          `json:"error,omitempty"`
}

// StripReasoning removes reasoning blocks from the target run's
// conversation before the model call at the fork point (for switching to a
// model that rejects another model's reasoning). ForkOptions.StripReasoning
// adds it.
type StripReasoning struct{}

func (SetConfig) overrideType() string      { return "set_config" }
func (InjectMessage) overrideType() string  { return "inject_message" }
func (SetTask) overrideType() string        { return "set_task" }
func (ReplaceOutcome) overrideType() string { return "replace_outcome" }
func (StripReasoning) overrideType() string { return "strip_reasoning" }

// MarshalOverrides encodes overrides as recorded in a fork origin: a JSON
// array of objects with a "type".
func MarshalOverrides(ovs []Override) (json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(ovs))
	for _, o := range ovs {
		raw, err := json.Marshal(o)
		if err != nil {
			return nil, err
		}
		var m map[string]json.RawMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
		if m == nil {
			m = map[string]json.RawMessage{}
		}
		m["type"], _ = json.Marshal(o.overrideType())
		b, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return json.Marshal(out)
}

// ParseOverrides decodes overrides recorded by MarshalOverrides.
func ParseOverrides(raw json.RawMessage) ([]Override, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("replay: overrides: %w", err)
	}
	out := make([]Override, 0, len(items))
	for _, it := range items {
		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(it, &head); err != nil {
			return nil, fmt.Errorf("replay: overrides: %w", err)
		}
		var o Override
		var err error
		switch head.Type {
		case "set_config":
			var v SetConfig
			err, o = json.Unmarshal(it, &v), &v
		case "inject_message":
			var v InjectMessage
			err, o = json.Unmarshal(it, &v), &v
		case "set_task":
			var v SetTask
			err, o = json.Unmarshal(it, &v), &v
		case "replace_outcome":
			var v ReplaceOutcome
			err, o = json.Unmarshal(it, &v), &v
		case "strip_reasoning":
			o = &StripReasoning{}
		default:
			return nil, fmt.Errorf("replay: overrides: unknown type %q", head.Type)
		}
		if err != nil {
			return nil, fmt.Errorf("replay: override %s: %w", head.Type, err)
		}
		out = append(out, deref(o))
	}
	return out, nil
}

func deref(o Override) Override {
	switch v := o.(type) {
	case *SetConfig:
		return *v
	case *InjectMessage:
		return *v
	case *SetTask:
		return *v
	case *ReplaceOutcome:
		return *v
	case *StripReasoning:
		return *v
	}
	return o
}

// kindOfKey returns the kind of the effect a replay key names.
func kindOfKey(key string) agentflow.EffectKind {
	if strings.Contains(key, "spawn:") {
		return agentflow.EffectSpawn
	}
	last := key[strings.LastIndex(key, "/")+1:]
	switch {
	case strings.HasPrefix(last, "llm#"):
		return agentflow.EffectLLM
	case strings.HasPrefix(last, "tool:"):
		return agentflow.EffectTool
	}
	return agentflow.EffectCustom
}

// toPatch validates overrides for an effect of this kind and turns them
// into the runtime's Patch.
func toPatch(ovs []Override, kind agentflow.EffectKind) (*agentflow.Patch, error) {
	if len(ovs) == 0 {
		return nil, nil
	}
	bad := func(o Override, why string) error {
		return fmt.Errorf("%w: %s at a %s effect: %s", ErrInvalidOverride, o.overrideType(), kind, why)
	}
	p := &agentflow.Patch{}
	for _, o := range ovs {
		if hasReasoning(o) {
			return nil, bad(o, "overrides can't contain reasoning blocks (their signatures can't be forged)")
		}
		switch v := o.(type) {
		case SetConfig:
			if kind != agentflow.EffectLLM {
				return nil, bad(o, "applies to a model call (llm#n)")
			}
			cp := &agentflow.ConfigPatch{Params: v.Params, DisableTools: v.DisableTools}
			if v.System != nil {
				sys := v.System
				cp.System = &sys
			}
			if v.Model != "" {
				m := v.Model
				cp.Model = &m
			}
			p.Config = cp
		case InjectMessage:
			if kind != agentflow.EffectLLM {
				return nil, bad(o, "applies to a model call (llm#n)")
			}
			m := v.Message
			m.Role = llm.RoleUser
			p.Message = &m
		case StripReasoning:
			if kind != agentflow.EffectLLM {
				return nil, bad(o, "applies to a model call (llm#n)")
			}
			p.StripReasoning = true
		case SetTask:
			if kind != agentflow.EffectSpawn {
				return nil, bad(o, "applies to a spawn (spawn:<path>)")
			}
			p.Task = v.Task
		case ReplaceOutcome:
			out, err := v.outcome(kind)
			if err != nil {
				return nil, bad(o, err.Error())
			}
			p.Outcome = out
		default:
			return nil, fmt.Errorf("%w: unknown override %T", ErrInvalidOverride, o)
		}
	}
	return p, nil
}

func (v ReplaceOutcome) outcome(kind agentflow.EffectKind) (*agentflow.Outcome, error) {
	set := 0
	for _, b := range []bool{v.Message != nil, v.ToolOutput != nil, v.Output != nil, v.ChildOutput != nil, v.Error != nil} {
		if b {
			set++
		}
	}
	if set != 1 {
		return nil, errors.New("set exactly one of Message, ToolOutput, Output, ChildOutput or Error")
	}
	if v.Error != nil {
		return &agentflow.Outcome{Failure: v.Error}, nil
	}
	switch {
	case kind == agentflow.EffectLLM && v.Message != nil:
		m := *v.Message
		m.Role = llm.RoleAssistant
		stop := llm.StopEndTurn
		for _, b := range m.Blocks {
			if _, ok := b.(llm.ToolUse); ok {
				stop = llm.StopToolUse
			}
		}
		return &agentflow.Outcome{Response: &llm.Response{Message: m, StopReason: stop}}, nil
	case kind == agentflow.EffectTool && v.ToolOutput != nil:
		return &agentflow.Outcome{Response: *v.ToolOutput}, nil
	case kind == agentflow.EffectCustom && v.Output != nil:
		return &agentflow.Outcome{Response: append(json.RawMessage(nil), v.Output...)}, nil
	case kind == agentflow.EffectSpawn && v.ChildOutput != nil:
		return &agentflow.Outcome{Response: &agentflow.RunResult{Status: trace.StatusCompleted, Output: v.ChildOutput}}, nil
	}
	return nil, errors.New("the field set doesn't fit this effect kind (Message: llm, ToolOutput: tool, Output: custom effect, ChildOutput: spawn)")
}

func hasReasoning(o Override) bool {
	var blocks []llm.Blocks
	switch v := o.(type) {
	case SetConfig:
		blocks = append(blocks, v.System)
	case InjectMessage:
		blocks = append(blocks, v.Message.Blocks)
	case SetTask:
		blocks = append(blocks, v.Task)
	case ReplaceOutcome:
		if v.Message != nil {
			blocks = append(blocks, v.Message.Blocks)
		}
		if v.ToolOutput != nil {
			blocks = append(blocks, v.ToolOutput.Content)
		}
		blocks = append(blocks, v.ChildOutput)
	}
	for _, bs := range blocks {
		for _, b := range bs {
			if _, ok := b.(llm.Reasoning); ok {
				return true
			}
		}
	}
	return false
}

// mergePatch returns a with b's fields applied over it (b wins).
func mergePatch(a, b *agentflow.Patch) *agentflow.Patch {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	out := *a
	if b.Config != nil {
		out.Config = b.Config
	}
	out.StripReasoning = a.StripReasoning || b.StripReasoning
	if b.Message != nil {
		out.Message = b.Message
	}
	if b.Task != nil {
		out.Task = b.Task
	}
	if b.Outcome != nil {
		out.Outcome = b.Outcome
	}
	return &out
}
