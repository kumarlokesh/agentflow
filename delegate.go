package agentflow

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/kumarlokesh/agentflow/llm"
)

// DelegateOptions configure a delegate tool.
type DelegateOptions struct {
	// ToolName is the name the model sees. Default: the agent's name.
	ToolName string
	// Description tells the model when to delegate.
	Description string
	// Output maps the child's result to the tool result. err is the child's
	// *RunError, or nil if it completed. It must be a pure function of its
	// arguments: replay re-applies it to the replayed child's result. The
	// default returns the child's output, or an error result naming how the
	// child ended.
	Output func(res *RunResult, err error) ToolOutput
}

// Delegate exposes e to the model as a tool with input {"task": string}.
// Each call starts a child run of e on the task, in the caller's tree, and
// returns the child's output as the tool result. The child has its own
// conversation: it sees the task, never the caller's history.
//
// The runtime knows a delegate tool starts exactly one child whose result
// is the tool's result, so a replay runs the child against its own
// recording instead of answering the tool call from the parent's.
func Delegate(e Entrypoint, opts DelegateOptions) Tool {
	name := opts.ToolName
	if name == "" {
		name = e.Name()
	}
	desc := opts.Description
	if desc == "" {
		desc = fmt.Sprintf("Delegate a task to the %s agent and get its answer.", e.Name())
	}
	out := opts.Output
	if out == nil {
		out = defaultDelegateOutput
	}
	return &delegateTool{e: e, output: out, spec: llm.ToolSpec{
		Name: name, Description: desc,
		InputSchema: json.RawMessage(`{"type":"object","properties":{"task":{"type":"string","description":"The task for the agent, with everything it needs to know."}},"required":["task"]}`),
	}}
}

type delegateTool struct {
	e      Entrypoint
	spec   llm.ToolSpec
	output func(*RunResult, error) ToolOutput
}

func (d *delegateTool) Spec() llm.ToolSpec { return d.spec }

func (d *delegateTool) Call(ctx context.Context, input json.RawMessage) (ToolOutput, error) {
	var in struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return ErrorOutput(fmt.Sprintf("invalid input: %v", err)), nil
	}
	res, err := Spawn(ctx, d.e, in.Task)
	if res == nil {
		return ToolOutput{}, err
	}
	return d.output(res, err), nil
}

func defaultDelegateOutput(res *RunResult, err error) ToolOutput {
	if err != nil {
		return ErrorOutput(fmt.Sprintf("%s did not complete: %v", res.Path, err))
	}
	content := append(llm.Blocks{}, res.Output...)
	return ToolOutput{Content: content}
}
