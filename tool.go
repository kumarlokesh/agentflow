package agentflow

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/kumarlokesh/agentflow/llm"
)

// Tool is a capability the model can call. Implementations must be safe for
// concurrent use when the agent's MaxParallelTools is above 1.
type Tool interface {
	Spec() llm.ToolSpec
	// Call runs the tool. Return ToolOutput.IsError for errors the model
	// should see and react to. A non-nil error is a tool failure, recorded
	// as tool.failed; the model is told the tool failed.
	Call(ctx context.Context, input json.RawMessage) (ToolOutput, error)
}

// ToolOutput is a tool's result as seen by the model.
type ToolOutput struct {
	Content llm.Blocks `json:"content"`
	IsError bool       `json:"is_error,omitempty"`
}

// TextOutput returns a successful text result.
func TextOutput(text string) ToolOutput {
	return ToolOutput{Content: llm.Blocks{llm.Text{Text: text}}}
}

// ErrorOutput returns a text result that tells the model the call failed.
func ErrorOutput(text string) ToolOutput {
	return ToolOutput{Content: llm.Blocks{llm.Text{Text: text}}, IsError: true}
}

// NewTool builds a Tool from a function with a typed input. The input JSON
// is decoded into In; decoding errors are reported to the model.
func NewTool[In any](spec llm.ToolSpec, fn func(ctx context.Context, in In) (ToolOutput, error)) Tool {
	return &funcTool[In]{spec: spec, fn: fn}
}

type funcTool[In any] struct {
	spec llm.ToolSpec
	fn   func(context.Context, In) (ToolOutput, error)
}

func (t *funcTool[In]) Spec() llm.ToolSpec { return t.spec }

func (t *funcTool[In]) Call(ctx context.Context, input json.RawMessage) (ToolOutput, error) {
	var in In
	if len(input) > 0 {
		if err := json.Unmarshal(input, &in); err != nil {
			return ErrorOutput(fmt.Sprintf("invalid input: %v", err)), nil
		}
	}
	return t.fn(ctx, in)
}
