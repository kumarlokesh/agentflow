package agentflow

import (
	"context"
	"encoding/json"
)

// Message represents a single message in a conversation with an LLM.
type Message struct {
	// Role is the message author: "system", "user", "assistant", or "tool".
	Role string `json:"role"`
	// Content is the text content of the message.
	Content string `json:"content"`
	// ToolCallID links a tool-result message back to the originating tool call.
	ToolCallID string `json:"tool_call_id,omitempty"`
	// ToolCalls holds tool invocation requests from an assistant message.
	ToolCalls []ToolCallRequest `json:"tool_calls,omitempty"`
}

// ToolCallRequest represents a single tool invocation requested by the LLM.
type ToolCallRequest struct {
	// ID uniquely identifies this tool call within a response.
	ID string `json:"id"`
	// Name is the name of the tool to invoke.
	Name string `json:"name"`
	// Arguments is the raw JSON arguments for the tool.
	Arguments json.RawMessage `json:"arguments"`
}

// LLMRequest is the input to a language model call.
type LLMRequest struct {
	Messages []Message    `json:"messages"`
	Tools    []ToolSchema `json:"tools,omitempty"`
}

// LLMResponse is the output from a language model call.
type LLMResponse struct {
	// Content is the text response (empty if the model chose to call tools).
	Content string `json:"content,omitempty"`
	// ToolCalls contains tool invocation requests from the model.
	ToolCalls []ToolCallRequest `json:"tool_calls,omitempty"`
	// Usage reports token consumption for this call.
	Usage *TokenUsage `json:"usage,omitempty"`
}

// LLM abstracts a language model capable of multi-turn conversation with tool use.
// Implementations must be safe for concurrent use.
type LLM interface {
	// ChatCompletion sends a conversation to the model and returns its response.
	// The request includes the message history and available tool schemas.
	ChatCompletion(ctx context.Context, req *LLMRequest) (*LLMResponse, error)
}
