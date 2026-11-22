package llm

import (
	"context"
	"encoding/json"
	"fmt"
)

// Model is a language model backend. Implementations must be safe for
// concurrent use.
type Model interface {
	Generate(ctx context.Context, req *Request) (*Response, error)
}

// Request is the input to one model call.
type Request struct {
	Model      string      `json:"model"`
	System     Blocks      `json:"system"`
	Messages   []Message   `json:"messages"`
	Tools      []ToolSpec  `json:"tools"`
	ToolChoice *ToolChoice `json:"tool_choice"`
	Params     Params      `json:"params"`
}

// Params are sampling and generation parameters. Pointer fields distinguish
// "not set" from a zero value.
type Params struct {
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature *float64         `json:"temperature,omitempty"`
	TopP        *float64         `json:"top_p,omitempty"`
	Stop        []string         `json:"stop,omitempty"`
	Reasoning   *ReasoningParams `json:"reasoning,omitempty"`
	Ext         Ext              `json:"ext,omitempty"`
}

// ReasoningParams configure model reasoning ("thinking").
type ReasoningParams struct {
	Effort       string `json:"effort,omitempty"`        // provider-specific, e.g. "low" | "medium" | "high"
	BudgetTokens int    `json:"budget_tokens,omitempty"` // provider-specific token budget
}

// ToolChoice constrains which tool the model may call.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode"`
	Name string         `json:"name,omitempty"` // for ToolChoiceTool
}

// ToolChoiceMode is how the model may use tools.
type ToolChoiceMode string

const (
	ToolChoiceAuto ToolChoiceMode = "auto"
	ToolChoiceAny  ToolChoiceMode = "any"
	ToolChoiceNone ToolChoiceMode = "none"
	ToolChoiceTool ToolChoiceMode = "tool"
)

// ToolSpec describes a tool to the model.
type ToolSpec struct {
	Name        string           `json:"name"`
	Description string           `json:"description"`
	InputSchema json.RawMessage  `json:"input_schema"`
	Annotations *ToolAnnotations `json:"annotations,omitempty"`
}

// ToolAnnotations are hints about a tool's behavior, aligned with MCP tool
// annotations. They are not sent to the model.
type ToolAnnotations struct {
	ReadOnly    bool `json:"read_only,omitempty"`
	Destructive bool `json:"destructive,omitempty"`
	Idempotent  bool `json:"idempotent,omitempty"`
}

// Response is the output of one model call.
type Response struct {
	Message    Message    `json:"message"` // Role is RoleAssistant
	StopReason StopReason `json:"stop_reason"`
	Usage      Usage      `json:"usage"`
	Model      string     `json:"model,omitempty"` // as reported by the provider
	Ext        Ext        `json:"ext,omitempty"`   // e.g. provider request id
}

// StopReason is why the model stopped generating.
type StopReason string

const (
	StopEndTurn      StopReason = "end_turn"
	StopToolUse      StopReason = "tool_use"
	StopMaxTokens    StopReason = "max_tokens"
	StopStopSequence StopReason = "stop_sequence"
	StopRefusal      StopReason = "refusal"
	StopOther        StopReason = "other"
)

// Usage is token consumption. ReasoningTokens are a subset of OutputTokens.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
}

// Add returns the sum of two usages.
func (u Usage) Add(o Usage) Usage {
	return Usage{
		InputTokens:      u.InputTokens + o.InputTokens,
		OutputTokens:     u.OutputTokens + o.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens + o.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens + o.CacheWriteTokens,
		ReasoningTokens:  u.ReasoningTokens + o.ReasoningTokens,
	}
}

// Total is input plus output tokens.
func (u Usage) Total() int { return u.InputTokens + u.OutputTokens }

// Error codes for model call failures.
const (
	ErrCodeRateLimited    = "rate_limited"
	ErrCodeOverloaded     = "overloaded"
	ErrCodeInvalidRequest = "invalid_request"
	ErrCodeAuth           = "auth"
	ErrCodeContextLength  = "context_length"
	ErrCodeServer         = "server"
	ErrCodeNetwork        = "network"
	ErrCodeCancelled      = "cancelled"
)

// Error is returned by providers so that failures can be recorded with a
// structured code.
type Error struct {
	Code      string
	Message   string
	Retryable bool
	Status    int // HTTP status, if any
}

func (e *Error) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("llm: %s (status %d): %s", e.Code, e.Status, e.Message)
	}
	return fmt.Sprintf("llm: %s: %s", e.Code, e.Message)
}
