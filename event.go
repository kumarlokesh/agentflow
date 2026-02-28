// Package agentflow provides a deterministic, observable runtime for building
// production-grade AI agents with event sourcing and deterministic replay.
package agentflow

import (
	"encoding/json"
	"fmt"
	"time"
)

// SchemaVersion is the current event schema version. Bumped when the Event
// struct changes in a backwards-incompatible way. Stored in every event to
// enable forward-compatible replay across versions.
const SchemaVersion = 1

// EventType identifies the kind of event recorded during an agent run.
type EventType string

const (
	// EventLLMRequest is emitted before calling the language model.
	EventLLMRequest EventType = "llm_request"
	// EventLLMResponse is emitted after receiving a language model response.
	EventLLMResponse EventType = "llm_response"
	// EventToolCall is emitted before executing a tool.
	EventToolCall EventType = "tool_call"
	// EventToolResult is emitted after a tool execution completes.
	EventToolResult EventType = "tool_result"
	// EventStepStart is emitted at the beginning of each agent loop iteration.
	EventStepStart EventType = "step_start"
	// EventStepEnd is emitted at the end of each agent loop iteration.
	EventStepEnd EventType = "step_end"
	// EventRunStart is emitted when an agent run begins.
	EventRunStart EventType = "run_start"
	// EventRunEnd is emitted when an agent run finishes (success or failure).
	EventRunEnd EventType = "run_end"
	// EventError is emitted when a non-fatal error occurs during execution.
	EventError EventType = "error"
)

// Event is the fundamental unit of the event log. Every action the agent takes
// — every LLM call, tool execution, and state transition — is recorded as an
// Event. This append-only log enables deterministic replay: given the same
// sequence of events, the agent produces identical behavior.
type Event struct {
	// ID is a unique identifier for this event (UUID v4).
	ID string `json:"id"`
	// SchemaVersion tracks the event format version for forward compatibility.
	SchemaVersion int `json:"schema_version"`
	// Type identifies what kind of event this is.
	Type EventType `json:"type"`
	// RunID links this event to a specific agent run.
	RunID string `json:"run_id"`
	// StepIndex is the zero-based index of the agent loop iteration.
	StepIndex int `json:"step_index"`
	// Timestamp is when the event was created (UTC).
	Timestamp time.Time `json:"timestamp"`
	// Data holds the type-specific event payload as raw JSON.
	Data json.RawMessage `json:"data"`
	// Metadata holds optional key-value pairs for extensibility.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// NewEvent creates a new Event with a generated ID, current timestamp, and
// the current schema version. The data parameter is JSON-marshalled into the
// Data field.
func NewEvent(eventType EventType, runID string, stepIndex int, data any) (Event, error) {
	raw, err := json.Marshal(data)
	if err != nil {
		return Event{}, fmt.Errorf("agentflow: marshal event data: %w", err)
	}
	return Event{
		ID:            newUUID(),
		SchemaVersion: SchemaVersion,
		Type:          eventType,
		RunID:         runID,
		StepIndex:     stepIndex,
		Timestamp:     nowUTC(),
		Data:          raw,
	}, nil
}

// --- Typed event data payloads ---

// LLMRequestData is the payload for EventLLMRequest.
type LLMRequestData struct {
	Messages []Message    `json:"messages"`
	Tools    []ToolSchema `json:"tools,omitempty"`
}

// LLMResponseData is the payload for EventLLMResponse.
type LLMResponseData struct {
	Content   string            `json:"content,omitempty"`
	ToolCalls []ToolCallRequest `json:"tool_calls,omitempty"`
	Usage     *TokenUsage       `json:"usage,omitempty"`
}

// ToolCallData is the payload for EventToolCall.
type ToolCallData struct {
	ToolName string          `json:"tool_name"`
	CallID   string          `json:"call_id"`
	Input    json.RawMessage `json:"input"`
}

// ToolResultData is the payload for EventToolResult.
type ToolResultData struct {
	ToolName string        `json:"tool_name"`
	CallID   string        `json:"call_id"`
	Output   string        `json:"output"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"duration_ns"`
}

// RunStartData is the payload for EventRunStart.
type RunStartData struct {
	Task         string   `json:"task"`
	Instructions string   `json:"instructions,omitempty"`
	Tools        []string `json:"tools"`
	MaxSteps     int      `json:"max_steps"`
}

// RunEndData is the payload for EventRunEnd.
type RunEndData struct {
	Status   string        `json:"status"` // "completed", "failed", "cancelled"
	Output   string        `json:"output,omitempty"`
	Error    string        `json:"error,omitempty"`
	Steps    int           `json:"steps"`
	Duration time.Duration `json:"duration_ns"`
}

// StepStartData is the payload for EventStepStart.
type StepStartData struct {
	StepIndex int `json:"step_index"`
}

// StepEndData is the payload for EventStepEnd.
type StepEndData struct {
	StepIndex int           `json:"step_index"`
	Duration  time.Duration `json:"duration_ns"`
}

// ErrorData is the payload for EventError.
type ErrorData struct {
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
}

// TokenUsage tracks token consumption for a single LLM call.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// DecodeData unmarshals the event's Data field into the provided target.
func (e *Event) DecodeData(target any) error {
	if e.Data == nil {
		return fmt.Errorf("agentflow: event %s has nil data", e.ID)
	}
	return json.Unmarshal(e.Data, target)
}
