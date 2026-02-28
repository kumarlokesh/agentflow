package agentflow

import (
	"errors"
	"fmt"
)

// Sentinel errors returned by the agent runtime.
var (
	// ErrMaxStepsExceeded is returned when the agent loop exceeds its step limit.
	ErrMaxStepsExceeded = errors.New("agentflow: max steps exceeded")
	// ErrToolNotFound is returned when the agent tries to call an unregistered tool.
	ErrToolNotFound = errors.New("agentflow: tool not found")
	// ErrInvalidToolParams is returned when tool parameters fail schema validation.
	ErrInvalidToolParams = errors.New("agentflow: invalid tool parameters")
	// ErrNoLLM is returned when an agent is started without an LLM configured.
	ErrNoLLM = errors.New("agentflow: no LLM configured")
	// ErrRunCancelled is returned when context cancellation stops a run.
	ErrRunCancelled = errors.New("agentflow: run cancelled")
	// ErrReplayMismatch is returned when replay produces different events than recorded.
	ErrReplayMismatch = errors.New("agentflow: replay mismatch")
	// ErrReplayExhausted is returned when replay runs out of recorded events.
	ErrReplayExhausted = errors.New("agentflow: no more recorded events for replay")
)

// ToolError wraps an error that occurred during tool execution, preserving the
// tool name and call ID for diagnostics.
type ToolError struct {
	ToolName string
	CallID   string
	Err      error
}

func (e *ToolError) Error() string {
	return fmt.Sprintf("agentflow: tool %q (call %s): %v", e.ToolName, e.CallID, e.Err)
}

func (e *ToolError) Unwrap() error { return e.Err }

// LLMError wraps an error from the language model layer.
type LLMError struct {
	Err error
}

func (e *LLMError) Error() string {
	return fmt.Sprintf("agentflow: llm: %v", e.Err)
}

func (e *LLMError) Unwrap() error { return e.Err }

// StoreError wraps an error from the event store layer.
type StoreError struct {
	Op  string // e.g. "append", "load"
	Err error
}

func (e *StoreError) Error() string {
	return fmt.Sprintf("agentflow: store %s: %v", e.Op, e.Err)
}

func (e *StoreError) Unwrap() error { return e.Err }
