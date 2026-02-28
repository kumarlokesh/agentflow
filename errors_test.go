package agentflow

import (
	"errors"
	"fmt"
	"testing"
)

func TestToolError(t *testing.T) {
	inner := fmt.Errorf("connection timeout")
	te := &ToolError{
		ToolName: "web_search",
		CallID:   "call-123",
		Err:      inner,
	}

	// Error string includes tool name and call ID.
	got := te.Error()
	if got == "" {
		t.Fatal("Error() returned empty string")
	}

	// Unwrap returns the inner error.
	if !errors.Is(te, inner) {
		t.Error("Unwrap() does not match inner error")
	}
}

func TestLLMError(t *testing.T) {
	inner := fmt.Errorf("rate limited")
	le := &LLMError{Err: inner}

	got := le.Error()
	if got == "" {
		t.Fatal("Error() returned empty string")
	}

	if !errors.Is(le, inner) {
		t.Error("Unwrap() does not match inner error")
	}
}

func TestStoreError(t *testing.T) {
	inner := fmt.Errorf("disk full")
	se := &StoreError{Op: "append", Err: inner}

	got := se.Error()
	if got == "" {
		t.Fatal("Error() returned empty string")
	}

	if !errors.Is(se, inner) {
		t.Error("Unwrap() does not match inner error")
	}
}

func TestSentinelErrors(t *testing.T) {
	// Verify sentinel errors are distinct and non-nil.
	sentinels := []error{
		ErrMaxStepsExceeded,
		ErrToolNotFound,
		ErrInvalidToolParams,
		ErrNoLLM,
		ErrRunCancelled,
		ErrReplayMismatch,
		ErrReplayExhausted,
	}

	for _, err := range sentinels {
		if err == nil {
			t.Error("sentinel error is nil")
		}
	}

	// All should be unique.
	seen := make(map[string]bool)
	for _, err := range sentinels {
		msg := err.Error()
		if seen[msg] {
			t.Errorf("duplicate sentinel error message: %q", msg)
		}
		seen[msg] = true
	}
}
