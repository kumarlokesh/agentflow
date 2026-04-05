package retry_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/providers/retry"
)

// --- Test helpers ---

type mockLLM struct {
	calls    atomic.Int32
	fn       func(attempt int) (*agentflow.LLMResponse, error)
}

func (m *mockLLM) ChatCompletion(ctx context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	n := int(m.calls.Add(1))
	return m.fn(n)
}

type retryableError struct{ msg string }

func (e *retryableError) Error() string    { return e.msg }
func (e *retryableError) IsRetryable() bool { return true }

type fatalError struct{ msg string }

func (e *fatalError) Error() string    { return e.msg }
func (e *fatalError) IsRetryable() bool { return false }

func successResp() *agentflow.LLMResponse {
	return &agentflow.LLMResponse{Content: "ok"}
}

// --- Tests ---

func TestRetry_SuccessOnFirstAttempt(t *testing.T) {
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		return successResp(), nil
	}}
	llm := retry.Wrap(mock, retry.WithMaxRetries(3), retry.WithBaseDelay(time.Millisecond))

	resp, err := llm.ChatCompletion(context.Background(), &agentflow.LLMRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("content = %q, want ok", resp.Content)
	}
	if mock.calls.Load() != 1 {
		t.Errorf("calls = %d, want 1", mock.calls.Load())
	}
}

func TestRetry_RetryThenSuccess(t *testing.T) {
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		if attempt < 3 {
			return nil, &retryableError{"transient"}
		}
		return successResp(), nil
	}}
	llm := retry.Wrap(mock, retry.WithMaxRetries(5), retry.WithBaseDelay(time.Millisecond))

	resp, err := llm.ChatCompletion(context.Background(), &agentflow.LLMRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("content = %q, want ok", resp.Content)
	}
	if mock.calls.Load() != 3 {
		t.Errorf("calls = %d, want 3", mock.calls.Load())
	}
}

func TestRetry_MaxRetriesExhausted(t *testing.T) {
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		return nil, &retryableError{"always fails"}
	}}
	llm := retry.Wrap(mock, retry.WithMaxRetries(2), retry.WithBaseDelay(time.Millisecond))

	_, err := llm.ChatCompletion(context.Background(), &agentflow.LLMRequest{})
	if err == nil {
		t.Fatal("expected error after max retries")
	}
	// 1 initial + 2 retries = 3 total calls
	if mock.calls.Load() != 3 {
		t.Errorf("calls = %d, want 3", mock.calls.Load())
	}
}

func TestRetry_NonRetryablePassesThroughImmediately(t *testing.T) {
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		return nil, &fatalError{"auth failed"}
	}}
	llm := retry.Wrap(mock, retry.WithMaxRetries(3), retry.WithBaseDelay(time.Millisecond))

	_, err := llm.ChatCompletion(context.Background(), &agentflow.LLMRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	// Should NOT retry — only 1 call.
	if mock.calls.Load() != 1 {
		t.Errorf("calls = %d, want 1 (non-retryable must not retry)", mock.calls.Load())
	}
	var fe *fatalError
	if !errors.As(err, &fe) {
		t.Errorf("expected *fatalError passthrough, got %T: %v", err, err)
	}
}

func TestRetry_PlainErrorNotRetried(t *testing.T) {
	// A plain error (not implementing Retryable) should not be retried.
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		return nil, errors.New("plain error")
	}}
	llm := retry.Wrap(mock, retry.WithMaxRetries(3), retry.WithBaseDelay(time.Millisecond))

	_, err := llm.ChatCompletion(context.Background(), &agentflow.LLMRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if mock.calls.Load() != 1 {
		t.Errorf("calls = %d, want 1 (plain error must not retry)", mock.calls.Load())
	}
}

func TestRetry_ContextCancelledBeforeStart(t *testing.T) {
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		return successResp(), nil
	}}
	llm := retry.Wrap(mock, retry.WithMaxRetries(3), retry.WithBaseDelay(time.Millisecond))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := llm.ChatCompletion(ctx, &agentflow.LLMRequest{})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestRetry_ContextCancelledDuringBackoff(t *testing.T) {
	mock := &mockLLM{fn: func(attempt int) (*agentflow.LLMResponse, error) {
		return nil, &retryableError{"transient"}
	}}
	// Long base delay so we reliably cancel during backoff.
	llm := retry.Wrap(mock, retry.WithMaxRetries(5), retry.WithBaseDelay(10*time.Second))

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := llm.ChatCompletion(ctx, &agentflow.LLMRequest{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error")
	}
	// Should have cancelled well before all retries completed.
	if elapsed > 5*time.Second {
		t.Errorf("took %v, expected cancellation within 200ms", elapsed)
	}
}
