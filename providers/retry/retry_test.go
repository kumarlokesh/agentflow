package retry_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/providers/retry"
)

type flaky struct {
	failures int
	err      error
	calls    atomic.Int32
}

func (f *flaky) Generate(context.Context, *llm.Request) (*llm.Response, error) {
	n := int(f.calls.Add(1))
	if n <= f.failures {
		return nil, f.err
	}
	return &llm.Response{StopReason: llm.StopEndTurn}, nil
}

var transient = &llm.Error{Code: llm.ErrCodeOverloaded, Message: "busy", Retryable: true}

func fast(m llm.Model, retries int) *retry.Model {
	return retry.Wrap(m, retry.WithMaxRetries(retries), retry.WithBaseDelay(time.Millisecond), retry.WithMaxDelay(2*time.Millisecond))
}

func TestRetriesTransientErrorsAndReportsAttempts(t *testing.T) {
	f := &flaky{failures: 2, err: transient}
	ctx, attempts := llm.WithAttemptCounter(context.Background())
	if _, err := fast(f, 3).Generate(ctx, &llm.Request{}); err != nil {
		t.Fatal(err)
	}
	if f.calls.Load() != 3 || attempts() != 3 {
		t.Errorf("calls %d, attempts %d, want 3", f.calls.Load(), attempts())
	}
}

func TestGivesUpWithTheLastError(t *testing.T) {
	f := &flaky{failures: 10, err: transient}
	_, err := fast(f, 2).Generate(context.Background(), &llm.Request{})
	var le *llm.Error
	if !errors.As(err, &le) || le.Code != llm.ErrCodeOverloaded {
		t.Fatalf("err = %v", err)
	}
	if f.calls.Load() != 3 {
		t.Errorf("calls = %d, want 3", f.calls.Load())
	}
}

func TestNonRetryableErrorsPassThrough(t *testing.T) {
	for _, err := range []error{&llm.Error{Code: llm.ErrCodeAuth, Message: "bad key"}, errors.New("plain")} {
		f := &flaky{failures: 10, err: err}
		if _, got := fast(f, 3).Generate(context.Background(), &llm.Request{}); !errors.Is(got, err) {
			t.Errorf("got %v, want %v", got, err)
		}
		if f.calls.Load() != 1 {
			t.Errorf("%v retried %d times", err, f.calls.Load())
		}
	}
}

func TestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := fast(&flaky{}, 3).Generate(ctx, &llm.Request{})
	var le *llm.Error
	if !errors.As(err, &le) || le.Code != llm.ErrCodeCancelled {
		t.Errorf("cancelled before start: %v", err)
	}

	ctx, cancel = context.WithCancel(context.Background())
	slow := retry.Wrap(&flaky{failures: 10, err: transient}, retry.WithBaseDelay(time.Hour), retry.WithMaxDelay(time.Hour))
	go func() { time.Sleep(10 * time.Millisecond); cancel() }()
	start := time.Now()
	if _, err := slow.Generate(ctx, &llm.Request{}); !errors.Is(err, transient) {
		t.Errorf("cancelled during backoff: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Error("backoff ignored cancellation")
	}
}
