// Package retry provides an agentflow.LLM wrapper that retries transient
// failures with exponential backoff and jitter.
//
// Usage:
//
//	base := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"))
//	llm := retry.Wrap(base,
//	    retry.WithMaxRetries(5),
//	    retry.WithBaseDelay(500*time.Millisecond),
//	)
package retry

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"time"

	"github.com/kumarlokesh/agentflow"
)

const (
	defaultMaxRetries = 3
	defaultBaseDelay  = 1 * time.Second
	defaultMaxDelay   = 60 * time.Second
)

// Retryable is implemented by API errors that know whether they are transient.
// Both anthropic.APIError and openai.APIError implement this interface.
type Retryable interface {
	IsRetryable() bool
}

// RetryLLM wraps an agentflow.LLM with exponential backoff retry logic.
// It is safe for concurrent use.
type RetryLLM struct {
	inner      agentflow.LLM
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
	logger     *slog.Logger
}

// Option configures a RetryLLM.
type Option func(*RetryLLM)

// WithMaxRetries sets the maximum number of retry attempts (not counting the
// initial attempt). Default is 3.
func WithMaxRetries(n int) Option {
	return func(r *RetryLLM) { r.maxRetries = n }
}

// WithBaseDelay sets the initial backoff delay. Default is 1s.
func WithBaseDelay(d time.Duration) Option {
	return func(r *RetryLLM) { r.baseDelay = d }
}

// WithMaxDelay caps the backoff delay. Default is 60s.
func WithMaxDelay(d time.Duration) Option {
	return func(r *RetryLLM) { r.maxDelay = d }
}

// WithLogger sets a structured logger for retry events.
func WithLogger(l *slog.Logger) Option {
	return func(r *RetryLLM) { r.logger = l }
}

// Wrap returns a RetryLLM that wraps llm with the given options.
func Wrap(llm agentflow.LLM, opts ...Option) *RetryLLM {
	r := &RetryLLM{
		inner:      llm,
		maxRetries: defaultMaxRetries,
		baseDelay:  defaultBaseDelay,
		maxDelay:   defaultMaxDelay,
		logger:     slog.Default(),
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// Ensure RetryLLM implements agentflow.LLM at compile time.
var _ agentflow.LLM = (*RetryLLM)(nil)

// ChatCompletion calls the wrapped LLM and retries on transient errors using
// exponential backoff with full jitter. Non-retryable errors pass through
// immediately without retry.
func (r *RetryLLM) ChatCompletion(ctx context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	var lastErr error
	for attempt := 0; attempt <= r.maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("retry: context cancelled before attempt %d: %w", attempt, err)
		}

		resp, err := r.inner.ChatCompletion(ctx, req)
		if err == nil {
			return resp, nil
		}

		// Check if the error is retryable.
		if re, ok := err.(Retryable); !ok || !re.IsRetryable() {
			return nil, err
		}

		lastErr = err

		if attempt == r.maxRetries {
			break
		}

		delay := r.backoffDelay(attempt)
		r.logger.Warn("retrying LLM call after transient error",
			"attempt", attempt+1,
			"max_retries", r.maxRetries,
			"delay", delay,
			"error", err,
		)

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("retry: context cancelled during backoff: %w", ctx.Err())
		case <-time.After(delay):
		}
	}

	return nil, fmt.Errorf("retry: all %d attempts failed: %w", r.maxRetries+1, lastErr)
}

// backoffDelay computes the delay for attempt n using exponential backoff
// with full jitter: delay = random(0, min(maxDelay, baseDelay * 2^n)).
func (r *RetryLLM) backoffDelay(attempt int) time.Duration {
	exp := r.baseDelay
	for i := 0; i < attempt; i++ {
		exp *= 2
		if exp > r.maxDelay {
			exp = r.maxDelay
			break
		}
	}
	// Full jitter: random value in [0, exp).
	jitter := time.Duration(rand.Int64N(int64(exp)))
	return jitter
}
