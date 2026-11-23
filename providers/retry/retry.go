// Package retry wraps an llm.Model so that transient failures (errors that
// are *llm.Error with Retryable set) are retried with exponential backoff
// and full jitter.
//
// Retries happen inside one recorded effect: the trace records only the
// final outcome, with the number of attempts (via llm.ReportAttempt).
//
// Usage:
//
//	model := retry.Wrap(anthropic.New(key), retry.WithMaxRetries(5))
package retry

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
)

const (
	defaultMaxRetries = 3
	defaultBaseDelay  = 1 * time.Second
	defaultMaxDelay   = 60 * time.Second
)

// Model wraps an llm.Model with retries. It is safe for concurrent use.
type Model struct {
	inner      llm.Model
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
	logger     *slog.Logger
}

var _ llm.Model = (*Model)(nil)

// Option configures a Model.
type Option func(*Model)

// WithMaxRetries sets the number of retries after the first attempt. Default 3.
func WithMaxRetries(n int) Option { return func(m *Model) { m.maxRetries = n } }

// WithBaseDelay sets the initial backoff. Default 1s.
func WithBaseDelay(d time.Duration) Option { return func(m *Model) { m.baseDelay = d } }

// WithMaxDelay caps the backoff. Default 60s.
func WithMaxDelay(d time.Duration) Option { return func(m *Model) { m.maxDelay = d } }

// WithLogger sets a logger for retry events. Default: discard.
func WithLogger(l *slog.Logger) Option { return func(m *Model) { m.logger = l } }

// Wrap returns inner with retries.
func Wrap(inner llm.Model, opts ...Option) *Model {
	m := &Model{inner: inner, maxRetries: defaultMaxRetries, baseDelay: defaultBaseDelay, maxDelay: defaultMaxDelay,
		logger: slog.New(slog.DiscardHandler)}
	for _, o := range opts {
		o(m)
	}
	return m
}

// Generate implements llm.Model. Non-retryable errors are returned at once;
// after the last retry the last error is returned unchanged, so its code is
// what gets recorded.
func (m *Model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, &llm.Error{Code: llm.ErrCodeCancelled, Message: err.Error()}
		}
		llm.ReportAttempt(ctx)
		resp, err := m.inner.Generate(ctx, req)
		if err == nil {
			return resp, nil
		}
		var le *llm.Error
		if !errors.As(err, &le) || !le.Retryable {
			return nil, err
		}
		lastErr = err
		if attempt == m.maxRetries {
			break
		}
		delay := m.backoff(attempt)
		m.logger.Warn("retrying model call", "attempt", attempt+1, "max_retries", m.maxRetries, "delay", delay, "error", err)
		select {
		case <-ctx.Done():
			return nil, lastErr
		case <-time.After(delay):
		}
	}
	return nil, lastErr
}

// backoff is random in [0, min(maxDelay, baseDelay·2^attempt)).
func (m *Model) backoff(attempt int) time.Duration {
	exp := m.baseDelay
	for i := 0; i < attempt && exp < m.maxDelay; i++ {
		exp *= 2
	}
	if exp > m.maxDelay {
		exp = m.maxDelay
	}
	if exp <= 0 {
		return 0
	}
	return time.Duration(rand.Int64N(int64(exp)))
}
