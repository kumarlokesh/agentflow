package llm

import (
	"context"
	"sync/atomic"
)

type attemptsKey struct{}

// WithAttemptCounter returns a context that counts ReportAttempt calls, and
// a function that reads the count. The runtime uses it to record how many
// provider attempts one model call took (e.g., with a retry wrapper). If
// nothing reports, the count is 0 and callers should treat it as 1.
func WithAttemptCounter(ctx context.Context) (context.Context, func() int) {
	c := new(atomic.Int32)
	return context.WithValue(ctx, attemptsKey{}, c), func() int { return int(c.Load()) }
}

// ReportAttempt records one attempt against the counter in ctx, if any.
// Retry wrappers call it once per attempt.
func ReportAttempt(ctx context.Context) {
	if c, ok := ctx.Value(attemptsKey{}).(*atomic.Int32); ok {
		c.Add(1)
	}
}
