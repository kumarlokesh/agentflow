// Package policy provides agentflow interceptors for common guardrails:
// tool permissions, token budgets, and rate limits.
//
// Permissions and budgets are deterministic given the trace so far, so they
// run in replay exactly as they did live. Rate limiting depends on wall-clock
// time and does nothing during replay.
package policy

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// Permissions allows or denies tool calls by name.
type Permissions struct {
	// Allow, if non-empty, lists the only tools that may run.
	Allow []string
	// Deny lists tools that may never run. Deny wins over Allow.
	Deny []string
}

// Intercept implements agentflow.Interceptor.
func (p Permissions) Intercept(ctx context.Context, call *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
	if call.Kind != agentflow.EffectTool {
		return next(ctx, call)
	}
	for _, n := range p.Deny {
		if n == call.Name {
			return agentflow.Fail(trace.ErrCodeDenied, fmt.Sprintf("tool %q is denied by policy", call.Name)), nil
		}
	}
	if len(p.Allow) > 0 {
		for _, n := range p.Allow {
			if n == call.Name {
				return next(ctx, call)
			}
		}
		return agentflow.Fail(trace.ErrCodeDenied, fmt.Sprintf("tool %q is not in the allow-list", call.Name)), nil
	}
	return next(ctx, call)
}

// Budget limits the tokens a run (or every run sharing the Budget) may use.
// It blocks model calls once the budget is spent: a blocked call fails with
// code "limit_exceeded", which fails the run. Usage is counted from model
// responses, which come from the recording during replay, so a budget
// behaves identically live and in replay.
type Budget struct {
	// MaxTotalTokens caps input plus output tokens. Zero means no cap.
	MaxTotalTokens int

	mu   sync.Mutex
	used llm.Usage
}

// Intercept implements agentflow.Interceptor.
func (b *Budget) Intercept(ctx context.Context, call *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
	if call.Kind != agentflow.EffectLLM {
		return next(ctx, call)
	}
	b.mu.Lock()
	spent := b.used.Total()
	b.mu.Unlock()
	if b.MaxTotalTokens > 0 && spent >= b.MaxTotalTokens {
		return agentflow.Fail(trace.ErrCodeLimitExceeded, fmt.Sprintf("token budget exhausted (%d of %d used)", spent, b.MaxTotalTokens)), nil
	}
	out, err := next(ctx, call)
	if err == nil && out != nil && out.Failure == nil {
		if resp, ok := out.Response.(*llm.Response); ok {
			b.mu.Lock()
			b.used = b.used.Add(resp.Usage)
			b.mu.Unlock()
		}
	}
	return out, err
}

// Used returns the tokens counted so far.
func (b *Budget) Used() llm.Usage {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.used
}

// RateLimit allows at most Max calls of the selected kinds per Window,
// waiting (or failing when the context ends) when the limit is reached. It
// does nothing during replay.
type RateLimit struct {
	Max    int
	Window time.Duration
	// Kinds limits which effects count; empty means model calls only.
	Kinds []agentflow.EffectKind

	mu    sync.Mutex
	times []time.Time
}

func (r *RateLimit) applies(k agentflow.EffectKind) bool {
	if len(r.Kinds) == 0 {
		return k == agentflow.EffectLLM
	}
	for _, x := range r.Kinds {
		if x == k {
			return true
		}
	}
	return false
}

// Intercept implements agentflow.Interceptor.
func (r *RateLimit) Intercept(ctx context.Context, call *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
	if call.Meta.Mode == agentflow.ModeReplay || r.Max <= 0 || !r.applies(call.Kind) {
		return next(ctx, call)
	}
	for {
		r.mu.Lock()
		now := time.Now()
		cut := now.Add(-r.Window)
		kept := r.times[:0]
		for _, t := range r.times {
			if t.After(cut) {
				kept = append(kept, t)
			}
		}
		r.times = kept
		if len(r.times) < r.Max {
			r.times = append(r.times, now)
			r.mu.Unlock()
			return next(ctx, call)
		}
		wait := r.times[0].Add(r.Window).Sub(now)
		r.mu.Unlock()
		select {
		case <-ctx.Done():
			return agentflow.Fail(trace.ErrCodeCancelled, "cancelled while rate limited"), nil
		case <-time.After(wait):
		}
	}
}
