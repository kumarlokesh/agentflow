// Package policy provides guardrails and enforcement for agent execution.
//
// It implements cost tracking, rate limiting, tool permissions, and timeout
// enforcement. The Policy is evaluated before each tool call and after each
// LLM response, giving fine-grained control over agent behavior.
//
// Design rationale: Rather than embedding policy logic inside the agent loop,
// policies are composable middleware. This keeps the agent focused on its
// Observe-Think-Act loop while allowing arbitrary guardrails to be layered on.
package policy

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// --- Sentinel errors ---

var (
	// ErrBudgetExceeded is returned when a token or cost budget is exhausted.
	ErrBudgetExceeded = errors.New("policy: budget exceeded")
	// ErrRateLimited is returned when a tool call rate limit is hit.
	ErrRateLimited = errors.New("policy: rate limited")
	// ErrToolDenied is returned when a tool call is blocked by permissions.
	ErrToolDenied = errors.New("policy: tool denied")
	// ErrToolTimeout is returned when a tool execution exceeds its deadline.
	ErrToolTimeout = errors.New("policy: tool execution timeout")
)

// --- Core types ---

// Decision represents the outcome of a policy check.
type Decision int

const (
	// Allow permits the action.
	Allow Decision = iota
	// Deny blocks the action.
	Deny
)

// ToolRequest describes a pending tool invocation for policy evaluation.
type ToolRequest struct {
	ToolName string
	CallID   string
	Step     int
}

// UsageReport describes token/cost consumption from an LLM call.
type UsageReport struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// Checker is the interface for a single policy rule. Multiple checkers
// can be composed into a Chain.
type Checker interface {
	// CheckTool evaluates whether a tool call should proceed.
	CheckTool(ctx context.Context, req ToolRequest) (Decision, error)
}

// Chain evaluates multiple policy checkers in order. If any checker denies,
// the chain short-circuits and returns Deny.
type Chain struct {
	checkers []Checker
}

// NewChain creates a policy chain from the given checkers.
func NewChain(checkers ...Checker) *Chain {
	return &Chain{checkers: checkers}
}

// CheckTool evaluates all checkers. Returns Deny on the first denial.
func (c *Chain) CheckTool(ctx context.Context, req ToolRequest) (Decision, error) {
	for _, checker := range c.checkers {
		decision, err := checker.CheckTool(ctx, req)
		if err != nil {
			return Deny, err
		}
		if decision == Deny {
			return Deny, nil
		}
	}
	return Allow, nil
}

// --- Cost Tracker ---

// CostTracker tracks cumulative token usage and enforces budgets.
type CostTracker struct {
	mu               sync.Mutex
	maxTotalTokens   int
	maxPromptTokens  int
	totalTokens      int
	promptTokens     int
	completionTokens int
	callCount        int
}

// CostTrackerConfig configures a CostTracker.
type CostTrackerConfig struct {
	// MaxTotalTokens is the maximum cumulative total tokens allowed.
	// Zero means unlimited.
	MaxTotalTokens int
	// MaxPromptTokens is the maximum cumulative prompt tokens allowed.
	// Zero means unlimited.
	MaxPromptTokens int
}

// NewCostTracker creates a CostTracker with the given budget limits.
func NewCostTracker(cfg CostTrackerConfig) *CostTracker {
	return &CostTracker{
		maxTotalTokens:  cfg.MaxTotalTokens,
		maxPromptTokens: cfg.MaxPromptTokens,
	}
}

// Record adds a usage report to the cumulative totals.
// Returns ErrBudgetExceeded if any budget limit is breached.
func (ct *CostTracker) Record(usage UsageReport) error {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.totalTokens += usage.TotalTokens
	ct.promptTokens += usage.PromptTokens
	ct.completionTokens += usage.CompletionTokens
	ct.callCount++

	if ct.maxTotalTokens > 0 && ct.totalTokens > ct.maxTotalTokens {
		return fmt.Errorf("%w: total tokens %d exceeds limit %d", ErrBudgetExceeded, ct.totalTokens, ct.maxTotalTokens)
	}
	if ct.maxPromptTokens > 0 && ct.promptTokens > ct.maxPromptTokens {
		return fmt.Errorf("%w: prompt tokens %d exceeds limit %d", ErrBudgetExceeded, ct.promptTokens, ct.maxPromptTokens)
	}
	return nil
}

// Snapshot returns a point-in-time copy of the usage counters.
func (ct *CostTracker) Snapshot() CostSnapshot {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return CostSnapshot{
		TotalTokens:      ct.totalTokens,
		PromptTokens:     ct.promptTokens,
		CompletionTokens: ct.completionTokens,
		CallCount:        ct.callCount,
	}
}

// CheckTool implements Checker. It denies if budget is already exceeded.
func (ct *CostTracker) CheckTool(_ context.Context, _ ToolRequest) (Decision, error) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if ct.maxTotalTokens > 0 && ct.totalTokens >= ct.maxTotalTokens {
		return Deny, fmt.Errorf("%w: total tokens %d reached limit %d", ErrBudgetExceeded, ct.totalTokens, ct.maxTotalTokens)
	}
	return Allow, nil
}

// CostSnapshot is a read-only view of cost tracking state.
type CostSnapshot struct {
	TotalTokens      int
	PromptTokens     int
	CompletionTokens int
	CallCount        int
}

// --- Rate Limiter ---

// RateLimiter enforces a maximum number of tool calls within a sliding window.
type RateLimiter struct {
	mu        sync.Mutex
	maxCalls  int
	window    time.Duration
	callTimes []time.Time
}

// NewRateLimiter creates a rate limiter allowing maxCalls within the given window.
func NewRateLimiter(maxCalls int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		maxCalls: maxCalls,
		window:   window,
	}
}

// CheckTool implements Checker. It denies if the rate limit would be exceeded.
func (rl *RateLimiter) CheckTool(_ context.Context, _ ToolRequest) (Decision, error) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-rl.window)

	// Prune expired entries.
	valid := rl.callTimes[:0]
	for _, t := range rl.callTimes {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	rl.callTimes = valid

	if len(rl.callTimes) >= rl.maxCalls {
		return Deny, fmt.Errorf("%w: %d calls in %s window (max %d)", ErrRateLimited, len(rl.callTimes), rl.window, rl.maxCalls)
	}

	rl.callTimes = append(rl.callTimes, now)
	return Allow, nil
}

// --- Permission System ---

// Permission defines the access level for a tool.
type Permission int

const (
	// PermAllow always permits the tool.
	PermAllow Permission = iota
	// PermDeny always blocks the tool.
	PermDeny
)

// PermissionChecker enforces tool-level access control.
type PermissionChecker struct {
	mu          sync.RWMutex
	permissions map[string]Permission
	defaultPerm Permission
}

// NewPermissionChecker creates a permission checker with the given default.
func NewPermissionChecker(defaultPerm Permission) *PermissionChecker {
	return &PermissionChecker{
		permissions: make(map[string]Permission),
		defaultPerm: defaultPerm,
	}
}

// SetPermission configures the permission for a specific tool.
func (pc *PermissionChecker) SetPermission(toolName string, perm Permission) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.permissions[toolName] = perm
}

// CheckTool implements Checker. It checks the tool's permission level.
func (pc *PermissionChecker) CheckTool(_ context.Context, req ToolRequest) (Decision, error) {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	perm, exists := pc.permissions[req.ToolName]
	if !exists {
		perm = pc.defaultPerm
	}

	switch perm {
	case PermDeny:
		return Deny, fmt.Errorf("%w: tool %q is not permitted", ErrToolDenied, req.ToolName)
	default:
		return Allow, nil
	}
}

// --- Timeout Enforcer ---

// TimeoutEnforcer wraps tool execution contexts with deadlines.
type TimeoutEnforcer struct {
	mu       sync.RWMutex
	defaults time.Duration
	perTool  map[string]time.Duration
}

// NewTimeoutEnforcer creates a timeout enforcer with the given default timeout.
func NewTimeoutEnforcer(defaultTimeout time.Duration) *TimeoutEnforcer {
	return &TimeoutEnforcer{
		defaults: defaultTimeout,
		perTool:  make(map[string]time.Duration),
	}
}

// SetToolTimeout configures a per-tool timeout override.
func (te *TimeoutEnforcer) SetToolTimeout(toolName string, timeout time.Duration) {
	te.mu.Lock()
	defer te.mu.Unlock()
	te.perTool[toolName] = timeout
}

// TimeoutFor returns the timeout duration for a given tool.
func (te *TimeoutEnforcer) TimeoutFor(toolName string) time.Duration {
	te.mu.RLock()
	defer te.mu.RUnlock()
	if d, ok := te.perTool[toolName]; ok {
		return d
	}
	return te.defaults
}

// WrapContext returns a child context with the appropriate timeout for the tool.
func (te *TimeoutEnforcer) WrapContext(ctx context.Context, toolName string) (context.Context, context.CancelFunc) {
	timeout := te.TimeoutFor(toolName)
	if timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, timeout)
}

// CheckTool implements Checker. Timeout enforcement is always Allow (the actual
// timeout is applied at execution time via WrapContext, not at check time).
func (te *TimeoutEnforcer) CheckTool(_ context.Context, _ ToolRequest) (Decision, error) {
	return Allow, nil
}
