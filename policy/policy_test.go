package policy

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// --- CostTracker Tests ---

func TestCostTracker_Record(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 1000})

	err := ct.Record(UsageReport{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150})
	if err != nil {
		t.Fatalf("Record() error = %v", err)
	}

	snap := ct.Snapshot()
	if snap.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150", snap.TotalTokens)
	}
	if snap.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", snap.PromptTokens)
	}
	if snap.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", snap.CompletionTokens)
	}
	if snap.CallCount != 1 {
		t.Errorf("CallCount = %d, want 1", snap.CallCount)
	}
}

func TestCostTracker_BudgetExceeded(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 100})

	// First call: 80 tokens — ok.
	if err := ct.Record(UsageReport{TotalTokens: 80}); err != nil {
		t.Fatalf("first Record() error = %v", err)
	}

	// Second call: 30 tokens — exceeds 100 limit.
	err := ct.Record(UsageReport{TotalTokens: 30})
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected ErrBudgetExceeded, got %v", err)
	}
}

func TestCostTracker_PromptBudget(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxPromptTokens: 200})

	ct.Record(UsageReport{PromptTokens: 150, TotalTokens: 200})
	err := ct.Record(UsageReport{PromptTokens: 60, TotalTokens: 100})
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected ErrBudgetExceeded for prompt budget, got %v", err)
	}
}

func TestCostTracker_UnlimitedBudget(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{}) // zero = unlimited

	for i := 0; i < 100; i++ {
		if err := ct.Record(UsageReport{TotalTokens: 10000}); err != nil {
			t.Fatalf("Record() should not fail with unlimited budget: %v", err)
		}
	}
}

func TestCostTracker_CheckTool_DeniesWhenExhausted(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 100})
	ct.Record(UsageReport{TotalTokens: 100})

	decision, err := ct.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Deny {
		t.Errorf("expected Deny, got %v", decision)
	}
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected ErrBudgetExceeded, got %v", err)
	}
}

func TestCostTracker_CheckTool_AllowsWithBudget(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 100})
	ct.Record(UsageReport{TotalTokens: 50})

	decision, err := ct.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Allow {
		t.Errorf("expected Allow, got %v", decision)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCostTracker_Concurrent(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 1000000})
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ct.Record(UsageReport{TotalTokens: 10})
			ct.Snapshot()
			ct.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
		}()
	}
	wg.Wait()
	snap := ct.Snapshot()
	if snap.TotalTokens != 1000 {
		t.Errorf("TotalTokens = %d, want 1000", snap.TotalTokens)
	}
}

// --- RateLimiter Tests ---

func TestRateLimiter_AllowsWithinLimit(t *testing.T) {
	rl := NewRateLimiter(5, time.Second)

	for i := 0; i < 5; i++ {
		decision, err := rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
		if decision != Allow {
			t.Errorf("call %d: expected Allow", i)
		}
		if err != nil {
			t.Errorf("call %d: unexpected error: %v", i, err)
		}
	}
}

func TestRateLimiter_DeniesOverLimit(t *testing.T) {
	rl := NewRateLimiter(3, time.Second)

	for i := 0; i < 3; i++ {
		rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	}

	decision, err := rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Deny {
		t.Errorf("expected Deny after limit exceeded")
	}
	if !errors.Is(err, ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", err)
	}
}

func TestRateLimiter_WindowExpiry(t *testing.T) {
	rl := NewRateLimiter(2, 50*time.Millisecond)

	rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})

	// Wait for window to expire.
	time.Sleep(60 * time.Millisecond)

	decision, _ := rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Allow {
		t.Error("expected Allow after window expired")
	}
}

func TestRateLimiter_Concurrent(t *testing.T) {
	rl := NewRateLimiter(100, time.Second)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rl.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
		}()
	}
	wg.Wait()
}

// --- PermissionChecker Tests ---

func TestPermissionChecker_DefaultAllow(t *testing.T) {
	pc := NewPermissionChecker(PermAllow)

	decision, err := pc.CheckTool(context.Background(), ToolRequest{ToolName: "unknown_tool"})
	if decision != Allow {
		t.Errorf("expected Allow for unknown tool with default PermAllow")
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPermissionChecker_DefaultDeny(t *testing.T) {
	pc := NewPermissionChecker(PermDeny)

	decision, err := pc.CheckTool(context.Background(), ToolRequest{ToolName: "unknown_tool"})
	if decision != Deny {
		t.Errorf("expected Deny for unknown tool with default PermDeny")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Errorf("expected ErrToolDenied, got %v", err)
	}
}

func TestPermissionChecker_ExplicitOverride(t *testing.T) {
	pc := NewPermissionChecker(PermDeny)
	pc.SetPermission("calculator", PermAllow)
	pc.SetPermission("file_delete", PermDeny)

	tests := []struct {
		tool     string
		wantPerm Decision
	}{
		{"calculator", Allow},
		{"file_delete", Deny},
		{"unknown", Deny}, // default
	}

	for _, tt := range tests {
		t.Run(tt.tool, func(t *testing.T) {
			decision, _ := pc.CheckTool(context.Background(), ToolRequest{ToolName: tt.tool})
			if decision != tt.wantPerm {
				t.Errorf("tool %q: decision = %v, want %v", tt.tool, decision, tt.wantPerm)
			}
		})
	}
}

func TestPermissionChecker_Concurrent(t *testing.T) {
	pc := NewPermissionChecker(PermAllow)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			pc.SetPermission("tool", PermAllow)
		}(i)
		go func() {
			defer wg.Done()
			pc.CheckTool(context.Background(), ToolRequest{ToolName: "tool"})
		}()
	}
	wg.Wait()
}

// --- TimeoutEnforcer Tests ---

func TestTimeoutEnforcer_Default(t *testing.T) {
	te := NewTimeoutEnforcer(5 * time.Second)

	timeout := te.TimeoutFor("any_tool")
	if timeout != 5*time.Second {
		t.Errorf("TimeoutFor() = %v, want 5s", timeout)
	}
}

func TestTimeoutEnforcer_PerTool(t *testing.T) {
	te := NewTimeoutEnforcer(5 * time.Second)
	te.SetToolTimeout("slow_tool", 30*time.Second)

	if te.TimeoutFor("slow_tool") != 30*time.Second {
		t.Error("expected 30s for slow_tool")
	}
	if te.TimeoutFor("normal_tool") != 5*time.Second {
		t.Error("expected 5s for normal_tool")
	}
}

func TestTimeoutEnforcer_WrapContext(t *testing.T) {
	te := NewTimeoutEnforcer(100 * time.Millisecond)

	ctx, cancel := te.WrapContext(context.Background(), "test_tool")
	defer cancel()

	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("expected deadline on wrapped context")
	}
	if time.Until(deadline) > 200*time.Millisecond {
		t.Error("deadline too far in the future")
	}
}

func TestTimeoutEnforcer_WrapContext_ZeroTimeout(t *testing.T) {
	te := NewTimeoutEnforcer(0) // zero means no timeout

	ctx, cancel := te.WrapContext(context.Background(), "test_tool")
	defer cancel()

	_, ok := ctx.Deadline()
	if ok {
		t.Error("expected no deadline for zero timeout")
	}
}

func TestTimeoutEnforcer_CheckTool_AlwaysAllow(t *testing.T) {
	te := NewTimeoutEnforcer(5 * time.Second)
	decision, err := te.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Allow || err != nil {
		t.Errorf("CheckTool should always Allow, got %v, %v", decision, err)
	}
}

// --- Chain Tests ---

func TestChain_AllAllow(t *testing.T) {
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 10000})
	rl := NewRateLimiter(100, time.Second)
	pc := NewPermissionChecker(PermAllow)

	chain := NewChain(ct, rl, pc)
	decision, err := chain.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Allow {
		t.Errorf("expected Allow, got %v", decision)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestChain_ShortCircuitsOnDeny(t *testing.T) {
	// Cost tracker has exhausted budget.
	ct := NewCostTracker(CostTrackerConfig{MaxTotalTokens: 100})
	ct.Record(UsageReport{TotalTokens: 100})

	// This should never be reached.
	pc := NewPermissionChecker(PermAllow)

	chain := NewChain(ct, pc)
	decision, err := chain.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Deny {
		t.Error("expected Deny from chain when first checker denies")
	}
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected ErrBudgetExceeded, got %v", err)
	}
}

func TestChain_PermissionDeny(t *testing.T) {
	pc := NewPermissionChecker(PermDeny)
	chain := NewChain(pc)

	decision, err := chain.CheckTool(context.Background(), ToolRequest{ToolName: "secret_tool"})
	if decision != Deny {
		t.Error("expected Deny")
	}
	if !errors.Is(err, ErrToolDenied) {
		t.Errorf("expected ErrToolDenied, got %v", err)
	}
}

func TestChain_Empty(t *testing.T) {
	chain := NewChain()
	decision, err := chain.CheckTool(context.Background(), ToolRequest{ToolName: "test"})
	if decision != Allow {
		t.Error("empty chain should Allow")
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
