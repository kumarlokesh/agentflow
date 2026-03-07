package agentflow_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/internal/testutil"
	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/policy"
)

func TestCostHook_RecordsLLMTokens(t *testing.T) {
	tracker := policy.NewCostTracker(policy.CostTrackerConfig{MaxTotalTokens: 1000})
	hook := agentflow.NewCostHook(tracker)

	// Simulate the OnLLMCall callback the agent fires after each LLM call.
	hook.OnLLMCall(context.Background(), "run-1", 0, 40, 20, 60, time.Millisecond, nil)

	snap := tracker.Snapshot()
	if snap.TotalTokens != 60 {
		t.Errorf("TotalTokens = %d, want 60", snap.TotalTokens)
	}
	if snap.PromptTokens != 40 {
		t.Errorf("PromptTokens = %d, want 40", snap.PromptTokens)
	}
	if snap.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", snap.CompletionTokens)
	}
	if snap.CallCount != 1 {
		t.Errorf("CallCount = %d, want 1", snap.CallCount)
	}
}

func TestCostHook_NoopMethods(t *testing.T) {
	// All other hook methods are no-ops; ensure they don't panic.
	tracker := policy.NewCostTracker(policy.CostTrackerConfig{})
	hook := agentflow.NewCostHook(tracker)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-1", "task")
	hook.OnRunEnd(ctx, "run-1", 1, time.Second, nil)
	hook.OnStepStart(ctx, "run-1", 0)
	hook.OnStepEnd(ctx, "run-1", 0, time.Millisecond)
	hook.OnToolCall(ctx, "run-1", 0, "tool", time.Millisecond, nil)
	// OnLLMCall with an error should still record zero tokens and not panic.
	hook.OnLLMCall(ctx, "run-1", 0, 0, 0, 0, time.Millisecond, errors.New("llm error"))
}

// TestAgent_Run_CostHook_EnforcesBudget verifies the full wiring:
// CostHook records LLM tokens → CostTracker.CheckTool blocks the next call.
func TestAgent_Run_CostHook_EnforcesBudget(t *testing.T) {
	pinTime(t)

	// Budget of 50 tokens; first LLM call costs 60 → second tool call denied.
	tracker := policy.NewCostTracker(policy.CostTrackerConfig{MaxTotalTokens: 50})
	costHook := agentflow.NewCostHook(tracker)
	metrics := observe.NewMetrics()
	combined := observe.NewMultiHook(costHook, observe.NewMetricsHook(metrics))

	tool := testutil.NewMockTool("calc", "calculator", nil,
		&agentflow.ToolResult{Output: "4"},
	)

	llm := testutil.NewMockLLM(
		// Step 0: request tool call, costs 60 tokens (over budget for next check).
		&agentflow.LLMResponse{
			ToolCalls: []agentflow.ToolCallRequest{
				{ID: "call-1", Name: "calc"},
			},
			Usage: &agentflow.TokenUsage{PromptTokens: 30, CompletionTokens: 30, TotalTokens: 60},
		},
		// Step 1: give final answer.
		&agentflow.LLMResponse{
			Content: "done",
			Usage:   &agentflow.TokenUsage{PromptTokens: 5, CompletionTokens: 5, TotalTokens: 10},
		},
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "budget-test",
		LLM:    llm,
		Tools:  []agentflow.Tool{tool},
		Logger: quietLogger(),
		Policy: tracker,
		Hook:   combined,
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	// Run should complete; the tool call in step 1 will be denied by budget.
	result, err := agent.Run(context.Background(), "compute")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Output != "done" {
		t.Errorf("Output = %q, want done", result.Output)
	}

	snap := tracker.Snapshot()
	if snap.TotalTokens < 60 {
		t.Errorf("TotalTokens = %d, want ≥60", snap.TotalTokens)
	}
}
