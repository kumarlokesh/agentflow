package agentflow

import (
	"context"
	"time"

	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/policy"
)

// CostHook is an observe.Hook that records LLM token usage into a
// policy.CostTracker. Wire it alongside any other hooks via observe.NewMultiHook.
//
// Without this bridge, a CostTracker used as AgentConfig.Policy will always
// see zero tokens, because Record() is never called automatically. CostHook
// closes that gap: every completed LLM call feeds its token counts into the
// tracker, so subsequent CheckTool calls reflect real consumption.
//
// Example:
//
//	tracker := policy.NewCostTracker(policy.CostTrackerConfig{MaxTotalTokens: 50_000})
//	hook := agentflow.NewCostHook(tracker)
//	agent, _ := agentflow.NewAgent(agentflow.AgentConfig{
//	    Policy: tracker,
//	    Hook:   observe.NewMultiHook(hook, otherHook),
//	    ...
//	})
type CostHook struct {
	tracker *policy.CostTracker
}

// NewCostHook creates a CostHook backed by the given tracker.
func NewCostHook(tracker *policy.CostTracker) *CostHook {
	return &CostHook{tracker: tracker}
}

// Ensure CostHook implements observe.Hook at compile time.
var _ observe.Hook = (*CostHook)(nil)

func (h *CostHook) OnRunStart(_ context.Context, _, _ string)                              {}
func (h *CostHook) OnRunEnd(_ context.Context, _ string, _ int, _ time.Duration, _ error)  {}
func (h *CostHook) OnStepStart(_ context.Context, _ string, _ int)                         {}
func (h *CostHook) OnStepEnd(_ context.Context, _ string, _ int, _ time.Duration)          {}
func (h *CostHook) OnToolCall(_ context.Context, _ string, _ int, _ string, _ time.Duration, _ error) {
}

// OnLLMCall records token usage into the CostTracker. Budget violations are
// surfaced on the next tool call via CheckTool — not here, because the hook
// fires after the LLM call has already completed.
func (h *CostHook) OnLLMCall(_ context.Context, _ string, _ int, promptTokens, completionTokens, totalTokens int, _ time.Duration, _ error) {
	// Ignore the error: budget violations are surfaced via CheckTool, not here.
	_ = h.tracker.Record(policy.UsageReport{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	})
}
