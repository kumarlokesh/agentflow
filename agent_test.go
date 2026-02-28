package agentflow_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/internal/testutil"
	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/policy"
	"github.com/kumarlokesh/agentflow/store"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
}

func pinTime(t *testing.T) {
	t.Helper()
	origUUID := agentflow.ExportNewUUID()
	origNow := agentflow.ExportNowUTC()
	counter := 0
	agentflow.SetNewUUID(func() string {
		counter++
		return fmt.Sprintf("test-uuid-%04d", counter)
	})
	fixedTime := time.Date(2025, 6, 15, 12, 0, 0, 0, time.UTC)
	tick := 0
	agentflow.SetNowUTC(func() time.Time {
		tick++
		return fixedTime.Add(time.Duration(tick) * time.Millisecond)
	})
	t.Cleanup(func() {
		agentflow.SetNewUUID(origUUID)
		agentflow.SetNowUTC(origNow)
	})
}

func TestNewAgent_NoLLM(t *testing.T) {
	_, err := agentflow.NewAgent(agentflow.AgentConfig{})
	if !errors.Is(err, agentflow.ErrNoLLM) {
		t.Errorf("expected ErrNoLLM, got %v", err)
	}
}

func TestNewAgent_DuplicateTools(t *testing.T) {
	llm := testutil.NewMockLLM()
	tool1 := testutil.NewMockTool("calc", "calculator", json.RawMessage(`{"type":"object"}`))
	tool2 := testutil.NewMockTool("calc", "duplicate", json.RawMessage(`{"type":"object"}`))

	_, err := agentflow.NewAgent(agentflow.AgentConfig{
		LLM:   llm,
		Tools: []agentflow.Tool{tool1, tool2},
	})
	if err == nil {
		t.Fatal("expected error for duplicate tool names")
	}
}

func TestAgent_Run_DirectAnswer(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		testutil.LLMResponseWithText("The answer is 42."),
	)

	memStore := store.NewMemory()
	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "test-agent",
		Instructions: "You are a helpful assistant.",
		LLM:          llm,
		MaxSteps:     5,
		Store:        memStore,
		Logger:       quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "What is the meaning of life?")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if result.Output != "The answer is 42." {
		t.Errorf("Output = %q, want %q", result.Output, "The answer is 42.")
	}
	if result.Steps != 1 {
		t.Errorf("Steps = %d, want 1", result.Steps)
	}
	if result.RunID == "" {
		t.Error("RunID is empty")
	}
	if result.Duration <= 0 {
		t.Error("Duration is zero or negative")
	}

	// Verify events were recorded.
	if len(result.Events) == 0 {
		t.Fatal("no events recorded")
	}

	// Verify event types sequence.
	expectedTypes := []agentflow.EventType{
		agentflow.EventRunStart,
		agentflow.EventStepStart,
		agentflow.EventLLMRequest,
		agentflow.EventLLMResponse,
		agentflow.EventStepEnd,
		agentflow.EventRunEnd,
	}

	if len(result.Events) != len(expectedTypes) {
		t.Fatalf("event count = %d, want %d", len(result.Events), len(expectedTypes))
	}
	for i, et := range expectedTypes {
		if result.Events[i].Type != et {
			t.Errorf("event[%d].Type = %q, want %q", i, result.Events[i].Type, et)
		}
	}

	// Verify events were persisted to store.
	storedEvents, _ := memStore.LoadEvents(context.Background(), result.RunID)
	if len(storedEvents) != len(expectedTypes) {
		t.Errorf("stored events = %d, want %d", len(storedEvents), len(expectedTypes))
	}

	// Verify LLM was called once.
	calls := llm.Calls()
	if len(calls) != 1 {
		t.Errorf("LLM calls = %d, want 1", len(calls))
	}
}

func TestAgent_Run_WithToolCalls(t *testing.T) {
	pinTime(t)

	calcTool := testutil.NewMockTool(
		"calculator", "Evaluates math expressions",
		json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
		&agentflow.ToolResult{Output: "4"},
	)

	llm := testutil.NewMockLLM(
		// Step 0: LLM requests a tool call.
		testutil.LLMResponseWithToolCalls(
			testutil.ToolCall("call-1", "calculator", map[string]string{"expression": "2+2"}),
		),
		// Step 1: LLM gives final answer after seeing tool result.
		testutil.LLMResponseWithText("2 + 2 = 4"),
	)

	memStore := store.NewMemory()
	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "calc-agent",
		Instructions: "Use the calculator tool.",
		LLM:          llm,
		Tools:        []agentflow.Tool{calcTool},
		MaxSteps:     5,
		Store:        memStore,
		Logger:       quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "What is 2+2?")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if result.Output != "2 + 2 = 4" {
		t.Errorf("Output = %q, want %q", result.Output, "2 + 2 = 4")
	}
	if result.Steps != 2 {
		t.Errorf("Steps = %d, want 2", result.Steps)
	}

	// Verify tool was called.
	toolCalls := calcTool.Calls()
	if len(toolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(toolCalls))
	}

	// Verify event sequence includes tool events.
	hasToolCall := false
	hasToolResult := false
	for _, e := range result.Events {
		switch e.Type {
		case agentflow.EventToolCall:
			hasToolCall = true
		case agentflow.EventToolResult:
			hasToolResult = true
		}
	}
	if !hasToolCall {
		t.Error("missing tool_call event")
	}
	if !hasToolResult {
		t.Error("missing tool_result event")
	}
}

func TestAgent_Run_MultipleToolCalls(t *testing.T) {
	pinTime(t)

	tool := testutil.NewMockTool(
		"lookup", "Looks up values",
		json.RawMessage(`{"type":"object"}`),
		&agentflow.ToolResult{Output: "result-a"},
		&agentflow.ToolResult{Output: "result-b"},
	)

	llm := testutil.NewMockLLM(
		// Step 0: Two tool calls in one response.
		testutil.LLMResponseWithToolCalls(
			testutil.ToolCall("call-1", "lookup", map[string]string{"key": "a"}),
			testutil.ToolCall("call-2", "lookup", map[string]string{"key": "b"}),
		),
		// Step 1: Final answer.
		testutil.LLMResponseWithText("Found both values."),
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "multi-tool-agent",
		LLM:    llm,
		Tools:  []agentflow.Tool{tool},
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "Look up a and b")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if result.Output != "Found both values." {
		t.Errorf("Output = %q", result.Output)
	}

	toolCalls := tool.Calls()
	if len(toolCalls) != 2 {
		t.Errorf("tool calls = %d, want 2", len(toolCalls))
	}
}

func TestAgent_Run_ToolNotFound(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		// LLM calls a tool that doesn't exist.
		testutil.LLMResponseWithToolCalls(
			testutil.ToolCall("call-1", "nonexistent", map[string]string{}),
		),
		// Then gives final answer (agent should continue after tool-not-found).
		testutil.LLMResponseWithText("I couldn't find that tool."),
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "test-agent",
		LLM:    llm,
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "Do something")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Agent should still complete despite tool-not-found.
	if result.Output != "I couldn't find that tool." {
		t.Errorf("Output = %q", result.Output)
	}
}

func TestAgent_Run_MaxStepsExceeded(t *testing.T) {
	pinTime(t)

	// LLM always requests tool calls, never gives a final answer.
	responses := make([]*agentflow.LLMResponse, 5)
	for i := range responses {
		responses[i] = testutil.LLMResponseWithToolCalls(
			testutil.ToolCall("call-1", "tool", map[string]string{}),
		)
	}

	llm := testutil.NewMockLLM(responses...)
	tool := testutil.NewMockTool("tool", "always called", json.RawMessage(`{"type":"object"}`))

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:     "loop-agent",
		LLM:      llm,
		Tools:    []agentflow.Tool{tool},
		MaxSteps: 3,
		Logger:   quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	_, err = agent.Run(context.Background(), "Loop forever")
	if !errors.Is(err, agentflow.ErrMaxStepsExceeded) {
		t.Errorf("expected ErrMaxStepsExceeded, got %v", err)
	}
}

func TestAgent_Run_ContextCancellation(t *testing.T) {
	pinTime(t)

	// LLM that blocks until context is cancelled.
	llm := &blockingLLM{}

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "cancel-agent",
		LLM:    llm,
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err = agent.Run(ctx, "Do something")
	if !errors.Is(err, agentflow.ErrRunCancelled) {
		t.Errorf("expected ErrRunCancelled, got %v", err)
	}
}

type blockingLLM struct{}

func (b *blockingLLM) ChatCompletion(ctx context.Context, _ *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestAgent_Run_LLMError(t *testing.T) {
	pinTime(t)

	llmErr := errors.New("model overloaded")
	llm := testutil.NewMockLLMWithErrors(nil, llmErr)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "error-agent",
		LLM:    llm,
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	_, err = agent.Run(context.Background(), "Fail")
	if err == nil {
		t.Fatal("expected error from LLM failure")
	}

	var llmError *agentflow.LLMError
	if !errors.As(err, &llmError) {
		t.Errorf("expected *LLMError, got %T: %v", err, err)
	}
}

func TestAgent_Run_ToolSchemaValidation(t *testing.T) {
	pinTime(t)

	tool := testutil.NewMockTool(
		"strict_tool", "requires name field",
		json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`),
		&agentflow.ToolResult{Output: "ok"},
	)

	llm := testutil.NewMockLLM(
		// LLM sends invalid params (missing required "name" field).
		testutil.LLMResponseWithToolCalls(
			testutil.ToolCall("call-1", "strict_tool", map[string]string{}),
		),
		// Then gives final answer.
		testutil.LLMResponseWithText("Validation failed, but I handled it."),
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "validation-agent",
		LLM:    llm,
		Tools:  []agentflow.Tool{tool},
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "Test validation")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Agent should still complete — validation errors are non-fatal.
	if result.Output == "" {
		t.Error("expected non-empty output")
	}

	// Tool should NOT have been called (params were invalid).
	if len(tool.Calls()) != 0 {
		t.Errorf("tool should not have been called, but was called %d times", len(tool.Calls()))
	}
}

func TestAgent_Run_NilStore(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		testutil.LLMResponseWithText("answer"),
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "no-store-agent",
		LLM:    llm,
		Store:  nil, // No store — events only kept in memory.
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "test")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Events should still be in the result.
	if len(result.Events) == 0 {
		t.Error("expected events even without a store")
	}
}

func TestAgent_Run_WithHook(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		testutil.LLMResponseWithToolCalls(testutil.ToolCall("call-1", "calc", map[string]any{"a": 1})),
		testutil.LLMResponseWithText("result is 42"),
	)
	calcTool := testutil.NewMockTool("calc", "calculator", nil,
		&agentflow.ToolResult{Output: "42"},
	)

	metrics := observe.NewMetrics()
	hook := observe.NewMetricsHook(metrics)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "hook-agent",
		LLM:    llm,
		Tools:  []agentflow.Tool{calcTool},
		Logger: quietLogger(),
		Hook:   hook,
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	_, err = agent.Run(context.Background(), "compute")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	snap := metrics.Snapshot()
	if snap.RunCount != 1 {
		t.Errorf("RunCount = %d, want 1", snap.RunCount)
	}
	if snap.StepCount < 1 {
		t.Errorf("StepCount = %d, want >=1", snap.StepCount)
	}
	if snap.LLMCallCount != 2 {
		t.Errorf("LLMCallCount = %d, want 2", snap.LLMCallCount)
	}
	if snap.ToolCallCount != 1 {
		t.Errorf("ToolCallCount = %d, want 1", snap.ToolCallCount)
	}
}

func TestAgent_Run_PolicyDeniesTool(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		testutil.LLMResponseWithToolCalls(testutil.ToolCall("call-1", "dangerous", nil)),
		testutil.LLMResponseWithText("ok, nevermind"),
	)
	dangerTool := testutil.NewMockTool("dangerous", "danger", nil,
		&agentflow.ToolResult{Output: "boom"},
	)

	pc := policy.NewPermissionChecker(policy.PermDeny)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "policy-agent",
		LLM:    llm,
		Tools:  []agentflow.Tool{dangerTool},
		Logger: quietLogger(),
		Policy: pc,
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "do something dangerous")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// The tool was denied, so agent should have received the error and produced a final answer.
	if result.Output != "ok, nevermind" {
		t.Errorf("Output = %q, want 'ok, nevermind'", result.Output)
	}

	// The dangerous tool should never have been called.
	if len(dangerTool.Calls()) != 0 {
		t.Errorf("dangerous tool was called %d times, want 0", len(dangerTool.Calls()))
	}
}

func TestAgent_Run_WithTimeout(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		testutil.LLMResponseWithToolCalls(testutil.ToolCall("call-1", "slow", nil)),
		testutil.LLMResponseWithText("done"),
	)
	slowTool := testutil.NewMockTool("slow", "slow tool", nil,
		&agentflow.ToolResult{Output: "ok"},
	)

	te := policy.NewTimeoutEnforcer(5 * time.Second)
	te.SetToolTimeout("slow", 50*time.Millisecond)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:            "timeout-agent",
		LLM:             llm,
		Tools:           []agentflow.Tool{slowTool},
		Logger:          quietLogger(),
		TimeoutEnforcer: te,
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	// The slow tool returns instantly in mock, so no timeout error expected.
	result, err := agent.Run(context.Background(), "do slow thing")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if result.Output != "done" {
		t.Errorf("Output = %q, want done", result.Output)
	}
}

func TestAgent_Run_EventSchemaVersion(t *testing.T) {
	pinTime(t)

	llm := testutil.NewMockLLM(
		testutil.LLMResponseWithText("done"),
	)

	memStore := store.NewMemory()
	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:   "schema-version-agent",
		LLM:    llm,
		Store:  memStore,
		Logger: quietLogger(),
	})
	if err != nil {
		t.Fatalf("NewAgent() error = %v", err)
	}

	result, err := agent.Run(context.Background(), "test")
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Every event should have the current schema version.
	for i, e := range result.Events {
		if e.SchemaVersion != agentflow.SchemaVersion {
			t.Errorf("event[%d].SchemaVersion = %d, want %d", i, e.SchemaVersion, agentflow.SchemaVersion)
		}
	}
}
