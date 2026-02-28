package replay

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/store"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
}

// buildRecordedRun creates a realistic event sequence in the store that
// simulates a completed agent run with one tool call.
func buildRecordedRun(t *testing.T, s *store.Memory, runID string) {
	t.Helper()
	ctx := context.Background()
	now := time.Now().UTC()
	step := 0

	mustEvent := func(eventType agentflow.EventType, stepIdx int, data any, offset time.Duration) {
		raw, err := json.Marshal(data)
		if err != nil {
			t.Fatalf("marshal data: %v", err)
		}
		err = s.Append(ctx, agentflow.Event{
			ID:            "evt-" + string(eventType) + "-" + runID,
			SchemaVersion: agentflow.SchemaVersion,
			Type:          eventType,
			RunID:         runID,
			StepIndex:     stepIdx,
			Timestamp:     now.Add(offset),
			Data:          raw,
		})
		if err != nil {
			t.Fatalf("append event: %v", err)
		}
	}

	// Run start
	mustEvent(agentflow.EventRunStart, step, agentflow.RunStartData{
		Task:         "What is 2+2?",
		Instructions: "Use the calculator.",
		Tools:        []string{"calculator"},
		MaxSteps:     10,
	}, 0)

	// Step 0: LLM requests tool call
	mustEvent(agentflow.EventStepStart, step, agentflow.StepStartData{StepIndex: 0}, time.Millisecond)
	mustEvent(agentflow.EventLLMRequest, step, agentflow.LLMRequestData{
		Messages: []agentflow.Message{{Role: "user", Content: "What is 2+2?"}},
	}, 2*time.Millisecond)
	mustEvent(agentflow.EventLLMResponse, step, agentflow.LLMResponseData{
		ToolCalls: []agentflow.ToolCallRequest{
			{ID: "call-1", Name: "calculator", Arguments: json.RawMessage(`{"expression":"2+2"}`)},
		},
	}, 3*time.Millisecond)
	mustEvent(agentflow.EventToolCall, step, agentflow.ToolCallData{
		ToolName: "calculator", CallID: "call-1", Input: json.RawMessage(`{"expression":"2+2"}`),
	}, 4*time.Millisecond)
	mustEvent(agentflow.EventToolResult, step, agentflow.ToolResultData{
		ToolName: "calculator", CallID: "call-1", Output: "4", Duration: time.Millisecond,
	}, 5*time.Millisecond)
	mustEvent(agentflow.EventStepEnd, step, agentflow.StepEndData{StepIndex: 0, Duration: 5 * time.Millisecond}, 6*time.Millisecond)

	// Step 1: LLM gives final answer
	step = 1
	mustEvent(agentflow.EventStepStart, step, agentflow.StepStartData{StepIndex: 1}, 7*time.Millisecond)
	mustEvent(agentflow.EventLLMRequest, step, agentflow.LLMRequestData{}, 8*time.Millisecond)
	mustEvent(agentflow.EventLLMResponse, step, agentflow.LLMResponseData{
		Content: "2 + 2 = 4",
	}, 9*time.Millisecond)
	mustEvent(agentflow.EventStepEnd, step, agentflow.StepEndData{StepIndex: 1, Duration: 2 * time.Millisecond}, 10*time.Millisecond)

	// Run end
	mustEvent(agentflow.EventRunEnd, step, agentflow.RunEndData{
		Status: "completed", Output: "2 + 2 = 4", Steps: 2, Duration: 10 * time.Millisecond,
	}, 11*time.Millisecond)
}

func TestEngine_Replay_ProducesIdenticalOutput(t *testing.T) {
	memStore := store.NewMemory()
	runID := "original-run-001"
	buildRecordedRun(t, memStore, runID)

	engine := NewEngine(memStore, quietLogger())
	result, err := engine.Replay(context.Background(), runID)
	if err != nil {
		t.Fatalf("Replay() error = %v", err)
	}

	if !result.Match {
		t.Error("Replay did not match original output")
	}
	if result.Output != "2 + 2 = 4" {
		t.Errorf("Output = %q, want %q", result.Output, "2 + 2 = 4")
	}
	if result.RunID != runID {
		t.Errorf("RunID = %q, want %q", result.RunID, runID)
	}
	if result.ReplayRunID == "" {
		t.Error("ReplayRunID is empty")
	}
	if result.ReplayRunID == runID {
		t.Error("ReplayRunID should differ from original RunID")
	}
	if result.Steps != 2 {
		t.Errorf("Steps = %d, want 2", result.Steps)
	}
	if result.Duration <= 0 {
		t.Error("Duration should be positive")
	}
}

func TestEngine_Replay_ProducesEvents(t *testing.T) {
	memStore := store.NewMemory()
	runID := "event-run-001"
	buildRecordedRun(t, memStore, runID)

	engine := NewEngine(memStore, quietLogger())
	result, err := engine.Replay(context.Background(), runID)
	if err != nil {
		t.Fatalf("Replay() error = %v", err)
	}

	if len(result.Events) == 0 {
		t.Fatal("Replay produced no events")
	}

	// Verify the replay run also has events in the store.
	replayEvents, err := memStore.LoadEvents(context.Background(), result.ReplayRunID)
	if err != nil {
		t.Fatalf("LoadEvents for replay: %v", err)
	}
	if len(replayEvents) == 0 {
		t.Error("Replay events not persisted to store")
	}
}

func TestEngine_Replay_NonexistentRun(t *testing.T) {
	memStore := store.NewMemory()
	engine := NewEngine(memStore, quietLogger())

	_, err := engine.Replay(context.Background(), "nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent run")
	}
}

func TestEngine_Replay_ContextCancellation(t *testing.T) {
	memStore := store.NewMemory()
	runID := "cancel-run-001"
	buildRecordedRun(t, memStore, runID)

	engine := NewEngine(memStore, quietLogger())

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := engine.Replay(ctx, runID)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

// buildDirectAnswerRun creates a simpler run with no tool calls.
func buildDirectAnswerRun(t *testing.T, s *store.Memory, runID, output string) {
	t.Helper()
	ctx := context.Background()
	now := time.Now().UTC()

	mustEvent := func(eventType agentflow.EventType, stepIdx int, data any, offset time.Duration) {
		raw, _ := json.Marshal(data)
		s.Append(ctx, agentflow.Event{
			ID:            "evt-" + string(eventType) + "-" + runID,
			SchemaVersion: agentflow.SchemaVersion,
			Type:          eventType,
			RunID:         runID,
			StepIndex:     stepIdx,
			Timestamp:     now.Add(offset),
			Data:          raw,
		})
	}

	mustEvent(agentflow.EventRunStart, 0, agentflow.RunStartData{
		Task: "Question", MaxSteps: 10,
	}, 0)
	mustEvent(agentflow.EventStepStart, 0, agentflow.StepStartData{StepIndex: 0}, time.Millisecond)
	mustEvent(agentflow.EventLLMRequest, 0, agentflow.LLMRequestData{}, 2*time.Millisecond)
	mustEvent(agentflow.EventLLMResponse, 0, agentflow.LLMResponseData{Content: output}, 3*time.Millisecond)
	mustEvent(agentflow.EventStepEnd, 0, agentflow.StepEndData{StepIndex: 0}, 4*time.Millisecond)
	mustEvent(agentflow.EventRunEnd, 0, agentflow.RunEndData{
		Status: "completed", Output: output, Steps: 1,
	}, 5*time.Millisecond)
}

func TestEngine_Replay_DirectAnswer(t *testing.T) {
	memStore := store.NewMemory()
	runID := "direct-run-001"
	buildDirectAnswerRun(t, memStore, runID, "Hello, World!")

	engine := NewEngine(memStore, quietLogger())
	result, err := engine.Replay(context.Background(), runID)
	if err != nil {
		t.Fatalf("Replay() error = %v", err)
	}

	if !result.Match {
		t.Error("expected match for direct answer replay")
	}
	if result.Output != "Hello, World!" {
		t.Errorf("Output = %q, want %q", result.Output, "Hello, World!")
	}
}
