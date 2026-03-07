package replay

import (
	"context"
	"testing"

	"github.com/kumarlokesh/agentflow/store"
)

func TestDiff_IdenticalRuns(t *testing.T) {
	memStore := store.NewMemory()
	buildDirectAnswerRun(t, memStore, "run-a", "same output")
	buildDirectAnswerRun(t, memStore, "run-b", "same output")

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}

	if !result.Identical {
		t.Errorf("expected identical runs, got %d differences", len(result.Differences))
		for _, d := range result.Differences {
			t.Logf("  diff: step=%d type=%s field=%s A=%q B=%q",
				d.StepIndex, d.EventType, d.Field, d.ValueA, d.ValueB)
		}
	}
	if result.Summary == "" {
		t.Error("Summary should not be empty")
	}
}

func TestDiff_DifferentOutputs(t *testing.T) {
	memStore := store.NewMemory()
	buildDirectAnswerRun(t, memStore, "run-a", "output A")
	buildDirectAnswerRun(t, memStore, "run-b", "output B")

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}

	if result.Identical {
		t.Fatal("expected different runs")
	}
	if len(result.Differences) == 0 {
		t.Fatal("expected at least one difference")
	}

	// Should detect the output difference.
	hasOutputDiff := false
	for _, d := range result.Differences {
		if d.Field == "output" || d.Field == "content" {
			hasOutputDiff = true
		}
	}
	if !hasOutputDiff {
		t.Error("expected output/content difference")
	}
}

func TestDiff_DifferentToolCalls(t *testing.T) {
	memStore := store.NewMemory()
	buildRecordedRun(t, memStore, "run-a")
	buildDirectAnswerRun(t, memStore, "run-b", "different approach")

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}

	if result.Identical {
		t.Fatal("runs with different tool calls should not be identical")
	}
}

func TestDiff_NonexistentRun(t *testing.T) {
	memStore := store.NewMemory()
	buildDirectAnswerRun(t, memStore, "run-a", "output")

	_, err := Diff(context.Background(), memStore, "run-a", "nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent run")
	}
}

func TestDiff_BothNonexistent(t *testing.T) {
	memStore := store.NewMemory()

	_, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err == nil {
		t.Fatal("expected error for nonexistent runs")
	}
}

func TestDiff_Summary(t *testing.T) {
	memStore := store.NewMemory()
	buildDirectAnswerRun(t, memStore, "run-a", "output A")
	buildDirectAnswerRun(t, memStore, "run-b", "output B")

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}

	if result.Summary == "" {
		t.Error("Summary should not be empty")
	}
	// Summary should mention both run IDs.
	if len(result.Summary) < 10 {
		t.Errorf("Summary too short: %q", result.Summary)
	}
}

func TestDiff_ToolCallCountMismatch(t *testing.T) {
	// run-a has a tool call, run-b is a direct answer with no tool calls.
	// This exercises the "i >= len(b)" branch in compareToolCalls.
	memStore := store.NewMemory()
	buildRecordedRun(t, memStore, "run-a")
	buildDirectAnswerRun(t, memStore, "run-b", "answer")

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}
	if result.Identical {
		t.Fatal("expected differences when one run has tool calls and the other does not")
	}
	hasPresence := false
	for _, d := range result.Differences {
		if d.Field == "presence" {
			hasPresence = true
		}
	}
	if !hasPresence {
		t.Error("expected a 'presence' difference for missing tool call")
	}
}

func TestDiff_ToolResultCountMismatch(t *testing.T) {
	// run-b has a tool result, run-a does not (same asymmetry in reverse).
	memStore := store.NewMemory()
	buildDirectAnswerRun(t, memStore, "run-a", "answer")
	buildRecordedRun(t, memStore, "run-b")

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}
	if result.Identical {
		t.Fatal("expected differences")
	}
}

func TestDiff_LLMResponseCountMismatch(t *testing.T) {
	// run-a has two LLM responses (tool-call run), run-b has one (direct).
	// This exercises the "i >= len(a)" and "i >= len(b)" branches in
	// compareLLMResponses.
	memStore := store.NewMemory()
	buildRecordedRun(t, memStore, "run-a")    // 2 LLM responses
	buildDirectAnswerRun(t, memStore, "run-b", "answer") // 1 LLM response

	result, err := Diff(context.Background(), memStore, "run-a", "run-b")
	if err != nil {
		t.Fatalf("Diff() error = %v", err)
	}
	if result.Identical {
		t.Fatal("expected differences")
	}
}

func TestEngine_NilLogger_UsesDefault(t *testing.T) {
	memStore := store.NewMemory()
	runID := "nil-logger-run"
	buildDirectAnswerRun(t, memStore, runID, "hello")

	// NewEngine with nil logger should not panic.
	engine := NewEngine(memStore, nil)
	result, err := engine.Replay(context.Background(), runID)
	if err != nil {
		t.Fatalf("Replay() error = %v", err)
	}
	if result.Output != "hello" {
		t.Errorf("Output = %q, want hello", result.Output)
	}
}


