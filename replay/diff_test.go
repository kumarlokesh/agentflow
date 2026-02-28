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
