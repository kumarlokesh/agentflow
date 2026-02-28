package store

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
)

func makeEvent(runID string, step int, eventType agentflow.EventType, ts time.Time) agentflow.Event {
	return agentflow.Event{
		ID:            "evt-" + runID + "-" + string(eventType),
		SchemaVersion: agentflow.SchemaVersion,
		Type:          eventType,
		RunID:         runID,
		StepIndex:     step,
		Timestamp:     ts,
		Data:          json.RawMessage(`{}`),
	}
}

func TestMemory_AppendAndLoad(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	now := time.Now().UTC()

	events := []agentflow.Event{
		makeEvent("run-1", 0, agentflow.EventRunStart, now),
		makeEvent("run-1", 0, agentflow.EventLLMRequest, now.Add(time.Millisecond)),
		makeEvent("run-1", 0, agentflow.EventLLMResponse, now.Add(2*time.Millisecond)),
		makeEvent("run-1", 1, agentflow.EventStepStart, now.Add(3*time.Millisecond)),
	}

	for _, e := range events {
		if err := m.Append(ctx, e); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
	}

	loaded, err := m.LoadEvents(ctx, "run-1")
	if err != nil {
		t.Fatalf("LoadEvents() error = %v", err)
	}
	if len(loaded) != 4 {
		t.Fatalf("LoadEvents() len = %d, want 4", len(loaded))
	}

	// Verify ordering: step 0 events before step 1.
	if loaded[0].StepIndex != 0 || loaded[3].StepIndex != 1 {
		t.Error("events not sorted by step index")
	}
}

func TestMemory_LoadEvents_Empty(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()

	events, err := m.LoadEvents(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("LoadEvents() error = %v", err)
	}
	if events != nil {
		t.Errorf("expected nil for nonexistent run, got %v", events)
	}
}

func TestMemory_LoadEventsByType(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	now := time.Now().UTC()

	m.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))
	m.Append(ctx, makeEvent("run-1", 0, agentflow.EventLLMRequest, now.Add(time.Millisecond)))
	m.Append(ctx, makeEvent("run-1", 0, agentflow.EventLLMResponse, now.Add(2*time.Millisecond)))
	m.Append(ctx, makeEvent("run-1", 1, agentflow.EventLLMRequest, now.Add(3*time.Millisecond)))

	llmReqs, err := m.LoadEventsByType(ctx, "run-1", agentflow.EventLLMRequest)
	if err != nil {
		t.Fatalf("LoadEventsByType() error = %v", err)
	}
	if len(llmReqs) != 2 {
		t.Fatalf("LoadEventsByType() len = %d, want 2", len(llmReqs))
	}
	for _, e := range llmReqs {
		if e.Type != agentflow.EventLLMRequest {
			t.Errorf("unexpected type: %s", e.Type)
		}
	}
}

func TestMemory_ListRuns(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	now := time.Now().UTC()

	m.Append(ctx, makeEvent("run-b", 0, agentflow.EventRunStart, now))
	m.Append(ctx, makeEvent("run-a", 0, agentflow.EventRunStart, now))

	runs, err := m.ListRuns(ctx)
	if err != nil {
		t.Fatalf("ListRuns() error = %v", err)
	}
	if len(runs) != 2 {
		t.Fatalf("ListRuns() len = %d, want 2", len(runs))
	}
	// Should be sorted.
	if runs[0] != "run-a" || runs[1] != "run-b" {
		t.Errorf("ListRuns() = %v, want [run-a, run-b]", runs)
	}
}

func TestMemory_LoadReturnsACopy(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	now := time.Now().UTC()

	m.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))

	loaded1, _ := m.LoadEvents(ctx, "run-1")
	loaded2, _ := m.LoadEvents(ctx, "run-1")

	// Mutating one slice should not affect the other.
	loaded1[0].StepIndex = 999
	if loaded2[0].StepIndex == 999 {
		t.Error("LoadEvents returns a reference to internal state, should return a copy")
	}
}

func TestMemory_ConcurrentAppendAndLoad(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	now := time.Now().UTC()

	var wg sync.WaitGroup

	// Concurrent appends.
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			m.Append(ctx, makeEvent("run-1", i, agentflow.EventStepStart, now.Add(time.Duration(i)*time.Millisecond)))
		}(i)
	}

	// Concurrent reads.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m.LoadEvents(ctx, "run-1")
			m.ListRuns(ctx)
		}()
	}

	wg.Wait()

	events, _ := m.LoadEvents(ctx, "run-1")
	if len(events) != 100 {
		t.Errorf("expected 100 events, got %d", len(events))
	}
}

func TestMemory_MultipleRuns(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	now := time.Now().UTC()

	m.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))
	m.Append(ctx, makeEvent("run-1", 0, agentflow.EventLLMRequest, now))
	m.Append(ctx, makeEvent("run-2", 0, agentflow.EventRunStart, now))

	events1, _ := m.LoadEvents(ctx, "run-1")
	events2, _ := m.LoadEvents(ctx, "run-2")

	if len(events1) != 2 {
		t.Errorf("run-1 events = %d, want 2", len(events1))
	}
	if len(events2) != 1 {
		t.Errorf("run-2 events = %d, want 1", len(events2))
	}
}
