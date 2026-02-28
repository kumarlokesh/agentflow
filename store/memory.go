// Package store provides EventStore implementations for the agentflow runtime.
package store

import (
	"context"
	"sort"
	"sync"

	"github.com/kumarlokesh/agentflow"
)

// Memory is a thread-safe, in-memory EventStore. It is suitable for tests,
// short-lived agents, and development. Data is lost when the process exits.
type Memory struct {
	mu     sync.RWMutex
	events map[string][]agentflow.Event // runID -> events
}

// NewMemory creates an empty in-memory event store.
func NewMemory() *Memory {
	return &Memory{
		events: make(map[string][]agentflow.Event),
	}
}

// Append adds an event to the in-memory store.
func (m *Memory) Append(_ context.Context, event agentflow.Event) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events[event.RunID] = append(m.events[event.RunID], event)
	return nil
}

// LoadEvents returns all events for a run, ordered by StepIndex then Timestamp.
func (m *Memory) LoadEvents(_ context.Context, runID string) ([]agentflow.Event, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	events := m.events[runID]
	if len(events) == 0 {
		return nil, nil
	}

	// Return a sorted copy to avoid mutating internal state.
	out := make([]agentflow.Event, len(events))
	copy(out, events)
	sortEvents(out)
	return out, nil
}

// LoadEventsByType returns events of a specific type for a run.
func (m *Memory) LoadEventsByType(_ context.Context, runID string, eventType agentflow.EventType) ([]agentflow.Event, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var out []agentflow.Event
	for _, e := range m.events[runID] {
		if e.Type == eventType {
			out = append(out, e)
		}
	}
	sortEvents(out)
	return out, nil
}

// ListRuns returns all run IDs that have at least one event.
func (m *Memory) ListRuns(_ context.Context) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	runs := make([]string, 0, len(m.events))
	for runID := range m.events {
		runs = append(runs, runID)
	}
	sort.Strings(runs)
	return runs, nil
}

func sortEvents(events []agentflow.Event) {
	sort.SliceStable(events, func(i, j int) bool {
		if events[i].StepIndex != events[j].StepIndex {
			return events[i].StepIndex < events[j].StepIndex
		}
		return events[i].Timestamp.Before(events[j].Timestamp)
	})
}
