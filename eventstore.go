package agentflow

import "context"

// EventStore persists and retrieves events. Implementations must be safe for
// concurrent use from multiple goroutines.
//
// The store is append-only by design: events are never modified or deleted.
// This guarantees that the event log for a given run is an immutable,
// replayable record of the agent's execution.
type EventStore interface {
	// Append persists a single event. Events must be retrievable immediately
	// after Append returns without error.
	Append(ctx context.Context, event Event) error

	// LoadEvents returns all events for the given run ID, ordered by
	// StepIndex then Timestamp.
	LoadEvents(ctx context.Context, runID string) ([]Event, error)

	// LoadEventsByType returns events of a specific type for a run, ordered
	// by StepIndex then Timestamp.
	LoadEventsByType(ctx context.Context, runID string, eventType EventType) ([]Event, error)

	// ListRuns returns all known run IDs.
	ListRuns(ctx context.Context) ([]string, error)
}
