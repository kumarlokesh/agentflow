package trace

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"
)

// Store persists traces. Implementations must be safe for
// concurrent use. Events returned by a Store are fully resolved: blob
// references are replaced by their content.
type Store interface {
	// Append writes events of a single run. events[0].Seq must equal the
	// run's next expected seq (0 for a new run), and seqs must be contiguous;
	// otherwise the error wraps ErrSeqConflict.
	Append(ctx context.Context, events ...Event) error
	// Events yields a run's events in seq order, starting at from.
	Events(ctx context.Context, runID string, from uint64) iter.Seq2[Event, error]
	// Runs lists runs, newest trace first; within a trace, in start order.
	Runs(ctx context.Context, f RunFilter) iter.Seq2[RunInfo, error]
	BlobStore
}

// BlobStore stores content-addressed blobs.
type BlobStore interface {
	PutBlob(ctx context.Context, data []byte) (BlobRef, error)
	Blob(ctx context.Context, ref BlobRef) ([]byte, error)
}

// RunInfo summarizes a run for listings.
type RunInfo struct {
	TraceID     string
	RunID       string
	Path        string // "/" for root runs
	Kind        RunKind
	ParentRunID string // empty for root runs
	Agent       string
	Origin      *Origin   // root runs only
	Status      RunStatus // run.finished status, or StatusIncomplete
	Started     time.Time
	Finished    time.Time // zero if incomplete
}

// RunFilter narrows Store.Runs.
type RunFilter struct {
	TraceID   string       // only this trace
	RootsOnly bool         // only root runs
	Origins   []OriginKind // only traces with these origins (root runs' origin)
	Limit     int          // maximum runs returned; 0 means no limit
}

// Match reports whether info passes the filter (ignoring Limit).
// traceOrigin is the origin recorded on the trace's root run (nil if the
// root is missing or has none, which counts as live); Origins filters whole
// traces by it.
func (f RunFilter) Match(info RunInfo, traceOrigin *Origin) bool {
	if f.TraceID != "" && info.TraceID != f.TraceID {
		return false
	}
	if f.RootsOnly && info.ParentRunID != "" {
		return false
	}
	if len(f.Origins) > 0 {
		kind := OriginLive
		if traceOrigin != nil {
			kind = traceOrigin.Kind
		}
		ok := false
		for _, o := range f.Origins {
			if o == kind {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	return true
}

// FilterRuns applies f (including Limit) to the runs of one or more traces,
// given in listing order, and yields the matches.
func FilterRuns(infos []RunInfo, f RunFilter, yield func(RunInfo, error) bool) {
	origins := map[string]*Origin{}
	for _, info := range infos {
		if info.ParentRunID == "" {
			origins[info.TraceID] = info.Origin
		}
	}
	n := 0
	for _, info := range infos {
		if !f.Match(info, origins[info.TraceID]) {
			continue
		}
		if f.Limit > 0 && n >= f.Limit {
			return
		}
		n++
		if !yield(info, nil) {
			return
		}
	}
}

// RunInfoFrom builds a RunInfo from a run's first event (run.started) and,
// if present, its run.finished event.
func RunInfoFrom(first Event, finished *Event) (RunInfo, error) {
	rs, ok := first.Body.(*RunStarted)
	if !ok {
		return RunInfo{}, fmt.Errorf("trace: run %s: first event is %s, not run.started", first.RunID, first.Kind())
	}
	info := RunInfo{
		TraceID: first.TraceID,
		RunID:   first.RunID,
		Path:    rs.Path,
		Kind:    rs.RunKind,
		Agent:   rs.Agent.Name,
		Origin:  rs.Origin,
		Status:  StatusIncomplete,
		Started: first.Time,
	}
	if rs.Parent != nil {
		info.ParentRunID = rs.Parent.RunID
	}
	if finished != nil {
		if rf, ok := finished.Body.(*RunFinished); ok {
			info.Status = rf.Status
			info.Finished = finished.Time
		}
	}
	return info, nil
}

// Collect drains an event iterator into a slice.
func Collect(seq iter.Seq2[Event, error]) ([]Event, error) {
	var out []Event
	for e, err := range seq {
		if err != nil {
			return out, err
		}
		out = append(out, e)
	}
	return out, nil
}

// Observer receives every event after it has been appended. Observe is
// called synchronously by the Writer and must not block.
type Observer interface {
	Observe(ctx context.Context, e Event)
}

// ObserverFunc adapts a function to Observer.
type ObserverFunc func(ctx context.Context, e Event)

// Observe implements Observer.
func (f ObserverFunc) Observe(ctx context.Context, e Event) { f(ctx, e) }

// WriterOptions configure a Writer.
type WriterOptions struct {
	// Now returns the wall-clock time for events. Defaults to time.Now.
	Now func() time.Time
	// Observers are notified after each successful append.
	Observers []Observer
}

// Writer appends events to one run, assigning seq numbers in order. It is
// safe for concurrent use; concurrent writes are serialized, so callers that
// need a deterministic order must provide it.
type Writer struct {
	mu        sync.Mutex
	store     Store
	traceID   string
	runID     string
	next      uint64
	now       func() time.Time
	observers []Observer
}

// NewWriter returns a Writer for a new run.
func NewWriter(store Store, traceID, runID string, opts WriterOptions) *Writer {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	return &Writer{store: store, traceID: traceID, runID: runID, now: now, observers: opts.Observers}
}

// TraceID returns the trace the writer writes to.
func (w *Writer) TraceID() string { return w.traceID }

// RunID returns the run the writer writes to.
func (w *Writer) RunID() string { return w.runID }

// Next returns the seq the next written event will get.
func (w *Writer) Next() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.next
}

// Write appends one event and returns its seq.
func (w *Writer) Write(ctx context.Context, body Body, cause *uint64) (uint64, error) {
	seqs, err := w.WriteBatch(ctx, []Pending{{Body: body, Cause: cause}})
	if err != nil {
		return 0, err
	}
	return seqs[0], nil
}

// Pending is an event body waiting to be written.
type Pending struct {
	Body  Body
	Cause *uint64
}

// WriteBatch appends several events atomically (one Store.Append) and
// returns their seqs.
func (w *Writer) WriteBatch(ctx context.Context, batch []Pending) ([]uint64, error) {
	if len(batch) == 0 {
		return nil, nil
	}
	w.mu.Lock()
	events := make([]Event, len(batch))
	seqs := make([]uint64, len(batch))
	for i, p := range batch {
		events[i] = Event{
			TraceID: w.traceID,
			RunID:   w.runID,
			Seq:     w.next + uint64(i),
			Time:    w.now().UTC(),
			Cause:   p.Cause,
			Body:    p.Body,
		}
		seqs[i] = events[i].Seq
	}
	if err := w.store.Append(ctx, events...); err != nil {
		w.mu.Unlock()
		return nil, err
	}
	w.next += uint64(len(batch))
	w.mu.Unlock()

	for _, e := range events {
		for _, o := range w.observers {
			o.Observe(ctx, e)
		}
	}
	return seqs, nil
}
