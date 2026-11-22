// Package memstore is an in-memory trace.Store, for tests and short-lived
// processes. Events are encoded on Append and decoded on read, exactly like
// the file store, so callers can't mutate stored events and anything that
// can't round-trip through the format fails at Append time.
package memstore

import (
	"context"
	"fmt"
	"iter"
	"sort"
	"sync"

	"github.com/kumarlokesh/agentflow/trace"
)

// Store is an in-memory trace.Store.
type Store struct {
	mu     sync.RWMutex
	runs   map[string]*run // by run ID
	traces map[string][]string
	blobs  map[trace.BlobRef][]byte
}

type run struct {
	traceID string
	lines   [][]byte
	first   trace.Event
	last    *trace.Event // latest run.finished, if any
}

var _ trace.Store = (*Store)(nil)

// New returns an empty store.
func New() *Store {
	return &Store{runs: map[string]*run{}, traces: map[string][]string{}, blobs: map[trace.BlobRef][]byte{}}
}

// Append implements trace.Store.
func (s *Store) Append(_ context.Context, events ...trace.Event) error {
	if len(events) == 0 {
		return nil
	}
	runID, traceID := events[0].RunID, events[0].TraceID
	lines := make([][]byte, len(events))
	for i, e := range events {
		if e.RunID != runID || e.TraceID != traceID {
			return fmt.Errorf("memstore: Append mixes runs (%s/%s and %s/%s)", traceID, runID, e.TraceID, e.RunID)
		}
		line, err := trace.EncodeEvent(e, trace.EncodeOptions{})
		if err != nil {
			return err
		}
		lines[i] = line
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[runID]
	next := uint64(0)
	if r != nil {
		next = uint64(len(r.lines))
		if r.traceID != traceID {
			return fmt.Errorf("memstore: run %s belongs to trace %s, not %s", runID, r.traceID, traceID)
		}
	}
	for i, e := range events {
		if e.Seq != next+uint64(i) {
			return fmt.Errorf("%w: run %s: got seq %d, want %d", trace.ErrSeqConflict, runID, e.Seq, next+uint64(i))
		}
	}
	if r == nil {
		if _, ok := events[0].Body.(*trace.RunStarted); !ok {
			return fmt.Errorf("memstore: run %s: first event must be run.started, got %s", runID, events[0].Kind())
		}
		// Keep decoded copies, never the caller's bodies.
		first, err := trace.DecodeEvent(lines[0], nil)
		if err != nil {
			return err
		}
		r = &run{traceID: traceID, first: first}
		s.runs[runID] = r
		s.traces[traceID] = append(s.traces[traceID], runID)
	}
	r.lines = append(r.lines, lines...)
	for i := range events {
		if _, ok := events[i].Body.(*trace.RunFinished); ok {
			last, err := trace.DecodeEvent(lines[i], nil)
			if err != nil {
				return err
			}
			r.last = &last
		}
	}
	return nil
}

// Events implements trace.Store.
func (s *Store) Events(_ context.Context, runID string, from uint64) iter.Seq2[trace.Event, error] {
	return func(yield func(trace.Event, error) bool) {
		s.mu.RLock()
		r := s.runs[runID]
		var lines [][]byte
		if r != nil {
			lines = r.lines
		}
		s.mu.RUnlock()
		if r == nil {
			yield(trace.Event{}, fmt.Errorf("%w: %s", trace.ErrRunNotFound, runID))
			return
		}
		for i := from; i < uint64(len(lines)); i++ {
			e, err := trace.DecodeEvent(lines[i], s.getBlob)
			if !yield(e, err) || err != nil {
				return
			}
		}
	}
}

// Runs implements trace.Store.
func (s *Store) Runs(_ context.Context, f trace.RunFilter) iter.Seq2[trace.RunInfo, error] {
	return func(yield func(trace.RunInfo, error) bool) {
		s.mu.RLock()
		traceIDs := make([]string, 0, len(s.traces))
		for id := range s.traces {
			traceIDs = append(traceIDs, id)
		}
		sort.Sort(sort.Reverse(sort.StringSlice(traceIDs))) // UUIDv7: newest first
		var infos []trace.RunInfo
		var buildErr error
		for _, tid := range traceIDs {
			if f.TraceID != "" && tid != f.TraceID {
				continue
			}
			for _, rid := range s.traces[tid] {
				r := s.runs[rid]
				info, err := trace.RunInfoFrom(r.first, r.last)
				if err != nil {
					buildErr = err
					break
				}
				infos = append(infos, info)
			}
		}
		s.mu.RUnlock()
		if buildErr != nil {
			yield(trace.RunInfo{}, buildErr)
			return
		}
		trace.FilterRuns(infos, f, yield)
	}
}

// PutBlob implements trace.BlobStore.
func (s *Store) PutBlob(_ context.Context, data []byte) (trace.BlobRef, error) {
	ref := trace.BlobRefOf(data)
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.blobs[ref]; !ok {
		s.blobs[ref] = append([]byte(nil), data...)
	}
	return ref, nil
}

// Blob implements trace.BlobStore.
func (s *Store) Blob(_ context.Context, ref trace.BlobRef) ([]byte, error) {
	return s.getBlob(ref)
}

func (s *Store) getBlob(ref trace.BlobRef) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.blobs[ref]
	if !ok {
		return nil, fmt.Errorf("%w: %s", trace.ErrBlobNotFound, ref)
	}
	return append([]byte(nil), data...), nil
}
