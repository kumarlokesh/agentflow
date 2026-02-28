package store

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/kumarlokesh/agentflow"
)

// File is a persistent EventStore backed by JSONL (JSON Lines) files.
// Each run is stored in its own file at <baseDir>/<runID>.jsonl.
// The format is append-only: one JSON-encoded Event per line.
//
// File is safe for concurrent use within a single process. It does not
// provide cross-process locking.
type File struct {
	mu      sync.Mutex
	baseDir string
}

// NewFile creates a File store rooted at baseDir. The directory is created
// if it does not exist.
func NewFile(baseDir string) (*File, error) {
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, fmt.Errorf("store: create base dir: %w", err)
	}
	return &File{baseDir: baseDir}, nil
}

// Append writes an event as a single JSON line to the run's JSONL file.
func (f *File) Append(_ context.Context, event agentflow.Event) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	path := f.runPath(event.RunID)
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("store: open %s: %w", path, err)
	}
	defer file.Close()

	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("store: marshal event: %w", err)
	}
	data = append(data, '\n')

	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("store: write event: %w", err)
	}
	return nil
}

// LoadEvents reads all events for a run from its JSONL file.
func (f *File) LoadEvents(_ context.Context, runID string) ([]agentflow.Event, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.loadEventsLocked(runID)
}

// LoadEventsByType returns events of a specific type for a run.
func (f *File) LoadEventsByType(_ context.Context, runID string, eventType agentflow.EventType) ([]agentflow.Event, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	all, err := f.loadEventsLocked(runID)
	if err != nil {
		return nil, err
	}

	var filtered []agentflow.Event
	for _, e := range all {
		if e.Type == eventType {
			filtered = append(filtered, e)
		}
	}
	return filtered, nil
}

// ListRuns returns all run IDs by scanning for .jsonl files in the base directory.
func (f *File) ListRuns(_ context.Context) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	entries, err := os.ReadDir(f.baseDir)
	if err != nil {
		return nil, fmt.Errorf("store: read dir: %w", err)
	}

	var runs []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := filepath.Ext(name)
		if ext == ".jsonl" {
			runs = append(runs, name[:len(name)-len(ext)])
		}
	}
	sort.Strings(runs)
	return runs, nil
}

func (f *File) loadEventsLocked(runID string) ([]agentflow.Event, error) {
	path := f.runPath(runID)

	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("store: open %s: %w", path, err)
	}
	defer file.Close()

	var events []agentflow.Event
	scanner := bufio.NewScanner(file)
	// Increase buffer for large events (1 MB max line).
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var event agentflow.Event
		if err := json.Unmarshal(line, &event); err != nil {
			return nil, fmt.Errorf("store: parse line %d in %s: %w", lineNum, path, err)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("store: scan %s: %w", path, err)
	}

	sortEvents(events)
	return events, nil
}

func (f *File) runPath(runID string) string {
	// Sanitize runID to prevent path traversal attacks.
	clean := filepath.Base(runID)
	return filepath.Join(f.baseDir, clean+".jsonl")
}
