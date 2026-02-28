package store

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
)

func tempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "agentflow-store-test-*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

func TestFile_NewFile_CreatesDir(t *testing.T) {
	dir := filepath.Join(tempDir(t), "nested", "store")

	fs, err := NewFile(dir)
	if err != nil {
		t.Fatalf("NewFile() error = %v", err)
	}
	if fs == nil {
		t.Fatal("NewFile() returned nil")
	}

	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("directory not created: %v", err)
	}
	if !info.IsDir() {
		t.Error("path is not a directory")
	}
}

func TestFile_AppendAndLoad(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()

	fs, err := NewFile(dir)
	if err != nil {
		t.Fatalf("NewFile() error = %v", err)
	}

	now := time.Now().UTC()
	events := []agentflow.Event{
		makeEvent("run-1", 0, agentflow.EventRunStart, now),
		makeEvent("run-1", 0, agentflow.EventLLMRequest, now.Add(time.Millisecond)),
		makeEvent("run-1", 1, agentflow.EventStepStart, now.Add(2*time.Millisecond)),
	}

	for _, e := range events {
		if err := fs.Append(ctx, e); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
	}

	loaded, err := fs.LoadEvents(ctx, "run-1")
	if err != nil {
		t.Fatalf("LoadEvents() error = %v", err)
	}
	if len(loaded) != 3 {
		t.Fatalf("LoadEvents() len = %d, want 3", len(loaded))
	}

	// Check file exists on disk.
	path := filepath.Join(dir, "run-1.jsonl")
	if _, err := os.Stat(path); err != nil {
		t.Errorf("JSONL file not created: %v", err)
	}
}

func TestFile_LoadEvents_Empty(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()

	fs, _ := NewFile(dir)

	events, err := fs.LoadEvents(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("LoadEvents() error = %v", err)
	}
	if events != nil {
		t.Errorf("expected nil for nonexistent run, got %d events", len(events))
	}
}

func TestFile_LoadEventsByType(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()
	fs, _ := NewFile(dir)
	now := time.Now().UTC()

	fs.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))
	fs.Append(ctx, makeEvent("run-1", 0, agentflow.EventLLMRequest, now.Add(time.Millisecond)))
	fs.Append(ctx, makeEvent("run-1", 1, agentflow.EventLLMRequest, now.Add(2*time.Millisecond)))

	llmReqs, err := fs.LoadEventsByType(ctx, "run-1", agentflow.EventLLMRequest)
	if err != nil {
		t.Fatalf("LoadEventsByType() error = %v", err)
	}
	if len(llmReqs) != 2 {
		t.Fatalf("len = %d, want 2", len(llmReqs))
	}
}

func TestFile_ListRuns(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()
	fs, _ := NewFile(dir)
	now := time.Now().UTC()

	fs.Append(ctx, makeEvent("run-beta", 0, agentflow.EventRunStart, now))
	fs.Append(ctx, makeEvent("run-alpha", 0, agentflow.EventRunStart, now))

	runs, err := fs.ListRuns(ctx)
	if err != nil {
		t.Fatalf("ListRuns() error = %v", err)
	}
	if len(runs) != 2 {
		t.Fatalf("ListRuns() len = %d, want 2", len(runs))
	}
	if runs[0] != "run-alpha" || runs[1] != "run-beta" {
		t.Errorf("ListRuns() = %v, want [run-alpha, run-beta]", runs)
	}
}

func TestFile_Persistence(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()
	now := time.Now().UTC()

	// Write with first store instance.
	fs1, _ := NewFile(dir)
	fs1.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))

	// Read with second store instance (simulating process restart).
	fs2, _ := NewFile(dir)
	events, err := fs2.LoadEvents(ctx, "run-1")
	if err != nil {
		t.Fatalf("LoadEvents() error = %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event after restart, got %d", len(events))
	}
}

func TestFile_JSONLFormat(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()
	now := time.Now().UTC()

	fs, _ := NewFile(dir)
	fs.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))

	// Read raw file and verify it's valid JSONL.
	path := filepath.Join(dir, "run-1.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}

	var event agentflow.Event
	if err := json.Unmarshal(data[:len(data)-1], &event); err != nil { // trim trailing newline
		t.Fatalf("invalid JSONL: %v", err)
	}
	if event.Type != agentflow.EventRunStart {
		t.Errorf("type = %q, want %q", event.Type, agentflow.EventRunStart)
	}
}

func TestFile_PathTraversal(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()
	now := time.Now().UTC()

	fs, _ := NewFile(dir)

	// A malicious runID with path traversal should be sanitized.
	maliciousID := "../../../etc/evil"
	fs.Append(ctx, makeEvent(maliciousID, 0, agentflow.EventRunStart, now))

	// The file should be written inside the base dir, not outside.
	// filepath.Base("../../../etc/evil") == "evil"
	expectedPath := filepath.Join(dir, "evil.jsonl")
	if _, err := os.Stat(expectedPath); err != nil {
		t.Errorf("expected sanitized file at %s, got error: %v", expectedPath, err)
	}

	// The escaped path should NOT exist.
	escapedPath := filepath.Join(dir, maliciousID+".jsonl")
	if _, err := os.Stat(escapedPath); err == nil {
		t.Error("path traversal was not prevented — file written outside base dir")
	}

	// Loading should also use the sanitized path.
	events, err := fs.LoadEvents(ctx, maliciousID)
	if err != nil {
		t.Fatalf("LoadEvents() error = %v", err)
	}
	if len(events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events))
	}
}

func TestFile_SortOrder(t *testing.T) {
	dir := tempDir(t)
	ctx := context.Background()
	now := time.Now().UTC()

	fs, _ := NewFile(dir)

	// Append out of order.
	fs.Append(ctx, makeEvent("run-1", 2, agentflow.EventStepStart, now.Add(2*time.Second)))
	fs.Append(ctx, makeEvent("run-1", 0, agentflow.EventRunStart, now))
	fs.Append(ctx, makeEvent("run-1", 1, agentflow.EventStepStart, now.Add(time.Second)))

	events, _ := fs.LoadEvents(ctx, "run-1")
	for i := 1; i < len(events); i++ {
		if events[i].StepIndex < events[i-1].StepIndex {
			t.Errorf("events not sorted: step[%d]=%d < step[%d]=%d",
				i, events[i].StepIndex, i-1, events[i-1].StepIndex)
		}
	}
}
