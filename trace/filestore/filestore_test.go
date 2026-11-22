package filestore_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
	"github.com/kumarlokesh/agentflow/trace/storetest"
)

func open(t *testing.T, dir string, opts filestore.Options) *filestore.Store {
	t.Helper()
	s, err := filestore.Open(dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestContract(t *testing.T) {
	storetest.Run(t, func(t *testing.T) trace.Store { return open(t, t.TempDir(), filestore.Options{}) })
}

func TestContractSmallBlobThreshold(t *testing.T) {
	// Externalize almost every string, to exercise blob paths everywhere.
	storetest.Run(t, func(t *testing.T) trace.Store { return open(t, t.TempDir(), filestore.Options{BlobThreshold: 4}) })
}

func TestTruncatedFinalLine(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir, filestore.Options{})
	tr, run := trace.NewID(), trace.NewID()
	ctx := context.Background()
	if err := s.Append(ctx,
		storetest.Ev(tr, run, 0, storetest.Started("/", "a", "")),
		storetest.Ev(tr, run, 1, &trace.Annotation{Key: "k", Value: json.RawMessage(`1`)})); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	// Simulate a crash in the middle of writing seq 2.
	path := filepath.Join(dir, "traces", tr, run+".jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"v":1,"trace_id":"` + tr + `","run_id":"` + run + `","seq":2,"time":"2026-`); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	s2 := open(t, dir, filestore.Options{})
	evs, err := trace.Collect(s2.Events(ctx, run, 0))
	if err != nil {
		t.Fatalf("reading a run with a truncated tail: %v", err)
	}
	if len(evs) != 2 {
		t.Fatalf("got %d events, want 2 (truncated line dropped)", len(evs))
	}
	for info, err := range s2.Runs(ctx, trace.RunFilter{TraceID: tr}) {
		if err != nil {
			t.Fatal(err)
		}
		if info.Status != trace.StatusIncomplete {
			t.Errorf("status = %s, want incomplete", info.Status)
		}
	}

	// Appending after reopen continues at seq 2 and stays well-formed.
	if err := s2.Append(ctx, storetest.Ev(tr, run, 2, storetest.Finished(trace.StatusCompleted, "ok"))); err != nil {
		t.Fatalf("append after crash: %v", err)
	}
	evs, err = trace.Collect(s2.Events(ctx, run, 0))
	if err != nil || len(evs) != 3 {
		t.Fatalf("after recovery: %d events, err %v", len(evs), err)
	}
	for info, err := range s2.Runs(ctx, trace.RunFilter{TraceID: tr}) {
		if err != nil || info.Status != trace.StatusCompleted {
			t.Errorf("after recovery: status %s, err %v", info.Status, err)
		}
	}
}

func TestBlobsAreExternalizedAndDeduplicated(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir, filestore.Options{})
	tr, run := trace.NewID(), trace.NewID()
	big := strings.Repeat("x", 100<<10)
	ctx := context.Background()
	if err := s.Append(ctx,
		storetest.Ev(tr, run, 0, storetest.Started("/", "a", "")),
		storetest.Ev(tr, run, 1, &trace.ToolCompleted{CallID: "c", Content: llm.Blocks{llm.Text{Text: big}}}),
		storetest.Ev(tr, run, 2, &trace.MessageAppended{Message: &llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{
			llm.ToolResult{CallID: "c", Content: llm.Blocks{llm.Text{Text: big}}}}}}),
	); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "traces", tr, run+".jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > 4<<10 {
		t.Errorf("run file is %d bytes; large strings should live in blobs", len(raw))
	}
	if !strings.Contains(string(raw), `"$blob":"sha256:`) {
		t.Error("run file has no blob reference")
	}
	var blobFiles int
	err = filepath.WalkDir(filepath.Join(dir, "blobs"), func(_ string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			blobFiles++
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if blobFiles != 1 {
		t.Errorf("got %d blob files, want 1 (the duplicate payload is stored once)", blobFiles)
	}
}

func TestPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permissions")
	}
	dir := filepath.Join(t.TempDir(), "store")
	s := open(t, dir, filestore.Options{BlobThreshold: 4})
	tr, run := trace.NewID(), trace.NewID()
	if err := s.Append(context.Background(), storetest.Ev(tr, run, 0, storetest.Started("/", "agent-with-long-name", ""))); err != nil {
		t.Fatal(err)
	}
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		perm := info.Mode().Perm()
		if d.IsDir() && perm != 0o700 {
			t.Errorf("%s: dir perm %o, want 700", path, perm)
		}
		if !d.IsDir() && perm != 0o600 {
			t.Errorf("%s: file perm %o, want 600", path, perm)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCorruptBlobDetected(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir, filestore.Options{})
	ref, err := s.PutBlob(context.Background(), []byte("original"))
	if err != nil {
		t.Fatal(err)
	}
	h := strings.TrimPrefix(string(ref), "sha256:")
	if err := os.WriteFile(filepath.Join(dir, "blobs", "sha256", h[:2], h[2:]), []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Blob(context.Background(), ref); err == nil {
		t.Fatal("corrupt blob was not detected")
	}
}

func TestRejectsPathTraversalIDs(t *testing.T) {
	s := open(t, t.TempDir(), filestore.Options{})
	for _, id := range []string{"../evil", "a/b", ".."} {
		e := storetest.Ev(trace.NewID(), id, 0, storetest.Started("/", "a", ""))
		if err := s.Append(context.Background(), e); err == nil {
			t.Errorf("run id %q accepted", id)
		}
	}
}
