// Package storetest is a contract test suite that every trace.Store
// implementation must pass.
package storetest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// Run runs the contract suite. newStore must return a fresh, empty store.
func Run(t *testing.T, newStore func(t *testing.T) trace.Store) {
	t.Helper()
	t.Run("AppendAndRead", func(t *testing.T) { testAppendAndRead(t, newStore(t)) })
	t.Run("SeqConflicts", func(t *testing.T) { testSeqConflicts(t, newStore(t)) })
	t.Run("EventsFrom", func(t *testing.T) { testEventsFrom(t, newStore(t)) })
	t.Run("RunsAndTree", func(t *testing.T) { testRunsAndTree(t, newStore(t)) })
	t.Run("RunFilters", func(t *testing.T) { testRunFilters(t, newStore(t)) })
	t.Run("Blobs", func(t *testing.T) { testBlobs(t, newStore(t)) })
	t.Run("LargeAndReservedContent", func(t *testing.T) { testLargeAndReserved(t, newStore(t)) })
	t.Run("AccessesAbsentVersusEmpty", func(t *testing.T) { testAccesses(t, newStore(t)) })
	t.Run("ConcurrentRuns", func(t *testing.T) { testConcurrentRuns(t, newStore(t)) })
	t.Run("UnknownRun", func(t *testing.T) { testUnknownRun(t, newStore(t)) })
}

var t0 = time.Date(2026, 11, 22, 10, 0, 0, 0, time.UTC)

// Ev builds an event with a deterministic time derived from seq.
func Ev(traceID, runID string, seq uint64, body trace.Body) trace.Event {
	return trace.Event{TraceID: traceID, RunID: runID, Seq: seq, Time: t0.Add(time.Duration(seq) * time.Millisecond), Body: body}
}

// Started returns a run.started body.
func Started(path, agent string, parent string) *trace.RunStarted {
	rs := &trace.RunStarted{
		RunKind: trace.RunKindAgent,
		Path:    path,
		Agent:   trace.AgentInfo{Name: agent},
		Task:    llm.Blocks{llm.Text{Text: "task for " + agent}},
		Config:  &trace.Config{Model: "m", System: llm.Blocks{llm.Text{Text: "sys"}}, Tools: []llm.ToolSpec{}, MaxSteps: 5},
	}
	if parent != "" {
		rs.Parent = &trace.ParentRef{RunID: parent}
	} else {
		rs.Origin = &trace.Origin{Kind: trace.OriginLive}
	}
	return rs
}

// Finished returns a run.finished body.
func Finished(status trace.RunStatus, text string) *trace.RunFinished {
	return &trace.RunFinished{Status: status, Output: llm.Blocks{llm.Text{Text: text}}, Usage: llm.Usage{InputTokens: 1, OutputTokens: 1}, Steps: 1}
}

func appendOK(t *testing.T, s trace.Store, events ...trace.Event) {
	t.Helper()
	if err := s.Append(context.Background(), events...); err != nil {
		t.Fatalf("Append: %v", err)
	}
}

func readAll(t *testing.T, s trace.Store, runID string) []trace.Event {
	t.Helper()
	evs, err := trace.Collect(s.Events(context.Background(), runID, 0))
	if err != nil {
		t.Fatalf("Events(%s): %v", runID, err)
	}
	return evs
}

// sameEvents compares events by their encoded form.
func sameEvents(t *testing.T, want, got []trace.Event) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("got %d events, want %d", len(got), len(want))
	}
	for i := range want {
		w, err := trace.EncodeEvent(want[i], trace.EncodeOptions{})
		if err != nil {
			t.Fatal(err)
		}
		g, err := trace.EncodeEvent(got[i], trace.EncodeOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if string(w) != string(g) {
			t.Fatalf("event %d differs\nwant %s\ngot  %s", i, w, g)
		}
	}
}

func testAppendAndRead(t *testing.T, s trace.Store) {
	tr, run := trace.NewID(), trace.NewID()
	input := json.RawMessage(`{"expression":"6*7"}`)
	events := []trace.Event{
		Ev(tr, run, 0, Started("/", "calc", "")),
		Ev(tr, run, 1, &trace.MessageAppended{Message: &llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "6*7?"}}}}),
		Ev(tr, run, 2, &trace.ToolStarted{Step: 0, CallID: "c1", Tool: "calc", Input: input, Fingerprint: "sha256:x"}),
	}
	events[2].Cause = trace.Cause(1)
	appendOK(t, s, events[0])
	appendOK(t, s, events[1:]...)
	got := readAll(t, s, run)
	sameEvents(t, events, got)
	if got[2].Cause == nil || *got[2].Cause != 1 {
		t.Errorf("cause not preserved: %v", got[2].Cause)
	}
	if !got[1].Time.Equal(events[1].Time) {
		t.Errorf("time = %v, want %v", got[1].Time, events[1].Time)
	}
}

func testSeqConflicts(t *testing.T, s trace.Store) {
	tr, run := trace.NewID(), trace.NewID()
	ctx := context.Background()
	if err := s.Append(ctx, Ev(tr, run, 1, Started("/", "a", ""))); !errors.Is(err, trace.ErrSeqConflict) {
		t.Errorf("first seq 1: got %v, want ErrSeqConflict", err)
	}
	appendOK(t, s, Ev(tr, run, 0, Started("/", "a", "")))
	if err := s.Append(ctx, Ev(tr, run, 0, &trace.Annotation{Key: "k", Value: json.RawMessage(`1`)})); !errors.Is(err, trace.ErrSeqConflict) {
		t.Errorf("duplicate seq: got %v, want ErrSeqConflict", err)
	}
	if err := s.Append(ctx, Ev(tr, run, 2, &trace.Annotation{Key: "k", Value: json.RawMessage(`1`)})); !errors.Is(err, trace.ErrSeqConflict) {
		t.Errorf("gap: got %v, want ErrSeqConflict", err)
	}
	err := s.Append(ctx,
		Ev(tr, run, 1, &trace.Annotation{Key: "k", Value: json.RawMessage(`1`)}),
		Ev(tr, run, 3, &trace.Annotation{Key: "k", Value: json.RawMessage(`2`)}))
	if !errors.Is(err, trace.ErrSeqConflict) {
		t.Errorf("non-contiguous batch: got %v, want ErrSeqConflict", err)
	}
	if n := len(readAll(t, s, run)); n != 1 {
		t.Errorf("rejected appends left %d events, want 1", n)
	}
	if err := s.Append(ctx, Ev(tr, trace.NewID(), 0, &trace.Annotation{Key: "k", Value: json.RawMessage(`1`)})); err == nil {
		t.Error("a run whose first event isn't run.started should be rejected")
	}
}

func testEventsFrom(t *testing.T, s trace.Store) {
	tr, run := trace.NewID(), trace.NewID()
	evs := []trace.Event{Ev(tr, run, 0, Started("/", "a", ""))}
	for i := uint64(1); i < 6; i++ {
		evs = append(evs, Ev(tr, run, i, &trace.Annotation{Key: "n", Value: json.RawMessage(fmt.Sprint(i))}))
	}
	appendOK(t, s, evs...)
	got, err := trace.Collect(s.Events(context.Background(), run, 3))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].Seq != 3 || got[2].Seq != 5 {
		t.Fatalf("Events(from=3) returned seqs %v", seqs(got))
	}
	// Stopping early must be safe.
	for _, err := range s.Events(context.Background(), run, 0) {
		if err != nil {
			t.Fatal(err)
		}
		break
	}
}

func seqs(evs []trace.Event) []uint64 {
	out := make([]uint64, len(evs))
	for i, e := range evs {
		out[i] = e.Seq
	}
	return out
}

func testRunsAndTree(t *testing.T, s trace.Store) {
	ctx := context.Background()
	tr := trace.NewID()
	root, child := trace.NewID(), trace.NewID()
	appendOK(t, s, Ev(tr, root, 0, Started("/", "lead", "")))
	appendOK(t, s,
		Ev(tr, root, 1, &trace.RunSpawned{ChildRunID: child, Path: "/worker#0", Agent: "worker", Task: llm.Blocks{llm.Text{Text: "go"}}, Fingerprint: "sha256:s"}))
	appendOK(t, s, Ev(tr, child, 0, Started("/worker#0", "worker", root)))
	appendOK(t, s, Ev(tr, child, 1, Finished(trace.StatusCompleted, "done")))
	appendOK(t, s,
		Ev(tr, root, 2, &trace.RunJoined{ChildRunID: child, Status: trace.StatusCompleted, Output: llm.Blocks{llm.Text{Text: "done"}}}),
		Ev(tr, root, 3, Finished(trace.StatusFailed, "x")))
	incomplete := trace.NewID()
	appendOK(t, s, Ev(tr, incomplete, 0, Started("/other#0", "other", root)))

	var infos []trace.RunInfo
	for info, err := range s.Runs(ctx, trace.RunFilter{TraceID: tr}) {
		if err != nil {
			t.Fatal(err)
		}
		infos = append(infos, info)
	}
	if len(infos) != 3 {
		t.Fatalf("got %d runs, want 3", len(infos))
	}
	byID := map[string]trace.RunInfo{}
	for _, i := range infos {
		byID[i.RunID] = i
	}
	r := byID[root]
	if r.Path != "/" || r.ParentRunID != "" || r.Status != trace.StatusFailed || r.Agent != "lead" || r.Kind != trace.RunKindAgent {
		t.Errorf("root info = %+v", r)
	}
	if r.Origin == nil || r.Origin.Kind != trace.OriginLive {
		t.Errorf("root origin = %+v", r.Origin)
	}
	if r.Finished.IsZero() || !r.Started.Equal(t0) {
		t.Errorf("root times = %v .. %v", r.Started, r.Finished)
	}
	c := byID[child]
	if c.Path != "/worker#0" || c.ParentRunID != root || c.Status != trace.StatusCompleted {
		t.Errorf("child info = %+v", c)
	}
	if byID[incomplete].Status != trace.StatusIncomplete {
		t.Errorf("run without run.finished: status %q, want incomplete", byID[incomplete].Status)
	}
	// Tree links resolve both ways.
	var spawned *trace.RunSpawned
	for _, e := range readAll(t, s, root) {
		if b, ok := e.Body.(*trace.RunSpawned); ok {
			spawned = b
		}
	}
	if spawned == nil || spawned.ChildRunID != child {
		t.Fatalf("parent's run.spawned doesn't name the child")
	}
	cs := readAll(t, s, child)[0].Body.(*trace.RunStarted)
	if cs.Parent == nil || cs.Parent.RunID != root {
		t.Fatalf("child's parent = %+v", cs.Parent)
	}
}

func testRunFilters(t *testing.T, s trace.Store) {
	ctx := context.Background()
	live, replay := trace.NewID(), trace.NewID()
	liveRoot, liveChild, replayRoot := trace.NewID(), trace.NewID(), trace.NewID()
	appendOK(t, s, Ev(live, liveRoot, 0, Started("/", "a", "")))
	appendOK(t, s, Ev(live, liveChild, 0, Started("/b#0", "b", liveRoot)))
	rs := Started("/", "a", "")
	rs.Origin = &trace.Origin{Kind: trace.OriginReplay, SourceTraceID: live, Mode: "strict"}
	appendOK(t, s, Ev(replay, replayRoot, 0, rs))

	collect := func(f trace.RunFilter) []string {
		var ids []string
		for info, err := range s.Runs(ctx, f) {
			if err != nil {
				t.Fatal(err)
			}
			ids = append(ids, info.RunID)
		}
		return ids
	}
	if got := collect(trace.RunFilter{RootsOnly: true}); len(got) != 2 {
		t.Errorf("RootsOnly: %v", got)
	}
	got := collect(trace.RunFilter{Origins: []trace.OriginKind{trace.OriginLive}})
	if !reflect.DeepEqual(sortedCopy(got), sortedCopy([]string{liveRoot, liveChild})) {
		t.Errorf("Origins=live: got %v", got)
	}
	if got := collect(trace.RunFilter{Origins: []trace.OriginKind{trace.OriginReplay}}); len(got) != 1 || got[0] != replayRoot {
		t.Errorf("Origins=replay: got %v", got)
	}
	if got := collect(trace.RunFilter{Limit: 1}); len(got) != 1 {
		t.Errorf("Limit=1: got %v", got)
	}
	// Newest trace first: replay was created after live.
	if got := collect(trace.RunFilter{RootsOnly: true}); got[0] != replayRoot {
		t.Errorf("newest trace should come first, got %v", got)
	}
}

func sortedCopy(s []string) []string {
	c := append([]string(nil), s...)
	for i := range c {
		for j := i + 1; j < len(c); j++ {
			if c[j] < c[i] {
				c[i], c[j] = c[j], c[i]
			}
		}
	}
	return c
}

func testBlobs(t *testing.T, s trace.Store) {
	ctx := context.Background()
	data := []byte("hello blob")
	ref, err := s.PutBlob(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	if ref != trace.BlobRefOf(data) {
		t.Errorf("ref = %s, want %s", ref, trace.BlobRefOf(data))
	}
	ref2, err := s.PutBlob(ctx, data)
	if err != nil || ref2 != ref {
		t.Errorf("second PutBlob = %s, %v", ref2, err)
	}
	got, err := s.Blob(ctx, ref)
	if err != nil || string(got) != string(data) {
		t.Errorf("Blob = %q, %v", got, err)
	}
	if _, err := s.Blob(ctx, trace.BlobRefOf([]byte("missing"))); !errors.Is(err, trace.ErrBlobNotFound) {
		t.Errorf("missing blob: got %v, want ErrBlobNotFound", err)
	}
}

func testLargeAndReserved(t *testing.T, s trace.Store) {
	tr, run := trace.NewID(), trace.NewID()
	big := strings.Repeat("0123456789abcdef", 8<<10) // 128 KiB
	weird := json.RawMessage(`{"$blob":"not a ref","$lit":1,"$redacted":"x","nested":{"$blob":"y","size":3}}`)
	events := []trace.Event{
		Ev(tr, run, 0, Started("/", "a", "")),
		Ev(tr, run, 1, &trace.ToolCompleted{CallID: "c", Content: llm.Blocks{llm.Text{Text: big}}}),
		Ev(tr, run, 2, &trace.EffectCompleted{Name: "e", Output: weird}),
		Ev(tr, run, 3, &trace.ToolStarted{CallID: "d", Tool: "t", Input: json.RawMessage(`{"blob":"` + big + `","n":1}`), Fingerprint: "sha256:z"}),
	}
	appendOK(t, s, events...)
	got := readAll(t, s, run)
	sameEvents(t, events, got)
	if out := string(got[2].Body.(*trace.EffectCompleted).Output); out != string(weird) {
		t.Errorf("reserved keys in application data not preserved:\n%s", out)
	}
}

func testAccesses(t *testing.T, s trace.Store) {
	tr, run := trace.NewID(), trace.NewID()
	empty := []trace.Access{}
	some := []trace.Access{{Resource: "file:/a", Mode: trace.AccessWrite}}
	appendOK(t, s,
		Ev(tr, run, 0, Started("/", "a", "")),
		Ev(tr, run, 1, &trace.ToolStarted{CallID: "1", Tool: "t", Input: json.RawMessage(`{}`), Fingerprint: "f"}),
		Ev(tr, run, 2, &trace.ToolStarted{CallID: "2", Tool: "t", Input: json.RawMessage(`{}`), Fingerprint: "f", Accesses: &empty}),
		Ev(tr, run, 3, &trace.ToolStarted{CallID: "3", Tool: "t", Input: json.RawMessage(`{}`), Fingerprint: "f", Accesses: &some}))
	got := readAll(t, s, run)
	if a := got[1].Body.(*trace.ToolStarted).Accesses; a != nil {
		t.Errorf("absent accesses decoded as %v, want nil (unknown)", *a)
	}
	if a := got[2].Body.(*trace.ToolStarted).Accesses; a == nil || len(*a) != 0 {
		t.Errorf("empty accesses decoded as %v, want non-nil empty", a)
	}
	if a := got[3].Body.(*trace.ToolStarted).Accesses; a == nil || !reflect.DeepEqual(*a, some) {
		t.Errorf("accesses decoded as %v", a)
	}
}

func testConcurrentRuns(t *testing.T, s trace.Store) {
	tr := trace.NewID()
	var wg sync.WaitGroup
	runs := make([]string, 8)
	errs := make(chan error, len(runs))
	for i := range runs {
		runs[i] = trace.NewID()
		wg.Add(1)
		go func(run string) {
			defer wg.Done()
			ctx := context.Background()
			if err := s.Append(ctx, Ev(tr, run, 0, Started(fmt.Sprintf("/w#%d", i), "w", ""))); err != nil {
				errs <- err
				return
			}
			for seq := uint64(1); seq <= 20; seq++ {
				if err := s.Append(ctx, Ev(tr, run, seq, &trace.Annotation{Key: "n", Value: json.RawMessage(fmt.Sprint(seq))})); err != nil {
					errs <- err
					return
				}
			}
		}(runs[i])
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	for _, run := range runs {
		if got := readAll(t, s, run); len(got) != 21 {
			t.Errorf("run %s has %d events, want 21", run, len(got))
		}
	}
}

func testUnknownRun(t *testing.T, s trace.Store) {
	_, err := trace.Collect(s.Events(context.Background(), trace.NewID(), 0))
	if !errors.Is(err, trace.ErrRunNotFound) {
		t.Errorf("unknown run: got %v, want ErrRunNotFound", err)
	}
}
