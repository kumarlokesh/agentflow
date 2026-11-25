package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"

	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

const cassette = "testdata/triage.trace.jsonl"

// TestTriageCassette replays the whole team strictly from a committed
// recording, with the read-only tools verified. Re-record with:
//
//	AGENTFLOW_RECORD=1 go test ./examples/team   (after deleting the cassette)
func TestTriageCassette(t *testing.T) {
	team, err := NewTriage(DefaultSQLInstructions, nil)
	if err != nil {
		t.Fatal(err)
	}
	res := agentflowtest.Cassette(t, cassette, team, DefaultTask, agentflowtest.Verify("tool:*"))
	// The recorded run is the team's failure: the sql agent's query misses
	// the spike (see TestForkFixesTheFailure).
	if got := res.OutputText(); got != "Checkout latency spiked at 14:05; likely cause: unknown." {
		t.Errorf("output = %q", got)
	}
	for _, p := range []string{"/logs#0", "/metrics#0", "/metrics#0/sql#0", "/summarizer#0"} {
		if res.Find(p) == nil {
			t.Errorf("no run at %s", p)
		}
	}
}

// Changing one sub-agent's prompt is reported in that sub-agent's run, and
// nowhere else.
func TestSubAgentChangeIsAttributed(t *testing.T) {
	data, err := os.ReadFile(cassette)
	if err != nil {
		t.Skip("no cassette yet")
	}
	team, err := NewTriage("Write efficient SQL.", nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := replay.RunBundle(context.Background(), team, bytes.NewReader(data), replay.Options{})
	var de *replay.DivergenceError
	if !errors.As(err, &de) || len(res.Divergences) != 1 {
		t.Fatalf("err %v, divergences %v", err, res.Divergences)
	}
	if d := de.Divergence; d.Path != "/metrics#0/sql#0" || d.Effect != "llm#0" || d.Kind != replay.RequestMismatch {
		t.Fatalf("%s", d.Summary)
	}
}

// Forking the recording at the sql agent's first model call with the fixed
// prompt fixes the failure, and everything the change doesn't reach is
// answered from the recording.
func TestForkFixesTheFailure(t *testing.T) {
	data, err := os.ReadFile(cassette)
	if err != nil {
		t.Skip("no cassette yet")
	}
	ctx := context.Background()
	store := memstore.New()
	traceID, err := trace.Import(ctx, bytes.NewReader(data), store)
	if err != nil {
		t.Fatal(err)
	}
	team, err := NewTriage(DefaultSQLInstructions, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := replay.Fork(ctx, team, store, traceID, ForkOptions(3))
	if err != nil {
		t.Fatal(err)
	}
	if res.Treatment.Passed != 3 || res.Baseline.Passed != 0 {
		t.Fatalf("%s", res)
	}
	if r := res.Treatment.Samples[0].Reuse; r.RunsReplayed != 1 || r.RunsNew != 1 {
		t.Fatalf("reuse %+v", r)
	}
}
