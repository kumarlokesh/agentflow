package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"testing"

	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/replay"
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
	if got := res.OutputText(); got != "Checkout latency spiked at 14:05; likely cause: payment-gateway timeouts." {
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
