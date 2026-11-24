package main

import (
	"testing"

	"github.com/kumarlokesh/agentflow/agentflowtest"
)

// TestCalculatorCassette replays a committed recording strictly. If the
// agent's prompt, tools, or behavior change, it fails with a report of the
// first difference. Verify("tool:*") also runs the (read-only) calculator
// tool and compares its results with the recording, so a bug introduced in
// the tool itself is caught too. Re-record with:
//
//	AGENTFLOW_RECORD=1 go test ./examples/calculator   (after deleting the cassette)
func TestCalculatorCassette(t *testing.T) {
	agent, err := NewCalculatorAgent(DefaultInstructions, nil)
	if err != nil {
		t.Fatal(err)
	}
	res := agentflowtest.Cassette(t, "testdata/calculator.trace.jsonl", agent, "What is (15 * 7) + (23 - 8)?",
		agentflowtest.Verify("tool:*"))
	if got := res.OutputText(); got != "15 × 7 = 105, 23 − 8 = 15, 105 + 15 = 120. The answer is 120." {
		t.Errorf("output = %q", got)
	}
}
