package replay

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/kumarlokesh/agentflow"
)

// DiffResult compares two agent runs event-by-event.
type DiffResult struct {
	// RunA is the first run ID.
	RunA string
	// RunB is the second run ID.
	RunB string
	// Identical is true if both runs produced the same logical output.
	Identical bool
	// Differences lists the specific differences found.
	Differences []Difference
	// Summary is a human-readable summary of the diff.
	Summary string
}

// Difference describes a single divergence between two runs.
type Difference struct {
	// StepIndex is the step where the difference was found.
	StepIndex int
	// EventType is the type of event that differs.
	EventType agentflow.EventType
	// Field is the specific field that differs (e.g. "output", "tool_calls").
	Field string
	// ValueA is the value from run A.
	ValueA string
	// ValueB is the value from run B.
	ValueB string
}

// Diff compares two runs and returns their differences. It focuses on
// semantically meaningful differences: LLM outputs, tool calls, tool results,
// and final outcomes. Timestamps and UUIDs are intentionally excluded from
// comparison since they vary between runs.
func Diff(ctx context.Context, store agentflow.EventStore, runA, runB string) (*DiffResult, error) {
	eventsA, err := store.LoadEvents(ctx, runA)
	if err != nil {
		return nil, fmt.Errorf("diff: load run %s: %w", runA, err)
	}
	eventsB, err := store.LoadEvents(ctx, runB)
	if err != nil {
		return nil, fmt.Errorf("diff: load run %s: %w", runB, err)
	}

	if len(eventsA) == 0 {
		return nil, fmt.Errorf("diff: no events found for run %s", runA)
	}
	if len(eventsB) == 0 {
		return nil, fmt.Errorf("diff: no events found for run %s", runB)
	}

	var diffs []Difference

	// Compare LLM responses.
	llmA := filterByType(eventsA, agentflow.EventLLMResponse)
	llmB := filterByType(eventsB, agentflow.EventLLMResponse)
	diffs = append(diffs, compareLLMResponses(llmA, llmB)...)

	// Compare tool calls.
	tcA := filterByType(eventsA, agentflow.EventToolCall)
	tcB := filterByType(eventsB, agentflow.EventToolCall)
	diffs = append(diffs, compareToolCalls(tcA, tcB)...)

	// Compare tool results.
	trA := filterByType(eventsA, agentflow.EventToolResult)
	trB := filterByType(eventsB, agentflow.EventToolResult)
	diffs = append(diffs, compareToolResults(trA, trB)...)

	// Compare final output.
	outputA := extractOriginalOutput(eventsA)
	outputB := extractOriginalOutput(eventsB)
	if outputA != outputB {
		diffs = append(diffs, Difference{
			StepIndex: -1,
			EventType: agentflow.EventRunEnd,
			Field:     "output",
			ValueA:    outputA,
			ValueB:    outputB,
		})
	}

	identical := len(diffs) == 0
	summary := buildSummary(runA, runB, diffs, identical)

	return &DiffResult{
		RunA:        runA,
		RunB:        runB,
		Identical:   identical,
		Differences: diffs,
		Summary:     summary,
	}, nil
}

// --- Internal helpers ---

func filterByType(events []agentflow.Event, eventType agentflow.EventType) []agentflow.Event {
	var out []agentflow.Event
	for _, e := range events {
		if e.Type == eventType {
			out = append(out, e)
		}
	}
	return out
}

func compareLLMResponses(a, b []agentflow.Event) []Difference {
	var diffs []Difference
	maxLen := max(len(a), len(b))

	for i := range maxLen {
		if i >= len(a) {
			diffs = append(diffs, Difference{
				StepIndex: i,
				EventType: agentflow.EventLLMResponse,
				Field:     "presence",
				ValueA:    "<missing>",
				ValueB:    summarizeEvent(b[i]),
			})
			continue
		}
		if i >= len(b) {
			diffs = append(diffs, Difference{
				StepIndex: i,
				EventType: agentflow.EventLLMResponse,
				Field:     "presence",
				ValueA:    summarizeEvent(a[i]),
				ValueB:    "<missing>",
			})
			continue
		}

		var dataA, dataB agentflow.LLMResponseData
		if err := json.Unmarshal(a[i].Data, &dataA); err != nil {
			continue
		}
		if err := json.Unmarshal(b[i].Data, &dataB); err != nil {
			continue
		}

		if dataA.Content != dataB.Content {
			diffs = append(diffs, Difference{
				StepIndex: a[i].StepIndex,
				EventType: agentflow.EventLLMResponse,
				Field:     "content",
				ValueA:    truncate(dataA.Content, 200),
				ValueB:    truncate(dataB.Content, 200),
			})
		}

		tcA, _ := json.Marshal(dataA.ToolCalls)
		tcB, _ := json.Marshal(dataB.ToolCalls)
		if string(tcA) != string(tcB) {
			diffs = append(diffs, Difference{
				StepIndex: a[i].StepIndex,
				EventType: agentflow.EventLLMResponse,
				Field:     "tool_calls",
				ValueA:    truncate(string(tcA), 200),
				ValueB:    truncate(string(tcB), 200),
			})
		}
	}
	return diffs
}

func compareToolCalls(a, b []agentflow.Event) []Difference {
	var diffs []Difference
	maxLen := max(len(a), len(b))

	for i := range maxLen {
		if i >= len(a) || i >= len(b) {
			valA, valB := "<missing>", "<missing>"
			if i < len(a) {
				valA = summarizeEvent(a[i])
			}
			if i < len(b) {
				valB = summarizeEvent(b[i])
			}
			diffs = append(diffs, Difference{
				StepIndex: i,
				EventType: agentflow.EventToolCall,
				Field:     "presence",
				ValueA:    valA,
				ValueB:    valB,
			})
			continue
		}

		var dataA, dataB agentflow.ToolCallData
		if err := json.Unmarshal(a[i].Data, &dataA); err != nil {
			continue
		}
		if err := json.Unmarshal(b[i].Data, &dataB); err != nil {
			continue
		}

		if dataA.ToolName != dataB.ToolName {
			diffs = append(diffs, Difference{
				StepIndex: a[i].StepIndex,
				EventType: agentflow.EventToolCall,
				Field:     "tool_name",
				ValueA:    dataA.ToolName,
				ValueB:    dataB.ToolName,
			})
		}
		if string(dataA.Input) != string(dataB.Input) {
			diffs = append(diffs, Difference{
				StepIndex: a[i].StepIndex,
				EventType: agentflow.EventToolCall,
				Field:     "input",
				ValueA:    truncate(string(dataA.Input), 200),
				ValueB:    truncate(string(dataB.Input), 200),
			})
		}
	}
	return diffs
}

func compareToolResults(a, b []agentflow.Event) []Difference {
	var diffs []Difference
	maxLen := max(len(a), len(b))

	for i := range maxLen {
		if i >= len(a) || i >= len(b) {
			valA, valB := "<missing>", "<missing>"
			if i < len(a) {
				valA = summarizeEvent(a[i])
			}
			if i < len(b) {
				valB = summarizeEvent(b[i])
			}
			diffs = append(diffs, Difference{
				StepIndex: i,
				EventType: agentflow.EventToolResult,
				Field:     "presence",
				ValueA:    valA,
				ValueB:    valB,
			})
			continue
		}

		var dataA, dataB agentflow.ToolResultData
		if err := json.Unmarshal(a[i].Data, &dataA); err != nil {
			continue
		}
		if err := json.Unmarshal(b[i].Data, &dataB); err != nil {
			continue
		}

		if dataA.Output != dataB.Output {
			diffs = append(diffs, Difference{
				StepIndex: a[i].StepIndex,
				EventType: agentflow.EventToolResult,
				Field:     "output",
				ValueA:    truncate(dataA.Output, 200),
				ValueB:    truncate(dataB.Output, 200),
			})
		}
		if dataA.Error != dataB.Error {
			diffs = append(diffs, Difference{
				StepIndex: a[i].StepIndex,
				EventType: agentflow.EventToolResult,
				Field:     "error",
				ValueA:    dataA.Error,
				ValueB:    dataB.Error,
			})
		}
	}
	return diffs
}

func buildSummary(runA, runB string, diffs []Difference, identical bool) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Diff: %s vs %s\n", runA[:min(8, len(runA))], runB[:min(8, len(runB))]))

	if identical {
		sb.WriteString("Result: IDENTICAL\n")
		return sb.String()
	}

	sb.WriteString(fmt.Sprintf("Result: %d difference(s) found\n\n", len(diffs)))
	for i, d := range diffs {
		sb.WriteString(fmt.Sprintf("[%d] step=%d type=%s field=%s\n", i+1, d.StepIndex, d.EventType, d.Field))
		sb.WriteString(fmt.Sprintf("  A: %s\n", d.ValueA))
		sb.WriteString(fmt.Sprintf("  B: %s\n", d.ValueB))
	}
	return sb.String()
}

func summarizeEvent(e agentflow.Event) string {
	return fmt.Sprintf("%s@step%d", e.Type, e.StepIndex)
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
