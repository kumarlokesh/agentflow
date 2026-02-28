// Package replay provides deterministic replay of recorded agent runs.
//
// The replay engine loads events from an EventStore, constructs mock LLM and
// tool implementations that return the recorded responses, and re-executes the
// agent loop. This guarantees that given the same event log, the agent produces
// identical behavior - enabling debugging, regression testing, and auditing.
//
// Design rationale: Rather than patching the agent internals, replay works by
// substituting the external dependencies (LLM, tools) with recorded versions.
// This keeps the agent code path identical between live and replay modes,
// ensuring high-fidelity reproduction.
package replay

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/kumarlokesh/agentflow"
)

// Engine replays recorded agent runs from an EventStore.
type Engine struct {
	store  agentflow.EventStore
	logger *slog.Logger
}

// NewEngine creates a replay engine backed by the given store.
func NewEngine(store agentflow.EventStore, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.Default()
	}
	return &Engine{store: store, logger: logger}
}

// Result is the outcome of a replay.
type Result struct {
	// RunID is the original run being replayed.
	RunID string
	// ReplayRunID is the new run ID assigned to this replay.
	ReplayRunID string
	// Output is the final agent output from the replay.
	Output string
	// Steps is the number of steps executed during replay.
	Steps int
	// Events are the events produced during replay.
	Events []agentflow.Event
	// Duration is the wall-clock time of the replay.
	Duration time.Duration
	// Match indicates whether the replay produced identical logical output.
	Match bool
}

// Replay re-executes the agent run identified by runID. It loads the recorded
// events, creates mock LLM and tool implementations from the recorded
// responses, and runs the agent loop. The replayed run is assigned a new RunID.
func (e *Engine) Replay(ctx context.Context, runID string) (*Result, error) {
	start := time.Now()
	log := e.logger.With("replay_run_id", runID)
	log.Info("starting replay")

	// Load recorded events.
	events, err := e.store.LoadEvents(ctx, runID)
	if err != nil {
		return nil, fmt.Errorf("replay: load events for run %s: %w", runID, err)
	}
	if len(events) == 0 {
		return nil, fmt.Errorf("replay: no events found for run %s", runID)
	}

	// Extract run configuration from RunStart event.
	runStart, err := findRunStart(events)
	if err != nil {
		return nil, err
	}

	// Build mock LLM from recorded LLM responses.
	mockLLM, err := buildMockLLM(events)
	if err != nil {
		return nil, err
	}

	// Build mock tools from recorded tool results.
	mockTools, err := buildMockTools(events)
	if err != nil {
		return nil, err
	}

	// Disable tool validation during replay — the recorded params were already
	// validated during the original run.
	validateParams := false

	// Create and run the agent.
	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:               "replay-" + truncateID(runID, 8),
		Instructions:       runStart.Instructions,
		LLM:                mockLLM,
		Tools:              mockTools,
		MaxSteps:           runStart.MaxSteps,
		Store:              e.store,
		Logger:             e.logger,
		ValidateToolParams: &validateParams,
	})
	if err != nil {
		return nil, fmt.Errorf("replay: create agent: %w", err)
	}

	result, err := agent.Run(ctx, runStart.Task)
	if err != nil {
		return nil, fmt.Errorf("replay: run agent: %w", err)
	}

	// Check if output matches.
	originalOutput := extractOriginalOutput(events)
	match := result.Output == originalOutput

	duration := time.Since(start)
	log.Info("replay completed",
		"match", match,
		"original_output_len", len(originalOutput),
		"replay_output_len", len(result.Output),
		"duration", duration,
	)

	return &Result{
		RunID:       runID,
		ReplayRunID: result.RunID,
		Output:      result.Output,
		Steps:       result.Steps,
		Events:      result.Events,
		Duration:    duration,
		Match:       match,
	}, nil
}

// --- Internal helpers ---

func findRunStart(events []agentflow.Event) (*agentflow.RunStartData, error) {
	for _, e := range events {
		if e.Type == agentflow.EventRunStart {
			var data agentflow.RunStartData
			if err := json.Unmarshal(e.Data, &data); err != nil {
				return nil, fmt.Errorf("replay: decode run_start: %w", err)
			}
			return &data, nil
		}
	}
	return nil, fmt.Errorf("replay: no run_start event found")
}

func truncateID(id string, maxLen int) string {
	if len(id) <= maxLen {
		return id
	}
	return id[:maxLen]
}

func extractOriginalOutput(events []agentflow.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Type == agentflow.EventRunEnd {
			var data agentflow.RunEndData
			if err := json.Unmarshal(events[i].Data, &data); err == nil {
				return data.Output
			}
		}
	}
	return ""
}

// --- Mock LLM ---

// mockLLM replays recorded LLM responses in order.
type mockLLM struct {
	responses []*agentflow.LLMResponse
	index     int
}

func buildMockLLM(events []agentflow.Event) (*mockLLM, error) {
	var responses []*agentflow.LLMResponse
	for _, e := range events {
		if e.Type == agentflow.EventLLMResponse {
			var data agentflow.LLMResponseData
			if err := json.Unmarshal(e.Data, &data); err != nil {
				return nil, fmt.Errorf("replay: decode llm_response: %w", err)
			}
			responses = append(responses, &agentflow.LLMResponse{
				Content:   data.Content,
				ToolCalls: data.ToolCalls,
				Usage:     data.Usage,
			})
		}
	}
	if len(responses) == 0 {
		return nil, fmt.Errorf("replay: no llm_response events found")
	}
	return &mockLLM{responses: responses}, nil
}

func (m *mockLLM) ChatCompletion(_ context.Context, _ *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	if m.index >= len(m.responses) {
		return nil, agentflow.ErrReplayExhausted
	}
	resp := m.responses[m.index]
	m.index++
	return resp, nil
}

// --- Mock Tools ---

// mockTool replays recorded tool results in order for a specific tool name.
type mockTool struct {
	name    string
	results []*agentflow.ToolResult
	index   int
}

func buildMockTools(events []agentflow.Event) ([]agentflow.Tool, error) {
	// Group tool results by tool name, preserving order.
	toolResults := make(map[string][]*agentflow.ToolResult)
	toolOrder := make([]string, 0)

	for _, e := range events {
		if e.Type == agentflow.EventToolResult {
			var data agentflow.ToolResultData
			if err := json.Unmarshal(e.Data, &data); err != nil {
				return nil, fmt.Errorf("replay: decode tool_result: %w", err)
			}
			if _, exists := toolResults[data.ToolName]; !exists {
				toolOrder = append(toolOrder, data.ToolName)
			}
			toolResults[data.ToolName] = append(toolResults[data.ToolName], &agentflow.ToolResult{
				Output: data.Output,
				Error:  data.Error,
			})
		}
	}

	tools := make([]agentflow.Tool, 0, len(toolOrder))
	for _, name := range toolOrder {
		tools = append(tools, &mockTool{
			name:    name,
			results: toolResults[name],
		})
	}
	return tools, nil
}

func (t *mockTool) Schema() agentflow.ToolSchema {
	return agentflow.ToolSchema{
		Name:        t.name,
		Description: "replay mock for " + t.name,
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
}

func (t *mockTool) Execute(_ context.Context, _ json.RawMessage) (*agentflow.ToolResult, error) {
	if t.index >= len(t.results) {
		return nil, fmt.Errorf("%w: tool %s", agentflow.ErrReplayExhausted, t.name)
	}
	result := t.results[t.index]
	t.index++
	return result, nil
}
