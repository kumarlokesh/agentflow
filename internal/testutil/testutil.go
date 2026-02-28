// Package testutil provides shared test helpers for agentflow packages.
package testutil

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/kumarlokesh/agentflow"
)

// MockLLM is a test double for the agentflow.LLM interface. It returns
// pre-configured responses in order, and records all requests for assertion.
type MockLLM struct {
	mu        sync.Mutex
	responses []*agentflow.LLMResponse
	errors    []error
	calls     []*agentflow.LLMRequest
	index     int
}

// NewMockLLM creates a MockLLM that returns the given responses in sequence.
func NewMockLLM(responses ...*agentflow.LLMResponse) *MockLLM {
	return &MockLLM{responses: responses}
}

// NewMockLLMWithErrors creates a MockLLM with paired responses and errors.
func NewMockLLMWithErrors(pairs ...any) *MockLLM {
	m := &MockLLM{}
	for i := 0; i < len(pairs); i += 2 {
		if pairs[i] != nil {
			m.responses = append(m.responses, pairs[i].(*agentflow.LLMResponse))
		} else {
			m.responses = append(m.responses, nil)
		}
		if i+1 < len(pairs) && pairs[i+1] != nil {
			m.errors = append(m.errors, pairs[i+1].(error))
		} else {
			m.errors = append(m.errors, nil)
		}
	}
	return m
}

// ChatCompletion implements agentflow.LLM.
func (m *MockLLM) ChatCompletion(_ context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.calls = append(m.calls, req)
	idx := m.index
	m.index++

	if idx < len(m.errors) && m.errors[idx] != nil {
		return nil, m.errors[idx]
	}
	if idx < len(m.responses) {
		return m.responses[idx], nil
	}
	return nil, agentflow.ErrReplayExhausted
}

// Calls returns all recorded LLM requests.
func (m *MockLLM) Calls() []*agentflow.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*agentflow.LLMRequest, len(m.calls))
	copy(out, m.calls)
	return out
}

// MockTool is a test double for the agentflow.Tool interface.
type MockTool struct {
	ToolSchema agentflow.ToolSchema
	Results    []*agentflow.ToolResult
	Errors     []error
	mu         sync.Mutex
	calls      []json.RawMessage
	index      int
}

// NewMockTool creates a MockTool with the given name, schema, and results.
func NewMockTool(name, description string, params json.RawMessage, results ...*agentflow.ToolResult) *MockTool {
	return &MockTool{
		ToolSchema: agentflow.ToolSchema{
			Name:        name,
			Description: description,
			Parameters:  params,
		},
		Results: results,
	}
}

// Schema implements agentflow.Tool.
func (t *MockTool) Schema() agentflow.ToolSchema {
	return t.ToolSchema
}

// Execute implements agentflow.Tool.
func (t *MockTool) Execute(_ context.Context, params json.RawMessage) (*agentflow.ToolResult, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.calls = append(t.calls, params)
	idx := t.index
	t.index++

	if idx < len(t.Errors) && t.Errors[idx] != nil {
		return nil, t.Errors[idx]
	}
	if idx < len(t.Results) {
		return t.Results[idx], nil
	}
	return &agentflow.ToolResult{Output: "mock output"}, nil
}

// Calls returns all recorded tool invocation parameters.
func (t *MockTool) Calls() []json.RawMessage {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]json.RawMessage, len(t.calls))
	copy(out, t.calls)
	return out
}

// LLMResponseWithText creates an LLMResponse with just text content (final answer).
func LLMResponseWithText(text string) *agentflow.LLMResponse {
	return &agentflow.LLMResponse{Content: text}
}

// LLMResponseWithToolCalls creates an LLMResponse with tool call requests.
func LLMResponseWithToolCalls(calls ...agentflow.ToolCallRequest) *agentflow.LLMResponse {
	return &agentflow.LLMResponse{ToolCalls: calls}
}

// ToolCall creates a ToolCallRequest for testing.
func ToolCall(id, name string, args any) agentflow.ToolCallRequest {
	raw, _ := json.Marshal(args)
	return agentflow.ToolCallRequest{
		ID:        id,
		Name:      name,
		Arguments: raw,
	}
}
