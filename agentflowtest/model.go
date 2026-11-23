// Package agentflowtest provides helpers for testing agentflow agents:
// scripted and function-backed models, and cassettes that turn a recorded
// trace into a `go test` regression test.
package agentflowtest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/kumarlokesh/agentflow/llm"
)

// ErrScriptExhausted is returned by a ScriptedModel with no responses left.
var ErrScriptExhausted = errors.New("agentflowtest: scripted model has no responses left")

// ScriptedModel returns pre-defined responses in order and records every
// request it receives. It is safe for concurrent use.
type ScriptedModel struct {
	mu       sync.Mutex
	steps    []Step
	next     int
	requests []*llm.Request
}

// Step is one scripted outcome: a response or an error.
type Step struct {
	Response *llm.Response
	Err      error
}

// NewScriptedModel returns a model that answers with responses in order.
func NewScriptedModel(responses ...*llm.Response) *ScriptedModel {
	m := &ScriptedModel{}
	for _, r := range responses {
		m.steps = append(m.steps, Step{Response: r})
	}
	return m
}

// Then appends a step and returns the model, for chaining.
func (m *ScriptedModel) Then(s Step) *ScriptedModel {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.steps = append(m.steps, s)
	return m
}

// Generate implements llm.Model.
func (m *ScriptedModel) Generate(_ context.Context, req *llm.Request) (*llm.Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := *req
	cp.Messages = append([]llm.Message(nil), req.Messages...)
	m.requests = append(m.requests, &cp)
	if m.next >= len(m.steps) {
		return nil, ErrScriptExhausted
	}
	s := m.steps[m.next]
	m.next++
	return s.Response, s.Err
}

// Requests returns the requests received so far.
func (m *ScriptedModel) Requests() []*llm.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*llm.Request(nil), m.requests...)
}

// Calls returns how many requests the model received.
func (m *ScriptedModel) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

// ModelFunc adapts a function to llm.Model; useful for fakes whose answer
// depends on the request.
type ModelFunc func(ctx context.Context, req *llm.Request) (*llm.Response, error)

// Generate implements llm.Model.
func (f ModelFunc) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return f(ctx, req)
}

// Reply returns a final text response.
func Reply(text string) *llm.Response {
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{llm.Text{Text: text}}},
		StopReason: llm.StopEndTurn,
		Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

// CallTools returns a response that calls tools. Each call is a ToolUse;
// see Use.
func CallTools(calls ...llm.ToolUse) *llm.Response {
	blocks := make(llm.Blocks, len(calls))
	for i, c := range calls {
		blocks[i] = c
	}
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
		StopReason: llm.StopToolUse,
		Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

// Use builds a ToolUse block; input is encoded as JSON.
func Use(callID, tool string, input any) llm.ToolUse {
	raw, err := json.Marshal(input)
	if err != nil {
		panic(fmt.Sprintf("agentflowtest.Use: %v", err))
	}
	return llm.ToolUse{CallID: callID, Name: tool, Input: raw}
}
