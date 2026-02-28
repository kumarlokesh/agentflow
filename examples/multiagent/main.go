// Command multiagent demonstrates multi-agent orchestration using agentflow.
//
// It creates a coordinator with three specialized agents:
//   - researcher: looks up facts
//   - calculator: solves math problems
//   - summarizer: combines results into a final answer
//
// The coordinator delegates tasks to each agent and collects results.
//
// Usage:
//
//	go run ./examples/multiagent
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/multi"
	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/policy"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// --- Set up observability ---
	metrics := observe.NewMetrics()
	hook := observe.NewMetricsHook(metrics)

	// --- Set up policy ---
	costTracker := policy.NewCostTracker(policy.CostTrackerConfig{MaxTotalTokens: 100000})
	rateLimit := policy.NewRateLimiter(50, time.Minute)
	policyChain := policy.NewChain(costTracker, rateLimit)

	// --- Create specialized agents ---
	researcher := newSimpleAgent("researcher",
		"You are a research assistant. Look up facts and return concise answers.",
		researcherResponses(), nil, hook, policyChain, logger)

	calculator := newSimpleAgent("calculator",
		"You are a calculator. Solve math problems step by step.",
		calculatorResponses(), []agentflow.Tool{&calcTool{}}, hook, policyChain, logger)

	summarizer := newSimpleAgent("summarizer",
		"You are a summarizer. Combine information into a clear summary.",
		summarizerResponses(), nil, hook, policyChain, logger)

	// --- Register agents with coordinator ---
	registry := multi.NewRegistry()
	registry.Register(researcher)
	registry.Register(calculator)
	registry.Register(summarizer)

	coord := multi.NewCoordinator(registry, multi.CoordinatorConfig{
		MaxDelegationDepth: 3,
		MaxConcurrent:      2,
		Logger:             logger,
	})

	ctx := context.Background()

	fmt.Println("=== Multi-Agent Orchestration Demo ===")
	fmt.Println()

	// --- Fan-out: parallel research and calculation ---
	fmt.Println("Step 1: Fan-out tasks to researcher and calculator...")
	results, err := coord.FanOut(ctx, []string{"researcher", "calculator"},
		"What is the population of Tokyo and what is 15% of 37 million?")
	if err != nil {
		logger.Error("fan-out failed", "error", err)
	}

	for _, task := range results {
		fmt.Printf("  [%s] %s -> %s\n", task.Status, task.AssignedTo, truncate(task.Result, 80))
	}

	// --- Delegate to summarizer ---
	fmt.Println()
	fmt.Println("Step 2: Delegate summary to summarizer...")
	summaryTask, err := coord.Delegate(ctx, "summarizer",
		"Summarize: Tokyo population is ~14 million. 15% of 37 million is 5,550,000.")
	if err != nil {
		logger.Error("summarize failed", "error", err)
	} else {
		fmt.Printf("  [%s] %s\n", summaryTask.Status, truncate(summaryTask.Result, 120))
	}

	// --- Send inter-agent message ---
	fmt.Println()
	fmt.Println("Step 3: Send message between agents...")
	coord.SendMessage("summarizer", "researcher", multi.MsgInfo,
		"Thank you for the population data!")

	mb, _ := registry.Mailbox("researcher")
	if msg, ok := mb.TryReceive(); ok {
		fmt.Printf("  Researcher received: [%s] from %s: %s\n", msg.Type, msg.From, msg.Content)
	}

	// --- Print metrics ---
	fmt.Println()
	fmt.Println("=== Metrics ===")
	snap := metrics.Snapshot()
	fmt.Printf("  Runs:       %d\n", snap.RunCount)
	fmt.Printf("  Steps:      %d\n", snap.StepCount)
	fmt.Printf("  LLM Calls:  %d\n", snap.LLMCallCount)
	fmt.Printf("  Tool Calls: %d\n", snap.ToolCallCount)
	fmt.Printf("  Tool Errors:%d\n", snap.ToolErrorCount)

	// --- Print task history ---
	fmt.Println()
	fmt.Println("=== Task History ===")
	for _, t := range coord.Tasks() {
		fmt.Printf("  %s: agent=%s status=%s depth=%d duration=%s\n",
			t.ID, t.AssignedTo, t.Status, t.Depth,
			t.CompletedAt.Sub(t.CreatedAt).Round(time.Millisecond))
	}

	fmt.Println()
	fmt.Println("Done.")
}

// --- Agent wrapper that implements multi.Runner ---

type simpleAgent struct {
	agent *agentflow.Agent
	name  string
}

func newSimpleAgent(
	name, instructions string,
	responses []*agentflow.LLMResponse,
	tools []agentflow.Tool,
	hook observe.Hook,
	pol policy.Checker,
	logger *slog.Logger,
) *simpleAgent {
	llm := &mockLLM{responses: responses}

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         name,
		Instructions: instructions,
		LLM:          llm,
		Tools:        tools,
		Logger:       logger,
		Hook:         hook,
		Policy:       pol,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create agent %q: %v", name, err))
	}
	return &simpleAgent{agent: agent, name: name}
}

func (a *simpleAgent) Name() string { return a.name }

func (a *simpleAgent) Run(ctx context.Context, task string) (string, error) {
	result, err := a.agent.Run(ctx, task)
	if err != nil {
		return "", err
	}
	return result.Output, nil
}

// --- Mock LLM for demo ---

type mockLLM struct {
	responses []*agentflow.LLMResponse
	idx       int
}

func (m *mockLLM) ChatCompletion(_ context.Context, _ *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	if m.idx >= len(m.responses) {
		return &agentflow.LLMResponse{Content: "(no more responses)"}, nil
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

// Pre-canned responses for demo agents.

func researcherResponses() []*agentflow.LLMResponse {
	return []*agentflow.LLMResponse{
		{
			Content: "The population of Tokyo metropolitan area is approximately 14 million people (city proper) or 37 million (greater metro area).",
			Usage:   &agentflow.TokenUsage{PromptTokens: 50, CompletionTokens: 30, TotalTokens: 80},
		},
	}
}

func calculatorResponses() []*agentflow.LLMResponse {
	return []*agentflow.LLMResponse{
		{
			Content: "",
			ToolCalls: []agentflow.ToolCallRequest{
				{ID: "calc-1", Name: "calc", Arguments: json.RawMessage(`{"op":"multiply","a":37000000,"b":0.15}`)},
			},
			Usage: &agentflow.TokenUsage{PromptTokens: 40, CompletionTokens: 20, TotalTokens: 60},
		},
		{
			Content: "15% of 37 million is 5,550,000.",
			Usage:   &agentflow.TokenUsage{PromptTokens: 60, CompletionTokens: 15, TotalTokens: 75},
		},
	}
}

func summarizerResponses() []*agentflow.LLMResponse {
	return []*agentflow.LLMResponse{
		{
			Content: "Tokyo has a city population of ~14 million. 15% of the greater metro population (37M) is 5,550,000 people.",
			Usage:   &agentflow.TokenUsage{PromptTokens: 55, CompletionTokens: 25, TotalTokens: 80},
		},
	}
}

// --- Calculator tool ---

type calcTool struct{}

func (t *calcTool) Schema() agentflow.ToolSchema {
	return agentflow.ToolSchema{
		Name:        "calc",
		Description: "Performs basic arithmetic operations",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"op": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
				"a":  {"type": "number"},
				"b":  {"type": "number"}
			},
			"required": ["op", "a", "b"]
		}`),
	}
}

func (t *calcTool) Execute(_ context.Context, params json.RawMessage) (*agentflow.ToolResult, error) {
	var p struct {
		Op string  `json:"op"`
		A  float64 `json:"a"`
		B  float64 `json:"b"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, fmt.Errorf("invalid params: %w", err)
	}

	var result float64
	switch p.Op {
	case "add":
		result = p.A + p.B
	case "subtract":
		result = p.A - p.B
	case "multiply":
		result = p.A * p.B
	case "divide":
		if p.B == 0 {
			return &agentflow.ToolResult{Error: "division by zero"}, nil
		}
		result = p.A / p.B
	default:
		return &agentflow.ToolResult{Error: fmt.Sprintf("unknown op: %s", p.Op)}, nil
	}

	formatted := strconv.FormatFloat(result, 'f', -1, 64)
	if strings.Contains(formatted, ".") {
		formatted = strings.TrimRight(formatted, "0")
		formatted = strings.TrimRight(formatted, ".")
	}

	return &agentflow.ToolResult{Output: formatted}, nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}
