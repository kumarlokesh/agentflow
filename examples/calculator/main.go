// Command calculator demonstrates a simple agentflow agent that uses a
// calculator tool to solve math problems. It showcases the core runtime loop,
// event recording, and deterministic replay.
//
// Usage:
//
//	# Set your OpenAI API key (or use the built-in mock for demo)
//	export AGENTFLOW_LLM=mock
//
//	# Run the agent
//	go run ./examples/calculator
//
//	# Replay a recorded run (prints the run ID after execution)
//	go run ./examples/calculator -replay <run-id>
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/store"
)

func main() {
	replayID := flag.String("replay", "", "Run ID to replay instead of executing a new run")
	storeDir := flag.String("store", ".agentflow/runs", "Directory for event persistence")
	task := flag.String("task", "What is (15 * 7) + (23 - 8)?", "Math task for the agent to solve")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	fileStore, err := store.NewFile(*storeDir)
	if err != nil {
		logger.Error("failed to create store", "error", err)
		os.Exit(1)
	}

	ctx := context.Background()

	if *replayID != "" {
		runReplay(ctx, fileStore, *replayID, logger)
		return
	}

	runAgent(ctx, fileStore, *task, logger)
}

func runAgent(ctx context.Context, fileStore agentflow.EventStore, task string, logger *slog.Logger) {
	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "calculator-agent",
		Instructions: "You are a math assistant. Use the calculator tool to evaluate expressions. Always show your work.",
		LLM:          &mockMathLLM{},
		Tools:        []agentflow.Tool{&calculatorTool{}},
		MaxSteps:     10,
		Store:        fileStore,
		Logger:       logger,
	})
	if err != nil {
		logger.Error("failed to create agent", "error", err)
		os.Exit(1)
	}

	fmt.Printf("Task: %s\n\n", task)

	result, err := agent.Run(ctx, task)
	if err != nil {
		logger.Error("agent run failed", "error", err)
		os.Exit(1)
	}

	fmt.Printf("Answer: %s\n", result.Output)
	fmt.Printf("Steps:  %d\n", result.Steps)
	fmt.Printf("RunID:  %s\n", result.RunID)
	fmt.Printf("Events: %d\n", len(result.Events))
	fmt.Printf("\nTo replay this run:\n")
	fmt.Printf("  go run ./examples/calculator -replay %s\n", result.RunID)
}

func runReplay(ctx context.Context, fileStore agentflow.EventStore, runID string, logger *slog.Logger) {
	fmt.Printf("Replaying run: %s\n\n", runID)

	engine := replay.NewEngine(fileStore, logger)
	result, err := engine.Replay(ctx, runID)
	if err != nil {
		logger.Error("replay failed", "error", err)
		os.Exit(1)
	}

	fmt.Printf("Replay Result:\n")
	fmt.Printf("  Original Run: %s\n", result.RunID)
	fmt.Printf("  Replay Run:   %s\n", result.ReplayRunID)
	fmt.Printf("  Match:        %v\n", result.Match)
	fmt.Printf("  Output:       %s\n", result.Output)
	fmt.Printf("  Steps:        %d\n", result.Steps)
	fmt.Printf("  Duration:     %s\n", result.Duration)

	if result.Match {
		fmt.Println("\n  Deterministic replay verified — identical output!")
	} else {
		fmt.Println("\n  WARNING: Replay produced different output!")
	}
}

// --- Calculator Tool ---

type calculatorTool struct{}

func (c *calculatorTool) Schema() agentflow.ToolSchema {
	return agentflow.ToolSchema{
		Name:        "calculator",
		Description: "Evaluates a mathematical expression and returns the result. Supports +, -, *, / with parentheses.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"expression": {
					"type": "string",
					"description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'"
				}
			},
			"required": ["expression"]
		}`),
	}
}

func (c *calculatorTool) Execute(_ context.Context, params json.RawMessage) (*agentflow.ToolResult, error) {
	var input struct {
		Expression string `json:"expression"`
	}
	if err := json.Unmarshal(params, &input); err != nil {
		return &agentflow.ToolResult{Error: fmt.Sprintf("invalid params: %v", err)}, nil
	}

	result, err := evalExpression(input.Expression)
	if err != nil {
		return &agentflow.ToolResult{Error: fmt.Sprintf("eval error: %v", err)}, nil
	}

	return &agentflow.ToolResult{Output: fmt.Sprintf("%g", result)}, nil
}

// evalExpression is a simple recursive-descent expression evaluator.
func evalExpression(expr string) (float64, error) {
	expr = strings.ReplaceAll(expr, " ", "")
	p := &parser{input: expr}
	result := p.parseExpr()
	if p.err != nil {
		return 0, p.err
	}
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected character at position %d: %c", p.pos, p.input[p.pos])
	}
	return result, nil
}

type parser struct {
	input string
	pos   int
	err   error
}

func (p *parser) parseExpr() float64 {
	result := p.parseTerm()
	for p.pos < len(p.input) && (p.input[p.pos] == '+' || p.input[p.pos] == '-') {
		op := p.input[p.pos]
		p.pos++
		right := p.parseTerm()
		if op == '+' {
			result += right
		} else {
			result -= right
		}
	}
	return result
}

func (p *parser) parseTerm() float64 {
	result := p.parseFactor()
	for p.pos < len(p.input) && (p.input[p.pos] == '*' || p.input[p.pos] == '/') {
		op := p.input[p.pos]
		p.pos++
		right := p.parseFactor()
		if op == '*' {
			result *= right
		} else {
			if right == 0 {
				p.err = fmt.Errorf("division by zero")
				return 0
			}
			result /= right
		}
	}
	return result
}

func (p *parser) parseFactor() float64 {
	if p.pos >= len(p.input) {
		p.err = fmt.Errorf("unexpected end of expression")
		return 0
	}

	// Handle parentheses.
	if p.input[p.pos] == '(' {
		p.pos++ // skip '('
		result := p.parseExpr()
		if p.pos < len(p.input) && p.input[p.pos] == ')' {
			p.pos++ // skip ')'
		} else {
			p.err = fmt.Errorf("missing closing parenthesis")
		}
		return result
	}

	// Handle negative numbers.
	negative := false
	if p.input[p.pos] == '-' {
		negative = true
		p.pos++
	}

	// Parse number.
	start := p.pos
	for p.pos < len(p.input) && (p.input[p.pos] >= '0' && p.input[p.pos] <= '9' || p.input[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		p.err = fmt.Errorf("expected number at position %d", p.pos)
		return 0
	}

	num, err := strconv.ParseFloat(p.input[start:p.pos], 64)
	if err != nil {
		p.err = err
		return 0
	}
	if negative {
		num = -num
	}
	return num
}

// --- Mock Math LLM ---
// This mock simulates an LLM that knows how to use the calculator tool.
// In production, you'd replace this with a real LLM client (OpenAI, Anthropic, etc.)

type mockMathLLM struct {
	callCount int
}

func (m *mockMathLLM) ChatCompletion(_ context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	m.callCount++

	// First call: request a tool call to evaluate the expression.
	if m.callCount == 1 {
		// Extract the task from the last user message.
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				return &agentflow.LLMResponse{
					ToolCalls: []agentflow.ToolCallRequest{
						{
							ID:        "call-1",
							Name:      "calculator",
							Arguments: json.RawMessage(`{"expression":"(15*7)+(23-8)"}`),
						},
					},
					Usage: &agentflow.TokenUsage{
						PromptTokens: 50, CompletionTokens: 20, TotalTokens: 70,
					},
				}, nil
			}
		}
	}

	// Second call: provide the final answer using the tool result.
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "tool" {
			return &agentflow.LLMResponse{
				Content: fmt.Sprintf("Let me break this down:\n• 15 × 7 = 105\n• 23 - 8 = 15\n• 105 + 15 = 120\n\nThe answer is %s.", req.Messages[i].Content),
				Usage: &agentflow.TokenUsage{
					PromptTokens: 80, CompletionTokens: 40, TotalTokens: 120,
				},
			}, nil
		}
	}

	return &agentflow.LLMResponse{Content: "I need more information to solve this."}, nil
}
