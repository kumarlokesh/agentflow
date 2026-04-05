// Command anthropic-agent demonstrates an end-to-end agentflow run using the
// Anthropic Claude API with the calculator tool, full observability hooks,
// and deterministic replay.
//
// Usage:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/anthropic-agent
//	go run ./examples/anthropic-agent -replay <run-id>
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
	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/store"
)

func main() {
	replayID := flag.String("replay", "", "Run ID to replay instead of executing a new run")
	storeDir := flag.String("store", ".agentflow/runs", "Directory for event persistence")
	task := flag.String("task", "What is 47 * 89 + 12?", "Math task for the agent to solve")
	flag.Parse()

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "error: ANTHROPIC_API_KEY environment variable is not set")
		os.Exit(1)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

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

	runAgent(ctx, fileStore, apiKey, *task, logger)
}

func runAgent(ctx context.Context, fileStore agentflow.EventStore, apiKey, task string, logger *slog.Logger) {
	// Wire the Anthropic provider with retry.
	anthropicClient := anthropic.New(apiKey,
		anthropic.WithModel("claude-haiku-4-5-20251001"),
	)
	llm := retry.Wrap(anthropicClient,
		retry.WithMaxRetries(3),
		retry.WithLogger(logger),
	)

	// Wire observability hooks.
	metrics := observe.NewMetrics()
	hook := observe.NewMultiHook(
		observe.NewMetricsHook(metrics),
		observe.NewTracingHook(),
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "anthropic-calculator-agent",
		Instructions: "You are a math assistant. Use the calculator tool to evaluate expressions. Always show your work step by step.",
		LLM:          llm,
		Tools:        []agentflow.Tool{&calculatorTool{}},
		MaxSteps:     10,
		Store:        fileStore,
		Logger:       logger,
		Hook:         hook,
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

	snap := metrics.Snapshot()
	fmt.Printf("Answer:  %s\n", result.Output)
	fmt.Printf("RunID:   %s\n", result.RunID)
	fmt.Printf("Steps:   %d\n", result.Steps)
	fmt.Printf("Tokens:  %d (prompt: %d, completion: %d)\n",
		snap.TotalTokens, snap.TotalPromptTokens, snap.TotalCompletionTokens)

	fmt.Printf("\nTo replay this run:\n")
	fmt.Printf("  go run ./examples/anthropic-agent -replay %s\n", result.RunID)

	// Verify replay determinism.
	fmt.Println("\nVerifying replay...")
	engine := replay.NewEngine(fileStore, logger)
	replayResult, err := engine.Replay(ctx, result.RunID)
	if err != nil {
		logger.Error("replay failed", "error", err)
		return
	}
	if replayResult.Match {
		fmt.Println("Deterministic replay verified — identical output!")
	} else {
		fmt.Println("WARNING: Replay produced different output!")
	}
}

func runReplay(ctx context.Context, fileStore agentflow.EventStore, runID string, logger *slog.Logger) {
	fmt.Printf("Replaying run: %s\n\n", runID)

	engine := replay.NewEngine(fileStore, logger)
	result, err := engine.Replay(ctx, runID)
	if err != nil {
		logger.Error("replay failed", "error", err)
		os.Exit(1)
	}

	fmt.Printf("Original Run: %s\n", result.RunID)
	fmt.Printf("Replay Run:   %s\n", result.ReplayRunID)
	fmt.Printf("Match:        %v\n", result.Match)
	fmt.Printf("Output:       %s\n", result.Output)
	fmt.Printf("Steps:        %d\n", result.Steps)
	fmt.Printf("Duration:     %s\n", result.Duration)

	if result.Match {
		fmt.Println("\nDeterministic replay verified — identical output!")
	} else {
		fmt.Println("\nWARNING: Replay produced different output!")
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
	if p.input[p.pos] == '(' {
		p.pos++
		result := p.parseExpr()
		if p.pos < len(p.input) && p.input[p.pos] == ')' {
			p.pos++
		} else {
			p.err = fmt.Errorf("missing closing parenthesis")
		}
		return result
	}
	negative := false
	if p.input[p.pos] == '-' {
		negative = true
		p.pos++
	}
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
