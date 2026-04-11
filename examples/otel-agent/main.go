// Command otel-agent demonstrates an end-to-end agentflow run with full
// OpenTelemetry observability: traces exported to Jaeger via OTLP gRPC, and
// metrics exposed on a Prometheus /metrics endpoint.
//
// Prerequisites:
//
//	docker compose up -d   # starts Jaeger + Prometheus (see docker-compose.yaml)
//
// Usage:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/otel-agent
//
// Then open:
//   - Jaeger UI:     http://localhost:16686
//   - Prometheus:    http://localhost:9090
//   - Agent metrics: http://localhost:2112/metrics
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/kumarlokesh/agentflow"
	agotel "github.com/kumarlokesh/agentflow/observe/otel"
	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/store"

	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	promexporter "go.opentelemetry.io/otel/exporters/prometheus"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	otlpEndpoint   = "localhost:4317"
	metricsAddr    = ":2112"
	storeDir       = ".agentflow/runs"
	task           = "What is 47 * 89 + 12?"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "error: ANTHROPIC_API_KEY environment variable is not set")
		os.Exit(1)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	// --- OTel resource ---
	res, _ := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName("agentflow-otel-example"),
			semconv.ServiceVersion("1.0.0"),
		),
	)

	// --- Trace provider (OTLP → Jaeger) ---
	traceExp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(otlpEndpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		logger.Error("failed to create OTLP trace exporter", "error", err)
		os.Exit(1)
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(traceExp),
		sdktrace.WithResource(res),
	)
	defer func() {
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		tp.Shutdown(shutCtx)
	}()

	// --- Metric provider (Prometheus exporter) ---
	promExp, err := promexporter.New()
	if err != nil {
		logger.Error("failed to create Prometheus exporter", "error", err)
		os.Exit(1)
	}
	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(promExp),
		sdkmetric.WithResource(res),
	)
	defer func() {
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		mp.Shutdown(shutCtx)
	}()

	// --- Start Prometheus /metrics endpoint ---
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		logger.Info("Prometheus metrics", "addr", "http://localhost"+metricsAddr+"/metrics")
		if err := http.ListenAndServe(metricsAddr, mux); err != nil && err != http.ErrServerClosed {
			logger.Error("metrics server error", "error", err)
		}
	}()

	// --- Build agentflow hooks ---
	tracingHook := agotel.NewTracingHook(tp)
	metricsHook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		logger.Error("failed to create metrics hook", "error", err)
		os.Exit(1)
	}
	// Compose with the existing in-process metrics for display.
	inprocMetrics := observe.NewMetrics()
	hook := observe.NewMultiHook(
		tracingHook,
		metricsHook,
		observe.NewMetricsHook(inprocMetrics),
		observe.NewTracingHook(),
	)

	// --- Event store ---
	fileStore, err := store.NewFile(storeDir)
	if err != nil {
		logger.Error("failed to create store", "error", err)
		os.Exit(1)
	}

	// --- LLM provider ---
	llm := retry.Wrap(
		anthropic.New(apiKey, anthropic.WithModel("claude-haiku-4-5-20251001")),
		retry.WithLogger(logger),
	)

	// --- Agent ---
	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "otel-calculator-agent",
		Instructions: "You are a math assistant. Use the calculator tool to evaluate expressions. Show your work.",
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
	logger.Info("Jaeger UI available at http://localhost:16686")

	result, err := agent.Run(ctx, task)
	if err != nil {
		logger.Error("agent run failed", "error", err)
		os.Exit(1)
	}

	snap := inprocMetrics.Snapshot()
	fmt.Printf("Answer:  %s\n", result.Output)
	fmt.Printf("RunID:   %s\n", result.RunID)
	fmt.Printf("Steps:   %d\n", result.Steps)
	fmt.Printf("Tokens:  %d (prompt: %d, completion: %d)\n",
		snap.TotalTokens, snap.TotalPromptTokens, snap.TotalCompletionTokens)
	fmt.Printf("\nView trace in Jaeger: http://localhost:16686/search?service=agentflow-otel-example\n")
	fmt.Printf("Prometheus metrics:   http://localhost%s/metrics\n", metricsAddr)

	// Verify replay.
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
					"description": "The mathematical expression to evaluate"
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
		return &agentflow.ToolResult{Error: err.Error()}, nil
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
		return 0, fmt.Errorf("unexpected character at position %d", p.pos)
	}
	return result, nil
}

type parser struct{ input string; pos int; err error }

func (p *parser) parseExpr() float64 {
	result := p.parseTerm()
	for p.pos < len(p.input) && (p.input[p.pos] == '+' || p.input[p.pos] == '-') {
		op := p.input[p.pos]; p.pos++
		right := p.parseTerm()
		if op == '+' { result += right } else { result -= right }
	}
	return result
}

func (p *parser) parseTerm() float64 {
	result := p.parseFactor()
	for p.pos < len(p.input) && (p.input[p.pos] == '*' || p.input[p.pos] == '/') {
		op := p.input[p.pos]; p.pos++
		right := p.parseFactor()
		if op == '*' {
			result *= right
		} else {
			if right == 0 { p.err = fmt.Errorf("division by zero"); return 0 }
			result /= right
		}
	}
	return result
}

func (p *parser) parseFactor() float64 {
	if p.pos >= len(p.input) { p.err = fmt.Errorf("unexpected end"); return 0 }
	if p.input[p.pos] == '(' {
		p.pos++
		result := p.parseExpr()
		if p.pos < len(p.input) && p.input[p.pos] == ')' { p.pos++ } else { p.err = fmt.Errorf("missing )") }
		return result
	}
	negative := false
	if p.input[p.pos] == '-' { negative = true; p.pos++ }
	start := p.pos
	for p.pos < len(p.input) && (p.input[p.pos] >= '0' && p.input[p.pos] <= '9' || p.input[p.pos] == '.') { p.pos++ }
	if start == p.pos { p.err = fmt.Errorf("expected number at pos %d", p.pos); return 0 }
	num, err := strconv.ParseFloat(p.input[start:p.pos], 64)
	if err != nil { p.err = err; return 0 }
	if negative { num = -num }
	return num
}
