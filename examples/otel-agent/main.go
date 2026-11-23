// Command otel-agent runs an agent with OpenTelemetry export: traces to
// Jaeger over OTLP/gRPC and metrics on a Prometheus /metrics endpoint. The
// exporters are agentflow trace observers, so the same run is also recorded
// in .agentflow/ for replay.
//
//	docker compose up -d          # Jaeger + Prometheus (see docker-compose.yaml)
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/otel-agent
//	open http://localhost:16686   # Jaeger UI, service "agentflow-otel-example"
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	agotel "github.com/kumarlokesh/agentflow/observe/otel"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	promexporter "go.opentelemetry.io/otel/exporters/prometheus"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

const (
	otlpEndpoint = "localhost:4317"
	metricsAddr  = ":2112"
	modelName    = "claude-haiku-5-5"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(logger); err != nil {
		logger.Error("run failed", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return errors.New("ANTHROPIC_API_KEY is not set")
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	res, err := resource.New(ctx, resource.WithAttributes(semconv.ServiceName("agentflow-otel-example")))
	if err != nil {
		return err
	}
	traceExp, err := otlptracegrpc.New(ctx, otlptracegrpc.WithEndpoint(otlpEndpoint), otlptracegrpc.WithInsecure())
	if err != nil {
		return fmt.Errorf("OTLP exporter: %w", err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(traceExp), sdktrace.WithResource(res))
	defer shutdown(logger, "tracer provider", tp.Shutdown)

	promExp, err := promexporter.New()
	if err != nil {
		return fmt.Errorf("prometheus exporter: %w", err)
	}
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(promExp), sdkmetric.WithResource(res))
	defer shutdown(logger, "meter provider", mp.Shutdown)
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		logger.Info("metrics", "url", "http://localhost"+metricsAddr+"/metrics")
		if err := http.ListenAndServe(metricsAddr, mux); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("metrics server", "error", err)
		}
	}()

	metrics, err := agotel.NewMetrics(mp)
	if err != nil {
		return err
	}
	store, err := filestore.Open(".agentflow", filestore.Options{Redact: trace.SecretPatterns})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	type pair struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	schema := json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}`)
	mul := agentflow.NewTool(llm.ToolSpec{Name: "multiply", Description: "Multiplies a and b.", InputSchema: schema},
		func(_ context.Context, p pair) (agentflow.ToolOutput, error) {
			return agentflow.TextOutput(fmt.Sprint(p.A * p.B)), nil
		})
	add := agentflow.NewTool(llm.ToolSpec{Name: "add", Description: "Adds a and b.", InputSchema: schema},
		func(_ context.Context, p pair) (agentflow.ToolOutput, error) {
			return agentflow.TextOutput(fmt.Sprint(p.A + p.B)), nil
		})

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "otel-calculator",
		Model:        retry.Wrap(anthropic.New(key, anthropic.WithModel(modelName))),
		ModelName:    modelName,
		Instructions: "Use the tools for all arithmetic.",
		Params:       llm.Params{MaxTokens: 1024},
		Tools:        []agentflow.Tool{mul, add},
		Store:        store,
		Observers:    []trace.Observer{agotel.NewTracer(tp), metrics},
	})
	if err != nil {
		return err
	}
	result, err := agent.Run(ctx, "What is 47 * 89 + 12?")
	if result != nil {
		fmt.Printf("%s\n\ntrace %s (%s); spans are in Jaeger under service agentflow-otel-example\n", result.OutputText(), result.TraceID, result.Status)
	}
	if err != nil {
		return err
	}
	fmt.Println("metrics stay available for 30s at http://localhost" + metricsAddr + "/metrics (Ctrl-C to exit)")
	select {
	case <-ctx.Done():
	case <-time.After(30 * time.Second):
	}
	return nil
}

func shutdown(logger *slog.Logger, what string, fn func(context.Context) error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := fn(ctx); err != nil {
		logger.Warn(what+" shutdown", "error", err)
	}
}
