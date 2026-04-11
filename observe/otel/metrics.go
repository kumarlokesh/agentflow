package otel

import (
	"context"
	"fmt"
	"time"

	"github.com/kumarlokesh/agentflow/observe"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// Ensure OTelMetricsHook satisfies observe.Hook at compile time.
var _ observe.Hook = (*OTelMetricsHook)(nil)

// OTelMetricsHook implements observe.Hook by recording OpenTelemetry metrics.
//
// Instruments:
//
//	agentflow_llm_calls_total        counter   {status=ok|error}
//	agentflow_tool_calls_total       counter   {tool_name, status=ok|error}
//	agentflow_tokens_total           counter   {token_type=prompt|completion}
//	agentflow_llm_duration_seconds   histogram {status=ok|error}
//	agentflow_tool_duration_seconds  histogram {tool_name, status=ok|error}
//
// It is safe for concurrent use.
type OTelMetricsHook struct {
	llmCalls    metric.Int64Counter
	toolCalls   metric.Int64Counter
	tokens      metric.Int64Counter
	llmLatency  metric.Float64Histogram
	toolLatency metric.Float64Histogram
}

// NewMetricsHook creates an OTelMetricsHook using instruments from mp.
// Returns an error if any instrument cannot be created (rare in practice).
func NewMetricsHook(mp metric.MeterProvider) (*OTelMetricsHook, error) {
	meter := mp.Meter(
		"github.com/kumarlokesh/agentflow",
		metric.WithInstrumentationVersion("1.0.0"),
	)

	llmCalls, err := meter.Int64Counter("agentflow_llm_calls_total",
		metric.WithDescription("Total number of LLM calls."),
		metric.WithUnit("{calls}"),
	)
	if err != nil {
		return nil, fmt.Errorf("agentflow/otel: create llm_calls_total: %w", err)
	}

	toolCalls, err := meter.Int64Counter("agentflow_tool_calls_total",
		metric.WithDescription("Total number of tool calls."),
		metric.WithUnit("{calls}"),
	)
	if err != nil {
		return nil, fmt.Errorf("agentflow/otel: create tool_calls_total: %w", err)
	}

	tokens, err := meter.Int64Counter("agentflow_tokens_total",
		metric.WithDescription("Total tokens consumed, partitioned by token_type (prompt|completion)."),
		metric.WithUnit("{tokens}"),
	)
	if err != nil {
		return nil, fmt.Errorf("agentflow/otel: create tokens_total: %w", err)
	}

	llmLatency, err := meter.Float64Histogram("agentflow_llm_duration_seconds",
		metric.WithDescription("LLM call duration in seconds."),
		metric.WithUnit("s"),
		metric.WithExplicitBucketBoundaries(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
	)
	if err != nil {
		return nil, fmt.Errorf("agentflow/otel: create llm_duration_seconds: %w", err)
	}

	toolLatency, err := meter.Float64Histogram("agentflow_tool_duration_seconds",
		metric.WithDescription("Tool execution duration in seconds."),
		metric.WithUnit("s"),
		metric.WithExplicitBucketBoundaries(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
	)
	if err != nil {
		return nil, fmt.Errorf("agentflow/otel: create tool_duration_seconds: %w", err)
	}

	return &OTelMetricsHook{
		llmCalls:    llmCalls,
		toolCalls:   toolCalls,
		tokens:      tokens,
		llmLatency:  llmLatency,
		toolLatency: toolLatency,
	}, nil
}

// No-ops for lifecycle hooks we don't measure at the metric level.
func (h *OTelMetricsHook) OnRunStart(_ context.Context, _, _ string)                              {}
func (h *OTelMetricsHook) OnRunEnd(_ context.Context, _ string, _ int, _ time.Duration, _ error) {}
func (h *OTelMetricsHook) OnStepStart(_ context.Context, _ string, _ int)                         {}
func (h *OTelMetricsHook) OnStepEnd(_ context.Context, _ string, _ int, _ time.Duration)          {}

func (h *OTelMetricsHook) OnLLMCall(ctx context.Context, _ string, _ int, promptTokens, completionTokens, _ int, latency time.Duration, err error) {
	status := statusAttr(err)
	h.llmCalls.Add(ctx, 1, metric.WithAttributes(status))
	h.tokens.Add(ctx, int64(promptTokens),
		metric.WithAttributes(attribute.String("token_type", "prompt")))
	h.tokens.Add(ctx, int64(completionTokens),
		metric.WithAttributes(attribute.String("token_type", "completion")))
	h.llmLatency.Record(ctx, latency.Seconds(), metric.WithAttributes(status))
}

func (h *OTelMetricsHook) OnToolCall(ctx context.Context, _ string, _ int, toolName string, duration time.Duration, err error) {
	status := statusAttr(err)
	tool := attribute.String("tool_name", toolName)
	h.toolCalls.Add(ctx, 1, metric.WithAttributes(tool, status))
	h.toolLatency.Record(ctx, duration.Seconds(), metric.WithAttributes(tool, status))
}

func statusAttr(err error) attribute.KeyValue {
	if err != nil {
		return attribute.String("status", "error")
	}
	return attribute.String("status", "ok")
}
