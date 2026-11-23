package otel

import (
	"context"
	"fmt"
	"sync"

	aftrace "github.com/kumarlokesh/agentflow/trace"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// Metrics turns recorded events into OpenTelemetry metrics:
//
//	agentflow_llm_calls_total        counter   {status=ok|error, error_type}
//	agentflow_tool_calls_total       counter   {tool_name, status=ok|error, error_type}
//	agentflow_tokens_total           counter   {token_type=input|output}
//	agentflow_llm_duration_seconds   histogram {status}
//	agentflow_tool_duration_seconds  histogram {tool_name, status}
//	agentflow_runs_total             counter   {status}
//
// It is safe for concurrent use.
type Metrics struct {
	llmCalls, toolCalls, tokens, runs metric.Int64Counter
	llmLatency, toolLatency           metric.Float64Histogram
	names                             *toolNames
}

var _ aftrace.Observer = (*Metrics)(nil)

// NewMetrics returns a Metrics observer that records with mp.
func NewMetrics(mp metric.MeterProvider) (*Metrics, error) {
	meter := mp.Meter(instrumentation)
	var err error
	m := &Metrics{names: newToolNames()}
	counter := func(name, desc, unit string) metric.Int64Counter {
		if err != nil {
			return nil
		}
		var c metric.Int64Counter
		c, err = meter.Int64Counter(name, metric.WithDescription(desc), metric.WithUnit(unit))
		if err != nil {
			err = fmt.Errorf("agentflow/otel: %s: %w", name, err)
		}
		return c
	}
	histogram := func(name, desc string, bounds ...float64) metric.Float64Histogram {
		if err != nil {
			return nil
		}
		var h metric.Float64Histogram
		h, err = meter.Float64Histogram(name, metric.WithDescription(desc), metric.WithUnit("s"), metric.WithExplicitBucketBoundaries(bounds...))
		if err != nil {
			err = fmt.Errorf("agentflow/otel: %s: %w", name, err)
		}
		return h
	}
	m.llmCalls = counter("agentflow_llm_calls_total", "Model calls.", "{calls}")
	m.toolCalls = counter("agentflow_tool_calls_total", "Tool calls.", "{calls}")
	m.tokens = counter("agentflow_tokens_total", "Tokens, by token_type (input|output).", "{tokens}")
	m.runs = counter("agentflow_runs_total", "Finished runs, by status.", "{runs}")
	m.llmLatency = histogram("agentflow_llm_duration_seconds", "Model call duration.", 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
	m.toolLatency = histogram("agentflow_tool_duration_seconds", "Tool call duration.", 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Observe implements trace.Observer.
func (m *Metrics) Observe(ctx context.Context, e aftrace.Event) {
	ok := attribute.String("status", "ok")
	failed := func(code string) metric.MeasurementOption {
		return metric.WithAttributes(attribute.String("status", "error"), attribute.String("error_type", code))
	}
	switch b := e.Body.(type) {
	case *aftrace.LLMCompleted:
		m.llmCalls.Add(ctx, 1, metric.WithAttributes(ok))
		m.llmLatency.Record(ctx, b.Duration.Seconds(), metric.WithAttributes(ok))
		m.tokens.Add(ctx, int64(b.Usage.InputTokens), metric.WithAttributes(attribute.String("token_type", "input")))
		m.tokens.Add(ctx, int64(b.Usage.OutputTokens), metric.WithAttributes(attribute.String("token_type", "output")))
	case *aftrace.LLMFailed:
		m.llmCalls.Add(ctx, 1, failed(b.Error.Code))
		m.llmLatency.Record(ctx, b.Duration.Seconds(), metric.WithAttributes(attribute.String("status", "error")))
	case *aftrace.ToolStarted:
		m.names.put(e.RunID, e.Seq, b.Tool)
	case *aftrace.ToolCompleted:
		name := attribute.String("tool_name", m.names.take(e))
		m.toolCalls.Add(ctx, 1, metric.WithAttributes(name, ok))
		m.toolLatency.Record(ctx, b.Duration.Seconds(), metric.WithAttributes(name, ok))
	case *aftrace.ToolFailed:
		name := attribute.String("tool_name", m.names.take(e))
		m.toolCalls.Add(ctx, 1, metric.WithAttributes(name, attribute.String("status", "error"), attribute.String("error_type", b.Error.Code)))
		m.toolLatency.Record(ctx, b.Duration.Seconds(), metric.WithAttributes(name, attribute.String("status", "error")))
	case *aftrace.RunFinished:
		m.runs.Add(ctx, 1, metric.WithAttributes(attribute.String("status", string(b.Status))))
	}
}

// toolNames remembers tool names from tool.started events, because
// completions carry only the call ID.
type toolNames struct {
	mu    sync.Mutex
	names map[spanKey]string
}

func newToolNames() *toolNames { return &toolNames{names: map[spanKey]string{}} }

func (n *toolNames) put(run string, seq uint64, name string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.names[spanKey{run, seq}] = name
}

func (n *toolNames) take(e aftrace.Event) string {
	if e.Cause == nil {
		return "unknown"
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	k := spanKey{e.RunID, *e.Cause}
	name, ok := n.names[k]
	if !ok {
		return "unknown"
	}
	delete(n.names, k)
	return name
}
