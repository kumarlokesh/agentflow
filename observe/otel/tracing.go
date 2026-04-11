// Package otel bridges agentflow's internal observe.Hook interface to
// OpenTelemetry traces and metrics. It is a separate Go module so that the
// root agentflow module's zero-dependency guarantee is preserved - only
// consumers that want OTel observability need to pay the import cost.
//
// Span hierarchy produced for each agent run:
//
//	agentflow.run      (root span — one per Run call)
//	└── agentflow.step (one per Observe-Think-Act iteration)
//	    ├── agentflow.llm              (one per LLM call, created and ended inline)
//	    └── agentflow.tool.{tool_name} (one per tool call, created and ended inline)
package otel

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/kumarlokesh/agentflow/observe"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Ensure OTelTracingHook satisfies observe.Hook at compile time.
var _ observe.Hook = (*OTelTracingHook)(nil)

// OTelTracingHook implements observe.Hook by emitting OpenTelemetry spans.
// It is safe for concurrent use.
type OTelTracingHook struct {
	tracer trace.Tracer
	mu     sync.Mutex
	spans  map[string]trace.Span // key → active span (run or step)
}

// NewTracingHook creates an OTelTracingHook using spans from tp.
func NewTracingHook(tp trace.TracerProvider) *OTelTracingHook {
	return &OTelTracingHook{
		tracer: tp.Tracer(
			"github.com/kumarlokesh/agentflow",
			trace.WithInstrumentationVersion("1.0.0"),
		),
		spans: make(map[string]trace.Span),
	}
}

func (h *OTelTracingHook) OnRunStart(ctx context.Context, runID, task string) {
	_, span := h.tracer.Start(ctx, "agentflow.run",
		trace.WithAttributes(
			attribute.String("agentflow.run_id", runID),
			attribute.String("agentflow.task", task),
		),
	)
	h.store(runKey(runID), span)
}

func (h *OTelTracingHook) OnRunEnd(_ context.Context, runID string, steps int, duration time.Duration, err error) {
	span, ok := h.delete(runKey(runID))
	if !ok {
		return
	}
	span.SetAttributes(
		attribute.Int("agentflow.total_steps", steps),
		attribute.Int64("agentflow.duration_ms", duration.Milliseconds()),
	)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetStatus(codes.Ok, "")
	}
	span.End()
}

func (h *OTelTracingHook) OnStepStart(_ context.Context, runID string, step int) {
	// Thread the run span as parent so steps nest correctly in the trace.
	ctx := h.spanContext(runKey(runID))
	_, span := h.tracer.Start(ctx, "agentflow.step",
		trace.WithAttributes(
			attribute.String("agentflow.run_id", runID),
			attribute.Int("agentflow.step_index", step),
		),
	)
	h.store(stepKey(runID, step), span)
}

func (h *OTelTracingHook) OnStepEnd(_ context.Context, runID string, step int, duration time.Duration) {
	span, ok := h.delete(stepKey(runID, step))
	if !ok {
		return
	}
	span.SetAttributes(attribute.Int64("agentflow.duration_ms", duration.Milliseconds()))
	span.SetStatus(codes.Ok, "")
	span.End()
}

func (h *OTelTracingHook) OnLLMCall(_ context.Context, runID string, step, promptTokens, completionTokens, totalTokens int, latency time.Duration, err error) {
	ctx := h.spanContext(stepKey(runID, step))
	_, span := h.tracer.Start(ctx, "agentflow.llm",
		trace.WithAttributes(
			attribute.String("agentflow.run_id", runID),
			attribute.Int("agentflow.step_index", step),
			attribute.Int("agentflow.prompt_tokens", promptTokens),
			attribute.Int("agentflow.completion_tokens", completionTokens),
			attribute.Int("agentflow.total_tokens", totalTokens),
			attribute.Int64("agentflow.duration_ms", latency.Milliseconds()),
		),
	)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetStatus(codes.Ok, "")
	}
	span.End()
}

func (h *OTelTracingHook) OnToolCall(_ context.Context, runID string, step int, toolName string, duration time.Duration, err error) {
	ctx := h.spanContext(stepKey(runID, step))
	_, span := h.tracer.Start(ctx, "agentflow.tool."+toolName,
		trace.WithAttributes(
			attribute.String("agentflow.run_id", runID),
			attribute.Int("agentflow.step_index", step),
			attribute.String("agentflow.tool_name", toolName),
			attribute.Int64("agentflow.duration_ms", duration.Milliseconds()),
			attribute.Bool("agentflow.success", err == nil),
		),
	)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
	} else {
		span.SetStatus(codes.Ok, "")
	}
	span.End()
}

// --- helpers ---

func (h *OTelTracingHook) store(key string, span trace.Span) {
	h.mu.Lock()
	h.spans[key] = span
	h.mu.Unlock()
}

func (h *OTelTracingHook) delete(key string) (trace.Span, bool) {
	h.mu.Lock()
	span, ok := h.spans[key]
	if ok {
		delete(h.spans, key)
	}
	h.mu.Unlock()
	return span, ok
}

// spanContext returns a context that carries the span stored under key as the
// active span, so that new child spans nest under it correctly. Falls back to
// context.Background() if no span is found.
func (h *OTelTracingHook) spanContext(key string) context.Context {
	h.mu.Lock()
	span := h.spans[key]
	h.mu.Unlock()
	if span == nil {
		return context.Background()
	}
	return trace.ContextWithSpan(context.Background(), span)
}

func runKey(runID string) string { return "run:" + runID }
func stepKey(runID string, step int) string {
	return fmt.Sprintf("step:%s:%d", runID, step)
}
