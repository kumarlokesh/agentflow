package otel_test

import (
	"context"
	"errors"
	"testing"
	"time"

	agotel "github.com/kumarlokesh/agentflow/observe/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func newTestTracerProvider(t *testing.T) (*sdktrace.TracerProvider, *tracetest.InMemoryExporter) {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { tp.Shutdown(context.Background()) })
	return tp, exp
}

func TestTracingHook_RunSpan(t *testing.T) {
	tp, exp := newTestTracerProvider(t)
	hook := agotel.NewTracingHook(tp)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-1", "solve task")
	hook.OnRunEnd(ctx, "run-1", 3, 2*time.Second, nil)

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	span := spans[0]
	if span.Name != "agentflow.run" {
		t.Errorf("name = %q, want agentflow.run", span.Name)
	}
	if span.Parent.IsValid() {
		t.Error("run span should have no parent")
	}
	assertAttr(t, span, "agentflow.run_id", "run-1")
	assertAttr(t, span, "agentflow.task", "solve task")
}

func TestTracingHook_StepSpanNestedUnderRun(t *testing.T) {
	tp, exp := newTestTracerProvider(t)
	hook := agotel.NewTracingHook(tp)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-2", "task")
	hook.OnStepStart(ctx, "run-2", 0)
	hook.OnStepEnd(ctx, "run-2", 0, 100*time.Millisecond)
	hook.OnRunEnd(ctx, "run-2", 1, time.Second, nil)

	spans := exp.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("expected 2 spans, got %d", len(spans))
	}

	// Find spans by name.
	var runSpan, stepSpan *tracetest.SpanStub
	for i := range spans {
		switch spans[i].Name {
		case "agentflow.run":
			runSpan = &spans[i]
		case "agentflow.step":
			stepSpan = &spans[i]
		}
	}
	if runSpan == nil || stepSpan == nil {
		t.Fatal("missing run or step span")
	}

	// Step span must be a child of the run span.
	if stepSpan.Parent.SpanID() != runSpan.SpanContext.SpanID() {
		t.Errorf("step parent = %s, want run span ID %s",
			stepSpan.Parent.SpanID(), runSpan.SpanContext.SpanID())
	}
	// They must share the same trace.
	if stepSpan.SpanContext.TraceID() != runSpan.SpanContext.TraceID() {
		t.Error("step and run spans have different trace IDs")
	}
}

func TestTracingHook_LLMSpanNestedUnderStep(t *testing.T) {
	tp, exp := newTestTracerProvider(t)
	hook := agotel.NewTracingHook(tp)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-3", "task")
	hook.OnStepStart(ctx, "run-3", 0)
	hook.OnLLMCall(ctx, "run-3", 0, 50, 20, 70, 500*time.Millisecond, nil)
	hook.OnStepEnd(ctx, "run-3", 0, time.Second)
	hook.OnRunEnd(ctx, "run-3", 1, 2*time.Second, nil)

	spans := exp.GetSpans()
	// Expect: run, step, llm = 3 spans.
	if len(spans) != 3 {
		t.Fatalf("expected 3 spans, got %d", len(spans))
	}

	var llmSpan, stepSpan *tracetest.SpanStub
	for i := range spans {
		switch spans[i].Name {
		case "agentflow.llm":
			llmSpan = &spans[i]
		case "agentflow.step":
			stepSpan = &spans[i]
		}
	}
	if llmSpan == nil || stepSpan == nil {
		t.Fatal("missing llm or step span")
	}
	if llmSpan.Parent.SpanID() != stepSpan.SpanContext.SpanID() {
		t.Errorf("llm parent = %s, want step span ID %s",
			llmSpan.Parent.SpanID(), stepSpan.SpanContext.SpanID())
	}
	assertAttr(t, *llmSpan, "agentflow.prompt_tokens", int64(50))
	assertAttr(t, *llmSpan, "agentflow.completion_tokens", int64(20))
	assertAttr(t, *llmSpan, "agentflow.total_tokens", int64(70))
}

func TestTracingHook_ToolSpanNestedUnderStep(t *testing.T) {
	tp, exp := newTestTracerProvider(t)
	hook := agotel.NewTracingHook(tp)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-4", "task")
	hook.OnStepStart(ctx, "run-4", 0)
	hook.OnToolCall(ctx, "run-4", 0, "calculator", 5*time.Millisecond, nil)
	hook.OnStepEnd(ctx, "run-4", 0, 50*time.Millisecond)
	hook.OnRunEnd(ctx, "run-4", 1, time.Second, nil)

	spans := exp.GetSpans()
	if len(spans) != 3 {
		t.Fatalf("expected 3 spans (run, step, tool), got %d", len(spans))
	}

	var toolSpan, stepSpan *tracetest.SpanStub
	for i := range spans {
		switch {
		case spans[i].Name == "agentflow.tool.calculator":
			toolSpan = &spans[i]
		case spans[i].Name == "agentflow.step":
			stepSpan = &spans[i]
		}
	}
	if toolSpan == nil || stepSpan == nil {
		t.Fatal("missing tool or step span")
	}
	if toolSpan.Parent.SpanID() != stepSpan.SpanContext.SpanID() {
		t.Errorf("tool parent = %s, want step span ID %s",
			toolSpan.Parent.SpanID(), stepSpan.SpanContext.SpanID())
	}
	assertAttr(t, *toolSpan, "agentflow.tool_name", "calculator")
	assertAttr(t, *toolSpan, "agentflow.success", true)
}

func TestTracingHook_ErrorPropagatedToSpan(t *testing.T) {
	tp, exp := newTestTracerProvider(t)
	hook := agotel.NewTracingHook(tp)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-5", "task")
	hook.OnRunEnd(ctx, "run-5", 0, 100*time.Millisecond, errors.New("timeout"))

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("status.Code = %v, want codes.Error (%v)", spans[0].Status.Code, codes.Error)
	}
}

func TestTracingHook_MultipleSteps(t *testing.T) {
	tp, exp := newTestTracerProvider(t)
	hook := agotel.NewTracingHook(tp)

	ctx := context.Background()
	hook.OnRunStart(ctx, "run-6", "task")
	for step := 0; step < 3; step++ {
		hook.OnStepStart(ctx, "run-6", step)
		hook.OnLLMCall(ctx, "run-6", step, 10, 5, 15, 100*time.Millisecond, nil)
		hook.OnStepEnd(ctx, "run-6", step, 200*time.Millisecond)
	}
	hook.OnRunEnd(ctx, "run-6", 3, time.Second, nil)

	spans := exp.GetSpans()
	// 1 run + 3 steps + 3 llm calls = 7 spans.
	if len(spans) != 7 {
		t.Errorf("expected 7 spans, got %d", len(spans))
	}
}

// --- helpers ---

func assertAttr(t *testing.T, span tracetest.SpanStub, key string, want any) {
	t.Helper()
	for _, kv := range span.Attributes {
		if string(kv.Key) != key {
			continue
		}
		var got any
		switch kv.Value.Type().String() {
		case "STRING":
			got = kv.Value.AsString()
		case "BOOL":
			got = kv.Value.AsBool()
		case "INT64":
			got = kv.Value.AsInt64()
		case "FLOAT64":
			got = kv.Value.AsFloat64()
		}
		if got != want {
			t.Errorf("attr %q = %v, want %v", key, got, want)
		}
		return
	}
	t.Errorf("attr %q not found in span %q", key, span.Name)
}
