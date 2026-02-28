package observe

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// --- Span Tests ---

func TestSpan_Duration(t *testing.T) {
	s := &Span{
		StartTime: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
		EndTime:   time.Date(2025, 1, 1, 0, 0, 1, 0, time.UTC),
	}
	if s.Duration() != time.Second {
		t.Errorf("Duration() = %v, want 1s", s.Duration())
	}
}

func TestSpan_Duration_ZeroEnd(t *testing.T) {
	s := &Span{StartTime: time.Now()}
	if s.Duration() != 0 {
		t.Error("Duration() should be 0 for unfinished span")
	}
}

// --- Tracer Tests ---

func TestTracer_StartAndEndSpan(t *testing.T) {
	tracer := NewTracer("trace-001")

	span := tracer.StartSpan("test.op", SpanKindRun, "")
	if span.TraceID != "trace-001" {
		t.Errorf("TraceID = %q", span.TraceID)
	}
	if span.SpanID == "" {
		t.Error("SpanID is empty")
	}
	if span.Name != "test.op" {
		t.Errorf("Name = %q", span.Name)
	}
	if tracer.ActiveSpans() != 1 {
		t.Errorf("ActiveSpans = %d, want 1", tracer.ActiveSpans())
	}

	tracer.EndSpan(span, "ok", "")
	if tracer.ActiveSpans() != 0 {
		t.Errorf("ActiveSpans = %d, want 0", tracer.ActiveSpans())
	}

	spans := tracer.Spans()
	if len(spans) != 1 {
		t.Fatalf("Spans() len = %d, want 1", len(spans))
	}
	if spans[0].Status != "ok" {
		t.Errorf("Status = %q, want ok", spans[0].Status)
	}
	if spans[0].Duration() <= 0 {
		t.Error("Duration should be positive")
	}
}

func TestTracer_ParentChild(t *testing.T) {
	tracer := NewTracer("trace-002")

	parent := tracer.StartSpan("parent", SpanKindRun, "")
	child := tracer.StartSpan("child", SpanKindStep, parent.SpanID)

	if child.ParentID != parent.SpanID {
		t.Errorf("ParentID = %q, want %q", child.ParentID, parent.SpanID)
	}

	tracer.EndSpan(child, "ok", "")
	tracer.EndSpan(parent, "ok", "")

	if len(tracer.Spans()) != 2 {
		t.Errorf("Spans() len = %d, want 2", len(tracer.Spans()))
	}
}

func TestTracer_ErrorSpan(t *testing.T) {
	tracer := NewTracer("trace-003")
	span := tracer.StartSpan("fail.op", SpanKindLLMCall, "")
	tracer.EndSpan(span, "error", "model timeout")

	spans := tracer.Spans()
	if spans[0].Status != "error" {
		t.Errorf("Status = %q, want error", spans[0].Status)
	}
	if spans[0].StatusMessage != "model timeout" {
		t.Errorf("StatusMessage = %q", spans[0].StatusMessage)
	}
}

func TestTracer_Concurrent(t *testing.T) {
	tracer := NewTracer("trace-concurrent")
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			span := tracer.StartSpan("op", SpanKindStep, "")
			tracer.EndSpan(span, "ok", "")
			tracer.Spans()
			tracer.ActiveSpans()
		}()
	}
	wg.Wait()

	if len(tracer.Spans()) != 50 {
		t.Errorf("Spans() len = %d, want 50", len(tracer.Spans()))
	}
}

// --- Metrics Tests ---

func TestMetrics_RecordLLMCall(t *testing.T) {
	m := NewMetrics()
	m.RecordLLMCall(100, 50, 150, 200*time.Millisecond)
	m.RecordLLMCall(80, 30, 110, 100*time.Millisecond)

	snap := m.Snapshot()
	if snap.TotalPromptTokens != 180 {
		t.Errorf("TotalPromptTokens = %d, want 180", snap.TotalPromptTokens)
	}
	if snap.TotalCompletionTokens != 80 {
		t.Errorf("TotalCompletionTokens = %d, want 80", snap.TotalCompletionTokens)
	}
	if snap.TotalTokens != 260 {
		t.Errorf("TotalTokens = %d, want 260", snap.TotalTokens)
	}
	if snap.LLMCallCount != 2 {
		t.Errorf("LLMCallCount = %d, want 2", snap.LLMCallCount)
	}
	if snap.AvgLLMLatency != 150*time.Millisecond {
		t.Errorf("AvgLLMLatency = %v, want 150ms", snap.AvgLLMLatency)
	}
}

func TestMetrics_RecordToolCall(t *testing.T) {
	m := NewMetrics()
	m.RecordToolCall("calc", 10*time.Millisecond, nil)
	m.RecordToolCall("calc", 20*time.Millisecond, nil)
	m.RecordToolCall("search", 100*time.Millisecond, errors.New("timeout"))

	snap := m.Snapshot()
	if snap.ToolCallCount != 3 {
		t.Errorf("ToolCallCount = %d, want 3", snap.ToolCallCount)
	}
	if snap.ToolErrorCount != 1 {
		t.Errorf("ToolErrorCount = %d, want 1", snap.ToolErrorCount)
	}

	calcStats := snap.ToolStats["calc"]
	if calcStats.CallCount != 2 {
		t.Errorf("calc CallCount = %d, want 2", calcStats.CallCount)
	}
	if calcStats.AverageTime != 15*time.Millisecond {
		t.Errorf("calc AverageTime = %v, want 15ms", calcStats.AverageTime)
	}

	searchStats := snap.ToolStats["search"]
	if searchStats.CallCount != 1 {
		t.Errorf("search CallCount = %d, want 1", searchStats.CallCount)
	}
}

func TestMetrics_StepAndRunCounters(t *testing.T) {
	m := NewMetrics()
	m.RecordRun()
	m.RecordRun()
	m.RecordStep()
	m.RecordStep()
	m.RecordStep()

	snap := m.Snapshot()
	if snap.RunCount != 2 {
		t.Errorf("RunCount = %d, want 2", snap.RunCount)
	}
	if snap.StepCount != 3 {
		t.Errorf("StepCount = %d, want 3", snap.StepCount)
	}
}

func TestMetrics_EmptySnapshot(t *testing.T) {
	m := NewMetrics()
	snap := m.Snapshot()
	if snap.AvgLLMLatency != 0 {
		t.Error("AvgLLMLatency should be 0 with no calls")
	}
	if len(snap.ToolStats) != 0 {
		t.Error("ToolStats should be empty")
	}
}

func TestMetrics_Concurrent(t *testing.T) {
	m := NewMetrics()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m.RecordLLMCall(10, 5, 15, time.Millisecond)
			m.RecordToolCall("t", time.Millisecond, nil)
			m.RecordStep()
			m.RecordRun()
			m.Snapshot()
		}()
	}
	wg.Wait()
	snap := m.Snapshot()
	if snap.LLMCallCount != 100 {
		t.Errorf("LLMCallCount = %d, want 100", snap.LLMCallCount)
	}
}

// --- MetricsHook Tests ---

func TestMetricsHook(t *testing.T) {
	m := NewMetrics()
	hook := NewMetricsHook(m)
	ctx := context.Background()

	hook.OnRunStart(ctx, "run-1", "task")
	hook.OnStepStart(ctx, "run-1", 0)
	hook.OnLLMCall(ctx, "run-1", 0, 100, 50, 150, 200*time.Millisecond, nil)
	hook.OnToolCall(ctx, "run-1", 0, "calc", 10*time.Millisecond, nil)
	hook.OnStepEnd(ctx, "run-1", 0, time.Second)
	hook.OnRunEnd(ctx, "run-1", 1, time.Second, nil)

	snap := m.Snapshot()
	if snap.RunCount != 1 {
		t.Errorf("RunCount = %d, want 1", snap.RunCount)
	}
	if snap.StepCount != 1 {
		t.Errorf("StepCount = %d, want 1", snap.StepCount)
	}
	if snap.LLMCallCount != 1 {
		t.Errorf("LLMCallCount = %d, want 1", snap.LLMCallCount)
	}
	if snap.ToolCallCount != 1 {
		t.Errorf("ToolCallCount = %d, want 1", snap.ToolCallCount)
	}
}

// --- TracingHook Tests ---

func TestTracingHook_FullLifecycle(t *testing.T) {
	hook := NewTracingHook()
	ctx := context.Background()

	hook.OnRunStart(ctx, "run-1", "solve math")
	hook.OnStepStart(ctx, "run-1", 0)
	hook.OnLLMCall(ctx, "run-1", 0, 100, 50, 150, 200*time.Millisecond, nil)
	hook.OnToolCall(ctx, "run-1", 0, "calculator", 10*time.Millisecond, nil)
	hook.OnStepEnd(ctx, "run-1", 0, time.Second)
	hook.OnRunEnd(ctx, "run-1", 1, time.Second, nil)

	tracer := hook.TracerFor("run-1")
	if tracer == nil {
		t.Fatal("tracer not found for run-1")
	}

	spans := tracer.Spans()
	if len(spans) < 4 {
		t.Fatalf("expected >=4 spans, got %d", len(spans))
	}

	// Verify span kinds exist.
	kindCounts := make(map[SpanKind]int)
	for _, s := range spans {
		kindCounts[s.Kind]++
	}

	if kindCounts[SpanKindRun] != 1 {
		t.Errorf("Run spans = %d, want 1", kindCounts[SpanKindRun])
	}
	if kindCounts[SpanKindStep] != 1 {
		t.Errorf("Step spans = %d, want 1", kindCounts[SpanKindStep])
	}
	if kindCounts[SpanKindLLMCall] != 1 {
		t.Errorf("LLM spans = %d, want 1", kindCounts[SpanKindLLMCall])
	}
	if kindCounts[SpanKindToolCall] != 1 {
		t.Errorf("Tool spans = %d, want 1", kindCounts[SpanKindToolCall])
	}
}

func TestTracingHook_ErrorRun(t *testing.T) {
	hook := NewTracingHook()
	ctx := context.Background()

	hook.OnRunStart(ctx, "run-err", "fail task")
	hook.OnRunEnd(ctx, "run-err", 0, time.Second, errors.New("budget exceeded"))

	tracer := hook.TracerFor("run-err")
	spans := tracer.Spans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].Status != "error" {
		t.Errorf("Status = %q, want error", spans[0].Status)
	}
}

func TestTracingHook_UnknownRun(t *testing.T) {
	hook := NewTracingHook()
	ctx := context.Background()
	// These should not panic.
	hook.OnStepStart(ctx, "unknown", 0)
	hook.OnStepEnd(ctx, "unknown", 0, time.Second)
	hook.OnLLMCall(ctx, "unknown", 0, 0, 0, 0, 0, nil)
	hook.OnToolCall(ctx, "unknown", 0, "tool", 0, nil)
	hook.OnRunEnd(ctx, "unknown", 0, 0, nil)
}

func TestTracingHook_TracerFor_Nil(t *testing.T) {
	hook := NewTracingHook()
	if hook.TracerFor("nonexistent") != nil {
		t.Error("expected nil for nonexistent run")
	}
}

// --- MultiHook Tests ---

func TestMultiHook(t *testing.T) {
	m := NewMetrics()
	mh := NewMetricsHook(m)
	th := NewTracingHook()

	multi := NewMultiHook(mh, th)
	ctx := context.Background()

	multi.OnRunStart(ctx, "run-1", "task")
	multi.OnStepStart(ctx, "run-1", 0)
	multi.OnLLMCall(ctx, "run-1", 0, 10, 5, 15, time.Millisecond, nil)
	multi.OnToolCall(ctx, "run-1", 0, "tool", time.Millisecond, nil)
	multi.OnStepEnd(ctx, "run-1", 0, time.Second)
	multi.OnRunEnd(ctx, "run-1", 1, time.Second, nil)

	// Metrics should be populated.
	snap := m.Snapshot()
	if snap.RunCount != 1 {
		t.Errorf("RunCount = %d", snap.RunCount)
	}

	// Tracing should have spans.
	tracer := th.TracerFor("run-1")
	if tracer == nil {
		t.Fatal("tracer nil")
	}
	if len(tracer.Spans()) < 4 {
		t.Errorf("spans = %d, want >=4", len(tracer.Spans()))
	}
}
