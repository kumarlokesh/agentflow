// Package observe provides observability primitives for agentflow agents.
//
// It implements tracing (OpenTelemetry-compatible spans), metrics collection
// (token usage, tool latency, step counts), and hooks that integrate with
// the agent runtime without modifying the core loop.
//
// Design rationale: Observability is implemented as composable hooks rather
// than hard-wired into the agent. This keeps the core runtime clean and lets
// users opt in to the level of instrumentation they need.
package observe

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// --- Span-based Tracing ---

// SpanKind classifies a span.
type SpanKind int

const (
	SpanKindRun      SpanKind = iota // Top-level agent run
	SpanKindStep                     // Single loop iteration
	SpanKindLLMCall                  // LLM request/response
	SpanKindToolCall                 // Tool execution
)

// Span represents a unit of work in the agent execution trace.
type Span struct {
	// TraceID groups all spans in a single run.
	TraceID string
	// SpanID uniquely identifies this span.
	SpanID string
	// ParentID links to the parent span (empty for root).
	ParentID string
	// Name is a human-readable name for this span.
	Name string
	// Kind classifies the span.
	Kind SpanKind
	// StartTime is when the span began.
	StartTime time.Time
	// EndTime is when the span ended (zero if still active).
	EndTime time.Time
	// Attributes holds key-value metadata.
	Attributes map[string]string
	// Status is the span outcome ("ok", "error").
	Status string
	// StatusMessage provides details on error status.
	StatusMessage string
}

// Duration returns the span's elapsed time.
func (s *Span) Duration() time.Duration {
	if s.EndTime.IsZero() {
		return 0
	}
	return s.EndTime.Sub(s.StartTime)
}

// Tracer records spans for a single agent run.
type Tracer struct {
	mu      sync.Mutex
	traceID string
	spans   []*Span
	active  map[string]*Span // spanID -> active span
	counter int
}

// NewTracer creates a tracer for the given trace/run ID.
func NewTracer(traceID string) *Tracer {
	return &Tracer{
		traceID: traceID,
		active:  make(map[string]*Span),
	}
}

// StartSpan begins a new span with the given name and kind.
func (t *Tracer) StartSpan(name string, kind SpanKind, parentID string) *Span {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.counter++
	span := &Span{
		TraceID:    t.traceID,
		SpanID:     fmt.Sprintf("span-%d", t.counter),
		ParentID:   parentID,
		Name:       name,
		Kind:       kind,
		StartTime:  time.Now(),
		Attributes: make(map[string]string),
		Status:     "ok",
	}
	t.active[span.SpanID] = span
	return span
}

// EndSpan completes a span, recording its end time and status.
func (t *Tracer) EndSpan(span *Span, status, statusMessage string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	span.EndTime = time.Now()
	span.Status = status
	span.StatusMessage = statusMessage
	delete(t.active, span.SpanID)
	t.spans = append(t.spans, span)
}

// Spans returns all completed spans in order.
func (t *Tracer) Spans() []*Span {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]*Span, len(t.spans))
	copy(out, t.spans)
	return out
}

// ActiveSpans returns the count of spans that haven't been ended.
func (t *Tracer) ActiveSpans() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.active)
}

// --- Metrics Collector ---

// Metrics collects quantitative measurements about agent execution.
type Metrics struct {
	mu sync.Mutex

	// Token metrics
	totalPromptTokens     int64
	totalCompletionTokens int64
	totalTokens           int64
	llmCallCount          int64

	// Tool metrics
	toolCallCount  int64
	toolErrorCount int64
	toolDurations  map[string][]time.Duration // tool_name -> durations
	toolCallCounts map[string]int64           // tool_name -> count

	// Step metrics
	stepCount int64
	runCount  int64

	// Latency metrics
	llmLatencies []time.Duration
}

// NewMetrics creates an empty metrics collector.
func NewMetrics() *Metrics {
	return &Metrics{
		toolDurations:  make(map[string][]time.Duration),
		toolCallCounts: make(map[string]int64),
	}
}

// RecordLLMCall records token usage and latency for an LLM call.
func (m *Metrics) RecordLLMCall(promptTokens, completionTokens, totalTokens int, latency time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.totalPromptTokens += int64(promptTokens)
	m.totalCompletionTokens += int64(completionTokens)
	m.totalTokens += int64(totalTokens)
	m.llmCallCount++
	m.llmLatencies = append(m.llmLatencies, latency)
}

// RecordToolCall records a tool execution with its duration and success/failure.
func (m *Metrics) RecordToolCall(toolName string, duration time.Duration, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.toolCallCount++
	m.toolCallCounts[toolName]++
	m.toolDurations[toolName] = append(m.toolDurations[toolName], duration)
	if err != nil {
		m.toolErrorCount++
	}
}

// RecordStep increments the step counter.
func (m *Metrics) RecordStep() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stepCount++
}

// RecordRun increments the run counter.
func (m *Metrics) RecordRun() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.runCount++
}

// Snapshot returns a point-in-time copy of all metrics.
func (m *Metrics) Snapshot() MetricsSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	toolStats := make(map[string]ToolStats, len(m.toolCallCounts))
	for name, count := range m.toolCallCounts {
		durations := m.toolDurations[name]
		var totalDur time.Duration
		for _, d := range durations {
			totalDur += d
		}
		var avgDur time.Duration
		if len(durations) > 0 {
			avgDur = totalDur / time.Duration(len(durations))
		}
		toolStats[name] = ToolStats{
			CallCount:   count,
			TotalTime:   totalDur,
			AverageTime: avgDur,
		}
	}

	var avgLLMLatency time.Duration
	if len(m.llmLatencies) > 0 {
		var total time.Duration
		for _, l := range m.llmLatencies {
			total += l
		}
		avgLLMLatency = total / time.Duration(len(m.llmLatencies))
	}

	return MetricsSnapshot{
		TotalPromptTokens:     m.totalPromptTokens,
		TotalCompletionTokens: m.totalCompletionTokens,
		TotalTokens:           m.totalTokens,
		LLMCallCount:          m.llmCallCount,
		AvgLLMLatency:         avgLLMLatency,
		ToolCallCount:         m.toolCallCount,
		ToolErrorCount:        m.toolErrorCount,
		ToolStats:             toolStats,
		StepCount:             m.stepCount,
		RunCount:              m.runCount,
	}
}

// MetricsSnapshot is a read-only view of collected metrics.
type MetricsSnapshot struct {
	TotalPromptTokens     int64
	TotalCompletionTokens int64
	TotalTokens           int64
	LLMCallCount          int64
	AvgLLMLatency         time.Duration
	ToolCallCount         int64
	ToolErrorCount        int64
	ToolStats             map[string]ToolStats
	StepCount             int64
	RunCount              int64
}

// ToolStats holds per-tool metrics.
type ToolStats struct {
	CallCount   int64
	TotalTime   time.Duration
	AverageTime time.Duration
}

// --- Hook Interface ---

// Hook is called at key points during agent execution.
// Implementations can record traces, collect metrics, or enforce policies.
type Hook interface {
	// OnRunStart is called when an agent run begins.
	OnRunStart(ctx context.Context, runID, task string)
	// OnRunEnd is called when an agent run completes.
	OnRunEnd(ctx context.Context, runID string, steps int, duration time.Duration, err error)
	// OnStepStart is called at the beginning of each loop iteration.
	OnStepStart(ctx context.Context, runID string, step int)
	// OnStepEnd is called at the end of each loop iteration.
	OnStepEnd(ctx context.Context, runID string, step int, duration time.Duration)
	// OnLLMCall is called after an LLM request completes.
	OnLLMCall(ctx context.Context, runID string, step int, promptTokens, completionTokens, totalTokens int, latency time.Duration, err error)
	// OnToolCall is called after a tool execution completes.
	OnToolCall(ctx context.Context, runID string, step int, toolName string, duration time.Duration, err error)
}

// --- MetricsHook ---

// MetricsHook is a Hook that feeds data into a Metrics collector.
type MetricsHook struct {
	metrics *Metrics
}

// NewMetricsHook creates a hook backed by the given metrics collector.
func NewMetricsHook(m *Metrics) *MetricsHook {
	return &MetricsHook{metrics: m}
}

func (h *MetricsHook) OnRunStart(_ context.Context, _, _ string) {
	h.metrics.RecordRun()
}
func (h *MetricsHook) OnRunEnd(_ context.Context, _ string, _ int, _ time.Duration, _ error) {}
func (h *MetricsHook) OnStepStart(_ context.Context, _ string, _ int) {
	h.metrics.RecordStep()
}
func (h *MetricsHook) OnStepEnd(_ context.Context, _ string, _ int, _ time.Duration) {}
func (h *MetricsHook) OnLLMCall(_ context.Context, _ string, _ int, prompt, completion, total int, latency time.Duration, _ error) {
	h.metrics.RecordLLMCall(prompt, completion, total, latency)
}
func (h *MetricsHook) OnToolCall(_ context.Context, _ string, _ int, toolName string, duration time.Duration, err error) {
	h.metrics.RecordToolCall(toolName, duration, err)
}

// --- TracingHook ---

// TracingHook is a Hook that records spans into a Tracer.
type TracingHook struct {
	mu      sync.Mutex
	tracers map[string]*Tracer // runID -> tracer
	spans   map[string]*Span   // runID or runID:step -> active span
}

// NewTracingHook creates a tracing hook.
func NewTracingHook() *TracingHook {
	return &TracingHook{
		tracers: make(map[string]*Tracer),
		spans:   make(map[string]*Span),
	}
}

func (h *TracingHook) OnRunStart(_ context.Context, runID, task string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	tracer := NewTracer(runID)
	h.tracers[runID] = tracer
	span := tracer.StartSpan("agent.run", SpanKindRun, "")
	span.Attributes["task"] = task
	h.spans[runID] = span
}

func (h *TracingHook) OnRunEnd(_ context.Context, runID string, steps int, duration time.Duration, err error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	tracer, ok := h.tracers[runID]
	if !ok {
		return
	}
	span := h.spans[runID]
	if span == nil {
		return
	}
	span.Attributes["steps"] = fmt.Sprintf("%d", steps)
	span.Attributes["duration"] = duration.String()
	status := "ok"
	msg := ""
	if err != nil {
		status = "error"
		msg = err.Error()
	}
	tracer.EndSpan(span, status, msg)
	delete(h.spans, runID)
}

func (h *TracingHook) OnStepStart(_ context.Context, runID string, step int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	tracer, ok := h.tracers[runID]
	if !ok {
		return
	}
	parentSpan := h.spans[runID]
	parentID := ""
	if parentSpan != nil {
		parentID = parentSpan.SpanID
	}
	span := tracer.StartSpan(fmt.Sprintf("step.%d", step), SpanKindStep, parentID)
	h.spans[stepKey(runID, step)] = span
}

func (h *TracingHook) OnStepEnd(_ context.Context, runID string, step int, _ time.Duration) {
	h.mu.Lock()
	defer h.mu.Unlock()

	tracer, ok := h.tracers[runID]
	if !ok {
		return
	}
	key := stepKey(runID, step)
	span := h.spans[key]
	if span == nil {
		return
	}
	tracer.EndSpan(span, "ok", "")
	delete(h.spans, key)
}

func (h *TracingHook) OnLLMCall(_ context.Context, runID string, step int, _, _, totalTokens int, latency time.Duration, err error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	tracer, ok := h.tracers[runID]
	if !ok {
		return
	}
	parentSpan := h.spans[stepKey(runID, step)]
	parentID := ""
	if parentSpan != nil {
		parentID = parentSpan.SpanID
	}
	span := tracer.StartSpan("llm.call", SpanKindLLMCall, parentID)
	span.Attributes["total_tokens"] = fmt.Sprintf("%d", totalTokens)
	span.Attributes["latency"] = latency.String()
	status, msg := "ok", ""
	if err != nil {
		status = "error"
		msg = err.Error()
	}
	tracer.EndSpan(span, status, msg)
}

func (h *TracingHook) OnToolCall(_ context.Context, runID string, step int, toolName string, duration time.Duration, err error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	tracer, ok := h.tracers[runID]
	if !ok {
		return
	}
	parentSpan := h.spans[stepKey(runID, step)]
	parentID := ""
	if parentSpan != nil {
		parentID = parentSpan.SpanID
	}
	span := tracer.StartSpan("tool."+toolName, SpanKindToolCall, parentID)
	span.Attributes["tool_name"] = toolName
	span.Attributes["duration"] = duration.String()
	status, msg := "ok", ""
	if err != nil {
		status = "error"
		msg = err.Error()
	}
	tracer.EndSpan(span, status, msg)
}

// TracerFor returns the Tracer for a specific run, or nil if not found.
func (h *TracingHook) TracerFor(runID string) *Tracer {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.tracers[runID]
}

func stepKey(runID string, step int) string {
	return fmt.Sprintf("%s:step:%d", runID, step)
}

// --- Multi-Hook ---

// MultiHook dispatches hook calls to multiple underlying hooks.
type MultiHook struct {
	hooks []Hook
}

// NewMultiHook creates a hook that forwards to all given hooks.
func NewMultiHook(hooks ...Hook) *MultiHook {
	return &MultiHook{hooks: hooks}
}

func (m *MultiHook) OnRunStart(ctx context.Context, runID, task string) {
	for _, h := range m.hooks {
		h.OnRunStart(ctx, runID, task)
	}
}
func (m *MultiHook) OnRunEnd(ctx context.Context, runID string, steps int, duration time.Duration, err error) {
	for _, h := range m.hooks {
		h.OnRunEnd(ctx, runID, steps, duration, err)
	}
}
func (m *MultiHook) OnStepStart(ctx context.Context, runID string, step int) {
	for _, h := range m.hooks {
		h.OnStepStart(ctx, runID, step)
	}
}
func (m *MultiHook) OnStepEnd(ctx context.Context, runID string, step int, duration time.Duration) {
	for _, h := range m.hooks {
		h.OnStepEnd(ctx, runID, step, duration)
	}
}
func (m *MultiHook) OnLLMCall(ctx context.Context, runID string, step int, prompt, completion, total int, latency time.Duration, err error) {
	for _, h := range m.hooks {
		h.OnLLMCall(ctx, runID, step, prompt, completion, total, latency, err)
	}
}
func (m *MultiHook) OnToolCall(ctx context.Context, runID string, step int, toolName string, duration time.Duration, err error) {
	for _, h := range m.hooks {
		h.OnToolCall(ctx, runID, step, toolName, duration, err)
	}
}
