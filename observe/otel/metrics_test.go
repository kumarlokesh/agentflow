package otel_test

import (
	"context"
	"errors"
	"testing"
	"time"

	agotel "github.com/kumarlokesh/agentflow/observe/otel"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

func newTestMeterProvider(t *testing.T) (*sdkmetric.MeterProvider, *sdkmetric.ManualReader) {
	t.Helper()
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() { mp.Shutdown(context.Background()) })
	return mp, reader
}

func collectMetrics(t *testing.T, reader *sdkmetric.ManualReader) metricdata.ResourceMetrics {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &rm); err != nil {
		t.Fatalf("collect metrics: %v", err)
	}
	return rm
}

func findMetric(rm metricdata.ResourceMetrics, name string) *metricdata.Metrics {
	for i := range rm.ScopeMetrics {
		for j := range rm.ScopeMetrics[i].Metrics {
			if rm.ScopeMetrics[i].Metrics[j].Name == name {
				return &rm.ScopeMetrics[i].Metrics[j]
			}
		}
	}
	return nil
}

func sumCounter(m *metricdata.Metrics) int64 {
	if m == nil {
		return 0
	}
	data, ok := m.Data.(metricdata.Sum[int64])
	if !ok {
		return 0
	}
	var total int64
	for _, dp := range data.DataPoints {
		total += dp.Value
	}
	return total
}

func TestMetricsHook_LLMCallCounted(t *testing.T) {
	mp, reader := newTestMeterProvider(t)
	hook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		t.Fatalf("NewMetricsHook: %v", err)
	}

	ctx := context.Background()
	hook.OnLLMCall(ctx, "run-1", 0, 50, 20, 70, 500*time.Millisecond, nil)
	hook.OnLLMCall(ctx, "run-1", 1, 30, 10, 40, 300*time.Millisecond, nil)

	rm := collectMetrics(t, reader)
	m := findMetric(rm, "agentflow_llm_calls_total")
	if m == nil {
		t.Fatal("agentflow_llm_calls_total metric not found")
	}
	if total := sumCounter(m); total != 2 {
		t.Errorf("llm_calls_total = %d, want 2", total)
	}
}

func TestMetricsHook_TokensCounted(t *testing.T) {
	mp, reader := newTestMeterProvider(t)
	hook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		t.Fatalf("NewMetricsHook: %v", err)
	}

	ctx := context.Background()
	// prompt: 50+30=80, completion: 20+10=30
	hook.OnLLMCall(ctx, "run-1", 0, 50, 20, 70, time.Second, nil)
	hook.OnLLMCall(ctx, "run-1", 1, 30, 10, 40, time.Second, nil)

	rm := collectMetrics(t, reader)
	m := findMetric(rm, "agentflow_tokens_total")
	if m == nil {
		t.Fatal("agentflow_tokens_total metric not found")
	}
	// Total across all token types should be 80+30=110.
	if total := sumCounter(m); total != 110 {
		t.Errorf("tokens_total sum = %d, want 110", total)
	}
}

func TestMetricsHook_ToolCallCounted(t *testing.T) {
	mp, reader := newTestMeterProvider(t)
	hook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		t.Fatalf("NewMetricsHook: %v", err)
	}

	ctx := context.Background()
	hook.OnToolCall(ctx, "run-1", 0, "calculator", 5*time.Millisecond, nil)
	hook.OnToolCall(ctx, "run-1", 0, "calculator", 3*time.Millisecond, nil)
	hook.OnToolCall(ctx, "run-1", 0, "search", 10*time.Millisecond, errors.New("not found"))

	rm := collectMetrics(t, reader)
	m := findMetric(rm, "agentflow_tool_calls_total")
	if m == nil {
		t.Fatal("agentflow_tool_calls_total metric not found")
	}
	if total := sumCounter(m); total != 3 {
		t.Errorf("tool_calls_total = %d, want 3", total)
	}
}

func TestMetricsHook_LLMErrorStatus(t *testing.T) {
	mp, reader := newTestMeterProvider(t)
	hook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		t.Fatalf("NewMetricsHook: %v", err)
	}

	ctx := context.Background()
	hook.OnLLMCall(ctx, "run-1", 0, 0, 0, 0, 100*time.Millisecond, errors.New("timeout"))

	rm := collectMetrics(t, reader)
	m := findMetric(rm, "agentflow_llm_calls_total")
	if m == nil {
		t.Fatal("agentflow_llm_calls_total not found")
	}

	data, ok := m.Data.(metricdata.Sum[int64])
	if !ok {
		t.Fatal("unexpected data type")
	}

	// Find the data point with status=error.
	found := false
	for _, dp := range data.DataPoints {
		for _, attr := range dp.Attributes.ToSlice() {
			if string(attr.Key) == "status" && attr.Value.AsString() == "error" {
				if dp.Value != 1 {
					t.Errorf("error count = %d, want 1", dp.Value)
				}
				found = true
			}
		}
	}
	if !found {
		t.Error("no data point with status=error found")
	}
}

func TestMetricsHook_LLMDurationRecorded(t *testing.T) {
	mp, reader := newTestMeterProvider(t)
	hook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		t.Fatalf("NewMetricsHook: %v", err)
	}

	ctx := context.Background()
	hook.OnLLMCall(ctx, "run-1", 0, 10, 5, 15, 750*time.Millisecond, nil)

	rm := collectMetrics(t, reader)
	m := findMetric(rm, "agentflow_llm_duration_seconds")
	if m == nil {
		t.Fatal("agentflow_llm_duration_seconds not found")
	}

	hist, ok := m.Data.(metricdata.Histogram[float64])
	if !ok {
		t.Fatalf("expected Histogram, got %T", m.Data)
	}
	if len(hist.DataPoints) == 0 {
		t.Fatal("no histogram data points")
	}
	if hist.DataPoints[0].Count != 1 {
		t.Errorf("histogram count = %d, want 1", hist.DataPoints[0].Count)
	}
}

func TestMetricsHook_AllInstrumentsPresent(t *testing.T) {
	mp, reader := newTestMeterProvider(t)
	hook, err := agotel.NewMetricsHook(mp)
	if err != nil {
		t.Fatalf("NewMetricsHook: %v", err)
	}

	ctx := context.Background()
	hook.OnLLMCall(ctx, "r", 0, 1, 1, 2, time.Millisecond, nil)
	hook.OnToolCall(ctx, "r", 0, "tool", time.Millisecond, nil)

	rm := collectMetrics(t, reader)
	want := []string{
		"agentflow_llm_calls_total",
		"agentflow_tool_calls_total",
		"agentflow_tokens_total",
		"agentflow_llm_duration_seconds",
		"agentflow_tool_duration_seconds",
	}
	for _, name := range want {
		if findMetric(rm, name) == nil {
			t.Errorf("metric %q not found", name)
		}
	}
}
