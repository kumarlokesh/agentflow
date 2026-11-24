package otel_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	agotel "github.com/kumarlokesh/agentflow/observe/otel"
	aftrace "github.com/kumarlokesh/agentflow/trace"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func run(t *testing.T, m llm.Model, obs ...aftrace.Observer) {
	t.Helper()
	tool := agentflow.NewTool(llm.ToolSpec{Name: "calc", InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, struct{}) (agentflow.ToolOutput, error) { return agentflow.TextOutput("42"), nil })
	broken := agentflow.NewTool(llm.ToolSpec{Name: "broken"},
		func(context.Context, struct{}) (agentflow.ToolOutput, error) {
			return agentflow.ToolOutput{}, errors.New("nope")
		})
	a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "calc-agent", Model: m, ModelName: "claude-opus-5-5",
		Tools: []agentflow.Tool{tool, broken}, Observers: obs})
	if err != nil {
		t.Fatal(err)
	}
	_, _ = a.Run(context.Background(), "6*7?")
}

func script() llm.Model {
	return agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("t1", "calc", struct{}{}), agentflowtest.Use("t2", "broken", struct{}{})),
		agentflowtest.Reply("42"))
}

func attr(kvs []attribute.KeyValue, key string) attribute.Value {
	for _, kv := range kvs {
		if string(kv.Key) == key {
			return kv.Value
		}
	}
	return attribute.Value{}
}

func TestTracerBuildsASpanTree(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	run(t, script(), agotel.NewTracer(tp))

	spans := exp.GetSpans()
	byName := map[string]tracetest.SpanStub{}
	llmSpans := 0
	for _, s := range spans {
		byName[s.Name] = s
		if s.Name == "agentflow.llm" {
			llmSpans++
		}
	}
	root, ok := byName["agentflow.run"]
	if !ok || root.Parent.IsValid() {
		t.Fatalf("missing root run span: %v", spans)
	}
	if len(spans) != 5 || llmSpans != 2 {
		t.Fatalf("got %d spans (%d llm), want 5 (run, 2 llm, 2 tools)", len(spans), llmSpans)
	}
	for _, name := range []string{"agentflow.llm", "agentflow.tool.calc", "agentflow.tool.broken"} {
		s := byName[name]
		if s.Parent.SpanID() != root.SpanContext.SpanID() {
			t.Errorf("%s is not a child of the run span", name)
		}
		if !s.EndTime.After(s.StartTime) && !s.EndTime.Equal(s.StartTime) {
			t.Errorf("%s has bad timing", name)
		}
	}
	llmSpan := byName["agentflow.llm"]
	if attr(llmSpan.Attributes, "gen_ai.request.model").AsString() != "claude-opus-5-5" || attr(llmSpan.Attributes, "gen_ai.usage.input_tokens").AsInt64() != 10 {
		t.Errorf("llm attributes = %v", llmSpan.Attributes)
	}
	if b := byName["agentflow.tool.broken"]; b.Status.Code != codes.Error || attr(b.Attributes, "error.type").AsString() != "error" {
		t.Errorf("failed tool span = %+v %v", b.Status, b.Attributes)
	}
	if root.Status.Code != codes.Ok || attr(root.Attributes, "agentflow.status").AsString() != "completed" {
		t.Errorf("run span = %+v %v", root.Status, root.Attributes)
	}
}

func TestMetrics(t *testing.T) {
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() { _ = mp.Shutdown(context.Background()) })
	m, err := agotel.NewMetrics(mp)
	if err != nil {
		t.Fatal(err)
	}
	run(t, script(), m)

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &rm); err != nil {
		t.Fatal(err)
	}
	sums := map[string]map[string]int64{}
	for _, sm := range rm.ScopeMetrics {
		for _, met := range sm.Metrics {
			s, ok := met.Data.(metricdata.Sum[int64])
			if !ok {
				continue
			}
			sums[met.Name] = map[string]int64{}
			for _, dp := range s.DataPoints {
				key := ""
				for _, kv := range dp.Attributes.ToSlice() {
					key += string(kv.Key) + "=" + kv.Value.Emit() + ","
				}
				sums[met.Name][key] = dp.Value
			}
		}
	}
	if got := sums["agentflow_llm_calls_total"]["status=ok,"]; got != 2 {
		t.Errorf("llm calls = %v", sums["agentflow_llm_calls_total"])
	}
	if got := sums["agentflow_tokens_total"]["token_type=input,"]; got != 20 {
		t.Errorf("input tokens = %v", sums["agentflow_tokens_total"])
	}
	tools := sums["agentflow_tool_calls_total"]
	if tools["status=ok,tool_name=calc,"] != 1 || tools["error_type=error,status=error,tool_name=broken,"] != 1 {
		t.Errorf("tool calls = %v", tools)
	}
	if sums["agentflow_runs_total"]["status=completed,"] != 1 {
		t.Errorf("runs = %v", sums["agentflow_runs_total"])
	}
}

func TestTracerLinksChildRuns(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	leaf := func(name string) *agentflow.Agent {
		a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: name, ModelName: "m", Model: agentflowtest.NewScriptedModel(agentflowtest.Reply(name))})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	sql := leaf("sql")
	lead, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "lead", ModelName: "m",
		Tools: []agentflow.Tool{agentflow.Delegate(sql, agentflow.DelegateOptions{})},
		Model: agentflowtest.NewScriptedModel(agentflowtest.CallTools(agentflowtest.Use("1", "sql", map[string]string{"task": "q"})), agentflowtest.Reply("ok"))})
	if err != nil {
		t.Fatal(err)
	}
	wf := agentflow.NewWorkflow("wf", func(ctx context.Context, _ string) (string, error) {
		_, err := agentflow.Spawn(ctx, lead, "go")
		return "", err
	})
	if _, err := wf.Run(context.Background(), "x", agentflow.WithObservers(agotel.NewTracer(tp))); err != nil {
		t.Fatal(err)
	}
	byPath := map[string]tracetest.SpanStub{}
	var spawn tracetest.SpanStub
	for _, s := range exp.GetSpans() {
		switch s.Name {
		case "agentflow.run":
			byPath[attr(s.Attributes, "agentflow.path").AsString()] = s
		case "agentflow.spawn.lead":
			spawn = s
		}
	}
	root, leadRun, sqlRun := byPath["/"], byPath["/lead#0"], byPath["/lead#0/sql#0"]
	if !root.SpanContext.IsValid() || !leadRun.SpanContext.IsValid() || !sqlRun.SpanContext.IsValid() {
		t.Fatalf("runs %v", byPath)
	}
	// Spawned from workflow code: under the spawn span. Delegated: under the
	// parent run.
	if spawn.Parent.SpanID() != root.SpanContext.SpanID() || leadRun.Parent.SpanID() != spawn.SpanContext.SpanID() {
		t.Error("workflow child is not under its spawn span")
	}
	if sqlRun.Parent.SpanID() != leadRun.SpanContext.SpanID() {
		t.Error("delegated child is not under its parent run")
	}
	if root.SpanContext.TraceID() != sqlRun.SpanContext.TraceID() {
		t.Error("the tree spans more than one OTel trace")
	}
}
