package agentflow_test

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

func TestAnnotationsAndExpectations(t *testing.T) {
	if err := agentflow.Annotate(context.Background(), "agentflow.x", 1); err == nil {
		t.Fatal("a reserved key was accepted")
	}
	if err := agentflow.Annotate(context.Background(), "k", 1); err != nil {
		t.Fatal("outside a run, Annotate must do nothing")
	}
	agentflow.Expect(context.Background(), "nothing", false, "") // no run: nothing happens

	check := func(name string) agentflow.Tool {
		return agentflow.NewTool(llm.ToolSpec{Name: name}, func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
			jitter(true)
			agentflow.Expect(ctx, name+" ok", name == "a", "")
			_, _ = agentflow.Do(ctx, "inner", 1, func(ctx context.Context, _ int) (int, error) {
				_ = agentflow.Annotate(ctx, "ignored", "inside an effect function")
				return 1, nil
			})
			return agentflow.TextOutput(name), nil
		})
	}
	build := func(m llm.Model) *agentflow.Agent {
		return mustAgent(t, agentflow.AgentConfig{Name: "x", ModelName: "m", Model: m, MaxParallelTools: 2, Tools: []agentflow.Tool{check("a"), check("b")}})
	}
	store := memstore.New()
	res, err := build(agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("1", "a", struct{}{}), agentflowtest.Use("2", "b", struct{}{})), agentflowtest.Reply("done"),
	)).Run(context.Background(), "go", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, e := range events(t, res) {
		if a, ok := e.Body.(*trace.Annotation); ok {
			if a.Key == "ignored" {
				t.Fatal("an annotation inside a Do function was recorded")
			}
			x, ok := trace.ExpectationOf(a)
			if !ok {
				t.Fatalf("annotation %+v", a)
			}
			raw, _ := json.Marshal(x)
			got = append(got, string(raw))
		}
	}
	// Buffered with each call's nested events, in batch order.
	if !slices.Equal(got, []string{`{"name":"a ok","pass":true}`, `{"name":"b ok","pass":false}`}) {
		t.Fatalf("expectations %v", got)
	}

	// A replay records them again (they aren't matched).
	rep, err := replay.Run(context.Background(), build(agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
		t.Error("model called")
		return nil, nil
	})), store, res.TraceID, replay.Options{Verify: []string{"tool:a", "tool:b"}})
	if err != nil || rep.Status != replay.Identical {
		t.Fatalf("%v", err)
	}
	evs, _ := rep.Run.Events(context.Background())
	n := 0
	for _, e := range evs {
		if strings.HasPrefix(string(e.Kind()), "annotation") {
			n++
		}
	}
	if n != 2 {
		t.Fatalf("replay recorded %d expectations", n)
	}
}
