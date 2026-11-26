package inspect_test

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/inspect"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

func lastText(req *llm.Request) string {
	m := req.Messages[len(req.Messages)-1]
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		switch b := m.Blocks[i].(type) {
		case llm.Text:
			return b.Text
		case llm.ToolResult:
			return llm.TextOf(b.Content)
		}
	}
	return ""
}

func hasResult(req *llm.Request) bool {
	for _, b := range req.Messages[len(req.Messages)-1].Blocks {
		if _, ok := b.(llm.ToolResult); ok {
			return true
		}
	}
	return false
}

// toolAgent calls tool once (with call ID prefix+"1"), then answers with
// the result. delay runs before the first model answer.
func toolAgent(t *testing.T, name, prefix string, tool agentflow.Tool, delay time.Duration) *agentflow.Agent {
	t.Helper()
	a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: name, ModelName: "m", Tools: []agentflow.Tool{tool},
		Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			if !hasResult(req) {
				time.Sleep(delay)
				r := agentflowtest.CallTools(agentflowtest.Use(prefix+"1", tool.Spec().Name, map[string]string{"task": lastText(req)}))
				r.Message.Blocks = append(llm.Blocks{llm.Reasoning{Text: "thinking " + prefix, Ext: llm.Ext{"p": json.RawMessage(`"sig-` + prefix + `"`)}}}, r.Message.Blocks...)
				return r, nil
			}
			return agentflowtest.Reply(name + ": " + lastText(req)), nil
		})})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func textTool(name, out string, isErr bool) agentflow.Tool {
	return agentflow.NewTool(llm.ToolSpec{Name: name}, func(context.Context, struct{}) (agentflow.ToolOutput, error) {
		if isErr {
			return agentflow.ErrorOutput(out), nil
		}
		return agentflow.TextOutput(out), nil
	})
}

func record(t *testing.T, e agentflow.Entrypoint) (*memstore.Store, *inspect.Trace) {
	t.Helper()
	store := memstore.New()
	res, _ := e.Run(context.Background(), "go", agentflow.WithStore(store))
	if res == nil {
		t.Fatal("no run")
	}
	tr, err := inspect.Load(context.Background(), store, res.TraceID)
	if err != nil {
		t.Fatal(err)
	}
	return store, tr
}

// A child's failed tool is the root cause; the parent's error result that
// it caused is a consequence.
func TestBlameRootCauseAndConsequence(t *testing.T) {
	broken := agentflow.NewTool(llm.ToolSpec{Name: "fetch"}, func(context.Context, struct{}) (agentflow.ToolOutput, error) {
		return agentflow.ToolOutput{}, fmt.Errorf("connection refused")
	})
	worker := toolAgent(t, "worker", "w", broken, 0)
	lead := toolAgent(t, "lead", "l", agentflow.Delegate(worker, agentflow.DelegateOptions{
		ToolName: "work",
		Output: func(r *agentflow.RunResult, err error) agentflow.ToolOutput {
			if strings.Contains(r.OutputText(), "Error") {
				return agentflow.ErrorOutput("worker couldn't do it")
			}
			return agentflow.TextOutput(r.OutputText())
		},
	}), 0)
	wf := agentflow.NewWorkflow("wf", func(ctx context.Context, task string) (string, error) {
		res, err := agentflow.Spawn(ctx, lead, task)
		if err != nil {
			return "", err
		}
		agentflow.Expect(ctx, "done", false, res.OutputText())
		return res.OutputText(), nil
	})
	_, tr := record(t, wf)
	b, err := inspect.Blame(tr, "")
	if err != nil {
		t.Fatal(err)
	}
	if b.Symptom.Kind != inspect.Expectation || len(b.RootCauses) != 1 || b.Consequences != 1 {
		t.Fatalf("%s", b)
	}
	rc := b.RootCauses[0]
	if rc.Anomaly.Path != "/lead#0/worker#0" || rc.Anomaly.Kind != inspect.Failure || rc.Anomaly.Effect != "tool:fetch(w1)" {
		t.Fatalf("root cause %s", rc.Anomaly)
	}
	if len(rc.Chain) != 2 || rc.Chain[0].Kind != inspect.HopUp || rc.Chain[0].Effect != "tool:l1/spawn:/lead#0/worker#0" || rc.Chain[1].To != "/" {
		t.Fatalf("chain %+v", rc.Chain)
	}
	if !strings.Contains(b.String(), "root cause 1: /lead#0/worker#0 tool:fetch(w1) — failure: error: connection refused") {
		t.Fatalf("%s", b)
	}
}

// Siblings run concurrently: one's anomaly is not upstream of the other's
// symptom, unless declared shared state links them.
func TestBlameSiblingsAndSharedState(t *testing.T) {
	build := func(declare bool) agentflow.Entrypoint {
		w := textTool("write", "wrote a partial file", true)
		// The reader checks what it read: an expectation inside a tool.
		r := agentflow.NewTool(llm.ToolSpec{Name: "read"}, func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
			agentflow.Expect(ctx, "data complete", false, "the file ends mid-record")
			return agentflow.TextOutput("garbage"), nil
		})
		if declare {
			w = declared{w, trace.AccessWrite}
			r = declared{r, trace.AccessRead}
		}
		writer := toolAgent(t, "writer", "w", w, 0)
		reader := toolAgent(t, "reader", "r", r, 30*time.Millisecond)
		return agentflow.NewWorkflow("wf", func(ctx context.Context, _ string) (string, error) {
			_, err := agentflow.SpawnAll(ctx, []agentflow.SpawnSpec{{Agent: writer, Task: "w"}, {Agent: reader, Task: "r"}})
			return "", err
		})
	}
	_, tr := record(t, build(false))
	b, err := inspect.Blame(tr, "/reader#0")
	if err != nil || b.Symptom == nil || b.Symptom.Kind != inspect.Expectation || len(b.RootCauses) != 0 {
		t.Fatalf("undeclared: %v %s", err, b)
	}

	_, tr = record(t, build(true))
	b, err = inspect.Blame(tr, "/reader#0")
	if err != nil || len(b.RootCauses) != 1 || b.RootCauses[0].SharedState != "file:x" || b.RootCauses[0].Anomaly.Path != "/writer#0" {
		t.Fatalf("declared: %v %s", err, b)
	}
	if !strings.Contains(b.String(), "shared state; ordered by recorded time") {
		t.Fatalf("%s", b)
	}
}

type declared struct {
	agentflow.Tool
	mode trace.AccessMode
}

func (d declared) Accesses(json.RawMessage) []trace.Access {
	return []trace.Access{{Resource: "file:x", Mode: d.mode}}
}

func TestBlameNothingFailed(t *testing.T) {
	_, tr := record(t, toolAgent(t, "a", "a", textTool("t", "ok", false), 0))
	b, err := inspect.Blame(tr, "")
	if err != nil || b.Symptom != nil || !strings.HasPrefix(b.String(), "BLAME  nothing failed") {
		t.Fatalf("%v %s", err, b)
	}
	if _, err := inspect.Blame(tr, "/nobody#0"); err == nil {
		t.Fatal("blame at an unknown run")
	}
}

// Two independent live runs that behaved the same, but whose providers
// assigned different call IDs and signatures, compare identical; -exact
// sees the differences.
func TestDiffNormalizesCallIDs(t *testing.T) {
	store := memstore.New()
	run := func(prefix string) string {
		res, err := toolAgent(t, "a", prefix, textTool("t", "ok", false), 0).Run(context.Background(), "go", agentflow.WithStore(store))
		if err != nil {
			t.Fatal(err)
		}
		return res.TraceID
	}
	a, b := run("x"), run("y")
	d, err := inspect.Diff(context.Background(), store, a, b, inspect.DiffOptions{})
	if err != nil || !d.Identical() {
		t.Fatalf("%v\n%s", err, d)
	}
	d, err = inspect.Diff(context.Background(), store, a, b, inspect.DiffOptions{Exact: true})
	if err != nil || d.Identical() {
		t.Fatalf("exact: %v\n%s", err, d)
	}
	// A real change is reported under the run's path.
	c, err := toolAgent(t, "a", "z", textTool("t", "not ok", false), 0).Run(context.Background(), "go", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	d, err = inspect.Diff(context.Background(), store, a, c.TraceID, inspect.DiffOptions{})
	if err != nil || len(d.Runs) != 1 || d.Runs[0].Path != "/" || d.Runs[0].Entries[0].Effect != "tool@0.0" || d.Runs[0].Entries[0].What != "outcome" {
		t.Fatalf("%v\n%s", err, d)
	}
}

func TestDiffRunsOnOneSide(t *testing.T) {
	store := memstore.New()
	leaf := toolAgent(t, "leaf", "l", textTool("t", "ok", false), 0)
	mk := func(n int) string {
		wf := agentflow.NewWorkflow("wf", func(ctx context.Context, _ string) (string, error) {
			for range n {
				if _, err := agentflow.Spawn(ctx, leaf, "x"); err != nil {
					return "", err
				}
			}
			return "", nil
		})
		res, err := wf.Run(context.Background(), "go", agentflow.WithStore(store))
		if err != nil {
			t.Fatal(err)
		}
		return res.TraceID
	}
	d, err := inspect.Diff(context.Background(), store, mk(1), mk(2), inspect.DiffOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(d.String(), "run /leaf#1 only in B") || !strings.Contains(d.String(), "spawn:/leaf#1 — only in b") {
		t.Fatalf("%s", d)
	}
}

func TestRenderTreeShowsForkModes(t *testing.T) {
	var calls atomic.Int32
	mk := func() *agentflow.Agent {
		a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "solo", ModelName: "m", Tools: []agentflow.Tool{textTool("t", "bad", true)},
			Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
				calls.Add(1)
				if !hasResult(req) {
					return agentflowtest.CallTools(agentflowtest.Use("c1", "t", struct{}{})), nil
				}
				return agentflowtest.Reply("answer: " + lastText(req)), nil
			})})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	store := memstore.New()
	rec, _ := mk().Run(context.Background(), "go", agentflow.WithStore(store))
	res, err := replay.Fork(context.Background(), mk(), store, rec.TraceID, replay.ForkOptions{At: &replay.Point{Effect: "llm#1"},
		Overrides: []replay.Override{replay.ReplaceOutcome{Message: &llm.Message{Blocks: llm.Blocks{llm.Text{Text: "fixed"}}}}}})
	if err != nil {
		t.Fatal(err)
	}
	tr, err := inspect.Load(context.Background(), store, res.Treatment.Samples[0].TraceID)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	if err := inspect.RenderTree(&sb, tr); err != nil {
		t.Fatal(err)
	}
	out := sb.String()
	for _, want := range []string{"fork of ", "at / llm#1", "(treatment sample 0)", "llm#0 tool:t! ↯fork_point llm#1⇢"} {
		if !strings.Contains(out, want) {
			t.Fatalf("tree lacks %q:\n%s", want, out)
		}
	}
	if e := tr.Root.Effects[0]; !e.Replayed {
		t.Fatalf("llm#0 should be answered from the recording: %+v", e)
	}
}

func TestHTMLIsSelfContainedAndEscaped(t *testing.T) {
	evil := "</script><script>alert(1)</script>"
	_, tr := record(t, toolAgent(t, "a", "a", textTool("t", evil, true), 0))
	b, _ := inspect.Blame(tr, "")
	var sb strings.Builder
	if err := inspect.WriteHTML(&sb, tr, b); err != nil {
		t.Fatal(err)
	}
	h := sb.String()
	if strings.Count(h, "</script>") != 2 || strings.Contains(h, evil) {
		t.Fatal("content broke out of its script element")
	}
	if regexp.MustCompile(`(src|href)\s*=\s*["']?(https?:)?//`).MatchString(h) {
		t.Fatal("the page references the network")
	}
	start := strings.Index(h, `<script type="application/json" id="data">`) + len(`<script type="application/json" id="data">`)
	end := strings.Index(h[start:], "</script>")
	var payload struct {
		Trace struct {
			ID   string `json:"id"`
			Runs []struct {
				Effects []struct {
					Output json.RawMessage `json:"output"`
				} `json:"effects"`
			} `json:"runs"`
		} `json:"trace"`
	}
	if err := json.Unmarshal([]byte(h[start:start+end]), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Trace.ID != tr.ID || !strings.Contains(string(payload.Trace.Runs[0].Effects[1].Output), "alert(1)") {
		t.Fatalf("payload %+v", payload.Trace)
	}
}
