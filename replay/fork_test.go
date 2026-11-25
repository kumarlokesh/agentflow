package replay_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

func text(s string) llm.Blocks { return llm.Blocks{llm.Text{Text: s}} }

// modes returns, per run path, the reasons of its run.mode_changed events.
func modes(t *testing.T, s trace.Store, traceID string) map[string][]string {
	t.Helper()
	out := map[string][]string{}
	for p, evs := range runsByPath(t, s, traceID) {
		out[p] = []string{}
		for _, e := range evs {
			if m, ok := e.Body.(*trace.RunModeChanged); ok {
				out[p] = append(out[p], m.Reason)
			}
		}
	}
	return out
}

func forkOnce(t *testing.T, e agentflow.Entrypoint, store trace.Store, src string, opts replay.ForkOptions) replay.Sample {
	t.Helper()
	res, err := replay.Fork(context.Background(), e, store, src, opts)
	if err != nil {
		t.Fatal(err)
	}
	s := res.Treatment.Samples[0]
	if s.Err != nil {
		t.Fatalf("sample errored: %v", s.Err)
	}
	for p, evs := range runsByPath(t, store, s.TraceID) {
		checkAnswered(t, p, evs)
	}
	return s
}

var sqlPoint = &replay.Point{Path: "/metrics#0/sql#0", Effect: "llm#0"}

// A fork in a sub-agent of the triage team spreads exactly as far
// as its change reaches.
func TestForkReuseIsExact(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	live := newTeam()
	s := forkOnce(t, live.workflow(t, true), store, src, replay.ForkOptions{
		At: sqlPoint, Overrides: []replay.Override{replay.SetConfig{System: text("Query carefully.")}},
	})
	if !reflect.DeepEqual(live.calls, map[string]int{"sql": 1, "metrics": 1, "summarizer": 1}) {
		t.Fatalf("live calls %v", live.calls)
	}
	got := modes(t, store, s.TraceID)
	want := map[string][]string{"/": {"divergence"}, "/logs#0": {}, "/metrics#0": {"divergence"}, "/metrics#0/sql#0": {"fork_point"}, "/summarizer#0": {"new_run"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("modes %v", got)
	}
	if r := s.Reuse; r.RunsReplayed != 1 || r.RunsLive != 3 || r.RunsNew != 1 || r.EffectsLive != 3 || r.TokensSaved == 0 {
		t.Fatalf("reuse %+v", r)
	}
	if !strings.Contains(s.Run.OutputText(), "Query carefully.") || s.Verdict == nil || !s.Verdict.Pass {
		t.Fatalf("output %q verdict %+v", s.Run.OutputText(), s.Verdict)
	}
	// logs is unaffected: its events equal the recording's.
	a, b := comparable(t, runsByPath(t, store, src)["/logs#0"]), comparable(t, runsByPath(t, store, s.TraceID)["/logs#0"])
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("logs differs:\n%s\nvs\n%s", strings.Join(a, "\n"), strings.Join(b, "\n"))
	}
	// sql: the config change is recorded before the call, which refers to it.
	sql := runsByPath(t, store, s.TraceID)["/metrics#0/sql#0"]
	var cfgSeq uint64
	for _, e := range sql {
		switch b := e.Body.(type) {
		case *trace.ConfigChanged:
			cfgSeq = e.Seq
		case *trace.LLMStarted:
			if cfgSeq == 0 || b.Context.ConfigSeq != cfgSeq {
				t.Fatalf("llm.started config_seq %d, config.changed at %d", b.Context.ConfigSeq, cfgSeq)
			}
		}
	}
	// Lineage.
	o := runsByPath(t, store, s.TraceID)["/"][0].Body.(*trace.RunStarted).Origin
	if o.Kind != trace.OriginFork || o.SourceTraceID != src || o.At.Path != "/metrics#0/sql#0" || o.At.Effect != "llm#0" || o.At.Seq == nil ||
		o.Arm != "treatment" || *o.Sample != 0 || o.Group == "" || !strings.Contains(string(o.Overrides), `"type":"set_config"`) {
		t.Fatalf("origin %+v", o)
	}
}

// A sample replays with its overrides re-applied, and makes no calls; so
// does a fork of a fork.
func TestForkSamplesReplayAndChain(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	s := forkOnce(t, newTeam().workflow(t, true), store, src, replay.ForkOptions{
		At: sqlPoint, Overrides: []replay.Override{replay.SetConfig{System: text("Query carefully.")}},
	})
	res, err := replay.Run(context.Background(), newTeam().workflow(t, false), store, s.TraceID, replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("replaying the sample: %v %v", err, res.Divergences)
	}

	live := newTeam()
	s2 := forkOnce(t, live.workflow(t, true), store, s.TraceID, replay.ForkOptions{
		At:        &replay.Point{Path: "/summarizer#0", Effect: "llm#0"},
		Overrides: []replay.Override{replay.InjectMessage{Message: llm.Message{Blocks: text("Be brief.")}}},
	})
	if !reflect.DeepEqual(live.calls, map[string]int{"summarizer": 1}) {
		t.Fatalf("fork of a fork: live calls %v", live.calls)
	}
	// Only the summarizer ran live: sql's recorded answer in the source
	// fork matched because its override was re-applied in the prefix.
	if s2.Run.OutputText() != "summarizer(Summarize.) saw Be brief." {
		t.Fatalf("output %q", s2.Run.OutputText())
	}
	res, err = replay.Run(context.Background(), newTeam().workflow(t, false), store, s2.TraceID, replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("replaying the fork of a fork: %v %v", err, res.Divergences)
	}
}

func TestForkOverrides(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	fork := func(at replay.Point, ovs ...replay.Override) (replay.Sample, *team) {
		live := newTeam()
		return forkOnce(t, live.workflow(t, true), store, src, replay.ForkOptions{At: &at, Overrides: ovs}), live
	}

	t.Run("inject message", func(t *testing.T) {
		s, _ := fork(replay.Point{Path: "/logs#0", Effect: "llm#0"}, replay.InjectMessage{Message: llm.Message{Blocks: text("Look harder.")}})
		evs := runsByPath(t, store, s.TraceID)["/logs#0"]
		got := kinds(evs)
		if !slices.Equal(got[:6], []trace.Kind{trace.KindRunStarted, trace.KindMessageAppended, trace.KindMessageAppended, trace.KindLLMStarted, trace.KindRunModeChanged, trace.KindLLMCompleted}) {
			t.Fatalf("events %v", got)
		}
		if m := evs[2].Body.(*trace.MessageAppended).Message; m.Role != llm.RoleUser || llm.TextOf(m.Blocks) != "Look harder." {
			t.Fatalf("injected %+v", m)
		}
	})
	t.Run("replace a model's answer", func(t *testing.T) {
		s, live := fork(replay.Point{Path: "/logs#0", Effect: "llm#0"}, replay.ReplaceOutcome{Message: &llm.Message{Blocks: text("nothing found")}})
		if live.calls["logs"] != 0 {
			t.Fatal("a replaced model call was made")
		}
		for _, e := range runsByPath(t, store, s.TraceID)["/logs#0"] {
			if c, ok := e.Body.(*trace.LLMCompleted); ok && (!c.Override || llm.TextOf(c.Message.Blocks) != "nothing found") {
				t.Fatalf("llm.completed %+v", c)
			}
		}
	})
	t.Run("replace a delegate tool's result", func(t *testing.T) {
		s, live := fork(replay.Point{Path: "/metrics#0", Effect: "tool:m1"}, replay.ReplaceOutcome{ToolOutput: &agentflow.ToolOutput{Content: text("no rows")}})
		if live.calls["sql"] != 0 || s.Run.Find("/metrics#0/sql#0") != nil {
			t.Fatal("the replaced delegate's child ran")
		}
		if !strings.Contains(s.Run.Find("/metrics#0").OutputText(), "no rows") {
			t.Fatalf("metrics: %q", s.Run.Find("/metrics#0").OutputText())
		}
	})
	t.Run("replace a child's result", func(t *testing.T) {
		s, live := fork(replay.Point{Path: "/", Effect: "spawn:/summarizer#0"}, replay.ReplaceOutcome{ChildOutput: text("all good")})
		if live.calls["summarizer"] != 0 || s.Run.OutputText() != "all good" {
			t.Fatalf("output %q calls %v", s.Run.OutputText(), live.calls)
		}
		if _, ran := runsByPath(t, store, s.TraceID)["/summarizer#0"]; ran {
			t.Fatal("the replaced child ran")
		}
	})
	t.Run("set a child's task", func(t *testing.T) {
		s, live := fork(replay.Point{Path: "/", Effect: "spawn:/summarizer#0"}, replay.SetTask{Task: text("summarize briefly")})
		if live.calls["summarizer"] != 1 || !strings.HasSuffix(s.Run.OutputText(), "saw summarize briefly") {
			t.Fatalf("output %q calls %v", s.Run.OutputText(), live.calls)
		}
		if m := modes(t, store, s.TraceID); !reflect.DeepEqual(m["/summarizer#0"], []string{"new_run"}) || !reflect.DeepEqual(m["/"], []string{"fork_point"}) {
			t.Fatalf("modes %v", m)
		}
	})
	t.Run("disable a tool", func(t *testing.T) {
		s, live := fork(replay.Point{Path: "/metrics#0", Effect: "llm#0"}, replay.SetConfig{DisableTools: []string{"query_db"}})
		if live.calls["sql"] != 0 {
			t.Fatal("a disabled tool ran")
		}
		for _, e := range runsByPath(t, store, s.TraceID)["/metrics#0"] {
			switch b := e.Body.(type) {
			case *trace.ConfigChanged:
				if b.Tools == nil || len(*b.Tools) != 0 {
					t.Fatalf("config.changed %+v", b)
				}
			case *trace.ToolFailed:
				if b.Error.Code != trace.ErrCodeNotFound {
					t.Fatalf("tool.failed %+v", b)
				}
			}
		}
	})
}

func TestForkValidation(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	cases := []struct {
		name string
		opts replay.ForkOptions
		want error
	}{
		{"task at a model call", replay.ForkOptions{At: sqlPoint, Overrides: []replay.Override{replay.SetTask{Task: text("x")}}}, replay.ErrInvalidOverride},
		{"reasoning", replay.ForkOptions{At: sqlPoint, Overrides: []replay.Override{replay.InjectMessage{Message: llm.Message{Blocks: llm.Blocks{llm.Reasoning{Text: "hmm"}}}}}}, replay.ErrInvalidOverride},
		{"outcome of the wrong kind", replay.ForkOptions{At: sqlPoint, Overrides: []replay.Override{replay.ReplaceOutcome{ChildOutput: text("x")}}}, replay.ErrInvalidOverride},
		{"overrides without a point", replay.ForkOptions{Overrides: []replay.Override{replay.SetTask{Task: text("x")}}}, replay.ErrInvalidOverride},
		{"unknown run", replay.ForkOptions{At: &replay.Point{Path: "/nobody#0", Effect: "llm#0"}}, replay.ErrForkPointUnknown},
		{"unknown effect", replay.ForkOptions{At: &replay.Point{Path: "/logs#0", Effect: "llm#7"}}, replay.ErrForkPointUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// The model fails the test if anything runs.
			_, err := replay.Fork(context.Background(), newTeam().workflow(t, false), store, src, c.opts)
			if !errors.Is(err, c.want) {
				t.Fatalf("err %v", err)
			}
		})
	}
	if _, err := replay.Fork(context.Background(), newTeam().workflow(t, false), store, src, replay.ForkOptions{Baseline: true}); err == nil {
		t.Fatal("a baseline without a point was accepted")
	}
}

func TestForkProbe(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	// metrics changed: it diverges before it delegates to sql.
	changed := newTeam()
	changed.prompts["metrics"] = "Find all anomalies."
	_, err := replay.Fork(context.Background(), changed.workflow(t, false), store, src, replay.ForkOptions{
		At: sqlPoint, Overrides: []replay.Override{replay.SetConfig{System: text("Query carefully.")}},
	})
	var early *replay.EarlyDivergenceError
	if !errors.As(err, &early) || !errors.Is(err, replay.ErrDivergedBeforeForkPoint) || early.Divergence.Path != "/metrics#0" || early.Divergence.Effect != "llm#0" {
		t.Fatalf("err %v", err)
	}

	live := newTeam()
	live.prompts["metrics"] = "Find all anomalies."
	res, err := replay.Fork(context.Background(), live.workflow(t, true), store, src, replay.ForkOptions{
		At: sqlPoint, Overrides: []replay.Override{replay.SetConfig{System: text("Query carefully.")}}, AcceptEarlierDivergence: true,
	})
	if err != nil || res.ActualAt == nil || res.ActualAt.Path != "/metrics#0" || res.Treatment.Samples[0].Err != nil {
		t.Fatalf("err %v actual %v sample %+v", err, res.ActualAt, res.Treatment.Samples[0].Err)
	}
	s := res.Treatment.Samples[0]
	o := runsByPath(t, store, s.TraceID)["/"][0].Body.(*trace.RunStarted).Origin
	if o.ActualAt == nil || o.ActualAt.Path != "/metrics#0" || o.At.Path != "/metrics#0/sql#0" {
		t.Fatalf("origin %+v", o)
	}
	// The overrides still apply at the point, reached live.
	if !strings.Contains(s.Run.OutputText(), "Query carefully.") {
		t.Fatalf("output %q", s.Run.OutputText())
	}
}

// A fork at an effect nested in a tool runs that tool, answers its earlier
// nested effects from the recording, and goes live at the point.
func TestForkAtNestedEffect(t *testing.T) {
	s := newShop()
	store, src := recordShop(t, s)
	srcID := ""
	for _, e := range runsByPath(t, store, src)["/"] {
		if c, ok := e.Body.(*trace.EffectCompleted); ok && c.Name == "id.new" {
			srcID = string(c.Output)
		}
	}
	live := newShop()
	model := agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
		return agentflowtest.Reply("done: " + lastText(req)), nil
	})
	res, err := replay.Fork(context.Background(), live.agent(t, model), store, src, replay.ForkOptions{
		At:        &replay.Point{Effect: "tool:2/effect:clock.now#0"},
		Overrides: []replay.Override{replay.ReplaceOutcome{Output: json.RawMessage(`"2030-01-01T09:30:00Z"`)}},
	})
	if err != nil || res.Treatment.Samples[0].Err != nil {
		t.Fatalf("err %v %v", err, res.Treatment.Samples[0].Err)
	}
	out := res.Treatment.Samples[0].Run.OutputText()
	var id string
	_ = json.Unmarshal([]byte(srcID), &id)
	if live.tickets.Load() != 1 || live.lookups.Load() != 0 || !strings.Contains(out, "at 09:30:00") || !strings.Contains(out, id) {
		t.Fatalf("tickets %d lookups %d output %q (recorded id %s)", live.tickets.Load(), live.lookups.Load(), out, id)
	}
	if m := modes(t, store, res.Treatment.Samples[0].TraceID)["/"]; !slices.Equal(m, []string{"fork_point"}) {
		t.Fatalf("modes %v", m)
	}
}

// writerReader is a workflow: a writer agent writes file:x, then a reader
// agent reads it. The writer always declares its write; the reader declares
// its read unless undeclared is set.
func writerReader(t *testing.T, live bool, undeclared bool, reads *atomic.Int32) *agentflow.Workflow {
	t.Helper()
	files := map[string]string{"x": "old"}
	var read agentflow.Tool = declaredRead{plainRead{files, reads}}
	if undeclared {
		read = plainRead{files, reads}
	}
	agent := func(name string, tool agentflow.Tool) *agentflow.Agent {
		a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: name, ModelName: "m", Tools: []agentflow.Tool{tool},
			Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
				if !live {
					t.Errorf("%s's model called", name)
				}
				if !hasToolResult(req.Messages[len(req.Messages)-1]) {
					return agentflowtest.CallTools(agentflowtest.Use(name+"-1", tool.Spec().Name, map[string]string{"text": lastText(req)})), nil
				}
				return agentflowtest.Reply(name + ": " + lastText(req)), nil
			})})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	writer := agent("writer", writeFile{files})
	reader := agent("reader", read)
	return agentflow.NewWorkflow("wr", func(ctx context.Context, _ string) (string, error) {
		if _, err := agentflow.Spawn(ctx, writer, "write"); err != nil {
			return "", err
		}
		res, err := agentflow.Spawn(ctx, reader, "read")
		if err != nil {
			return "", err
		}
		return res.OutputText(), nil
	})
}

type writeFile struct{ files map[string]string }

func (writeFile) Spec() llm.ToolSpec { return llm.ToolSpec{Name: "write_file"} }
func (w writeFile) Call(_ context.Context, in json.RawMessage) (agentflow.ToolOutput, error) {
	var v struct{ Text string }
	_ = json.Unmarshal(in, &v)
	w.files["x"] = v.Text
	return agentflow.TextOutput("ok"), nil
}
func (writeFile) Accesses(json.RawMessage) []trace.Access {
	return []trace.Access{{Resource: "file:x", Mode: trace.AccessWrite}}
}

type plainRead struct {
	files map[string]string
	reads *atomic.Int32
}

func (plainRead) Spec() llm.ToolSpec {
	return llm.ToolSpec{Name: "read_file", Annotations: &llm.ToolAnnotations{ReadOnly: true}}
}
func (r plainRead) Call(context.Context, json.RawMessage) (agentflow.ToolOutput, error) {
	r.reads.Add(1)
	return agentflow.TextOutput(r.files["x"]), nil
}

type declaredRead struct{ plainRead }

func (declaredRead) Accesses(json.RawMessage) []trace.Access {
	return []trace.Access{{Resource: "file:x", Mode: trace.AccessRead}}
}

func TestForkSharedState(t *testing.T) {
	writerFork := replay.ForkOptions{
		At:        &replay.Point{Path: "/writer#0", Effect: "llm#0"},
		Overrides: []replay.Override{replay.InjectMessage{Message: llm.Message{Blocks: text("write something else")}}},
	}
	record := func(undeclared bool) (trace.Store, string) {
		store := memstore.New()
		var n atomic.Int32
		rec, err := writerReader(t, true, undeclared, &n).Run(context.Background(), "go", agentflow.WithStore(store))
		if err != nil {
			t.Fatal(err)
		}
		return store, rec.TraceID
	}

	// The writer goes live and writes file:x; the reader's matching call,
	// which declares it reads file:x, can't be answered from the recording.
	store, src := record(false)
	var reads atomic.Int32
	s := forkOnce(t, writerReader(t, true, false, &reads), store, src, writerFork)
	if reads.Load() != 1 || !strings.Contains(s.Run.OutputText(), "write something else") {
		t.Fatalf("reads %d output %q", reads.Load(), s.Run.OutputText())
	}
	if m := modes(t, store, s.TraceID)["/reader#0"]; !slices.Equal(m, []string{"shared_state"}) {
		t.Fatalf("reader modes %v", m)
	}
	if len(s.SideEffects) != 1 || s.SideEffects[0].Tool != "write_file" || s.SideEffects[0].Path != "/writer#0" {
		t.Fatalf("side effects %+v", s.SideEffects)
	}

	// Undeclared reads: served with an assumption (default), or live.
	store, src = record(true)
	reads.Store(0)
	s = forkOnce(t, writerReader(t, true, true, &reads), store, src, writerFork)
	if reads.Load() != 0 || len(s.Assumptions) != 1 || s.Assumptions[0].Path != "/reader#0" || s.Assumptions[0].Tool != "read_file" {
		t.Fatalf("optimistic: reads %d assumptions %+v", reads.Load(), s.Assumptions)
	}
	conservative := writerFork
	conservative.UnknownAccess = replay.Conservative
	s = forkOnce(t, writerReader(t, true, true, &reads), store, src, conservative)
	if reads.Load() != 1 || len(s.Assumptions) != 0 {
		t.Fatalf("conservative: reads %d assumptions %+v", reads.Load(), s.Assumptions)
	}
}

func hasToolResult(m llm.Message) bool {
	for _, b := range m.Blocks {
		if _, ok := b.(llm.ToolResult); ok {
			return true
		}
	}
	return false
}

func kinds(evs []trace.Event) []trace.Kind {
	out := make([]trace.Kind, len(evs))
	for i, e := range evs {
		out[i] = e.Kind()
	}
	return out
}

// opsAgent calls a destructive "delete" tool, then answers.
func opsAgent(t *testing.T, deletes *atomic.Int32, live bool) *agentflow.Agent {
	t.Helper()
	del := agentflow.NewTool(llm.ToolSpec{Name: "delete", Annotations: &llm.ToolAnnotations{Destructive: true}},
		func(context.Context, struct{}) (agentflow.ToolOutput, error) {
			deletes.Add(1)
			return agentflow.TextOutput("deleted"), nil
		})
	a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "ops", ModelName: "m", Tools: []agentflow.Tool{del},
		Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			if !live {
				t.Error("model called")
			}
			if !hasToolResult(req.Messages[len(req.Messages)-1]) {
				return agentflowtest.CallTools(agentflowtest.Use("d1", "delete", struct{}{})), nil
			}
			return agentflowtest.Reply("done: " + lastText(req)), nil
		})})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestForkToolSafety(t *testing.T) {
	var deletes atomic.Int32
	store := memstore.New()
	rec, err := opsAgent(t, &deletes, true).Run(context.Background(), "clean up", agentflow.WithStore(store))
	if err != nil || deletes.Load() != 1 {
		t.Fatal(err)
	}
	inject := []replay.Override{replay.InjectMessage{Message: llm.Message{Blocks: text("carefully")}}}

	deletes.Store(0)
	s := forkOnce(t, opsAgent(t, &deletes, true), store, rec.TraceID, replay.ForkOptions{At: &replay.Point{Effect: "llm#0"}, Overrides: inject})
	if deletes.Load() != 0 || !strings.Contains(s.Run.OutputText(), "destructive tool") {
		t.Fatalf("denied: deletes %d output %q", deletes.Load(), s.Run.OutputText())
	}
	// The denial is an outcome, not an interceptor's: the sample replays.
	if res, err := replay.Run(context.Background(), opsAgent(t, &deletes, false), store, s.TraceID, replay.Options{}); err != nil || res.Status != replay.Identical {
		t.Fatalf("replaying the sample: %v", err)
	}

	s = forkOnce(t, opsAgent(t, &deletes, true), store, rec.TraceID, replay.ForkOptions{At: &replay.Point{Effect: "llm#0"}, Overrides: inject, AllowTools: []string{"delete"}})
	if deletes.Load() != 1 || len(s.SideEffects) != 1 {
		t.Fatalf("allowed: deletes %d side effects %v", deletes.Load(), s.SideEffects)
	}

	// Before the fork point it is answered from the recording.
	deletes.Store(0)
	forkOnce(t, opsAgent(t, &deletes, true), store, rec.TraceID, replay.ForkOptions{At: &replay.Point{Effect: "llm#1"}, Overrides: inject})
	if deletes.Load() != 0 {
		t.Fatal("a recorded destructive call ran")
	}
}

// solver answers "right" or "wrong"; with a "fix" in its prompt it is
// mostly right, without it mostly wrong (deterministically, by call count).
func solver(t *testing.T, calls *atomic.Int32) *agentflow.Agent {
	a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "solver", ModelName: "m", Instructions: "Solve.",
		Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			n := calls.Add(1)
			right := n%5 == 0
			if strings.Contains(llm.TextOf(req.System), "fix") {
				right = !right
			}
			if right {
				return agentflowtest.Reply("right"), nil
			}
			return agentflowtest.Reply("wrong"), nil
		})})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestForkSamplesAndBaseline(t *testing.T) {
	var calls atomic.Int32
	store := memstore.New()
	rec, err := solver(t, &calls).Run(context.Background(), "2+2", agentflow.WithStore(store))
	if err != nil || rec.OutputText() != "wrong" {
		t.Fatal(err)
	}
	calls.Store(0)
	res, err := replay.Fork(context.Background(), solver(t, &calls), store, rec.TraceID, replay.ForkOptions{
		At: &replay.Point{Effect: "llm#0"}, Overrides: []replay.Override{replay.SetConfig{System: text("Solve. fix")}},
		Samples: 5, Baseline: true,
		Check: func(r *agentflow.RunResult) replay.Verdict { return replay.Verdict{Pass: r.OutputText() == "right"} },
	})
	if err != nil {
		t.Fatal(err)
	}
	tr, bl := res.Treatment, res.Baseline
	if tr.Passed != 4 || tr.Failed != 1 || bl == nil || bl.Passed != 1 || bl.Failed != 4 {
		t.Fatalf("treatment %d/%d, baseline %+v", tr.Passed, tr.Failed, bl)
	}
	if math.Abs(tr.Low95-0.3755) > 1e-3 || math.Abs(tr.High95-0.9638) > 1e-3 {
		t.Fatalf("interval [%f, %f]", tr.Low95, tr.High95)
	}
	report := res.String()
	for _, want := range []string{"treatment  4/5 pass   80%  [38%,  96%]", "baseline   1/5 pass   20%  [ 4%,  62%]", "small samples"} {
		if !strings.Contains(report, want) {
			t.Fatalf("report lacks %q:\n%s", want, report)
		}
	}
	groups := map[string]bool{}
	for i, s := range append(append([]replay.Sample{}, tr.Samples...), bl.Samples...) {
		o := runsByPath(t, store, s.TraceID)["/"][0].Body.(*trace.RunStarted).Origin
		groups[o.Group] = true
		if wantArm := []string{"treatment", "baseline"}[i/5]; o.Arm != wantArm || *o.Sample != i%5 || (o.Arm == "baseline") != (o.Overrides == nil) {
			t.Fatalf("sample %d origin %+v", i, o)
		}
	}
	if len(groups) != 1 {
		t.Fatalf("groups %v", groups)
	}
}

func TestForkBudget(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	res, err := replay.Fork(context.Background(), newTeam().workflow(t, true), store, src, replay.ForkOptions{
		At: sqlPoint, Overrides: []replay.Override{replay.SetConfig{System: text("Query carefully.")}}, MaxTokens: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if s := res.Treatment.Samples[0]; !errors.Is(s.Err, replay.ErrSampleBudget) || res.Treatment.Errored != 1 || res.Treatment.Passed+res.Treatment.Failed != 0 {
		t.Fatalf("sample %+v", s.Err)
	}
}

func TestForkStripReasoning(t *testing.T) {
	var seen []*llm.Request
	think := llm.Reasoning{Text: "let me think"}
	model := func(live bool) llm.Model {
		return agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			seen = append(seen, req)
			if len(req.Messages) == 1 {
				r := agentflowtest.CallTools(agentflowtest.Use("c1", "calc", struct{}{}))
				r.Message.Blocks = append(llm.Blocks{think}, r.Message.Blocks...)
				return r, nil
			}
			return agentflowtest.Reply("42"), nil
		})
	}
	agent := func(m llm.Model) *agentflow.Agent {
		calc := agentflow.NewTool(llm.ToolSpec{Name: "calc"}, func(context.Context, struct{}) (agentflow.ToolOutput, error) {
			return agentflow.TextOutput("42"), nil
		})
		a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "calc", ModelName: "m", Model: m, Tools: []agentflow.Tool{calc}})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	store := memstore.New()
	rec, err := agent(model(true)).Run(context.Background(), "6*7", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	seen = nil
	s := forkOnce(t, agent(model(true)), store, rec.TraceID, replay.ForkOptions{
		At: &replay.Point{Effect: "llm#1"}, Overrides: []replay.Override{replay.SetConfig{Model: "other-model"}}, StripReasoning: true,
	})
	if len(seen) != 1 || seen[0].Model != "other-model" {
		t.Fatalf("live calls %d", len(seen))
	}
	for _, m := range seen[0].Messages {
		for _, b := range m.Blocks {
			if _, ok := b.(llm.Reasoning); ok {
				t.Fatal("reasoning sent after StripReasoning")
			}
		}
	}
	var replaced bool
	for _, e := range runsByPath(t, store, s.TraceID)["/"] {
		_, ok := e.Body.(*trace.ConversationReplaced)
		replaced = replaced || ok
	}
	if !replaced {
		t.Fatal("no conversation.replaced")
	}
	// Promoted to a cassette: the exported sample replays from the bundle alone.
	var buf strings.Builder
	if err := trace.Export(context.Background(), &buf, store, s.TraceID, trace.BundleOptions{}); err != nil {
		t.Fatal(err)
	}
	res, err := replay.RunBundle(context.Background(), agent(noModel(t)), strings.NewReader(buf.String()), replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("cassette: %v", err)
	}
}
