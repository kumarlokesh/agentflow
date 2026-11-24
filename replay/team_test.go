package replay_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/policy"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// team is the RFC's triage team as "code": a workflow fans out to logs and
// metrics (metrics delegates to sql), then a summarizer combines them.
// Tests change fields between recording and replay.
type team struct {
	prompts    map[string]string // agent → system prompt
	sumTask    string            // prefix of the summarizer's task
	skipLogs   bool
	extraAgent bool
	denySQL    bool
	jitter     bool

	mu    sync.Mutex
	calls map[string]int // live model calls per agent
}

func newTeam() *team {
	return &team{prompts: map[string]string{"logs": "Find errors.", "metrics": "Find anomalies.", "sql": "Query.", "summarizer": "Summarize."},
		sumTask: "summarize", calls: map[string]int{}}
}

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

// model answers deterministically from the request. With live false it
// fails the test if called: a strict replay must not reach models.
func (tm *team) model(t *testing.T, name string, live bool) llm.Model {
	return agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
		if !live {
			t.Errorf("model of %s called during a strict replay", name)
			return nil, errors.New("no model")
		}
		tm.mu.Lock()
		tm.calls[name]++
		tm.mu.Unlock()
		if tm.jitter {
			time.Sleep(time.Duration(rand.IntN(300)) * time.Microsecond)
		}
		if name == "metrics" && len(req.Messages) == 1 {
			return agentflowtest.CallTools(agentflowtest.Use("m1", "query_db", map[string]string{"task": "rows for " + lastText(req)})), nil
		}
		// The answer depends on the prompt, so a prompt change changes it.
		return agentflowtest.Reply(fmt.Sprintf("%s(%s) saw %s", name, llm.TextOf(req.System), lastText(req))), nil
	})
}

func (tm *team) agent(t *testing.T, name string, live bool, tools ...agentflow.Tool) *agentflow.Agent {
	t.Helper()
	cfg := agentflow.AgentConfig{Name: name, ModelName: "m", Instructions: tm.prompts[name], Model: tm.model(t, name, live), Tools: tools}
	if name == "metrics" && tm.denySQL {
		cfg.Interceptors = []agentflow.Interceptor{agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
			if c.Kind == agentflow.EffectSpawn && c.Name == "sql" {
				return agentflow.Fail(trace.ErrCodeDenied, "no sql"), nil
			}
			return next(ctx, c)
		})}
	}
	a, err := agentflow.NewAgent(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func (tm *team) workflow(t *testing.T, live bool) *agentflow.Workflow {
	sql := tm.agent(t, "sql", live)
	logs := tm.agent(t, "logs", live)
	metrics := tm.agent(t, "metrics", live, agentflow.Delegate(sql, agentflow.DelegateOptions{ToolName: "query_db"}))
	summarizer := tm.agent(t, "summarizer", live)
	audit := tm.agent(t, "audit", live)
	return agentflow.NewWorkflow("triage", func(ctx context.Context, task string) (string, error) {
		var specs []agentflow.SpawnSpec
		if !tm.skipLogs {
			specs = append(specs, agentflow.SpawnSpec{Agent: logs, Task: "errors for " + task})
		}
		specs = append(specs, agentflow.SpawnSpec{Agent: metrics, Task: "anomalies for " + task})
		if tm.extraAgent {
			specs = append(specs, agentflow.SpawnSpec{Agent: audit, Task: "audit " + task})
		}
		found, err := agentflow.SpawnAll(ctx, specs)
		if err != nil {
			return "", err
		}
		// The summarizer's task doesn't depend on which children ran, so
		// missing and unexpected children are reported on their own.
		sum, err := agentflow.Spawn(ctx, summarizer, tm.sumTask+": "+found[len(found)-1].OutputText())
		if err != nil {
			return "", err
		}
		return sum.OutputText(), nil
	})
}

func recordTeam(t *testing.T, tm *team) (*memstore.Store, string) {
	t.Helper()
	store := memstore.New()
	res, err := tm.workflow(t, true).Run(context.Background(), "spike", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	return store, res.TraceID
}

func runsByPath(t *testing.T, s trace.Store, traceID string) map[string][]trace.Event {
	t.Helper()
	ctx := context.Background()
	out := map[string][]trace.Event{}
	for info, err := range s.Runs(ctx, trace.RunFilter{TraceID: traceID}) {
		if err != nil {
			t.Fatal(err)
		}
		evs, err := trace.Collect(s.Events(ctx, info.RunID, 0))
		if err != nil {
			t.Fatal(err)
		}
		out[info.Path] = evs
	}
	return out
}

// comparable renders events without times, IDs, durations and the origin.
func comparable(t *testing.T, evs []trace.Event) []string {
	t.Helper()
	out := make([]string, len(evs))
	for i, e := range evs {
		raw, _ := json.Marshal(e.Body)
		var m map[string]any
		_ = json.Unmarshal(raw, &m)
		for _, k := range []string{"child_run_id", "duration_ns", "parent", "origin"} {
			delete(m, k)
		}
		b, _ := json.Marshal(m)
		cause := "-"
		if e.Cause != nil {
			cause = fmt.Sprint(*e.Cause)
		}
		out[i] = fmt.Sprintf("%s %s %s", e.Kind(), cause, b)
	}
	return out
}

func TestTeamReplayIsIdentical(t *testing.T) {
	tm := newTeam()
	store, src := recordTeam(t, tm)
	res, err := replay.Run(context.Background(), newTeam().workflow(t, false), store, src, replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("status %s err %v", res.Status, err)
	}
	// Every run of the tree was reproduced, event by event.
	want, got := runsByPath(t, store, src), runsByPath(t, store, res.TraceID)
	if len(got) != 5 || len(got) != len(want) {
		t.Fatalf("replay has %d runs, recording %d", len(got), len(want))
	}
	for p, evs := range want {
		a, b := comparable(t, evs), comparable(t, got[p])
		if !reflect.DeepEqual(a, b) {
			t.Errorf("%s differs:\n%s\nvs\n%s", p, strings.Join(a, "\n"), strings.Join(b, "\n"))
		}
	}
	if res.Run.Find("/metrics#0/sql#0") == nil {
		t.Fatal("the delegated child is missing from the result")
	}
}

func TestTeamDivergenceIsAttributedToTheChild(t *testing.T) {
	tm := newTeam()
	store, src := recordTeam(t, tm)
	changed := newTeam()
	changed.prompts["sql"] = "Query carefully."
	res, err := replay.Run(context.Background(), changed.workflow(t, false), store, src, replay.Options{})
	var de *replay.DivergenceError
	if !errors.As(err, &de) {
		t.Fatalf("err %v", err)
	}
	if len(res.Divergences) != 1 {
		t.Fatalf("divergences: %v", res.Divergences)
	}
	d := res.Divergences[0]
	if d.Path != "/metrics#0/sql#0" || d.Effect != "llm#0" || d.Kind != replay.RequestMismatch || d.Diffs[0].Path != "system[0].text" {
		t.Fatalf("%s", d.Summary)
	}
	if !strings.HasPrefix(d.Summary, "DIVERGED in /metrics#0/sql#0 at llm#0 (step 0)") {
		t.Errorf("summary %s", d.Summary)
	}
	// The diverged child and its ancestors end diverged; its sibling
	// finished; the summarizer never started.
	r := res.Run
	if r.Status != trace.StatusDiverged || r.Find("/metrics#0").Status != trace.StatusDiverged ||
		r.Find("/metrics#0/sql#0").Status != trace.StatusDiverged || r.Find("/logs#0").Status != trace.StatusCompleted || r.Find("/summarizer#0") != nil {
		t.Fatalf("statuses: %+v", r.Children)
	}
	for p, evs := range runsByPath(t, store, res.TraceID) {
		checkAnswered(t, p, evs)
	}
}

func checkAnswered(t *testing.T, path string, evs []trace.Event) {
	t.Helper()
	open := map[uint64]bool{}
	for _, e := range evs {
		switch e.Kind() {
		case trace.KindLLMStarted, trace.KindToolStarted, trace.KindEffectStarted, trace.KindRunSpawned:
			open[e.Seq] = true
		case trace.KindLLMCompleted, trace.KindLLMFailed, trace.KindToolCompleted, trace.KindToolFailed,
			trace.KindEffectCompleted, trace.KindEffectFailed, trace.KindRunJoined:
			delete(open, *e.Cause)
		case trace.KindRunFinished:
			if len(open) > 0 {
				t.Errorf("%s: finished with unanswered starts %v", path, open)
			}
		}
	}
}

func TestTeamSpawnRequestMismatch(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	changed := newTeam()
	changed.sumTask = "summarise"
	res, err := replay.Run(context.Background(), changed.workflow(t, false), store, src, replay.Options{})
	if err == nil || len(res.Divergences) != 1 {
		t.Fatalf("err %v divergences %v", err, res.Divergences)
	}
	d := res.Divergences[0]
	if d.Path != "/" || d.Effect != "spawn:/summarizer#0" || d.Kind != replay.RequestMismatch || d.Diffs[0].Path != "task[0].text" {
		t.Fatalf("%s", d.Summary)
	}
	if !strings.HasPrefix(d.Summary, "DIVERGED in / at spawn:/summarizer#0 — request_mismatch") {
		t.Errorf("summary %s", d.Summary)
	}
}

func TestTeamMissingAndUnexpectedChildren(t *testing.T) {
	store, src := recordTeam(t, newTeam())

	missing := newTeam()
	missing.skipLogs = true
	res, err := replay.Run(context.Background(), missing.workflow(t, false), store, src, replay.Options{})
	if err == nil || len(res.Divergences) != 1 {
		t.Fatalf("err %v divergences %v", err, res.Divergences)
	}
	if d := res.Divergences[0]; d.Kind != replay.MissingEffect || d.Effect != "spawn:/logs#0" || d.Path != "/" {
		t.Fatalf("%s", d.Summary)
	}

	extra := newTeam()
	extra.extraAgent = true
	res, err = replay.Run(context.Background(), extra.workflow(t, false), store, src, replay.Options{})
	if err == nil || len(res.Divergences) != 1 {
		t.Fatalf("err %v divergences %v", err, res.Divergences)
	}
	if d := res.Divergences[0]; d.Kind != replay.UnexpectedEffect || d.Effect != "spawn:/audit#0" {
		t.Fatalf("%s", d.Summary)
	}
}

func TestTeamOpaqueChild(t *testing.T) {
	store, src := recordTeam(t, newTeam())
	changed := newTeam()
	changed.prompts["sql"] = "Query carefully." // not re-executed, so not seen
	res, err := replay.Run(context.Background(), changed.workflow(t, false), store, src, replay.Options{Opaque: []string{"sql"}})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("status %s err %v", res.Status, err)
	}
	got := runsByPath(t, store, res.TraceID)
	if _, ok := got["/metrics#0/sql#0"]; ok {
		t.Fatal("an opaque child ran")
	}
	if len(got) != 4 {
		t.Fatalf("runs %d", len(got))
	}
}

func TestTeamGoLiveIsPerRun(t *testing.T) {
	store, src := recordTeam(t, newTeam())

	// logs changes, but its answer doesn't feed the summarizer: only logs
	// goes live; the other runs keep replaying.
	changed := newTeam()
	changed.prompts["logs"] = "Find all errors."
	res, err := replay.Run(context.Background(), changed.workflow(t, true), store, src, replay.Options{OnDivergence: replay.GoLive})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(changed.calls, map[string]int{"logs": 1}) {
		t.Fatalf("live calls %v", changed.calls)
	}
	if len(res.Divergences) != 1 || res.Divergences[0].Path != "/logs#0" {
		t.Fatalf("divergences %v", res.Divergences)
	}

	// metrics' sub-agent changes: the change reaches the summarizer's task,
	// so the root goes live at that spawn, and the summarizer runs live.
	changed = newTeam()
	changed.prompts["sql"] = "Query carefully."
	res, err = replay.Run(context.Background(), changed.workflow(t, true), store, src, replay.Options{OnDivergence: replay.GoLive})
	if err != nil {
		t.Fatal(err)
	}
	var where []string
	for _, d := range res.Divergences {
		where = append(where, d.Path+" "+d.Effect+" "+string(d.Kind))
	}
	// The delegate's tool result isn't compared (the child reported it);
	// metrics sees the consequence at its next request.
	want := []string{"/ spawn:/summarizer#0 request_mismatch", "/metrics#0 llm#1 request_mismatch", "/metrics#0/sql#0 llm#0 request_mismatch"}
	if !slices.Equal(where, want) {
		t.Fatalf("divergences %v", where)
	}
	if !reflect.DeepEqual(changed.calls, map[string]int{"sql": 1, "metrics": 1, "summarizer": 1}) {
		t.Fatalf("live calls %v", changed.calls)
	}
}

func TestTeamRefusedSpawnReplays(t *testing.T) {
	denied := newTeam()
	denied.denySQL = true
	store, src := recordTeam(t, denied)

	again := newTeam()
	again.denySQL = true
	res, err := replay.Run(context.Background(), again.workflow(t, false), store, src, replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("same policy: status %s err %v", res.Status, err)
	}

	res, err = replay.Run(context.Background(), newTeam().workflow(t, false), store, src, replay.Options{})
	if err == nil || len(res.Divergences) != 1 {
		t.Fatalf("err %v divergences %v", err, res.Divergences)
	}
	if d := res.Divergences[0]; d.Path != "/metrics#0" || d.Kind != replay.UnexpectedEffect || d.Effect != "tool:m1/spawn:/metrics#0/sql#0" ||
		!strings.Contains(d.Summary, "now it reaches the world") {
		t.Fatalf("%s", d.Summary)
	}
}

func TestTeamReplayIsDeterministicUnderConcurrency(t *testing.T) {
	rec := newTeam()
	rec.jitter = true
	store, src := recordTeam(t, rec)
	var first []string
	for i := range 30 {
		// Two siblings change; both are reported, in path order, whatever
		// order they ran in.
		changed := newTeam()
		changed.prompts["logs"] = "Find all errors."
		changed.prompts["sql"] = "Query carefully."
		res, _ := replay.Run(context.Background(), changed.workflow(t, false), store, src, replay.Options{})
		var got []string
		for _, d := range res.Divergences {
			got = append(got, d.Summary)
		}
		if len(got) != 2 || !strings.Contains(got[0], "/logs#0") || !strings.Contains(got[1], "/metrics#0/sql#0") {
			t.Fatalf("divergences %v", got)
		}
		if i == 0 {
			first = got
		} else if !slices.Equal(got, first) {
			t.Fatalf("run %d differs", i)
		}
	}
}

// A child started inside an ordinary tool is covered by the tool's recorded
// result: it doesn't run during replay, and isn't missing. Verifying the
// tool runs it, against its own recording.
func TestChildInsideOrdinaryTool(t *testing.T) {
	build := func(live bool, helperPrompt string) *agentflow.Agent {
		helper, _ := agentflow.NewAgent(agentflow.AgentConfig{Name: "helper", ModelName: "m", Instructions: helperPrompt,
			Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
				if !live {
					t.Error("helper's model called during replay")
				}
				return agentflowtest.Reply("helper(" + llm.TextOf(req.System) + ") saw " + lastText(req)), nil
			})})
		ask := agentflow.NewTool(llm.ToolSpec{Name: "ask"}, func(ctx context.Context, in struct{ Q string }) (agentflow.ToolOutput, error) {
			res, err := helper.Run(ctx, in.Q)
			if err != nil {
				return agentflow.ToolOutput{}, err
			}
			return agentflow.TextOutput(res.OutputText()), nil
		})
		m := noModel(t)
		if live {
			m = agentflowtest.NewScriptedModel(agentflowtest.CallTools(agentflowtest.Use("1", "ask", map[string]string{"Q": "why"})), agentflowtest.Reply("done"))
		}
		lead, _ := agentflow.NewAgent(agentflow.AgentConfig{Name: "lead", ModelName: "m", Model: m, Tools: []agentflow.Tool{ask}})
		return lead
	}
	store := memstore.New()
	rec, err := build(true, "Help.").Run(context.Background(), "go", agentflow.WithStore(store))
	if err != nil || rec.Find("/helper@0.0.0") == nil {
		t.Fatalf("err %v", err)
	}

	res, err := replay.Run(context.Background(), build(false, "Help."), store, rec.TraceID, replay.Options{})
	if err != nil || res.Status != replay.Identical || len(res.Run.Children) != 0 {
		t.Fatalf("status %s err %v children %d", res.Status, err, len(res.Run.Children))
	}

	res, err = replay.Run(context.Background(), build(false, "Help."), store, rec.TraceID, replay.Options{Verify: []string{"tool:ask"}})
	if err != nil || res.Status != replay.Identical || res.Run.Find("/helper@0.0.0") == nil || !slices.Equal(res.Verified, []string{"tool:ask(1)"}) {
		t.Fatalf("verified: status %s err %v verified %v", res.Status, err, res.Verified)
	}

	res, _ = replay.Run(context.Background(), build(false, "Help more."), store, rec.TraceID, replay.Options{Verify: []string{"tool:ask"}})
	if len(res.Divergences) != 1 || res.Divergences[0].Path != "/helper@0.0.0" || res.Divergences[0].Effect != "llm#0" {
		t.Fatalf("divergences %v", res.Divergences)
	}
}

// A budget passed to the root with WithInterceptors is shared by the whole
// tree: it stops a grandchild mid-tree, and the replay, given the same
// budget, reproduces every decision.
func TestSharedBudgetAcrossTheTree(t *testing.T) {
	build := func(live bool) *agentflow.Agent {
		mk := func(name string, tools ...agentflow.Tool) *agentflow.Agent {
			a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: name, ModelName: "m", Tools: tools,
				Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
					if !live {
						t.Errorf("%s's model called during replay", name)
					}
					if len(tools) > 0 && len(req.Messages) == 1 {
						return agentflowtest.CallTools(agentflowtest.Use(name+"-1", tools[0].Spec().Name, map[string]string{"task": "t"})), nil
					}
					return agentflowtest.Reply(name + " got " + lastText(req)), nil
				})})
			if err != nil {
				t.Fatal(err)
			}
			return a
		}
		a2 := mk("a2")
		a1 := mk("a1", agentflow.Delegate(a2, agentflow.DelegateOptions{}))
		return mk("a0", agentflow.Delegate(a1, agentflow.DelegateOptions{}))
	}
	store := memstore.New()
	res, err := build(true).Run(context.Background(), "go", agentflow.WithStore(store),
		agentflow.WithInterceptors(&policy.Budget{MaxTotalTokens: 30})) // two calls of 15 tokens
	if err == nil {
		t.Fatal("the budget didn't stop the tree")
	}
	a2 := res.Find("/a1#0/a2#0")
	if a2 == nil || a2.Status != trace.StatusFailed || a2.Steps != 1 {
		t.Fatalf("a2: %+v", a2)
	}
	for _, e := range runsByPath(t, store, res.TraceID)["/a1#0/a2#0"] {
		if f, ok := e.Body.(*trace.LLMFailed); ok && (!f.Intercepted || f.Error.Code != trace.ErrCodeLimitExceeded) {
			t.Fatalf("llm.failed %+v", f)
		}
	}

	rep, err := replay.Run(context.Background(), build(false), store, res.TraceID,
		replay.Options{Interceptors: []agentflow.Interceptor{&policy.Budget{MaxTotalTokens: 30}}})
	if err != nil || rep.Status != replay.Identical {
		t.Fatalf("status %s err %v divergences %v", rep.Status, err, rep.Divergences)
	}
}
