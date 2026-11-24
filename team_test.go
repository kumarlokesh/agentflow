package agentflow_test

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
	"sync/atomic"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// lastText is the text of the last message's last text block or tool result.
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

// jitter sleeps a little, to shake out scheduling dependence.
func jitter(on bool) {
	if on {
		time.Sleep(time.Duration(rand.IntN(300)) * time.Microsecond)
	}
}

// leaf answers "<name> saw <task>".
func leaf(t *testing.T, name string, slow bool) *agentflow.Agent {
	t.Helper()
	return mustAgent(t, agentflow.AgentConfig{Name: name, ModelName: "m", Instructions: "You are " + name + ".",
		Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			jitter(slow)
			return agentflowtest.Reply(name + " saw " + lastText(req)), nil
		})})
}

// delegator calls each of tools (all at once) with {"task": "<tool> for <task>"},
// then answers with what they returned.
func delegator(t *testing.T, name string, slow bool, tools ...agentflow.Tool) *agentflow.Agent {
	t.Helper()
	return mustAgent(t, agentflow.AgentConfig{Name: name, ModelName: "m", Tools: tools, MaxParallelTools: len(tools),
		Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			jitter(slow)
			if len(req.Messages) == 1 {
				var uses []llm.ToolUse
				for i, tl := range tools {
					tn := tl.Spec().Name
					uses = append(uses, agentflowtest.Use(fmt.Sprintf("%s-%d", name, i), tn, map[string]string{"task": tn + " for " + lastText(req)}))
				}
				return agentflowtest.CallTools(uses...), nil
			}
			var got []string
			for _, b := range req.Messages[len(req.Messages)-1].Blocks {
				got = append(got, llm.TextOf(b.(llm.ToolResult).Content))
			}
			return agentflowtest.Reply(name + " got [" + strings.Join(got, "; ") + "]"), nil
		})})
}

func mustAgent(t *testing.T, cfg agentflow.AgentConfig) *agentflow.Agent {
	t.Helper()
	a, err := agentflow.NewAgent(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// triage is the RFC's example team: a workflow fans out to logs and
// metrics (metrics delegates to sql), then a summarizer combines them.
func triage(t *testing.T, slow bool) *agentflow.Workflow {
	sql := leaf(t, "sql", slow)
	logs := leaf(t, "logs", slow)
	metrics := delegator(t, "metrics", slow, agentflow.Delegate(sql, agentflow.DelegateOptions{ToolName: "query_db"}))
	summarizer := leaf(t, "summarizer", slow)
	return agentflow.NewWorkflow("triage", func(ctx context.Context, task string) (string, error) {
		found, err := agentflow.SpawnAll(ctx, []agentflow.SpawnSpec{
			{Agent: logs, Task: "errors for " + task},
			{Agent: metrics, Task: "anomalies for " + task},
		})
		if err != nil {
			return "", err
		}
		sum, err := agentflow.Spawn(ctx, summarizer, found[0].OutputText()+" | "+found[1].OutputText())
		if err != nil {
			return "", err
		}
		return sum.OutputText(), nil
	})
}

// treeEvents returns every run's events in a trace, by path.
func treeEvents(t *testing.T, s trace.Store, traceID string) map[string][]trace.Event {
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
		if _, dup := out[info.Path]; dup {
			t.Fatalf("two runs at %s", info.Path)
		}
		out[info.Path] = evs
	}
	return out
}

// normalized renders events without times, IDs and durations, which differ
// between runs that behaved the same.
func normalized(t *testing.T, evs []trace.Event) []string {
	t.Helper()
	out := make([]string, len(evs))
	for i, e := range evs {
		raw, err := json.Marshal(e.Body)
		if err != nil {
			t.Fatal(err)
		}
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
		out[i] = fmt.Sprintf("%d %s %s %s", e.Seq, e.Kind(), cause, b)
	}
	return out
}

// checkAnswered checks trace rule 5.3: in a finished run every start has
// exactly one answer, and every run.spawned exactly one run.joined.
func checkAnswered(t *testing.T, path string, evs []trace.Event) {
	t.Helper()
	open := map[uint64]trace.Kind{}
	finished := false
	for _, e := range evs {
		switch e.Kind() {
		case trace.KindLLMStarted, trace.KindToolStarted, trace.KindEffectStarted, trace.KindRunSpawned:
			open[e.Seq] = e.Kind()
		case trace.KindLLMCompleted, trace.KindLLMFailed, trace.KindToolCompleted, trace.KindToolFailed,
			trace.KindEffectCompleted, trace.KindEffectFailed, trace.KindRunJoined:
			if e.Cause == nil {
				t.Fatalf("%s seq %d: %s without cause", path, e.Seq, e.Kind())
			}
			if _, ok := open[*e.Cause]; !ok {
				t.Fatalf("%s seq %d: %s answers %d, which isn't open", path, e.Seq, e.Kind(), *e.Cause)
			}
			delete(open, *e.Cause)
		case trace.KindRunFinished:
			finished = true
		}
	}
	if finished && len(open) > 0 {
		t.Errorf("%s: finished with unanswered starts %v", path, open)
	}
}

func TestWorkflowRecordsOneTree(t *testing.T) {
	store := memstore.New()
	res, err := triage(t, false).Run(context.Background(), "latency spike", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	if res.Kind != trace.RunKindWorkflow || res.Path != "/" {
		t.Fatalf("root %s %s", res.Kind, res.Path)
	}
	want := "summarizer saw logs saw errors for latency spike | metrics got [sql saw query_db for anomalies for latency spike]"
	if res.OutputText() != want {
		t.Fatalf("output %q", res.OutputText())
	}
	var paths []string
	for _, c := range res.Children {
		paths = append(paths, c.Path)
	}
	if !slices.Equal(paths, []string{"/logs#0", "/metrics#0", "/summarizer#0"}) {
		t.Fatalf("children %v", paths)
	}
	sql := res.Find("/metrics#0/sql#0")
	if sql == nil || sql.Kind != trace.RunKindAgent || sql.Status != trace.StatusCompleted {
		t.Fatalf("sql %+v", sql)
	}
	// Each agent call uses 10/5 tokens: logs 1, metrics 2, sql 1, summarizer 1.
	if u := res.TreeUsage(); u.InputTokens != 50 || res.Usage.InputTokens != 0 {
		t.Errorf("tree usage %+v, own %+v", u, res.Usage)
	}

	tree := treeEvents(t, store, res.TraceID)
	if len(tree) != 5 {
		t.Fatalf("runs %v", slices.Collect(func(yield func(string) bool) {
			for p := range tree {
				yield(p)
			}
		}))
	}
	ids := map[string]string{}
	for p, evs := range tree {
		checkAnswered(t, p, evs)
		ids[p] = evs[0].RunID
	}
	for p, evs := range tree {
		rs := evs[0].Body.(*trace.RunStarted)
		if p == "/" {
			if rs.Parent != nil || rs.Config != nil || rs.RunKind != trace.RunKindWorkflow {
				t.Errorf("root run.started %+v", rs)
			}
			continue
		}
		parent := p[:strings.LastIndex(p, "/")]
		if parent == "" {
			parent = "/"
		}
		if rs.Parent == nil || rs.Parent.RunID != ids[parent] {
			t.Errorf("%s: parent %+v, want %s", p, rs.Parent, ids[parent])
		}
	}
	// The workflow's own events: the SpawnAll batch (both spawned, then both
	// joined), then the summarizer.
	got := kinds(tree["/"])
	wantKinds := []trace.Kind{trace.KindRunStarted, trace.KindRunSpawned, trace.KindRunSpawned, trace.KindRunJoined, trace.KindRunJoined,
		trace.KindRunSpawned, trace.KindRunJoined, trace.KindRunFinished}
	if !slices.Equal(got, wantKinds) {
		t.Fatalf("root events %v", got)
	}
	sp := tree["/"][1].Body.(*trace.RunSpawned)
	if sp.Path != "/logs#0" || sp.Agent != "logs" || sp.ChildRunID != ids["/logs#0"] || llm.TextOf(sp.Task) != "errors for latency spike" {
		t.Errorf("run.spawned %+v", sp)
	}
	j := tree["/"][3]
	if *j.Cause != 1 || j.Body.(*trace.RunJoined).Status != trace.StatusCompleted {
		t.Errorf("run.joined %+v cause %d", j.Body, *j.Cause)
	}
}

func TestDelegationBatchLayout(t *testing.T) {
	// Two delegations in one response, run in parallel with random
	// latencies: the layout and the paths never depend on which finishes
	// first.
	var first []string
	for i := range 50 {
		researcher := leaf(t, "researcher", true)
		lead := delegator(t, "lead", true,
			agentflow.Delegate(researcher, agentflow.DelegateOptions{ToolName: "research_a"}),
			agentflow.Delegate(researcher, agentflow.DelegateOptions{ToolName: "research_b"}))
		store := memstore.New()
		res, err := lead.Run(context.Background(), "q", agentflow.WithStore(store))
		if err != nil {
			t.Fatal(err)
		}
		tree := treeEvents(t, store, res.TraceID)
		root := tree["/"]
		got := kinds(root)
		want := []trace.Kind{trace.KindRunStarted, trace.KindMessageAppended, trace.KindLLMStarted, trace.KindLLMCompleted, trace.KindMessageAppended,
			trace.KindToolStarted, trace.KindToolStarted,
			trace.KindRunSpawned, trace.KindRunJoined, trace.KindToolCompleted,
			trace.KindRunSpawned, trace.KindRunJoined, trace.KindToolCompleted,
			trace.KindMessageAppended, trace.KindLLMStarted, trace.KindLLMCompleted, trace.KindMessageAppended, trace.KindRunFinished}
		if !slices.Equal(got, want) {
			t.Fatalf("layout %v", got)
		}
		// Causes: run.spawned → its tool.started; run.joined → run.spawned.
		if *root[7].Cause != 5 || *root[8].Cause != 7 || *root[9].Cause != 5 || *root[10].Cause != 6 || *root[11].Cause != 10 || *root[12].Cause != 6 {
			t.Fatalf("causes %v", normalized(t, root[5:13]))
		}
		if p := root[7].Body.(*trace.RunSpawned).Path; p != "/researcher#0" {
			t.Fatalf("first delegation path %s", p)
		}
		if p := root[10].Body.(*trace.RunSpawned).Path; p != "/researcher#1" {
			t.Fatalf("second delegation path %s", p)
		}
		if tree["/researcher#0"][0].RunID != root[7].Body.(*trace.RunSpawned).ChildRunID {
			t.Fatal("child not linked")
		}
		if llm.TextOf(tree["/researcher#1"][0].Body.(*trace.RunStarted).Task) != "research_b for q" {
			t.Fatal("ordinals don't follow response order")
		}
		for p, evs := range tree {
			checkAnswered(t, p, evs)
		}
		n := normalized(t, root)
		if i == 0 {
			first = n
		} else if !slices.Equal(n, first) {
			t.Fatalf("run %d differs:\n%s\nvs\n%s", i, strings.Join(n, "\n"), strings.Join(first, "\n"))
		}
	}
}

func TestRunInsideToolJoinsTreeWithGeneratedPath(t *testing.T) {
	for _, parallel := range []int{1, 8} {
		var first []string
		for i := range 30 {
			helper := leaf(t, "helper", true)
			ask := agentflow.NewTool(llm.ToolSpec{Name: "ask"}, func(ctx context.Context, in struct{ Q string }) (agentflow.ToolOutput, error) {
				res, err := helper.Run(ctx, in.Q) // an ordinary tool starting an agent
				if err != nil {
					return agentflow.ToolOutput{}, err
				}
				return agentflow.TextOutput(res.OutputText()), nil
			})
			lead := mustAgent(t, agentflow.AgentConfig{Name: "lead", ModelName: "m", Tools: []agentflow.Tool{ask}, MaxParallelTools: parallel,
				Model: agentflowtest.NewScriptedModel(
					agentflowtest.CallTools(agentflowtest.Use("a", "ask", map[string]string{"Q": "one"}), agentflowtest.Use("b", "ask", map[string]string{"Q": "two"})),
					agentflowtest.Reply("done"))})
			store := memstore.New()
			res, err := lead.Run(context.Background(), "go", agentflow.WithStore(store))
			if err != nil {
				t.Fatal(err)
			}
			tree := treeEvents(t, store, res.TraceID)
			one, two := tree["/helper@0.0.0"], tree["/helper@0.1.0"]
			if one == nil || two == nil {
				t.Fatalf("paths: %v", res.Children)
			}
			if llm.TextOf(one[0].Body.(*trace.RunStarted).Task) != "one" || llm.TextOf(two[0].Body.(*trace.RunStarted).Task) != "two" {
				t.Fatal("generated keys don't follow call position")
			}
			n := normalized(t, tree["/"])
			if i == 0 {
				first = n
			} else if !slices.Equal(n, first) {
				t.Fatalf("parallel %d run %d differs", parallel, i)
			}
		}
	}
}

func TestConcurrentSpawnIsRejected(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	slow := mustAgent(t, agentflow.AgentConfig{Name: "slow", ModelName: "m",
		Model: agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
			close(entered)
			<-release
			return agentflowtest.Reply("ok"), nil
		})})
	other := leaf(t, "other", false)
	var second error
	wf := agentflow.NewWorkflow("racy", func(ctx context.Context, _ string) (string, error) {
		var wg sync.WaitGroup
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = agentflow.Spawn(ctx, slow, "a") }()
		<-entered
		_, second = agentflow.Spawn(ctx, other, "b")
		close(release)
		wg.Wait()
		return "", nil
	})
	if _, err := wf.Run(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(second, agentflow.ErrConcurrentEffect) {
		t.Fatalf("second Spawn: %v", second)
	}
}

func TestSpawnErrors(t *testing.T) {
	a1 := leaf(t, "same", false)
	a2 := leaf(t, "same", false)
	cases := []struct {
		name string
		fn   agentflow.WorkflowFunc
		want error
		msg  string
	}{
		{"name conflict", func(ctx context.Context, _ string) (string, error) {
			if _, err := agentflow.Spawn(ctx, a1, "x"); err != nil {
				return "", err
			}
			_, err := agentflow.Spawn(ctx, a2, "x")
			return "", err
		}, agentflow.ErrAgentNameConflict, ""},
		{"generated key shape", func(ctx context.Context, _ string) (string, error) {
			_, err := agentflow.Spawn(ctx, a1, "x", agentflow.SpawnKey("2.1.0"))
			return "", err
		}, nil, "generated shape"},
		{"repeated key", func(ctx context.Context, _ string) (string, error) {
			if _, err := agentflow.Spawn(ctx, a1, "x", agentflow.SpawnKey("k")); err != nil {
				return "", err
			}
			_, err := agentflow.Spawn(ctx, a1, "y", agentflow.SpawnKey("k"))
			return "", err
		}, nil, "already used"},
		{"inside Do", func(ctx context.Context, _ string) (string, error) {
			return agentflow.Do(ctx, "wrap", 1, func(ctx context.Context, _ int) (string, error) {
				_, err := agentflow.Spawn(ctx, a1, "x")
				return "", err
			})
		}, nil, "child run started inside an effect function"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := agentflow.NewWorkflow("wf", c.fn).Run(context.Background(), "t")
			if err == nil || (c.want != nil && !errors.Is(err, c.want)) || !strings.Contains(err.Error(), c.msg) {
				t.Fatalf("err %v", err)
			}
		})
	}
	// A run started inside a run joins its tree: tree-wide options are refused.
	wf := agentflow.NewWorkflow("wf", func(ctx context.Context, _ string) (string, error) {
		_, err := a1.Run(ctx, "x", agentflow.WithStore(memstore.New()))
		return "", err
	})
	if _, err := wf.Run(context.Background(), "t"); err == nil || !strings.Contains(err.Error(), "only WithTaskBlocks") {
		t.Fatalf("err %v", err)
	}
}

func TestTreeLimits(t *testing.T) {
	// a0 → a1 → a2 by delegation; MaxDepth 1 refuses a2, and a1's model sees
	// the refusal as an error result.
	a2 := leaf(t, "a2", false)
	a1 := delegator(t, "a1", false, agentflow.Delegate(a2, agentflow.DelegateOptions{}))
	a0 := delegator(t, "a0", false, agentflow.Delegate(a1, agentflow.DelegateOptions{}))
	store := memstore.New()
	res, err := a0.Run(context.Background(), "go", agentflow.WithStore(store), agentflow.WithLimits(agentflow.TreeLimits{MaxDepth: 1}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.OutputText(), "limit_exceeded") {
		t.Fatalf("output %q", res.OutputText())
	}
	tree := treeEvents(t, store, res.TraceID)
	if _, ok := tree["/a1#0/a2#0"]; ok {
		t.Fatal("a refused child ran")
	}
	var joined *trace.RunJoined
	for _, e := range tree["/a1#0"] {
		if b, ok := e.Body.(*trace.RunJoined); ok {
			joined = b
		}
	}
	if joined == nil || joined.Status != trace.StatusFailed || !joined.Intercepted || joined.Error == nil || joined.Error.Code != trace.ErrCodeLimitExceeded {
		t.Fatalf("run.joined %+v", joined)
	}
	for p, evs := range tree {
		checkAnswered(t, p, evs)
	}

	// MaxRuns counts the root: 2 allows one child.
	wf := agentflow.NewWorkflow("wf", func(ctx context.Context, _ string) (string, error) {
		_, err := agentflow.SpawnAll(ctx, []agentflow.SpawnSpec{{Agent: a2, Task: "1"}, {Agent: a2, Task: "2"}})
		return "", err
	})
	res, err = wf.Run(context.Background(), "go", agentflow.WithLimits(agentflow.TreeLimits{MaxRuns: 2}))
	var ee *agentflow.EffectError
	if !errors.As(err, &ee) || ee.Code != trace.ErrCodeLimitExceeded || len(res.Children) != 1 {
		t.Fatalf("err %v, children %d", err, len(res.Children))
	}
}

func TestMaxConcurrentRuns(t *testing.T) {
	// Three agents that each delegate (deeper than the limit of 1): no
	// deadlock, and never more than one child executing at a time.
	var active, peak atomic.Int32
	model := func(name string) llm.Model {
		return agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			n := active.Add(1)
			defer active.Add(-1)
			for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
			}
			time.Sleep(200 * time.Microsecond)
			if len(req.Messages) == 1 && strings.HasPrefix(name, "mid") {
				return agentflowtest.CallTools(agentflowtest.Use("c", "leaf", map[string]string{"task": "t"})), nil
			}
			return agentflowtest.Reply(name), nil
		})
	}
	leafA := mustAgent(t, agentflow.AgentConfig{Name: "leaf", ModelName: "m", Model: model("leaf")})
	var specs []agentflow.SpawnSpec
	for i := range 3 {
		name := fmt.Sprintf("mid%d", i)
		mid := mustAgent(t, agentflow.AgentConfig{Name: name, ModelName: "m", Model: model(name),
			Tools: []agentflow.Tool{agentflow.Delegate(leafA, agentflow.DelegateOptions{})}})
		specs = append(specs, agentflow.SpawnSpec{Agent: mid, Task: "go"})
	}
	wf := agentflow.NewWorkflow("fan", func(ctx context.Context, _ string) (string, error) {
		_, err := agentflow.SpawnAll(ctx, specs)
		return "", err
	})
	done := make(chan error, 1)
	go func() {
		_, err := wf.Run(context.Background(), "x", agentflow.WithLimits(agentflow.TreeLimits{MaxConcurrentRuns: 1}))
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("deadlock")
	}
	if p := peak.Load(); p != 1 {
		t.Fatalf("peak concurrent model calls %d, want 1", p)
	}
}

func TestTreeIsDeterministicUnderConcurrency(t *testing.T) {
	var first map[string][]string
	for i := range 100 {
		store := memstore.New()
		res, err := triage(t, true).Run(context.Background(), "spike", agentflow.WithStore(store),
			agentflow.WithLimits(agentflow.TreeLimits{MaxConcurrentRuns: 2}))
		if err != nil {
			t.Fatal(err)
		}
		got := map[string][]string{}
		for p, evs := range treeEvents(t, store, res.TraceID) {
			got[p] = normalized(t, evs)
		}
		if i == 0 {
			first = got
		} else if !reflect.DeepEqual(got, first) {
			t.Fatalf("run %d differs from run 0", i)
		}
	}
}

func TestRefusedSpawnAndInheritedInterceptors(t *testing.T) {
	sql := leaf(t, "sql", false)
	metrics := delegator(t, "metrics", false, agentflow.Delegate(sql, agentflow.DelegateOptions{ToolName: "query_db"}))
	var seen []string
	var mu sync.Mutex
	deny := agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
		mu.Lock()
		seen = append(seen, c.Meta.Path+" "+string(c.Kind)+":"+c.Name)
		mu.Unlock()
		if c.Kind == agentflow.EffectSpawn && c.Name == "sql" {
			return agentflow.Fail(trace.ErrCodeDenied, "metrics may not query sql"), nil
		}
		return next(ctx, c)
	})
	store := memstore.New()
	res, err := metrics.Run(context.Background(), "q", agentflow.WithStore(store), agentflow.WithInterceptors(deny))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.OutputText(), "denied") || len(res.Children) != 0 {
		t.Fatalf("output %q children %d", res.OutputText(), len(res.Children))
	}
	tree := treeEvents(t, store, res.TraceID)
	if len(tree) != 1 {
		t.Fatal("a denied child ran")
	}
	for _, e := range tree["/"] {
		if j, ok := e.Body.(*trace.RunJoined); ok && (!j.Intercepted || j.Error.Code != trace.ErrCodeDenied || j.Status != trace.StatusFailed) {
			t.Fatalf("run.joined %+v", j)
		}
	}
	if !slices.Contains(seen, "/ spawn:sql") {
		t.Fatalf("interceptor saw %v", seen)
	}
}

func TestAbandonedBatchIsFlushed(t *testing.T) {
	// An interceptor error aborts the run in the middle of a batch whose
	// other call started a child: the batch is still written in order, the
	// child is linked, and every start is answered.
	researcher := leaf(t, "researcher", false)
	other := agentflow.NewTool(llm.ToolSpec{Name: "other"}, func(context.Context, struct{}) (agentflow.ToolOutput, error) {
		return agentflow.TextOutput("x"), nil
	})
	boom := agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
		if c.Name == "other" {
			return nil, errors.New("interceptor broke")
		}
		return next(ctx, c)
	})
	lead := mustAgent(t, agentflow.AgentConfig{Name: "lead", ModelName: "m", MaxParallelTools: 2, Interceptors: []agentflow.Interceptor{boom},
		Tools: []agentflow.Tool{agentflow.Delegate(researcher, agentflow.DelegateOptions{}), other},
		Model: agentflowtest.NewScriptedModel(agentflowtest.CallTools(
			agentflowtest.Use("1", "researcher", map[string]string{"task": "t"}), agentflowtest.Use("2", "other", struct{}{})))})
	store := memstore.New()
	res, err := lead.Run(context.Background(), "go", agentflow.WithStore(store))
	if err == nil || res.Status != trace.StatusFailed {
		t.Fatalf("status %s err %v", res.Status, err)
	}
	tree := treeEvents(t, store, res.TraceID)
	root := tree["/"]
	for p, evs := range tree {
		checkAnswered(t, p, evs)
	}
	got := kinds(root[5:])
	want := []trace.Kind{trace.KindToolStarted, trace.KindToolStarted, trace.KindRunSpawned, trace.KindRunJoined, trace.KindToolCompleted, trace.KindToolFailed, trace.KindRunFinished}
	if !slices.Equal(got, want) {
		t.Fatalf("events %v", got)
	}
	if f := root[10].Body.(*trace.ToolFailed); f.Error.Code != trace.ErrCodeCancelled {
		t.Fatalf("tool.failed %+v", f)
	}
	if len(res.Children) != 1 || res.Children[0].Path != "/researcher#0" {
		t.Fatalf("children %v", res.Children)
	}
}

func TestAccessDeclarations(t *testing.T) {
	m := agentflowtest.NewScriptedModel(agentflowtest.CallTools(agentflowtest.Use("1", "edit", map[string]string{"file": "a.go"})), agentflowtest.Reply("ok"))
	a := mustAgent(t, agentflow.AgentConfig{Name: "dev", ModelName: "m", Model: m, Tools: []agentflow.Tool{editTool{}}})
	res, err := a.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range events(t, res) {
		if ts, ok := e.Body.(*trace.ToolStarted); ok {
			want := []trace.Access{{Resource: "file:a.go", Mode: trace.AccessWrite}}
			if ts.Accesses == nil || !reflect.DeepEqual(*ts.Accesses, want) {
				t.Fatalf("accesses %v", ts.Accesses)
			}
			return
		}
	}
	t.Fatal("no tool.started")
}

type editTool struct{}

func (editTool) Spec() llm.ToolSpec { return llm.ToolSpec{Name: "edit"} }
func (editTool) Call(context.Context, json.RawMessage) (agentflow.ToolOutput, error) {
	return agentflow.TextOutput("edited"), nil
}
func (editTool) Accesses(input json.RawMessage) []trace.Access {
	var in struct{ File string }
	_ = json.Unmarshal(input, &in)
	return []trace.Access{{Resource: "file:" + in.File, Mode: trace.AccessWrite}}
}
