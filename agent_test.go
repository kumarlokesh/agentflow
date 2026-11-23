package agentflow_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

type calcIn struct {
	A  float64 `json:"a"`
	B  float64 `json:"b"`
	Op string  `json:"op"`
}

var calcSpec = llm.ToolSpec{
	Name:        "calc",
	Description: "Arithmetic on two numbers.",
	InputSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"},"op":{"type":"string","enum":["+","*"]}},"required":["a","b","op"]}`),
}

func calcTool() agentflow.Tool {
	return agentflow.NewTool(calcSpec, func(_ context.Context, in calcIn) (agentflow.ToolOutput, error) {
		switch in.Op {
		case "+":
			return agentflow.TextOutput(fmt.Sprint(in.A + in.B)), nil
		case "*":
			return agentflow.TextOutput(fmt.Sprint(in.A * in.B)), nil
		}
		return agentflow.ErrorOutput("unknown op"), nil
	})
}

func newAgent(t *testing.T, m llm.Model, mutate func(*agentflow.AgentConfig)) *agentflow.Agent {
	t.Helper()
	cfg := agentflow.AgentConfig{Name: "calculator", Model: m, ModelName: "test-model", Instructions: "You calculate.", Tools: []agentflow.Tool{calcTool()}}
	if mutate != nil {
		mutate(&cfg)
	}
	a, err := agentflow.NewAgent(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func kinds(evs []trace.Event) []trace.Kind {
	out := make([]trace.Kind, len(evs))
	for i, e := range evs {
		out[i] = e.Kind()
	}
	return out
}

func events(t *testing.T, res *agentflow.RunResult) []trace.Event {
	t.Helper()
	evs, err := res.Events(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return evs
}

func TestRunRecordsExactRequests(t *testing.T) {
	m := agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("t1", "calc", calcIn{6, 7, "*"})),
		agentflowtest.Reply("6 × 7 = 42"),
	)
	res, err := newAgent(t, m, nil).Run(context.Background(), "What is 6*7?")
	if err != nil {
		t.Fatal(err)
	}
	if res.OutputText() != "6 × 7 = 42" || res.Steps != 2 || res.Status != trace.StatusCompleted {
		t.Fatalf("result = %+v", res)
	}
	if res.Usage.InputTokens != 20 || res.Usage.OutputTokens != 10 {
		t.Errorf("usage = %+v", res.Usage)
	}
	evs := events(t, res)
	want := []trace.Kind{
		trace.KindRunStarted, trace.KindMessageAppended,
		trace.KindLLMStarted, trace.KindLLMCompleted, trace.KindMessageAppended,
		trace.KindToolStarted, trace.KindToolCompleted, trace.KindMessageAppended,
		trace.KindLLMStarted, trace.KindLLMCompleted, trace.KindMessageAppended,
		trace.KindRunFinished,
	}
	if got := kinds(evs); !reflect.DeepEqual(got, want) {
		t.Fatalf("event kinds:\n got %v\nwant %v", got, want)
	}
	if tc := evs[6].Body.(*trace.ToolCompleted); llm.TextOf(tc.Content) != "42" {
		t.Errorf("tool result = %q", llm.TextOf(tc.Content))
	}

	// Every recorded request folds back to exactly what the model received.
	sent := m.Requests()
	i := 0
	for _, e := range evs {
		ls, ok := e.Body.(*trace.LLMStarted)
		if !ok {
			continue
		}
		folded, err := trace.Request(evs, e.Seq)
		if err != nil {
			t.Fatal(err)
		}
		a, _ := json.Marshal(sent[i])
		b, _ := json.Marshal(folded)
		if string(a) != string(b) {
			t.Fatalf("request %d:\nsent   %s\nfolded %s", i, a, b)
		}
		if fp, _ := trace.RequestFingerprint(sent[i]); fp != ls.Fingerprint {
			t.Errorf("request %d fingerprint mismatch", i)
		}
		i++
	}
	fin := evs[len(evs)-1].Body.(*trace.RunFinished)
	if fin.Steps != 2 || fin.Usage.InputTokens != 20 {
		t.Errorf("run.finished = %+v", fin)
	}
}

func TestParallelBatchIsDeterministic(t *testing.T) {
	sleepy := agentflow.NewTool(llm.ToolSpec{Name: "sleepy", InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(ctx context.Context, in struct{ N int }) (agentflow.ToolOutput, error) {
			time.Sleep(time.Duration(rand.IntN(3)) * time.Millisecond)
			// Effects inside a tool are nested and buffered (R7).
			v, err := agentflow.Do(ctx, "lookup", in.N, func(_ context.Context, n int) (int, error) { return n * 10, nil })
			if err != nil {
				return agentflow.ToolOutput{}, err
			}
			return agentflow.TextOutput(fmt.Sprint(v)), nil
		})
	var reference []string
	for run := 0; run < 50; run++ {
		m := agentflowtest.NewScriptedModel(
			agentflowtest.CallTools(
				agentflowtest.Use("a", "sleepy", map[string]int{"N": 1}),
				agentflowtest.Use("b", "sleepy", map[string]int{"N": 2}),
				agentflowtest.Use("c", "sleepy", map[string]int{"N": 3}),
			),
			agentflowtest.Reply("done"),
		)
		res, err := newAgent(t, m, func(c *agentflow.AgentConfig) {
			c.Tools = []agentflow.Tool{sleepy}
			c.MaxParallelTools = 3
		}).Run(context.Background(), "go")
		if err != nil {
			t.Fatal(err)
		}
		var shape []string
		for _, e := range events(t, res) {
			s := string(e.Kind())
			switch b := e.Body.(type) {
			case *trace.ToolStarted:
				s += ":" + b.CallID
			case *trace.ToolCompleted:
				s += ":" + b.CallID + "=" + llm.TextOf(b.Content)
			case *trace.EffectCompleted:
				s += ":" + string(b.Output)
			}
			if e.Cause != nil {
				s += fmt.Sprintf("<-%d", *e.Cause)
			}
			shape = append(shape, s)
		}
		if reference == nil {
			reference = shape
			continue
		}
		if !reflect.DeepEqual(shape, reference) {
			t.Fatalf("run %d differs:\n%v\nvs\n%v", run, shape, reference)
		}
	}
	// All starts precede the batch's completions; nested effects precede
	// their tool's completion.
	joined := strings.Join(reference, " ")
	if !strings.Contains(joined, "tool.started:a<-3 tool.started:b<-3 tool.started:c<-3 effect.started<-5 effect.completed:10<-8 tool.completed:a=10<-5") {
		t.Errorf("unexpected batch layout:\n%s", strings.Join(reference, "\n"))
	}
	// And the second request contains the results in response order.
}

func TestToolFailuresAreRecordedAndSeenByTheModel(t *testing.T) {
	panicky := agentflow.NewTool(llm.ToolSpec{Name: "panicky"}, func(context.Context, struct{}) (agentflow.ToolOutput, error) {
		panic("kaboom")
	})
	failing := agentflow.NewTool(llm.ToolSpec{Name: "failing"}, func(context.Context, struct{}) (agentflow.ToolOutput, error) {
		return agentflow.ToolOutput{}, errors.New("disk full")
	})
	m := agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(
			agentflowtest.Use("1", "missing", map[string]any{}),
			agentflowtest.Use("2", "calc", map[string]any{"a": 1}), // fails schema validation
			agentflowtest.Use("3", "panicky", map[string]any{}),
			agentflowtest.Use("4", "failing", map[string]any{}),
		),
		agentflowtest.Reply("recovered"),
	)
	res, err := newAgent(t, m, func(c *agentflow.AgentConfig) {
		c.Tools = append(c.Tools, panicky, failing)
	}).Run(context.Background(), "try everything")
	if err != nil {
		t.Fatalf("tool failures must not fail the run: %v", err)
	}
	codes := map[string]string{}
	for _, e := range events(t, res) {
		if f, ok := e.Body.(*trace.ToolFailed); ok {
			codes[f.CallID] = f.Error.Code
		}
	}
	want := map[string]string{"1": trace.ErrCodeNotFound, "2": trace.ErrCodeInvalidInput, "3": trace.ErrCodePanic, "4": "error"}
	if !reflect.DeepEqual(codes, want) {
		t.Errorf("failure codes = %v, want %v", codes, want)
	}
	// The model's second request carries four error results.
	last := m.Requests()[1].Messages
	results := last[len(last)-1].Blocks
	if len(results) != 4 {
		t.Fatalf("got %d tool results", len(results))
	}
	for _, b := range results {
		if r := b.(llm.ToolResult); !r.IsError || !strings.HasPrefix(llm.TextOf(r.Content), "Error: ") {
			t.Errorf("result %s = %+v", r.CallID, r)
		}
	}
}

func TestModelFailureFailsTheRun(t *testing.T) {
	m := agentflowtest.NewScriptedModel().Then(agentflowtest.Step{Err: &llm.Error{Code: llm.ErrCodeRateLimited, Message: "slow down", Retryable: true}})
	res, err := newAgent(t, m, nil).Run(context.Background(), "hi")
	var re *agentflow.RunError
	if !errors.As(err, &re) || re.Status != trace.StatusFailed || res.Status != trace.StatusFailed {
		t.Fatalf("err = %v, status %v", err, res.Status)
	}
	evs := events(t, res)
	f, ok := evs[3].Body.(*trace.LLMFailed)
	if !ok || f.Error.Code != llm.ErrCodeRateLimited || !f.Error.Retryable {
		t.Fatalf("event 3 = %#v", evs[3].Body)
	}
	fin := evs[len(evs)-1].Body.(*trace.RunFinished)
	if fin.Status != trace.StatusFailed || fin.Error == nil {
		t.Errorf("run.finished = %+v", fin)
	}
}

func TestMaxStepsAndCancellation(t *testing.T) {
	loop := agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
		return agentflowtest.CallTools(agentflowtest.Use("x", "calc", calcIn{1, 1, "+"})), nil
	})
	res, err := newAgent(t, loop, func(c *agentflow.AgentConfig) { c.MaxSteps = 3 }).Run(context.Background(), "loop")
	if res.Status != trace.StatusMaxSteps || res.Steps != 3 || err == nil {
		t.Errorf("max steps: status %s steps %d err %v", res.Status, res.Steps, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	blocking := agentflowtest.ModelFunc(func(c context.Context, _ *llm.Request) (*llm.Response, error) {
		cancel()
		<-c.Done()
		return nil, c.Err()
	})
	res, err = newAgent(t, blocking, nil).Run(ctx, "hi")
	if res.Status != trace.StatusCancelled || err == nil {
		t.Errorf("cancel: status %s err %v", res.Status, err)
	}
	evs := events(t, res)
	if evs[len(evs)-1].Body.(*trace.RunFinished).Status != trace.StatusCancelled {
		t.Error("run.finished must be written even after cancellation")
	}
}

type memory struct{ calls int }

func (m *memory) Name() string { return "memory" }
func (m *memory) Provide(_ context.Context, conv []llm.Message) ([]trace.Ephemeral, error) {
	m.calls++
	return []trace.Ephemeral{{Index: 0, Message: llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{
		llm.Text{Text: fmt.Sprintf("memory #%d (conversation has %d messages)", m.calls, len(conv))}}}}}, nil
}

func TestContextProvidersAreRecordedEffects(t *testing.T) {
	m := agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("t1", "calc", calcIn{1, 2, "+"})),
		agentflowtest.Reply("3"),
	)
	res, err := newAgent(t, m, func(c *agentflow.AgentConfig) {
		c.ContextProviders = []agentflow.ContextProvider{&memory{}}
	}).Run(context.Background(), "1+2?")
	if err != nil {
		t.Fatal(err)
	}
	sent := m.Requests()
	if got := llm.TextOf(sent[1].Messages[0].Blocks); got != "memory #2 (conversation has 3 messages)" {
		t.Errorf("ephemeral not first in request 2: %q", got)
	}
	evs := events(t, res)
	n := 0
	for _, e := range evs {
		if es, ok := e.Body.(*trace.EffectStarted); ok && es.Name == "context.memory" {
			n++
			if strings.Contains(string(es.Input), "memory #") {
				t.Error("provider input must reference the conversation, not copy it")
			}
		}
		if ls, ok := e.Body.(*trace.LLMStarted); ok {
			folded, err := trace.Request(evs, e.Seq)
			if err != nil {
				t.Fatal(err)
			}
			if fp, _ := trace.RequestFingerprint(folded); fp != ls.Fingerprint {
				t.Errorf("seq %d: folded request with ephemerals doesn't match", e.Seq)
			}
		}
	}
	if n != 2 {
		t.Errorf("got %d context.memory effects, want 2", n)
	}
}

func TestInterceptorsSeeEveryEffect(t *testing.T) {
	var mu sync.Mutex
	var keys []string
	record := agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
		mu.Lock()
		keys = append(keys, c.Meta.Key)
		mu.Unlock()
		return next(ctx, c)
	})
	deny := agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
		if c.Kind == agentflow.EffectTool && c.Name == "calc" {
			return agentflow.Fail(trace.ErrCodeDenied, "calc is not allowed"), nil
		}
		return next(ctx, c)
	})
	m := agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("t1", "calc", calcIn{1, 2, "+"})),
		agentflowtest.Reply("ok"),
	)
	res, err := newAgent(t, m, func(c *agentflow.AgentConfig) {
		c.Interceptors = []agentflow.Interceptor{record, deny}
	}).Run(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(keys, []string{"llm#0", "tool:t1", "llm#1"}) {
		t.Errorf("interceptor saw %v", keys)
	}
	for _, e := range events(t, res) {
		if f, ok := e.Body.(*trace.ToolFailed); ok && f.Error.Code != trace.ErrCodeDenied {
			t.Errorf("denial recorded as %s", f.Error.Code)
		}
	}

	boom := agentflow.InterceptorFunc(func(context.Context, *agentflow.Call, agentflow.Invoker) (*agentflow.Outcome, error) {
		return nil, errors.New("interceptor broke")
	})
	res, err = newAgent(t, agentflowtest.NewScriptedModel(agentflowtest.Reply("x")), func(c *agentflow.AgentConfig) {
		c.Interceptors = []agentflow.Interceptor{boom}
	}).Run(context.Background(), "x")
	if err == nil || res.Status != trace.StatusFailed {
		t.Errorf("an interceptor error must abort the run: %v %v", err, res.Status)
	}
}

func TestToolTimeout(t *testing.T) {
	slow := agentflow.NewTool(llm.ToolSpec{Name: "slow"}, func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
		<-ctx.Done()
		return agentflow.ToolOutput{}, ctx.Err()
	})
	m := agentflowtest.NewScriptedModel(agentflowtest.CallTools(agentflowtest.Use("s", "slow", struct{}{})), agentflowtest.Reply("gave up"))
	res, err := newAgent(t, m, func(c *agentflow.AgentConfig) {
		c.Tools = []agentflow.Tool{slow}
		c.Timeouts = agentflow.Timeouts{PerTool: map[string]time.Duration{"slow": 20 * time.Millisecond}}
	}).Run(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, e := range events(t, res) {
		if f, ok := e.Body.(*trace.ToolFailed); ok && f.Error.Code == trace.ErrCodeTimeout {
			found = true
		}
	}
	if !found {
		t.Error("timeout not recorded as tool.failed{timeout}")
	}
}

func TestConcurrentEffectsAreRejected(t *testing.T) {
	var got error
	racy := agentflow.NewTool(llm.ToolSpec{Name: "racy"}, func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
		release := make(chan struct{})
		inside := make(chan struct{})
		go func() {
			_, _ = agentflow.Do(ctx, "slow", 0, func(context.Context, int) (int, error) {
				close(inside)
				<-release
				return 1, nil
			})
		}()
		<-inside
		_, got = agentflow.Do(ctx, "second", 0, func(context.Context, int) (int, error) { return 2, nil })
		close(release)
		return agentflow.TextOutput("ok"), nil
	})
	m := agentflowtest.NewScriptedModel(agentflowtest.CallTools(agentflowtest.Use("r", "racy", struct{}{})), agentflowtest.Reply("x"))
	if _, err := newAgent(t, m, func(c *agentflow.AgentConfig) { c.Tools = []agentflow.Tool{racy} }).Run(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(got, agentflow.ErrConcurrentEffect) {
		t.Errorf("second concurrent Do returned %v, want ErrConcurrentEffect", got)
	}
}

func TestDoOutsideARunJustCalls(t *testing.T) {
	v, err := agentflow.Do(context.Background(), "x", 2, func(_ context.Context, n int) (int, error) { return n * 3, nil })
	if err != nil || v != 6 {
		t.Errorf("Do outside a run = %v, %v", v, err)
	}
}

func TestObserversAndFinalOutput(t *testing.T) {
	var seen []trace.Kind
	obs := trace.ObserverFunc(func(_ context.Context, e trace.Event) { seen = append(seen, e.Kind()) })
	thinking := &llm.Response{Message: llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{
		llm.Reasoning{Text: "hmm", Ext: llm.Ext{"anthropic": json.RawMessage(`{"signature":"s"}`)}}, llm.Text{Text: "answer"}}}, StopReason: llm.StopEndTurn}
	res, err := newAgent(t, agentflowtest.NewScriptedModel(thinking), func(c *agentflow.AgentConfig) {
		c.Observers = []trace.Observer{obs}
	}).Run(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Output) != 1 || res.OutputText() != "answer" {
		t.Errorf("reasoning must not be part of the output: %#v", res.Output)
	}
	if len(seen) != len(events(t, res)) {
		t.Errorf("observer saw %d events, trace has %d", len(seen), len(events(t, res)))
	}
}

func TestNewAgentValidation(t *testing.T) {
	if _, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "x"}); !errors.Is(err, agentflow.ErrNoModel) {
		t.Errorf("no model: %v", err)
	}
	m := agentflowtest.NewScriptedModel()
	if _, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "x", Model: m, Tools: []agentflow.Tool{calcTool(), calcTool()}}); err == nil {
		t.Error("duplicate tools accepted")
	}
	if _, err := agentflow.NewAgent(agentflow.AgentConfig{Model: m}); err == nil {
		t.Error("empty name accepted")
	}
}
