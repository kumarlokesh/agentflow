package replay_test

import (
	"context"
	"errors"
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

// shop is a small agent for verification tests. Its fields are the "code"
// that tests change between recording and replay.
type shop struct {
	status     string // what lookup reports
	policy     string // what the context provider injects
	clockCalls int    // how many times ticket reads the clock
	parallel   int
	deny       string // if set, an interceptor denies ticket with this message

	lookups, tickets *atomic.Int32 // executions, to check what ran
}

func newShop() shop {
	return shop{status: "shipped", policy: "refunds within 30 days", clockCalls: 1, lookups: new(atomic.Int32), tickets: new(atomic.Int32)}
}

type note struct{ text string }

func (n note) Name() string { return "policy" }
func (n note) Provide(context.Context, []llm.Message) ([]trace.Ephemeral, error) {
	return []trace.Ephemeral{{Index: 0, Message: llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: n.text}}}}}, nil
}

func (s shop) agent(t *testing.T, m llm.Model) *agentflow.Agent {
	t.Helper()
	lookup := agentflow.NewTool(llm.ToolSpec{Name: "lookup", Annotations: &llm.ToolAnnotations{ReadOnly: true}},
		func(context.Context, struct{}) (agentflow.ToolOutput, error) {
			s.lookups.Add(1)
			return agentflow.TextOutput("order A-1: " + s.status), nil
		})
	// ticket has side effects (not ReadOnly) and its own nondeterminism.
	ticket := agentflow.NewTool(llm.ToolSpec{Name: "ticket"},
		func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
			s.tickets.Add(1)
			id := agentflow.NewID(ctx)
			var at string
			for range s.clockCalls {
				at = agentflow.Now(ctx).Format("15:04:05.000000000")
			}
			return agentflow.TextOutput("ticket " + id + " at " + at), nil
		})
	cfg := agentflow.AgentConfig{Name: "shop", Model: m, ModelName: "test-model", Tools: []agentflow.Tool{lookup, ticket},
		ContextProviders: []agentflow.ContextProvider{note{s.policy}}, MaxParallelTools: s.parallel}
	if s.deny != "" {
		msg := s.deny
		cfg.Interceptors = []agentflow.Interceptor{agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
			if c.Name == "ticket" {
				return agentflow.Fail(trace.ErrCodeDenied, msg), nil
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

func shopScript() *agentflowtest.ScriptedModel {
	return agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("1", "lookup", struct{}{}), agentflowtest.Use("2", "ticket", struct{}{})),
		agentflowtest.Reply("done"),
	)
}

func recordShop(t *testing.T, s shop) (trace.Store, string) {
	t.Helper()
	store := memstore.New()
	res, err := s.agent(t, shopScript()).Run(context.Background(), "where is my order?", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	return store, res.TraceID
}

func replayShop(t *testing.T, s shop, store trace.Store, src string, opts replay.Options) (*replay.Result, *replay.DivergenceError, error) {
	t.Helper()
	res, err := replay.Run(context.Background(), s.agent(t, noModel(t)), store, src, opts)
	var de *replay.DivergenceError
	if errors.As(err, &de) {
		return res, de, nil
	}
	return res, nil, err
}

func hasDiff(d replay.Divergence, path string) bool {
	for _, f := range d.Diffs {
		if f.Path == path {
			return true
		}
	}
	return false
}

func TestVerifyCatchesToolImplementationChange(t *testing.T) {
	store, src := recordShop(t, newShop())
	changed := newShop()
	changed.status = "delivered"

	// Without Verify the change is invisible: the outcome comes from the tape.
	res, de, err := replayShop(t, changed, store, src, replay.Options{})
	if err != nil || de != nil || res.Status != replay.Identical {
		t.Fatalf("no verify: %v %v", err, de)
	}
	if changed.lookups.Load() != 0 {
		t.Error("lookup ran without Verify")
	}

	for _, pattern := range []string{"tool:lookup", "tool:*", "tool:look*"} {
		_, de, err = replayShop(t, changed, store, src, replay.Options{Verify: []string{pattern}})
		if err != nil || de == nil {
			t.Fatalf("%s: err %v, divergence %v", pattern, err, de)
		}
		d := de.Divergence
		if d.Kind != replay.OutcomeMismatch || d.Effect != "tool:lookup(1)" || !hasDiff(d, "outcome.content[0].text") {
			t.Fatalf("%s: %s", pattern, d.Summary)
		}
		if !strings.Contains(d.Summary, "verified") {
			t.Errorf("summary: %s", d.Summary)
		}
	}
}

func TestVerifyUnchangedCodeIsIdentical(t *testing.T) {
	store, src := recordShop(t, newShop())
	s := newShop()
	res, de, err := replayShop(t, s, store, src, replay.Options{Verify: []string{"tool:lookup", "tool:ticket", "effect:context.*"}})
	if err != nil || de != nil || res.Status != replay.Identical {
		t.Fatalf("err %v, divergence %v", err, de)
	}
	// ticket ran, and its NewID and Now were answered from the recording.
	if s.lookups.Load() != 1 || s.tickets.Load() != 1 {
		t.Errorf("executions: lookup %d, ticket %d", s.lookups.Load(), s.tickets.Load())
	}
	want := []string{"effect:context.policy#0", "tool:lookup(1)", "tool:ticket(2)", "effect:context.policy#1"}
	if !slices.Equal(res.Verified, want) {
		t.Errorf("Verified = %v, want %v", res.Verified, want)
	}
}

func TestVerifyContextProvider(t *testing.T) {
	store, src := recordShop(t, newShop())
	changed := newShop()
	changed.policy = "refunds within 14 days"
	res, de, err := replayShop(t, changed, store, src, replay.Options{})
	if err != nil || de != nil || res.Status != replay.Identical {
		t.Fatalf("no verify: %v %v", err, de)
	}
	_, de, err = replayShop(t, changed, store, src, replay.Options{Verify: []string{"effect:context.*"}})
	if err != nil || de == nil {
		t.Fatalf("err %v, divergence %v", err, de)
	}
	if d := de.Divergence; d.Kind != replay.OutcomeMismatch || d.Effect != "effect:context.policy#0" || !hasDiff(d, "outcome.output[0].message.blocks[0].text") {
		t.Fatalf("%s", d.Summary)
	}
}

func TestVerifiedToolNestedEffects(t *testing.T) {
	store, src := recordShop(t, newShop())
	opts := replay.Options{Verify: []string{"tool:ticket"}}

	twice := newShop()
	twice.clockCalls = 2
	_, de, err := replayShop(t, twice, store, src, opts)
	if err != nil || de == nil {
		t.Fatalf("extra nested effect: err %v", err)
	}
	if d := de.Divergence; d.Kind != replay.UnexpectedEffect || d.Effect != "tool:2/effect:clock.now#1" {
		t.Fatalf("extra nested effect: %s", d.Summary)
	}

	never := newShop()
	never.clockCalls = 0
	_, de, err = replayShop(t, never, store, src, opts)
	if err != nil || de == nil {
		t.Fatalf("missing nested effect: err %v", err)
	}
	if d := de.Divergence; d.Kind != replay.MissingEffect || d.Effect != "tool:2/effect:clock.now#0" {
		t.Fatalf("missing nested effect: %s", d.Summary)
	}
}

func TestVerifyWildcardSkipsToolsThatAreNotReadOnly(t *testing.T) {
	store, src := recordShop(t, newShop())
	s := newShop()
	res, de, err := replayShop(t, s, store, src, replay.Options{Verify: []string{"tool:*"}})
	if err != nil || de != nil || res.Status != replay.Identical {
		t.Fatalf("err %v, divergence %v", err, de)
	}
	if s.tickets.Load() != 0 {
		t.Error("a wildcard executed a tool that isn't ReadOnly")
	}
	if !slices.Equal(res.Unverified, []string{"ticket"}) || !slices.Equal(res.Verified, []string{"tool:lookup(1)"}) {
		t.Errorf("Verified %v, Unverified %v", res.Verified, res.Unverified)
	}
}

func TestVerifyRejectsBadPatterns(t *testing.T) {
	store, src := recordShop(t, newShop())
	for _, p := range []string{"effect:clock.now", "effect:id.new", "llm#0", "tool:", "lookup"} {
		s := newShop()
		if _, err := replay.Run(context.Background(), s.agent(t, noModel(t)), store, src, replay.Options{Verify: []string{p}}); err == nil {
			t.Errorf("%q: no error", p)
		}
	}
	// Wildcards never select the clock or ID effects.
	s := newShop()
	res, de, err := replayShop(t, s, store, src, replay.Options{Verify: []string{"tool:ticket", "effect:*"}})
	if err != nil || de != nil || res.Status != replay.Identical {
		t.Fatalf("err %v, divergence %v", err, de)
	}
}

func TestVerifyMismatchGoLiveAndIgnore(t *testing.T) {
	store, src := recordShop(t, newShop())
	changed := newShop()
	changed.status = "delivered"

	// GoLive: the new outcome is used, the switch precedes the completion,
	// and the next model call is live.
	live := agentflowtest.NewScriptedModel(agentflowtest.Reply("it was delivered"))
	res, err := replay.Run(context.Background(), changed.agent(t, live), store, src, replay.Options{Verify: []string{"tool:lookup"}, OnDivergence: replay.GoLive})
	if err != nil {
		t.Fatal(err)
	}
	if live.Calls() != 1 || res.Run.OutputText() != "it was delivered" {
		t.Fatalf("live calls %d, output %q", live.Calls(), res.Run.OutputText())
	}
	evs, err := res.Run.Events(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, e := range evs {
		switch b := e.Body.(type) {
		case *trace.RunModeChanged:
			kinds = append(kinds, "switch")
		case *trace.ToolCompleted:
			kinds = append(kinds, "completed:"+llm.TextOf(b.Content))
		}
	}
	want := []string{"switch", "completed:order A-1: delivered"}
	if len(kinds) < 2 || !slices.Equal(kinds[:2], want) {
		t.Errorf("events %v, want %v first", kinds, want)
	}

	// Ignore: the recorded outcome is used and the divergence is reported.
	res, err = replay.Run(context.Background(), changed.agent(t, noModel(t)), store, src, replay.Options{Verify: []string{"tool:lookup"}, OnDivergence: replay.Ignore})
	if err != nil || res.Status != replay.Diverged || len(res.Divergences) != 1 {
		t.Fatalf("ignore: %v %v", err, res.Divergences)
	}
	evs, err = res.Run.Events(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range evs {
		if b, ok := e.Body.(*trace.ToolCompleted); ok && b.CallID == "1" && llm.TextOf(b.Content) != "order A-1: shipped" {
			t.Errorf("ignore used %q, want the recorded outcome", llm.TextOf(b.Content))
		}
	}
}

func TestInterceptorAnsweringDifferentlyIsCaught(t *testing.T) {
	rec := newShop()
	rec.deny = "no tickets on weekends"
	store, src := recordShop(t, rec)

	same := newShop()
	same.deny = "no tickets on weekends"
	if res, de, err := replayShop(t, same, store, src, replay.Options{}); err != nil || de != nil || res.Status != replay.Identical {
		t.Fatalf("same policy: %v %v", err, de)
	}
	other := newShop()
	other.deny = "no tickets today"
	_, de, err := replayShop(t, other, store, src, replay.Options{})
	if err != nil || de == nil {
		t.Fatalf("err %v", err)
	}
	if d := de.Divergence; d.Kind != replay.OutcomeMismatch || d.Effect != "tool:ticket(2)" || !strings.Contains(d.Summary, "answered it differently") {
		t.Fatalf("%s", d.Summary)
	}
}

func TestVerifyParallelBatchIsDeterministic(t *testing.T) {
	rec := newShop()
	rec.parallel = 4
	store, src := recordShop(t, rec)
	for i := range 100 {
		s := newShop()
		s.parallel = 4
		res, de, err := replayShop(t, s, store, src, replay.Options{Verify: []string{"tool:lookup", "tool:ticket"}})
		if err != nil || de != nil || res.Status != replay.Identical {
			t.Fatalf("run %d: %v %v", i, err, de)
		}
	}
}
