package replay_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

type calcIn struct {
	A, B float64
	Op   string
}

var calcSchema = `{"type":"object","properties":{"A":{"type":"number"},"B":{"type":"number"},"Op":{"type":"string"}},"required":["A","B","Op"]}`

func calc(schema string) agentflow.Tool {
	return agentflow.NewTool(llm.ToolSpec{Name: "calc", Description: "Arithmetic.", InputSchema: json.RawMessage(schema)},
		func(_ context.Context, in calcIn) (agentflow.ToolOutput, error) {
			if in.Op == "*" {
				return agentflow.TextOutput(fmt.Sprint(in.A * in.B)), nil
			}
			return agentflow.TextOutput(fmt.Sprint(in.A + in.B)), nil
		})
}

type memory struct{ name string }

func (m memory) Name() string { return m.name }
func (m memory) Provide(_ context.Context, conv []llm.Message) ([]trace.Ephemeral, error) {
	return []trace.Ephemeral{{Index: 0, Message: llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "user prefers digits"}}}}}, nil
}

// spec describes an agent; tests mutate it to simulate code changes.
type spec struct {
	instructions string
	schema       string
	tools        bool
	temperature  *float64
	providers    []agentflow.ContextProvider
	maxSteps     int
	interceptors []agentflow.Interceptor
}

func baseline() spec {
	return spec{instructions: "You calculate.", schema: calcSchema, tools: true, providers: []agentflow.ContextProvider{memory{"memory"}}}
}

func (s spec) agent(t *testing.T, m llm.Model) *agentflow.Agent {
	t.Helper()
	cfg := agentflow.AgentConfig{Name: "calculator", Model: m, ModelName: "test-model", Instructions: s.instructions,
		ContextProviders: s.providers, MaxSteps: s.maxSteps, Interceptors: s.interceptors}
	if s.tools {
		cfg.Tools = []agentflow.Tool{calc(s.schema)}
	}
	if s.temperature != nil {
		cfg.Params.Temperature = s.temperature
	}
	a, err := agentflow.NewAgent(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func script() *agentflowtest.ScriptedModel {
	return agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("t1", "calc", calcIn{6, 7, "*"})),
		agentflowtest.Reply("6 × 7 = 42"),
	)
}

// record runs the baseline live and returns the store and trace ID.
func record(t *testing.T, s spec) (trace.Store, string) {
	t.Helper()
	store := memstore.New()
	res, err := s.agent(t, script()).Run(context.Background(), "What is 6*7?", agentflow.WithStore(store))
	if err != nil && res == nil {
		t.Fatal(err)
	}
	return store, res.TraceID
}

// noModel fails the test if the model is ever called.
func noModel(t *testing.T) llm.Model {
	return agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
		t.Error("model called during replay")
		return nil, errors.New("no model in strict replay")
	})
}

func TestIdenticalReplayMakesNoCalls(t *testing.T) {
	store, src := record(t, baseline())
	res, err := replay.Run(context.Background(), baseline().agent(t, noModel(t)), store, src, replay.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != replay.Identical || len(res.Divergences) != 0 {
		t.Fatalf("status %s, divergences %v", res.Status, res.Divergences)
	}
	if res.Run.OutputText() != "6 × 7 = 42" {
		t.Errorf("output = %q", res.Run.OutputText())
	}
	// The replay's trace equals the source event by event, ignoring time and IDs.
	a := normalized(t, store, src)
	b := normalized(t, store, res.TraceID)
	if len(a) != len(b) {
		t.Fatalf("source has %d events, replay %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("event %d differs:\n%s\n%s", i, a[i], b[i])
		}
	}
	// Its origin says what it is.
	for info, err := range store.Runs(context.Background(), trace.RunFilter{TraceID: res.TraceID}) {
		if err != nil {
			t.Fatal(err)
		}
		if info.Origin == nil || info.Origin.Kind != trace.OriginReplay || info.Origin.SourceTraceID != src || info.Origin.Mode != "strict" {
			t.Errorf("origin = %+v", info.Origin)
		}
	}
}

// normalized encodes a trace's root-run events without time, IDs, and the
// root origin, which differ between a recording and its replay.
func normalized(t *testing.T, s trace.Store, traceID string) []string {
	t.Helper()
	var out []string
	for info, err := range s.Runs(context.Background(), trace.RunFilter{TraceID: traceID, RootsOnly: true}) {
		if err != nil {
			t.Fatal(err)
		}
		evs, err := trace.Collect(s.Events(context.Background(), info.RunID, 0))
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range evs {
			if rs, ok := e.Body.(*trace.RunStarted); ok {
				cp := *rs
				cp.Origin = nil
				e.Body = &cp
			}
			e.TraceID, e.RunID, e.Time = "", "", time.Time{}
			line, err := trace.EncodeEvent(e, trace.EncodeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			// Durations are wall-clock measurements.
			out = append(out, durationRE(string(line)))
		}
	}
	return out
}

func durationRE(s string) string {
	var b strings.Builder
	for {
		i := strings.Index(s, `"duration_ns":`)
		if i < 0 {
			b.WriteString(s)
			return b.String()
		}
		b.WriteString(s[:i+len(`"duration_ns":`)])
		s = s[i+len(`"duration_ns":`):]
		j := strings.IndexAny(s, ",}")
		b.WriteString("0")
		s = s[j:]
	}
}

func f64(v float64) *float64 { return &v }

// The RFC-0002 mutation suite.
func TestMutationsAreDetectedPrecisely(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*spec)
		task   *string
		kind   replay.DivergenceKind
		effect string
		path   string
	}{
		{"system prompt", func(s *spec) { s.instructions = "You calculate precisely." }, nil, replay.RequestMismatch, "llm#0", "system[0].text"},
		{"tool schema", func(s *spec) {
			s.schema = strings.Replace(calcSchema, `"Op":{"type":"string"}`, `"Op":{"type":"string","enum":["+","*"]}`, 1)
		}, nil, replay.RequestMismatch, "llm#0", "tools{calc}.input_schema.properties.Op.enum"},
		{"tool removed", func(s *spec) { s.tools = false }, nil, replay.RequestMismatch, "llm#0", "tools{calc}"},
		{"temperature", func(s *spec) { s.temperature = f64(0) }, nil, replay.RequestMismatch, "llm#0", "params.temperature"},
		{"context provider added", func(s *spec) {
			s.providers = append(s.providers, memory{"docs"})
		}, nil, replay.UnexpectedEffect, "effect:context.docs#0", ""},
		// The first effect of the new run is llm#0, whose request lacks the
		// injected message: that is the earliest difference.
		{"context provider removed", func(s *spec) { s.providers = nil }, nil, replay.RequestMismatch, "llm#0", "messages[0]"},
		{"task (provider sees a different conversation)", func(*spec) {}, ptr("What is 6*8?"), replay.RequestMismatch, "effect:context.memory#0", "input.conversation"},
		{"stops one call earlier", func(s *spec) { s.maxSteps = 1 }, nil, replay.MissingEffect, "effect:context.memory#1", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			store, src := record(t, baseline())
			s := baseline()
			c.mutate(&s)
			res, err := replay.Run(context.Background(), s.agent(t, noModel(t)), store, src, replay.Options{Task: c.task})
			var de *replay.DivergenceError
			if !errors.As(err, &de) {
				t.Fatalf("err = %v, want *DivergenceError", err)
			}
			if !errors.Is(err, agentflow.ErrDiverged) {
				t.Error("DivergenceError must wrap agentflow.ErrDiverged")
			}
			d := res.Divergences[0]
			if d.Kind != c.kind || d.Effect != c.effect || d.Path != "/" {
				t.Fatalf("divergence = %s at %s in %s, want %s at %s\n%s", d.Kind, d.Effect, d.Path, c.kind, c.effect, d.Summary)
			}
			if c.path != "" {
				found := false
				for _, f := range d.Diffs {
					if f.Path == c.path {
						found = true
					}
				}
				if !found {
					t.Fatalf("no diff at %s:\n%s", c.path, d.Summary)
				}
			}
			if res.Run.Status != trace.StatusDiverged || res.Status != replay.Diverged {
				t.Errorf("run status %s, replay status %s", res.Run.Status, res.Status)
			}
		})
	}
}

func ptr(s string) *string { return &s }

func TestExtraCallIsUnexpected(t *testing.T) {
	short := baseline()
	short.maxSteps = 1 // the recording stops after one model call (max_steps)
	store, src := record(t, short)
	_, err := replay.Run(context.Background(), baseline().agent(t, noModel(t)), store, src, replay.Options{})
	var de *replay.DivergenceError
	if !errors.As(err, &de) || de.Divergence.Kind != replay.UnexpectedEffect || de.Divergence.Effect != "effect:context.memory#1" {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(de.Divergence.Summary, "the recorded run had no more effects") {
		t.Errorf("summary should say the recording is exhausted:\n%s", de.Divergence.Summary)
	}
}

func TestIgnorePatterns(t *testing.T) {
	store, src := record(t, baseline())
	s := baseline()
	s.temperature = f64(0.5)
	res, err := replay.Run(context.Background(), s.agent(t, noModel(t)), store, src, replay.Options{Ignore: []string{"params.temperature"}})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("ignored difference still diverged: %v %v", err, res.Divergences)
	}
}

func TestGoLiveSwitchesOnceAndCallsTheModel(t *testing.T) {
	store, src := record(t, baseline())
	s := baseline()
	s.instructions = "You calculate precisely."
	live := script()
	res, err := replay.Run(context.Background(), s.agent(t, live), store, src, replay.Options{OnDivergence: replay.GoLive})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != replay.Diverged || len(res.Divergences) != 1 || res.Run.Status != trace.StatusCompleted {
		t.Fatalf("status %s/%s, %d divergences", res.Status, res.Run.Status, len(res.Divergences))
	}
	if live.Calls() != 2 {
		t.Errorf("live model called %d times, want 2", live.Calls())
	}
	evs, err := res.Run.Events(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	switches := 0
	for _, e := range evs {
		if mc, ok := e.Body.(*trace.RunModeChanged); ok {
			switches++
			if mc.Reason != trace.ReasonDivergence || !strings.Contains(string(mc.Detail), "system[0].text") {
				t.Errorf("mode change = %+v %s", mc, mc.Detail)
			}
		}
	}
	if switches != 1 {
		t.Errorf("got %d run.mode_changed events, want 1", switches)
	}
}

func TestIgnoreModeServesRecordingButReportsDivergence(t *testing.T) {
	store, src := record(t, baseline())
	s := baseline()
	s.instructions = "Something else entirely."
	res, err := replay.Run(context.Background(), s.agent(t, noModel(t)), store, src, replay.Options{OnDivergence: replay.Ignore})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != replay.Diverged || res.Run.OutputText() != "6 × 7 = 42" {
		t.Fatalf("status %s, output %q", res.Status, res.Run.OutputText())
	}
	if len(res.Divergences) != 2 { // both model calls see the new system prompt
		t.Errorf("got %d divergences, want 2", len(res.Divergences))
	}
}

func TestPolicyChangesAreCaught(t *testing.T) {
	deny := agentflow.InterceptorFunc(func(ctx context.Context, c *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
		if c.Kind == agentflow.EffectTool {
			return agentflow.Fail(trace.ErrCodeDenied, "no tools today"), nil
		}
		return next(ctx, c)
	})
	withDeny := baseline()
	withDeny.interceptors = []agentflow.Interceptor{deny}
	store, src := record(t, withDeny)

	// Same policy: the denial replays identically (no false missing_effect).
	res, err := replay.Run(context.Background(), withDeny.agent(t, noModel(t)), store, src, replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("same policy: %v %v", err, res.Divergences)
	}
	// Policy removed: the call now reaches the world, but the recording only
	// has the interceptor's denial, which must not be served as a world result.
	_, err = replay.Run(context.Background(), baseline().agent(t, noModel(t)), store, src, replay.Options{})
	var de *replay.DivergenceError
	if !errors.As(err, &de) || de.Divergence.Kind != replay.UnexpectedEffect || de.Divergence.Effect != "tool:calc(t1)" {
		t.Fatalf("policy removed: %v", err)
	}
	if !strings.Contains(de.Divergence.Summary, "an interceptor answered this call (denied: no tools today)") {
		t.Errorf("summary:\n%s", de.Divergence.Summary)
	}
	// Policy added: the interceptor now denies a call that reached the world
	// in the recording; the tool result differs, so the next request differs.
	plain, plainSrc := record(t, baseline())
	_, err = replay.Run(context.Background(), withDeny.agent(t, noModel(t)), plain, plainSrc, replay.Options{})
	if !errors.As(err, &de) || de.Divergence.Effect != "effect:context.memory#1" {
		t.Fatalf("policy added: %v", err)
	}
}

func TestRecordedFailureReplaysIdentically(t *testing.T) {
	store := memstore.New()
	failing := agentflowtest.NewScriptedModel().Then(agentflowtest.Step{Err: &llm.Error{Code: llm.ErrCodeOverloaded, Message: "busy"}})
	res, err := baseline().agent(t, failing).Run(context.Background(), "x", agentflow.WithStore(store))
	if err == nil {
		t.Fatal("expected the recorded run to fail")
	}
	rep, err := replay.Run(context.Background(), baseline().agent(t, noModel(t)), store, res.TraceID, replay.Options{})
	if err != nil || rep.Status != replay.Identical || rep.Run.Status != trace.StatusFailed {
		t.Fatalf("err %v, status %s/%s", err, rep.Status, rep.Run.Status)
	}
}

func TestNestedEffectsAreCoveredByTheirToolCall(t *testing.T) {
	var lookups atomic.Int32
	tool := agentflow.NewTool(llm.ToolSpec{Name: "lookup"}, func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
		v, err := agentflow.Do(ctx, "db.get", "k", func(context.Context, string) (string, error) {
			lookups.Add(1)
			return "value", nil
		})
		if err != nil {
			return agentflow.ToolOutput{}, err
		}
		return agentflow.TextOutput(v), nil
	})
	mk := func(m llm.Model) *agentflow.Agent {
		a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "a", Model: m, Tools: []agentflow.Tool{tool}})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	store := memstore.New()
	m := agentflowtest.NewScriptedModel(agentflowtest.CallTools(agentflowtest.Use("1", "lookup", struct{}{})), agentflowtest.Reply("done"))
	res, err := mk(m).Run(context.Background(), "x", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	if lookups.Load() != 1 {
		t.Fatal("tool didn't run live")
	}
	rep, err := replay.Run(context.Background(), mk(noModel(t)), store, res.TraceID, replay.Options{})
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if rep.Status != replay.Identical {
		t.Fatalf("divergences: %v", rep.Divergences)
	}
	if lookups.Load() != 1 {
		t.Error("the tool (and its nested effect) ran during replay")
	}
}

func TestRunBundleAndLiveGuard(t *testing.T) {
	store, src := record(t, baseline())
	var bundle bytes.Buffer
	if err := trace.Export(context.Background(), &bundle, store, src, trace.BundleOptions{}); err != nil {
		t.Fatal(err)
	}
	res, err := replay.RunBundle(context.Background(), baseline().agent(t, noModel(t)), bytes.NewReader(bundle.Bytes()), replay.Options{})
	if err != nil || res.Status != replay.Identical {
		t.Fatalf("bundle replay: %v", err)
	}
}

func TestSummaryIsReadable(t *testing.T) {
	store, src := record(t, baseline())
	s := baseline()
	s.instructions = "You calculate precisely."
	_, err := replay.Run(context.Background(), s.agent(t, noModel(t)), store, src, replay.Options{})
	want := "DIVERGED in / at llm#0 (step 0) — request_mismatch\n  system[0].text changed:\n    - \"You calculate.\"\n    + \"You calculate precisely.\""
	if err == nil || err.Error() != want {
		t.Errorf("summary:\n%v\nwant:\n%s", err, want)
	}
}
