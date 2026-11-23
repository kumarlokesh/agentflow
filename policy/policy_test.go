package policy_test

import (
	"context"
	"encoding/json"
	"errors"
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

func echo(name string) agentflow.Tool {
	return agentflow.NewTool(llm.ToolSpec{Name: name, InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(context.Context, struct{}) (agentflow.ToolOutput, error) { return agentflow.TextOutput(name), nil })
}

func agent(t *testing.T, m llm.Model, is ...agentflow.Interceptor) *agentflow.Agent {
	t.Helper()
	a, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "a", Model: m, Tools: []agentflow.Tool{echo("read"), echo("delete")}, Interceptors: is})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func failures(t *testing.T, res *agentflow.RunResult) map[string]string {
	t.Helper()
	evs, err := res.Events(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	for _, e := range evs {
		if f, ok := e.Body.(*trace.ToolFailed); ok {
			out[f.CallID] = f.Error.Code
		}
	}
	return out
}

func TestPermissions(t *testing.T) {
	m := agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("1", "read", struct{}{}), agentflowtest.Use("2", "delete", struct{}{})),
		agentflowtest.Reply("ok"))
	res, err := agent(t, m, policy.Permissions{Deny: []string{"delete"}}).Run(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if got := failures(t, res); len(got) != 1 || got["2"] != trace.ErrCodeDenied {
		t.Errorf("deny: %v", got)
	}

	m = agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("1", "read", struct{}{}), agentflowtest.Use("2", "delete", struct{}{})),
		agentflowtest.Reply("ok"))
	res, err = agent(t, m, policy.Permissions{Allow: []string{"read"}}).Run(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if got := failures(t, res); len(got) != 1 || got["2"] != trace.ErrCodeDenied {
		t.Errorf("allow-list: %v", got)
	}
}

// The prototype's cost tracker never blocked a model call; this budget does.
func TestBudgetBlocksModelCalls(t *testing.T) {
	loop := agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
		return agentflowtest.CallTools(agentflowtest.Use("x", "read", struct{}{})), nil // 15 tokens each
	})
	budget := &policy.Budget{MaxTotalTokens: 40}
	store := memstore.New()
	res, err := agent(t, loop, budget).Run(context.Background(), "x", agentflow.WithStore(store))
	var re *agentflow.RunError
	if !errors.As(err, &re) || re.Status != trace.StatusFailed {
		t.Fatalf("err = %v", err)
	}
	if res.Steps != 4 || budget.Used().Total() != 45 {
		t.Errorf("steps %d, used %d: want the 4th call blocked after 45 tokens", res.Steps, budget.Used().Total())
	}

	// In replay the budget sees the recorded usage and blocks at the same call.
	replayBudget := &policy.Budget{MaxTotalTokens: 40}
	rep, err := replay.Run(context.Background(), agent(t, agentflowtest.NewScriptedModel(), replayBudget), store, res.TraceID, replay.Options{})
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if rep.Status != replay.Identical {
		t.Fatalf("replay diverged: %v", rep.Divergences)
	}
	if replayBudget.Used().Total() != 45 {
		t.Errorf("replayed budget used %d", replayBudget.Used().Total())
	}
}

func TestRateLimitWaitsLiveAndIsSkippedInReplay(t *testing.T) {
	rl := &policy.RateLimit{Max: 1, Window: 60 * time.Millisecond}
	m := agentflowtest.NewScriptedModel(
		agentflowtest.CallTools(agentflowtest.Use("1", "read", struct{}{})),
		agentflowtest.Reply("ok"))
	store := memstore.New()
	start := time.Now()
	res, err := agent(t, m, rl).Run(context.Background(), "x", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) < 50*time.Millisecond {
		t.Error("second model call wasn't delayed")
	}
	slow := &policy.RateLimit{Max: 1, Window: time.Hour}
	start = time.Now()
	rep, err := replay.Run(context.Background(), agent(t, agentflowtest.NewScriptedModel(), slow), store, res.TraceID, replay.Options{})
	if err != nil || rep.Status != replay.Identical {
		t.Fatalf("replay: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Error("rate limit applied during replay")
	}
}
