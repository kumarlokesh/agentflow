package replay

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// trace.Effects (the spec's labels) and the replay tape's keys agree:
// nested effects, repeated call IDs, delegations.
func TestEffectLabelsMatchTapeKeys(t *testing.T) {
	helper, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "helper", ModelName: "m", Model: &script{reply("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	stamp := agentflow.NewTool(llm.ToolSpec{Name: "stamp"}, func(ctx context.Context, _ struct{}) (agentflow.ToolOutput, error) {
		agentflow.Now(ctx)
		agentflow.Now(ctx)
		agentflow.Expect(ctx, "stamped", true, "")
		return agentflow.TextOutput(agentflow.NewID(ctx)), nil
	})
	m := &script{
		calls(use("a", "stamp", `{}`), use("b", "helper", `{"task":"x"}`)),
		calls(use("a", "stamp", `{}`)), // a repeated call ID
		reply("done")}
	lead, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "lead", ModelName: "m", Model: m, MaxParallelTools: 2,
		Tools: []agentflow.Tool{stamp, agentflow.Delegate(helper, agentflow.DelegateOptions{})}})
	if err != nil {
		t.Fatal(err)
	}
	store := memstore.New()
	res, err := lead.Run(context.Background(), "go", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	events, err := res.Events(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	tp, err := buildTape(events, store)
	if err != nil {
		t.Fatal(err)
	}
	var keys, labels []string
	for _, e := range tp.entries {
		keys = append(keys, e.key)
	}
	for _, r := range trace.Effects(events) {
		labels = append(labels, r.Label)
		if r.End == nil {
			t.Errorf("%s unanswered", r.Label)
		}
	}
	if !slices.Equal(keys, labels) {
		t.Fatalf("tape %v\neffects %v", keys, labels)
	}
	want := []string{"llm#0", "tool:a", "tool:b", "tool:a/effect:clock.now#0", "tool:a/effect:clock.now#1", "tool:a/effect:id.new#0",
		"tool:b/spawn:/helper#0", "llm#1", "tool:a#1", "tool:a#1/effect:clock.now#0", "tool:a#1/effect:clock.now#1", "tool:a#1/effect:id.new#0", "llm#2"}
	if !slices.Equal(labels, want) {
		t.Fatalf("labels %v", labels)
	}
	for _, r := range trace.Effects(events) {
		if r.Label == "tool:b" && (r.Step != 0 || r.Position != 1) {
			t.Fatalf("tool:b step %d position %d", r.Step, r.Position)
		}
	}
}

// script answers with responses in order (agentflowtest imports replay).
type script []*llm.Response

func (s *script) Generate(context.Context, *llm.Request) (*llm.Response, error) {
	r := (*s)[0]
	*s = (*s)[1:]
	return r, nil
}

func reply(text string) *llm.Response {
	return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{llm.Text{Text: text}}}, StopReason: llm.StopEndTurn}
}

func use(id, name, input string) llm.ToolUse {
	return llm.ToolUse{CallID: id, Name: name, Input: json.RawMessage(input)}
}

func calls(us ...llm.ToolUse) *llm.Response {
	bs := llm.Blocks{}
	for _, u := range us {
		bs = append(bs, u)
	}
	return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant, Blocks: bs}, StopReason: llm.StopToolUse}
}
