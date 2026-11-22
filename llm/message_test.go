package llm_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/kumarlokesh/agentflow/llm"
)

func TestBlocksRoundTrip(t *testing.T) {
	msg := llm.Message{
		Role: llm.RoleAssistant,
		Blocks: llm.Blocks{
			llm.Reasoning{Text: "check the math", Ext: llm.Ext{"anthropic": json.RawMessage(`{"signature":"sig-abc"}`)}},
			llm.Text{Text: "Let me compute."},
			llm.ToolUse{CallID: "toolu_01", Name: "calculator", Input: json.RawMessage(`{"expression":"6*7"}`)},
			llm.Media{MediaType: "image/png", Data: []byte{0x89, 'P', 'N', 'G'}},
			llm.Data{Value: json.RawMessage(`{"answer":42}`)},
			llm.ToolResult{CallID: "toolu_01", Content: llm.Blocks{llm.Text{Text: "42"}}, IsError: true},
		},
	}

	raw, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var got llm.Message
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal %s: %v", raw, err)
	}
	if !reflect.DeepEqual(msg, got) {
		t.Fatalf("round trip mismatch\nwant %#v\ngot  %#v", msg, got)
	}

	// Re-encoding must be byte-identical.
	raw2, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != string(raw2) {
		t.Fatalf("re-encode differs\n%s\n%s", raw, raw2)
	}
}

func TestBlockJSONShape(t *testing.T) {
	raw, err := json.Marshal(llm.Text{Text: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(raw), `{"type":"text","text":"hi"}`; got != want {
		t.Errorf("got %s, want %s", got, want)
	}

	raw, err = json.Marshal(llm.ToolUse{CallID: "c1", Name: "noop"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(raw), `{"type":"tool_use","call_id":"c1","name":"noop","input":{}}`; got != want {
		t.Errorf("empty input: got %s, want %s", got, want)
	}
}

func TestUnknownBlockPreserved(t *testing.T) {
	in := `{"role":"assistant","blocks":[{"type":"hologram","frames":[1,2,3]},{"type":"text","text":"ok"}]}`
	var msg llm.Message
	if err := json.Unmarshal([]byte(in), &msg); err != nil {
		t.Fatal(err)
	}
	u, ok := msg.Blocks[0].(llm.Unknown)
	if !ok {
		t.Fatalf("block 0 is %T, want llm.Unknown", msg.Blocks[0])
	}
	if u.Type != "hologram" {
		t.Errorf("unknown type = %q", u.Type)
	}
	out, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != in {
		t.Fatalf("unknown block not preserved\nin  %s\nout %s", in, out)
	}
}

func TestBlockErrors(t *testing.T) {
	for _, in := range []string{
		`{"role":"user","blocks":[{"text":"no type"}]}`,
		`{"role":"user","blocks":{"type":"text"}}`,
	} {
		var msg llm.Message
		if err := json.Unmarshal([]byte(in), &msg); err == nil {
			t.Errorf("expected error for %s", in)
		}
	}
}

func TestNilBlocksEncodeAsEmptyArray(t *testing.T) {
	raw, err := json.Marshal(llm.Message{Role: llm.RoleUser})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(raw), `{"role":"user","blocks":[]}`; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	raw, err = json.Marshal(llm.ToolResult{CallID: "c"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(raw), `{"type":"tool_result","call_id":"c","content":[]}`; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestTextOfAndUsage(t *testing.T) {
	bs := llm.Blocks{llm.Text{Text: "a"}, llm.ToolUse{CallID: "x", Name: "y"}, llm.Text{Text: "b"}}
	if got := llm.TextOf(bs); got != "a\nb" {
		t.Errorf("TextOf = %q", got)
	}
	u := llm.Usage{InputTokens: 10, OutputTokens: 5}.Add(llm.Usage{InputTokens: 1, OutputTokens: 2, CacheReadTokens: 3})
	if u.InputTokens != 11 || u.OutputTokens != 7 || u.CacheReadTokens != 3 || u.Total() != 18 {
		t.Errorf("usage = %+v", u)
	}
}
