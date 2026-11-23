package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
)

// fakeAPI records the last request body and answers with a fixed response.
func fakeAPI(t *testing.T, status int, response string) (*anthropic.Client, *[]byte) {
	t.Helper()
	var last []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("x-api-key") != "k" || r.Header.Get("anthropic-version") == "" {
			t.Errorf("bad request: %s %v", r.URL.Path, r.Header)
		}
		last, _ = io.ReadAll(r.Body)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)
	return anthropic.New("k", anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-5-5")), &last
}

const thinkingResponse = `{
  "id": "msg_01", "model": "claude-opus-5-5", "stop_reason": "tool_use",
  "content": [
    {"type": "thinking", "thinking": "I should multiply.", "signature": "EqQBCkYIBxgCKkA="},
    {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4a"},
    {"type": "text", "text": "Let me compute that."},
    {"type": "tool_use", "id": "toolu_01", "name": "calc", "input": {"expression": "6*7"}},
    {"type": "server_widget", "spin": true}
  ],
  "usage": {"input_tokens": 120, "output_tokens": 40, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7}
}`

func TestResponseMapping(t *testing.T) {
	c, _ := fakeAPI(t, 200, thinkingResponse)
	resp, err := c.Generate(context.Background(), &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "6*7?"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	bs := resp.Message.Blocks
	if len(bs) != 5 {
		t.Fatalf("got %d blocks", len(bs))
	}
	r := bs[0].(llm.Reasoning)
	if r.Text != "I should multiply." || string(r.Ext["anthropic"]) != `{"signature":"EqQBCkYIBxgCKkA="}` {
		t.Errorf("thinking = %+v %s", r, r.Ext["anthropic"])
	}
	if rr := bs[1].(llm.Reasoning); !rr.Redacted || string(rr.Ext["anthropic"]) != `{"data":"EmwKAhgBEgy3va3pzix/LafPsn4a"}` {
		t.Errorf("redacted = %+v", rr)
	}
	if u := bs[3].(llm.ToolUse); u.CallID != "toolu_01" || u.Name != "calc" || string(u.Input) != `{"expression": "6*7"}` {
		t.Errorf("tool_use = %+v", u)
	}
	if u, ok := bs[4].(llm.Unknown); !ok || u.Type != "server_widget" {
		t.Errorf("unknown block = %#v", bs[4])
	}
	if resp.StopReason != llm.StopToolUse || resp.Usage.CacheReadTokens != 100 || resp.Usage.CacheWriteTokens != 7 || resp.Model != "claude-opus-5-5" {
		t.Errorf("response = %+v", resp)
	}
}

// The M1 exit criterion: a reasoning block with its signature survives
// response → trace → next request byte-for-byte.
func TestReasoningRoundTrip(t *testing.T) {
	c, last := fakeAPI(t, 200, thinkingResponse)
	user := llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "6*7?"}}}
	resp, err := c.Generate(context.Background(), &llm.Request{Messages: []llm.Message{user}})
	if err != nil {
		t.Fatal(err)
	}
	// Through the trace encoding, as the runtime would store it.
	raw, err := json.Marshal(resp.Message)
	if err != nil {
		t.Fatal(err)
	}
	var stored llm.Message
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatal(err)
	}
	result := llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.ToolResult{CallID: "toolu_01", Content: llm.Blocks{llm.Text{Text: "42"}}}}}
	if _, err := c.Generate(context.Background(), &llm.Request{Messages: []llm.Message{user, stored, result}}); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(*last, &sent); err != nil {
		t.Fatal(err)
	}
	assistant := sent.Messages[1]
	if assistant.Role != "assistant" {
		t.Fatalf("role = %s", assistant.Role)
	}
	want := []string{
		`{"type":"thinking","thinking":"I should multiply.","signature":"EqQBCkYIBxgCKkA="}`,
		`{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3pzix/LafPsn4a"}`,
		`{"type":"text","text":"Let me compute that."}`,
		`{"type":"tool_use","id":"toolu_01","name":"calc","input":{"expression":"6*7"}}`,
	}
	for i, w := range want {
		if string(assistant.Content[i]) != w {
			t.Errorf("block %d sent as\n%s\nwant\n%s", i, assistant.Content[i], w)
		}
	}
	if got := string(sent.Messages[2].Content[0]); got != `{"type":"tool_result","tool_use_id":"toolu_01","content":[{"type":"text","text":"42"}]}` {
		t.Errorf("tool result sent as %s", got)
	}
}

func TestRequestMapping(t *testing.T) {
	c, last := fakeAPI(t, 200, `{"id":"m","model":"x","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	temp := 0.3
	foreign := llm.Reasoning{Text: "other provider", Ext: llm.Ext{"openai": json.RawMessage(`{"id":"rs_1"}`)}}
	_, err := c.Generate(context.Background(), &llm.Request{
		Model:  "claude-haiku-5-5",
		System: llm.Blocks{llm.Text{Text: "Be brief."}},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "hi"}, llm.Media{MediaType: "image/png", Data: []byte{1, 2, 3}}}},
			{Role: llm.RoleAssistant, Blocks: llm.Blocks{foreign, llm.Text{Text: "hello"}}},
		},
		Tools:      []llm.ToolSpec{{Name: "calc", Description: "d", Annotations: &llm.ToolAnnotations{ReadOnly: true}}},
		ToolChoice: &llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "calc"},
		Params:     llm.Params{MaxTokens: 256, Temperature: &temp, Stop: []string{"END"}, Reasoning: &llm.ReasoningParams{BudgetTokens: 1024}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"model":"claude-haiku-5-5","max_tokens":256,"system":[{"type":"text","text":"Be brief."}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}}]},{"role":"assistant","content":[{"type":"text","text":"hello"}]}],"tools":[{"name":"calc","description":"d","input_schema":{"type":"object","properties":{}}}],"tool_choice":{"type":"tool","name":"calc"},"temperature":0.3,"stop_sequences":["END"],"thinking":{"type":"enabled","budget_tokens":1024}}`
	if string(*last) != want {
		t.Errorf("request body\n got %s\nwant %s", *last, want)
	}
}

func TestErrors(t *testing.T) {
	cases := []struct {
		status    int
		body      string
		code      string
		retryable bool
	}{
		{429, `{"error":{"type":"rate_limit_error","message":"slow down"}}`, llm.ErrCodeRateLimited, true},
		{529, `{"error":{"type":"overloaded_error","message":"busy"}}`, llm.ErrCodeOverloaded, true},
		{401, `{"error":{"type":"authentication_error","message":"bad key"}}`, llm.ErrCodeAuth, false},
		{400, `{"error":{"type":"invalid_request_error","message":"prompt is too long: 300000 tokens"}}`, llm.ErrCodeContextLength, false},
		{400, `{"error":{"type":"invalid_request_error","message":"bad field"}}`, llm.ErrCodeInvalidRequest, false},
		{503, `upstream down`, llm.ErrCodeServer, true},
	}
	for _, c := range cases {
		cl, _ := fakeAPI(t, c.status, c.body)
		_, err := cl.Generate(context.Background(), &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "x"}}}}})
		var e *llm.Error
		if !errors.As(err, &e) || e.Code != c.code || e.Retryable != c.retryable || e.Status != c.status {
			t.Errorf("%d %s: got %#v", c.status, c.body, err)
		}
		if anthropic.IsRetryable(err) != c.retryable {
			t.Errorf("%d: IsRetryable mismatch", c.status)
		}
	}
}
