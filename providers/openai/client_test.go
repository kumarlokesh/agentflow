package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/providers/openai"
)

func fakeAPI(t *testing.T, status int, response string) (*openai.Client, *[]byte) {
	t.Helper()
	var last []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("bad request: %s %v", r.URL.Path, r.Header)
		}
		last, _ = io.ReadAll(r.Body)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)
	return openai.New("k", openai.WithBaseURL(srv.URL)), &last
}

func TestResponseMapping(t *testing.T) {
	c, _ := fakeAPI(t, 200, `{"id":"chatcmpl-1","model":"gpt-6.2-astra","choices":[{"finish_reason":"tool_calls","message":{"content":"Checking.",
	  "tool_calls":[{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"x\":1}"}},
	                {"id":"call_2","type":"function","function":{"name":"calc","arguments":"{not json"}}]}}],
	  "usage":{"prompt_tokens":50,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":32},"completion_tokens_details":{"reasoning_tokens":8}}}`)
	resp, err := c.Generate(context.Background(), &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "x"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	bs := resp.Message.Blocks
	if len(bs) != 3 || llm.TextOf(bs) != "Checking." {
		t.Fatalf("blocks = %#v", bs)
	}
	if u := bs[1].(llm.ToolUse); u.CallID != "call_1" || string(u.Input) != `{"x":1}` {
		t.Errorf("tool call = %+v", u)
	}
	if u := bs[2].(llm.ToolUse); string(u.Input) != `"{not json"` {
		t.Errorf("invalid arguments should be kept as a JSON string, got %s", u.Input)
	}
	if resp.StopReason != llm.StopToolUse || resp.Usage.CacheReadTokens != 32 || resp.Usage.ReasoningTokens != 8 || resp.Usage.InputTokens != 50 {
		t.Errorf("response = %+v", resp)
	}
}

func TestRequestMapping(t *testing.T) {
	c, last := fakeAPI(t, 200, `{"id":"x","model":"m","choices":[{"finish_reason":"stop","message":{"content":"ok"}}],"usage":{}}`)
	_, err := c.Generate(context.Background(), &llm.Request{
		Model:  "gpt-6-luna",
		System: llm.Blocks{llm.Text{Text: "Be brief."}},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "6*7?"}, llm.Media{MediaType: "image/png", Data: []byte{1, 2, 3}}}},
			{Role: llm.RoleAssistant, Blocks: llm.Blocks{
				llm.Reasoning{Text: "hidden", Ext: llm.Ext{"anthropic": json.RawMessage(`{"signature":"s"}`)}},
				llm.ToolUse{CallID: "call_1", Name: "calc", Input: json.RawMessage(`{"x":1}`)}}},
			{Role: llm.RoleUser, Blocks: llm.Blocks{
				llm.ToolResult{CallID: "call_1", Content: llm.Blocks{llm.Text{Text: "42"}}},
				llm.ToolResult{CallID: "call_2", Content: llm.Blocks{llm.Text{Text: "boom"}}, IsError: true},
				llm.Text{Text: "and now?"}}},
		},
		Tools:      []llm.ToolSpec{{Name: "calc", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: &llm.ToolChoice{Mode: llm.ToolChoiceAny},
		Params:     llm.Params{MaxTokens: 100, Reasoning: &llm.ReasoningParams{Effort: "low"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"model":"gpt-6-luna","messages":[` +
		`{"role":"system","content":"Be brief."},` +
		`{"role":"user","content":[{"type":"text","text":"6*7?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"x\":1}"}}]},` +
		`{"role":"tool","content":"42","tool_call_id":"call_1"},` +
		`{"role":"tool","content":"Error: boom","tool_call_id":"call_2"},` +
		`{"role":"user","content":[{"type":"text","text":"and now?"}]}],` +
		`"tools":[{"type":"function","function":{"name":"calc","description":"d","parameters":{"type":"object"}}}],` +
		`"tool_choice":"required","max_completion_tokens":100,"reasoning_effort":"low"}`
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
		{429, `{"error":{"message":"slow down","type":"requests"}}`, llm.ErrCodeRateLimited, true},
		{401, `{"error":{"message":"bad key"}}`, llm.ErrCodeAuth, false},
		{400, `{"error":{"message":"too long","code":"context_length_exceeded"}}`, llm.ErrCodeContextLength, false},
		{400, `{"error":{"message":"bad"}}`, llm.ErrCodeInvalidRequest, false},
		{502, `bad gateway`, llm.ErrCodeServer, true},
	}
	for _, c := range cases {
		cl, _ := fakeAPI(t, c.status, c.body)
		_, err := cl.Generate(context.Background(), &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "x"}}}}})
		var e *llm.Error
		if !errors.As(err, &e) || e.Code != c.code || e.Retryable != c.retryable {
			t.Errorf("%d: got %#v", c.status, err)
		}
	}
}
