package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/providers/openai"
)

func openAITextResp(content string, promptTokens, completionTokens int) []byte {
	resp := map[string]any{
		"id":     "chatcmpl-test",
		"object": "chat.completion",
		"model":  "gpt-4o",
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "stop",
				"message": map[string]any{
					"role":    "assistant",
					"content": content,
				},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func openAIToolCallResp(id, name, arguments string) []byte {
	resp := map[string]any{
		"id":     "chatcmpl-test",
		"object": "chat.completion",
		"model":  "gpt-4o",
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []map[string]any{
						{
							"id":   id,
							"type": "function",
							"function": map[string]any{
								"name":      name,
								"arguments": arguments,
							},
						},
					},
				},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     40,
			"completion_tokens": 15,
			"total_tokens":      55,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func openAIErrResp(errType, message string) []byte {
	resp := map[string]any{
		"error": map[string]any{
			"type":    errType,
			"message": message,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func newTestClient(t *testing.T, handler http.HandlerFunc) (*openai.Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	client := openai.New("test-key",
		openai.WithBaseURL(srv.URL),
		openai.WithModel("gpt-4o"),
	)
	return client, srv
}

func TestChatCompletion_TextResponse(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Error("missing or wrong Authorization header")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(openAITextResp("Hello!", 10, 5))
	})

	resp, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello!" {
		t.Errorf("content = %q, want %q", resp.Content, "Hello!")
	}
	if resp.Usage == nil {
		t.Fatal("expected usage")
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v, want {10,5,15}", resp.Usage)
	}
}

func TestChatCompletion_ToolCallResponse(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(openAIToolCallResp("call-1", "calculator", `{"expression":"3*7"}`))
	})

	resp, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "user", Content: "What is 3*7?"},
		},
		Tools: []agentflow.ToolSchema{
			{
				Name:        "calculator",
				Description: "Evaluates math.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call-1" || tc.Name != "calculator" {
		t.Errorf("tool call = {%q, %q}, want {call-1, calculator}", tc.ID, tc.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["expression"] != "3*7" {
		t.Errorf("expression = %q, want %q", args["expression"], "3*7")
	}
}

func TestChatCompletion_MultipleToolCalls(t *testing.T) {
	resp2 := map[string]any{
		"id": "chatcmpl-test",
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": "tool_calls",
			"message": map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []map[string]any{
					{"id": "c1", "type": "function", "function": map[string]any{"name": "add", "arguments": `{"a":1,"b":2}`}},
					{"id": "c2", "type": "function", "function": map[string]any{"name": "mul", "arguments": `{"a":3,"b":4}`}},
				},
			},
		}},
		"usage": map[string]any{"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
	}
	body, _ := json.Marshal(resp2)
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	})

	resp, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "Do math."}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(resp.ToolCalls))
	}
}

func TestChatCompletion_RateLimitError(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write(openAIErrResp("rate_limit_exceeded", "Rate limit reached"))
	})

	_, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	apiErr, ok := err.(*openai.APIError)
	if !ok {
		t.Fatalf("expected *openai.APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("status = %d, want 429", apiErr.StatusCode)
	}
	if !apiErr.IsRetryable() {
		t.Error("rate limit should be retryable")
	}
}

func TestChatCompletion_AuthError(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write(openAIErrResp("invalid_api_key", "Incorrect API key"))
	})

	_, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	apiErr, ok := err.(*openai.APIError)
	if !ok {
		t.Fatalf("expected *openai.APIError, got %T", err)
	}
	if apiErr.IsRetryable() {
		t.Error("auth error should not be retryable")
	}
}

func TestChatCompletion_EmptyChoices(t *testing.T) {
	resp := map[string]any{
		"id":      "chatcmpl-test",
		"choices": []any{},
		"usage":   map[string]any{"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
	}
	body, _ := json.Marshal(resp)
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	})

	_, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for empty choices")
	}
}

func TestChatCompletion_MalformedJSON(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("{invalid json"))
	})

	_, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestChatCompletion_ContextCancellation(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.ChatCompletion(ctx, &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestChatCompletion_ToolResultConversion(t *testing.T) {
	var captured map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write(openAITextResp("The answer is 21.", 30, 10))
	})

	client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "user", Content: "What is 3*7?"},
			{Role: "assistant", ToolCalls: []agentflow.ToolCallRequest{
				{ID: "call-1", Name: "calculator", Arguments: json.RawMessage(`{"expression":"3*7"}`)},
			}},
			{Role: "tool", Content: "21", ToolCallID: "call-1"},
		},
	})

	msgs, _ := captured["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}

	// Tool message
	toolMsg, _ := msgs[2].(map[string]any)
	if toolMsg["role"] != "tool" {
		t.Errorf("role = %q, want tool", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "call-1" {
		t.Errorf("tool_call_id = %q, want call-1", toolMsg["tool_call_id"])
	}
}
