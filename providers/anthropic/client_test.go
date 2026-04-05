package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
)

// anthropicResp builds a minimal Anthropic API response JSON.
func anthropicTextResp(content string, inputTokens, outputTokens int) []byte {
	resp := map[string]any{
		"id":          "msg_test",
		"type":        "message",
		"role":        "assistant",
		"stop_reason": "end_turn",
		"model":       "claude-opus-4-6-20251101",
		"content": []map[string]any{
			{"type": "text", "text": content},
		},
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func anthropicToolUseResp(id, name string, input map[string]any) []byte {
	inputJSON, _ := json.Marshal(input)
	resp := map[string]any{
		"id":          "msg_test",
		"type":        "message",
		"role":        "assistant",
		"stop_reason": "tool_use",
		"model":       "claude-opus-4-6-20251101",
		"content": []map[string]any{
			{
				"type":  "tool_use",
				"id":    id,
				"name":  name,
				"input": json.RawMessage(inputJSON),
			},
		},
		"usage": map[string]any{
			"input_tokens":  50,
			"output_tokens": 20,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func anthropicErrResp(errType, message string) []byte {
	resp := map[string]any{
		"error": map[string]any{
			"type":    errType,
			"message": message,
		},
	}
	b, _ := json.Marshal(resp)
	return b
}

func newTestClient(t *testing.T, handler http.HandlerFunc) (*anthropic.Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	client := anthropic.New("test-key",
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-opus-4-6-20251101"),
	)
	return client, srv
}

func TestChatCompletion_TextResponse(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key" {
			t.Error("missing api key header")
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Error("missing anthropic-version header")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(anthropicTextResp("Hello, world!", 10, 5))
	})

	resp, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello, world!" {
		t.Errorf("content = %q, want %q", resp.Content, "Hello, world!")
	}
	if resp.Usage == nil {
		t.Fatal("expected usage")
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v, want {10,5,15}", resp.Usage)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(resp.ToolCalls))
	}
}

func TestChatCompletion_ToolUseResponse(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(anthropicToolUseResp("call-1", "calculator", map[string]any{"expression": "2+2"}))
	})

	resp, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "user", Content: "What is 2+2?"},
		},
		Tools: []agentflow.ToolSchema{
			{
				Name:        "calculator",
				Description: "Evaluates a math expression.",
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
	if args["expression"] != "2+2" {
		t.Errorf("expression = %q, want %q", args["expression"], "2+2")
	}
}

func TestChatCompletion_MultipleToolCalls(t *testing.T) {
	resp2 := map[string]any{
		"id":          "msg_test2",
		"type":        "message",
		"role":        "assistant",
		"stop_reason": "tool_use",
		"model":       "claude-opus-4-6-20251101",
		"content": []map[string]any{
			{"type": "tool_use", "id": "call-1", "name": "add", "input": map[string]any{"a": 1, "b": 2}},
			{"type": "tool_use", "id": "call-2", "name": "mul", "input": map[string]any{"a": 3, "b": 4}},
		},
		"usage": map[string]any{"input_tokens": 30, "output_tokens": 15},
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
		w.Write(anthropicErrResp("rate_limit_error", "Too many requests"))
	})

	_, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	apiErr, ok := err.(*anthropic.APIError)
	if !ok {
		t.Fatalf("expected *anthropic.APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", apiErr.StatusCode)
	}
	if !apiErr.IsRetryable() {
		t.Error("rate limit error should be retryable")
	}
}

func TestChatCompletion_AuthError(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write(anthropicErrResp("authentication_error", "Invalid API key"))
	})

	_, err := client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	apiErr, ok := err.(*anthropic.APIError)
	if !ok {
		t.Fatalf("expected *anthropic.APIError, got %T", err)
	}
	if apiErr.IsRetryable() {
		t.Error("auth error should not be retryable")
	}
}

func TestChatCompletion_MalformedJSON(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not json at all {{{"))
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
		// Block until client cancels.
		<-r.Context().Done()
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := client.ChatCompletion(ctx, &agentflow.LLMRequest{
		Messages: []agentflow.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestChatCompletion_SystemPromptConversion(t *testing.T) {
	var captured map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write(anthropicTextResp("ok", 5, 3))
	})

	client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Hello."},
		},
	})

	if captured["system"] != "You are helpful." {
		t.Errorf("system = %q, want %q", captured["system"], "You are helpful.")
	}
	msgs, _ := captured["messages"].([]any)
	if len(msgs) != 1 {
		t.Errorf("messages length = %d, want 1 (system must not appear in messages array)", len(msgs))
	}
}

func TestChatCompletion_ToolResultConversion(t *testing.T) {
	var captured map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		w.Write(anthropicTextResp("The answer is 4.", 20, 8))
	})

	client.ChatCompletion(context.Background(), &agentflow.LLMRequest{
		Messages: []agentflow.Message{
			{Role: "user", Content: "What is 2+2?"},
			{Role: "assistant", ToolCalls: []agentflow.ToolCallRequest{
				{ID: "call-1", Name: "calculator", Arguments: json.RawMessage(`{"expression":"2+2"}`)},
			}},
			{Role: "tool", Content: "4", ToolCallID: "call-1"},
		},
	})

	msgs, _ := captured["messages"].([]any)
	// Expect: user, assistant (tool_use), user (tool_result)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages in request, got %d", len(msgs))
	}
	lastMsg, _ := msgs[2].(map[string]any)
	if lastMsg["role"] != "user" {
		t.Errorf("tool result role = %q, want user", lastMsg["role"])
	}
	content, _ := lastMsg["content"].([]any)
	if len(content) == 0 {
		t.Fatal("tool result content is empty")
	}
	block, _ := content[0].(map[string]any)
	if block["type"] != "tool_result" {
		t.Errorf("block type = %q, want tool_result", block["type"])
	}
	if block["tool_use_id"] != "call-1" {
		t.Errorf("tool_use_id = %q, want call-1", block["tool_use_id"])
	}
}
