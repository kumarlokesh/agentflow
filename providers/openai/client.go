// Package openai provides an agentflow.LLM implementation backed by the
// OpenAI Chat Completions API. It supports text responses, tool/function
// calling, and token usage reporting.
//
// Usage:
//
//	client := openai.New(os.Getenv("OPENAI_API_KEY"),
//	    openai.WithModel("gpt-4o"),
//	)
//	agent, _ := agentflow.NewAgent(agentflow.AgentConfig{LLM: client, ...})
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/kumarlokesh/agentflow"
)

const (
	defaultBaseURL = "https://api.openai.com"
	defaultModel   = "gpt-4o"
)

// Client implements agentflow.LLM using the OpenAI Chat Completions API.
// It is safe for concurrent use.
type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithModel sets the OpenAI model to use.
func WithModel(model string) Option {
	return func(c *Client) { c.model = model }
}

// WithBaseURL overrides the OpenAI API base URL (useful for testing).
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// New creates a Client with the given API key and options.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		apiKey:  apiKey,
		model:   defaultModel,
		baseURL: defaultBaseURL,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Ensure Client implements agentflow.LLM at compile time.
var _ agentflow.LLM = (*Client)(nil)

// --- OpenAI wire types ---

type openAIRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
	Tools    []openAITool    `json:"tools,omitempty"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content"`              // string or nil
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

type openAITool struct {
	Type     string           `json:"type"`
	Function openAIFunctionDef `json:"function"`
}

type openAIFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type openAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
	Error   *openAIError   `json:"error,omitempty"`
}

type openAIChoice struct {
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

// ChatCompletion sends a conversation to the OpenAI Chat Completions API.
func (c *Client) ChatCompletion(ctx context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	oreq := c.buildRequest(req)

	body, err := json.Marshal(oreq)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, parseHTTPError(resp.StatusCode, respBody)
	}

	var oresp openAIResponse
	if err := json.Unmarshal(respBody, &oresp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	if len(oresp.Choices) == 0 {
		return nil, fmt.Errorf("openai: response has no choices")
	}

	return c.buildResponse(&oresp), nil
}

// buildRequest converts an agentflow.LLMRequest into an OpenAI API request.
func (c *Client) buildRequest(req *agentflow.LLMRequest) *openAIRequest {
	oreq := &openAIRequest{
		Model: c.model,
	}

	// Convert tools.
	for _, t := range req.Tools {
		params := t.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		oreq.Tools = append(oreq.Tools, openAITool{
			Type: "function",
			Function: openAIFunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}

	// Convert messages.
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "user":
			oreq.Messages = append(oreq.Messages, openAIMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})

		case "assistant":
			om := openAIMessage{
				Role:    "assistant",
				Content: msg.Content,
			}
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					args := string(tc.Arguments)
					if args == "" {
						args = "{}"
					}
					om.ToolCalls = append(om.ToolCalls, openAIToolCall{
						ID:   tc.ID,
						Type: "function",
						Function: openAIFunctionCall{
							Name:      tc.Name,
							Arguments: args,
						},
					})
				}
			}
			oreq.Messages = append(oreq.Messages, om)

		case "tool":
			oreq.Messages = append(oreq.Messages, openAIMessage{
				Role:       "tool",
				Content:    msg.Content,
				ToolCallID: msg.ToolCallID,
			})
		}
	}

	return oreq
}

// buildResponse converts an OpenAI API response into an agentflow.LLMResponse.
func (c *Client) buildResponse(oresp *openAIResponse) *agentflow.LLMResponse {
	choice := oresp.Choices[0]
	resp := &agentflow.LLMResponse{
		Usage: &agentflow.TokenUsage{
			PromptTokens:     oresp.Usage.PromptTokens,
			CompletionTokens: oresp.Usage.CompletionTokens,
			TotalTokens:      oresp.Usage.TotalTokens,
		},
	}

	// Content can be nil when the model returns only tool calls.
	if s, ok := choice.Message.Content.(string); ok {
		resp.Content = s
	}

	for _, tc := range choice.Message.ToolCalls {
		args := json.RawMessage(tc.Function.Arguments)
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		resp.ToolCalls = append(resp.ToolCalls, agentflow.ToolCallRequest{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
		})
	}

	return resp
}

// APIError represents an error returned by the OpenAI API.
type APIError struct {
	StatusCode int
	Type       string
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: API error %d: %s: %s", e.StatusCode, e.Type, e.Message)
}

// IsRetryable reports whether the error is a transient failure that may succeed
// on retry.
func (e *APIError) IsRetryable() bool {
	switch e.StatusCode {
	case http.StatusTooManyRequests,       // 429
		http.StatusInternalServerError,    // 500
		http.StatusBadGateway,            // 502
		http.StatusServiceUnavailable:    // 503
		return true
	}
	return false
}

func parseHTTPError(status int, body []byte) error {
	var wrapper struct {
		Error openAIError `json:"error"`
	}
	apiErr := &APIError{StatusCode: status}
	if json.Unmarshal(body, &wrapper) == nil && wrapper.Error.Message != "" {
		apiErr.Type = wrapper.Error.Type
		apiErr.Message = wrapper.Error.Message
	} else {
		apiErr.Type = "unknown"
		apiErr.Message = string(body)
	}
	return apiErr
}
