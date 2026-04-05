// Package anthropic provides an agentflow.LLM implementation backed by the
// Anthropic Messages API. It supports text responses, tool use (function
// calling), and token usage reporting.
//
// Usage:
//
//	client := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"),
//	    anthropic.WithModel("claude-opus-4-6-20251101"),
//	)
//	agent, _ := agentflow.NewAgent(agentflow.AgentConfig{LLM: client, ...})
package anthropic

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
	defaultBaseURL = "https://api.anthropic.com"
	defaultModel   = "claude-opus-4-6-20251101"
	anthropicVersion = "2023-06-01"
)

// Client implements agentflow.LLM using the Anthropic Messages API.
// It is safe for concurrent use.
type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
	maxTokens  int
}

// Option configures a Client.
type Option func(*Client)

// WithModel sets the Anthropic model to use.
func WithModel(model string) Option {
	return func(c *Client) { c.model = model }
}

// WithBaseURL overrides the Anthropic API base URL (useful for testing).
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// WithMaxTokens sets the maximum tokens for each response.
func WithMaxTokens(n int) Option {
	return func(c *Client) { c.maxTokens = n }
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
		maxTokens: 4096,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Ensure Client implements agentflow.LLM at compile time.
var _ agentflow.LLM = (*Client)(nil)

// --- Anthropic wire types ---

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
	Tools     []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string           `json:"role"`
	Content anthropicContent `json:"content"`
}

// anthropicContent can be a plain string or a slice of content blocks.
// We use a custom type to handle both during marshalling.
type anthropicContent struct {
	text   string
	blocks []anthropicBlock
}

func (c anthropicContent) MarshalJSON() ([]byte, error) {
	if len(c.blocks) > 0 {
		return json.Marshal(c.blocks)
	}
	return json.Marshal(c.text)
}

type anthropicBlock struct {
	Type      string          `json:"type"`
	// For text blocks
	Text      string          `json:"text,omitempty"`
	// For tool_use blocks (assistant → tool call)
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	// For tool_result blocks (user → tool result)
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicResponse struct {
	ID           string           `json:"id"`
	Type         string           `json:"type"`
	Role         string           `json:"role"`
	Content      []anthropicBlock `json:"content"`
	Model        string           `json:"model"`
	StopReason   string           `json:"stop_reason"`
	Usage        anthropicUsage   `json:"usage"`
	Error        *anthropicError  `json:"error,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// ChatCompletion sends a conversation to the Anthropic Messages API.
func (c *Client) ChatCompletion(ctx context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
	areq, err := c.buildRequest(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build request: %w", err)
	}

	body, err := json.Marshal(areq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", anthropicVersion)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, parseHTTPError(resp.StatusCode, respBody)
	}

	var aresp anthropicResponse
	if err := json.Unmarshal(respBody, &aresp); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}

	return c.buildResponse(&aresp), nil
}

// buildRequest converts an agentflow.LLMRequest into an Anthropic API request.
func (c *Client) buildRequest(req *agentflow.LLMRequest) (*anthropicRequest, error) {
	areq := &anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
	}

	// Convert tools.
	for _, t := range req.Tools {
		// The Parameters field is a JSON Schema object. Anthropic expects it
		// verbatim as input_schema.
		schema := t.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		areq.Tools = append(areq.Tools, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		})
	}

	// Convert messages. Anthropic separates the system prompt from messages.
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			// Anthropic takes system as a top-level field (last one wins if multiple).
			areq.System = msg.Content

		case "user":
			areq.Messages = append(areq.Messages, anthropicMessage{
				Role:    "user",
				Content: anthropicContent{text: msg.Content},
			})

		case "assistant":
			// Assistant messages may carry tool call requests.
			if len(msg.ToolCalls) > 0 {
				blocks := make([]anthropicBlock, 0, len(msg.ToolCalls)+1)
				if msg.Content != "" {
					blocks = append(blocks, anthropicBlock{Type: "text", Text: msg.Content})
				}
				for _, tc := range msg.ToolCalls {
					input := tc.Arguments
					if len(input) == 0 {
						input = json.RawMessage(`{}`)
					}
					blocks = append(blocks, anthropicBlock{
						Type:  "tool_use",
						ID:    tc.ID,
						Name:  tc.Name,
						Input: input,
					})
				}
				areq.Messages = append(areq.Messages, anthropicMessage{
					Role:    "assistant",
					Content: anthropicContent{blocks: blocks},
				})
			} else {
				areq.Messages = append(areq.Messages, anthropicMessage{
					Role:    "assistant",
					Content: anthropicContent{text: msg.Content},
				})
			}

		case "tool":
			// Tool results must be grouped into a user message with tool_result blocks.
			// Consecutive tool messages are merged into the same user message.
			block := anthropicBlock{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Content:   msg.Content,
			}
			if len(areq.Messages) > 0 {
				last := &areq.Messages[len(areq.Messages)-1]
				if last.Role == "user" && len(last.Content.blocks) > 0 && last.Content.blocks[0].Type == "tool_result" {
					last.Content.blocks = append(last.Content.blocks, block)
					continue
				}
			}
			areq.Messages = append(areq.Messages, anthropicMessage{
				Role:    "user",
				Content: anthropicContent{blocks: []anthropicBlock{block}},
			})
		}
	}

	return areq, nil
}

// buildResponse converts an Anthropic API response into an agentflow.LLMResponse.
func (c *Client) buildResponse(aresp *anthropicResponse) *agentflow.LLMResponse {
	resp := &agentflow.LLMResponse{
		Usage: &agentflow.TokenUsage{
			PromptTokens:     aresp.Usage.InputTokens,
			CompletionTokens: aresp.Usage.OutputTokens,
			TotalTokens:      aresp.Usage.InputTokens + aresp.Usage.OutputTokens,
		},
	}

	for _, block := range aresp.Content {
		switch block.Type {
		case "text":
			resp.Content += block.Text
		case "tool_use":
			args := block.Input
			if len(args) == 0 {
				args = json.RawMessage(`{}`)
			}
			resp.ToolCalls = append(resp.ToolCalls, agentflow.ToolCallRequest{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: args,
			})
		}
	}

	return resp
}

// APIError represents an error returned by the Anthropic API.
type APIError struct {
	StatusCode int
	Type       string
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("anthropic: API error %d: %s: %s", e.StatusCode, e.Type, e.Message)
}

// IsRetryable reports whether the error is a transient failure that may succeed
// on retry (rate limit, overloaded, or server error).
func (e *APIError) IsRetryable() bool {
	switch e.StatusCode {
	case http.StatusTooManyRequests, // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		529:                            // Anthropic overloaded
		return true
	}
	return false
}

func parseHTTPError(status int, body []byte) error {
	var wrapper struct {
		Error anthropicError `json:"error"`
	}
	apiErr := &APIError{StatusCode: status}
	if json.Unmarshal(body, &wrapper) == nil && wrapper.Error.Type != "" {
		apiErr.Type = wrapper.Error.Type
		apiErr.Message = wrapper.Error.Message
	} else {
		apiErr.Type = "unknown"
		apiErr.Message = string(body)
	}
	return apiErr
}
