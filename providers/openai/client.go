// Package openai implements llm.Model with the OpenAI Chat Completions API
// (and compatible servers).
//
// Mapping notes: tool results, which agentflow keeps as llm.ToolResult
// blocks in a user message, become "tool" messages; images become
// "image_url" parts; reasoning blocks are not sent (Chat Completions has no
// reasoning input), and other unsupported blocks are rejected.
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
)

const (
	defaultBaseURL = "https://api.openai.com"
	defaultModel   = "gpt-6.2-astra"
	extKey         = "openai"
)

// Client implements llm.Model. It is safe for concurrent use.
type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

var _ llm.Model = (*Client)(nil)

// Option configures a Client.
type Option func(*Client)

// WithModel sets the model used when a request doesn't name one.
func WithModel(model string) Option { return func(c *Client) { c.model = model } }

// WithBaseURL overrides the API base URL (for testing or compatible servers).
func WithBaseURL(url string) Option { return func(c *Client) { c.baseURL = url } }

// WithHTTPClient sets the HTTP client.
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// New returns a Client.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{apiKey: apiKey, model: defaultModel, baseURL: defaultBaseURL, httpClient: &http.Client{Timeout: 10 * time.Minute}}
	for _, o := range opts {
		o(c)
	}
	return c
}

// --- wire types ---

type wireRequest struct {
	Model               string        `json:"model"`
	Messages            []wireMessage `json:"messages"`
	Tools               []wireTool    `json:"tools,omitempty"`
	ToolChoice          any           `json:"tool_choice,omitempty"`
	MaxCompletionTokens int           `json:"max_completion_tokens,omitempty"`
	Temperature         *float64      `json:"temperature,omitempty"`
	TopP                *float64      `json:"top_p,omitempty"`
	Stop                []string      `json:"stop,omitempty"`
	ReasoningEffort     string        `json:"reasoning_effort,omitempty"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content"` // string, []wirePart, or nil
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wirePart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *wireImageURL `json:"image_url,omitempty"`
}

type wireImageURL struct {
	URL string `json:"url"`
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type wireResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content   *string        `json:"content"`
			Refusal   *string        `json:"refusal"`
			ToolCalls []wireToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

// Generate implements llm.Model.
func (c *Client) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	wreq, err := c.buildRequest(req)
	if err != nil {
		return nil, &llm.Error{Code: llm.ErrCodeInvalidRequest, Message: err.Error()}
	}
	body, err := json.Marshal(wreq)
	if err != nil {
		return nil, &llm.Error{Code: llm.ErrCodeInvalidRequest, Message: err.Error()}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return nil, &llm.Error{Code: llm.ErrCodeCancelled, Message: err.Error()}
		}
		return nil, &llm.Error{Code: llm.ErrCodeNetwork, Message: err.Error(), Retryable: true}
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &llm.Error{Code: llm.ErrCodeNetwork, Message: err.Error(), Retryable: true}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, apiError(resp.StatusCode, respBody)
	}
	var wresp wireResponse
	if err := json.Unmarshal(respBody, &wresp); err != nil {
		return nil, &llm.Error{Code: llm.ErrCodeServer, Message: "decode response: " + err.Error()}
	}
	if len(wresp.Choices) == 0 {
		return nil, &llm.Error{Code: llm.ErrCodeServer, Message: "response has no choices"}
	}
	return buildResponse(&wresp), nil
}

func (c *Client) buildRequest(req *llm.Request) (*wireRequest, error) {
	w := &wireRequest{
		Model: req.Model, MaxCompletionTokens: req.Params.MaxTokens,
		Temperature: req.Params.Temperature, TopP: req.Params.TopP, Stop: req.Params.Stop,
	}
	if w.Model == "" {
		w.Model = c.model
	}
	if r := req.Params.Reasoning; r != nil {
		w.ReasoningEffort = r.Effort
	}
	if len(req.System) > 0 {
		text, err := textOnly(req.System)
		if err != nil {
			return nil, fmt.Errorf("system: %w", err)
		}
		w.Messages = append(w.Messages, wireMessage{Role: "system", Content: text})
	}
	for _, t := range req.Tools {
		var wt wireTool
		wt.Type = "function"
		wt.Function.Name, wt.Function.Description = t.Name, t.Description
		wt.Function.Parameters = t.InputSchema
		if len(wt.Function.Parameters) == 0 {
			wt.Function.Parameters = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		w.Tools = append(w.Tools, wt)
	}
	if tc := req.ToolChoice; tc != nil {
		switch tc.Mode {
		case llm.ToolChoiceAuto:
			w.ToolChoice = "auto"
		case llm.ToolChoiceAny:
			w.ToolChoice = "required"
		case llm.ToolChoiceNone:
			w.ToolChoice = "none"
		case llm.ToolChoiceTool:
			w.ToolChoice = map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
		}
	}
	for _, m := range req.Messages {
		msgs, err := toWire(m)
		if err != nil {
			return nil, err
		}
		w.Messages = append(w.Messages, msgs...)
	}
	return w, nil
}

// toWire converts one message. A user message's tool results become "tool"
// messages, emitted before any remaining user content.
func toWire(m llm.Message) ([]wireMessage, error) {
	var out []wireMessage
	if m.Role == llm.RoleAssistant {
		am := wireMessage{Role: "assistant"}
		var text []string
		for _, b := range m.Blocks {
			switch v := b.(type) {
			case llm.Text:
				text = append(text, v.Text)
			case llm.Data:
				text = append(text, string(v.Value))
			case llm.ToolUse:
				var tc wireToolCall
				tc.ID, tc.Type = v.CallID, "function"
				tc.Function.Name = v.Name
				tc.Function.Arguments = string(v.Input)
				if tc.Function.Arguments == "" {
					tc.Function.Arguments = "{}"
				}
				am.ToolCalls = append(am.ToolCalls, tc)
			case llm.Reasoning, llm.Unknown:
				// Not representable in Chat Completions input.
			default:
				return nil, fmt.Errorf("unsupported assistant block %q", b.BlockType())
			}
		}
		if len(text) > 0 {
			am.Content = strings.Join(text, "\n")
		}
		return []wireMessage{am}, nil
	}

	var parts []wirePart
	for _, b := range m.Blocks {
		switch v := b.(type) {
		case llm.ToolResult:
			text, err := textOnly(v.Content)
			if err != nil {
				return nil, fmt.Errorf("tool result %s: %w", v.CallID, err)
			}
			if v.IsError && !strings.HasPrefix(text, "Error") {
				text = "Error: " + text
			}
			out = append(out, wireMessage{Role: "tool", ToolCallID: v.CallID, Content: text})
		case llm.Text:
			parts = append(parts, wirePart{Type: "text", Text: v.Text})
		case llm.Data:
			parts = append(parts, wirePart{Type: "text", Text: string(v.Value)})
		case llm.Media:
			if !strings.HasPrefix(v.MediaType, "image/") {
				return nil, fmt.Errorf("unsupported media type %q", v.MediaType)
			}
			url := v.URL
			if url == "" {
				url = "data:" + v.MediaType + ";base64," + base64.StdEncoding.EncodeToString(v.Data)
			}
			parts = append(parts, wirePart{Type: "image_url", ImageURL: &wireImageURL{URL: url}})
		default:
			return nil, fmt.Errorf("unsupported user block %q", b.BlockType())
		}
	}
	if len(parts) > 0 {
		out = append(out, wireMessage{Role: "user", Content: parts})
	}
	return out, nil
}

func textOnly(bs llm.Blocks) (string, error) {
	var out []string
	for _, b := range bs {
		switch v := b.(type) {
		case llm.Text:
			out = append(out, v.Text)
		case llm.Data:
			out = append(out, string(v.Value))
		default:
			return "", fmt.Errorf("only text is supported here, got %q", b.BlockType())
		}
	}
	return strings.Join(out, "\n"), nil
}

func buildResponse(w *wireResponse) *llm.Response {
	ch := w.Choices[0]
	blocks := llm.Blocks{}
	if ch.Message.Content != nil && *ch.Message.Content != "" {
		blocks = append(blocks, llm.Text{Text: *ch.Message.Content})
	}
	if ch.Message.Refusal != nil && *ch.Message.Refusal != "" {
		blocks = append(blocks, llm.Text{Text: *ch.Message.Refusal})
	}
	for _, tc := range ch.Message.ToolCalls {
		args := json.RawMessage(tc.Function.Arguments)
		if len(args) == 0 || !json.Valid(args) {
			// Models occasionally emit invalid JSON; keep it as a string so
			// the tool's input validation can report it.
			quoted, _ := json.Marshal(tc.Function.Arguments)
			args = quoted
		}
		blocks = append(blocks, llm.ToolUse{CallID: tc.ID, Name: tc.Function.Name, Input: args})
	}
	stop := llm.StopOther
	switch ch.FinishReason {
	case "stop":
		stop = llm.StopEndTurn
	case "tool_calls", "function_call":
		stop = llm.StopToolUse
	case "length":
		stop = llm.StopMaxTokens
	case "content_filter":
		stop = llm.StopRefusal
	}
	if ch.Message.Refusal != nil && *ch.Message.Refusal != "" {
		stop = llm.StopRefusal
	}
	meta, _ := json.Marshal(map[string]string{"id": w.ID})
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
		StopReason: stop,
		Usage: llm.Usage{
			InputTokens: w.Usage.PromptTokens, OutputTokens: w.Usage.CompletionTokens,
			CacheReadTokens: w.Usage.PromptTokensDetails.CachedTokens, ReasoningTokens: w.Usage.CompletionTokensDetails.ReasoningTokens,
		},
		Model: w.Model,
		Ext:   llm.Ext{extKey: meta},
	}
}

func apiError(status int, body []byte) error {
	var wrapper struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := strings.TrimSpace(string(body))
	code := ""
	if json.Unmarshal(body, &wrapper) == nil && wrapper.Error.Message != "" {
		msg, code = wrapper.Error.Message, wrapper.Error.Code
	}
	e := &llm.Error{Message: msg, Status: status}
	switch {
	case status == http.StatusTooManyRequests:
		e.Code, e.Retryable = llm.ErrCodeRateLimited, true
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		e.Code = llm.ErrCodeAuth
	case code == "context_length_exceeded":
		e.Code = llm.ErrCodeContextLength
	case status >= 500:
		e.Code, e.Retryable = llm.ErrCodeServer, true
	default:
		e.Code = llm.ErrCodeInvalidRequest
	}
	return e
}

// IsRetryable reports whether err is a retryable *llm.Error.
func IsRetryable(err error) bool {
	var e *llm.Error
	return errors.As(err, &e) && e.Retryable
}
