// Package anthropic implements llm.Model with the Anthropic Messages API.
//
// It maps every llm content block, including reasoning: "thinking" and
// "redacted_thinking" blocks are returned as llm.Reasoning with their
// signature (or encrypted data) in Ext["anthropic"], and sent back unchanged,
// as the API requires for multi-turn tool use. Reasoning blocks from other
// providers are dropped.
//
// Usage:
//
//	model := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), anthropic.WithModel("claude-opus-5-5"))
//	agent, _ := agentflow.NewAgent(agentflow.AgentConfig{Name: "a", Model: model, ModelName: "claude-opus-5-5"})
package anthropic

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
	defaultBaseURL   = "https://api.anthropic.com"
	defaultModel     = "claude-opus-5-5"
	defaultMaxTokens = 4096
	apiVersion       = "2023-06-01"
	extKey           = "anthropic"
)

// Client implements llm.Model. It is safe for concurrent use.
type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
	maxTokens  int
}

var _ llm.Model = (*Client)(nil)

// Option configures a Client.
type Option func(*Client)

// WithModel sets the model used when a request doesn't name one.
func WithModel(model string) Option { return func(c *Client) { c.model = model } }

// WithBaseURL overrides the API base URL (useful for testing).
func WithBaseURL(url string) Option { return func(c *Client) { c.baseURL = url } }

// WithHTTPClient sets the HTTP client.
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// WithMaxTokens sets max_tokens when a request doesn't set Params.MaxTokens.
func WithMaxTokens(n int) Option { return func(c *Client) { c.maxTokens = n } }

// New returns a Client.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		apiKey: apiKey, model: defaultModel, baseURL: defaultBaseURL,
		httpClient: &http.Client{Timeout: 10 * time.Minute}, maxTokens: defaultMaxTokens,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// --- wire types ---

type wireRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	System        []wireBlock     `json:"system,omitempty"`
	Messages      []wireMessage   `json:"messages"`
	Tools         []wireTool      `json:"tools,omitempty"`
	ToolChoice    *wireToolChoice `json:"tool_choice,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Thinking      *wireThinking   `json:"thinking,omitempty"`
}

type wireThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

type wireToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type wireMessage struct {
	Role    string            `json:"role"`
	Content []json.RawMessage `json:"content"`
}

type wireSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type wireBlock struct {
	Type string `json:"type"`
	// text
	Text string `json:"text,omitempty"`
	// image, document
	Source *wireSource `json:"source,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID string            `json:"tool_use_id,omitempty"`
	Content   []json.RawMessage `json:"content,omitempty"`
	IsError   bool              `json:"is_error,omitempty"`
	// thinking, redacted_thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type wireResponse struct {
	ID         string            `json:"id"`
	Model      string            `json:"model"`
	StopReason string            `json:"stop_reason"`
	Content    []json.RawMessage `json:"content"`
	Usage      struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

// reasoningExt is what Ext["anthropic"] holds for reasoning blocks.
type reasoningExt struct {
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"` // redacted_thinking payload
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
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", apiVersion)

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
	return buildResponse(&wresp)
}

func (c *Client) buildRequest(req *llm.Request) (*wireRequest, error) {
	w := &wireRequest{
		Model: req.Model, MaxTokens: req.Params.MaxTokens,
		Temperature: req.Params.Temperature, TopP: req.Params.TopP, StopSequences: req.Params.Stop,
	}
	if w.Model == "" {
		w.Model = c.model
	}
	if w.MaxTokens == 0 {
		w.MaxTokens = c.maxTokens
	}
	if r := req.Params.Reasoning; r != nil && r.BudgetTokens > 0 {
		w.Thinking = &wireThinking{Type: "enabled", BudgetTokens: r.BudgetTokens}
	}
	for _, b := range req.System {
		t, ok := b.(llm.Text)
		if !ok {
			return nil, fmt.Errorf("system blocks must be text, got %s", b.BlockType())
		}
		w.System = append(w.System, wireBlock{Type: "text", Text: t.Text})
	}
	for _, t := range req.Tools {
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		w.Tools = append(w.Tools, wireTool{Name: t.Name, Description: t.Description, InputSchema: schema})
	}
	if tc := req.ToolChoice; tc != nil {
		w.ToolChoice = &wireToolChoice{Type: string(tc.Mode), Name: tc.Name}
	}
	for _, m := range req.Messages {
		blocks, err := toWire(m.Blocks)
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		w.Messages = append(w.Messages, wireMessage{Role: string(m.Role), Content: blocks})
	}
	return w, nil
}

// toWire converts blocks to API content items. Unknown blocks (block types
// agentflow doesn't model, which came from this API) are sent back verbatim.
func toWire(bs llm.Blocks) ([]json.RawMessage, error) {
	var blocks []wireBlock
	out := make([]json.RawMessage, 0, len(bs))
	flush := func() error {
		for _, wb := range blocks {
			raw, err := json.Marshal(wb)
			if err != nil {
				return err
			}
			out = append(out, raw)
		}
		blocks = blocks[:0]
		return nil
	}
	for _, b := range bs {
		if u, ok := b.(llm.Unknown); ok {
			if err := flush(); err != nil {
				return nil, err
			}
			out = append(out, u.Raw)
			continue
		}
		switch v := b.(type) {
		case llm.Text:
			blocks = append(blocks, wireBlock{Type: "text", Text: v.Text})
		case llm.Media:
			wb, err := mediaToWire(v)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, wb)
		case llm.ToolUse:
			input := v.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			blocks = append(blocks, wireBlock{Type: "tool_use", ID: v.CallID, Name: v.Name, Input: input})
		case llm.ToolResult:
			content, err := toWire(v.Content)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, wireBlock{Type: "tool_result", ToolUseID: v.CallID, Content: content, IsError: v.IsError})
		case llm.Reasoning:
			raw, ok := v.Ext[extKey]
			if !ok {
				continue // another provider's reasoning: its signature can't be verified here
			}
			var ext reasoningExt
			if err := json.Unmarshal(raw, &ext); err != nil {
				return nil, fmt.Errorf("reasoning ext: %w", err)
			}
			if v.Redacted {
				blocks = append(blocks, wireBlock{Type: "redacted_thinking", Data: ext.Data})
			} else {
				blocks = append(blocks, wireBlock{Type: "thinking", Thinking: v.Text, Signature: ext.Signature})
			}
		case llm.Data:
			blocks = append(blocks, wireBlock{Type: "text", Text: string(v.Value)})
		default:
			return nil, fmt.Errorf("unsupported block type %q", b.BlockType())
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return out, nil
}

func mediaToWire(m llm.Media) (wireBlock, error) {
	kind := "image"
	switch {
	case strings.HasPrefix(m.MediaType, "image/"):
	case m.MediaType == "application/pdf":
		kind = "document"
	default:
		return wireBlock{}, fmt.Errorf("unsupported media type %q", m.MediaType)
	}
	if m.URL != "" {
		return wireBlock{Type: kind, Source: &wireSource{Type: "url", URL: m.URL}}, nil
	}
	return wireBlock{Type: kind, Source: &wireSource{Type: "base64", MediaType: m.MediaType, Data: base64.StdEncoding.EncodeToString(m.Data)}}, nil
}

func buildResponse(w *wireResponse) (*llm.Response, error) {
	blocks := make(llm.Blocks, 0, len(w.Content))
	for _, raw := range w.Content {
		var wb wireBlock
		if err := json.Unmarshal(raw, &wb); err != nil {
			return nil, &llm.Error{Code: llm.ErrCodeServer, Message: "decode content block: " + err.Error()}
		}
		switch wb.Type {
		case "text":
			blocks = append(blocks, llm.Text{Text: wb.Text})
		case "tool_use":
			input := wb.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			blocks = append(blocks, llm.ToolUse{CallID: wb.ID, Name: wb.Name, Input: input})
		case "thinking":
			ext, _ := json.Marshal(reasoningExt{Signature: wb.Signature})
			blocks = append(blocks, llm.Reasoning{Text: wb.Thinking, Ext: llm.Ext{extKey: ext}})
		case "redacted_thinking":
			ext, _ := json.Marshal(reasoningExt{Data: wb.Data})
			blocks = append(blocks, llm.Reasoning{Redacted: true, Ext: llm.Ext{extKey: ext}})
		default:
			blocks = append(blocks, llm.Unknown{Type: llm.BlockType(wb.Type), Raw: raw})
		}
	}
	meta, _ := json.Marshal(map[string]string{"id": w.ID})
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Blocks: blocks},
		StopReason: stopReason(w.StopReason),
		Usage: llm.Usage{
			InputTokens: w.Usage.InputTokens, OutputTokens: w.Usage.OutputTokens,
			CacheReadTokens: w.Usage.CacheReadInputTokens, CacheWriteTokens: w.Usage.CacheCreationInputTokens,
		},
		Model: w.Model,
		Ext:   llm.Ext{extKey: meta},
	}, nil
}

func stopReason(s string) llm.StopReason {
	switch s {
	case "end_turn":
		return llm.StopEndTurn
	case "tool_use":
		return llm.StopToolUse
	case "max_tokens":
		return llm.StopMaxTokens
	case "stop_sequence":
		return llm.StopStopSequence
	case "refusal":
		return llm.StopRefusal
	}
	return llm.StopOther
}

// apiError maps an HTTP error response to *llm.Error.
func apiError(status int, body []byte) error {
	var wrapper struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := strings.TrimSpace(string(body))
	if json.Unmarshal(body, &wrapper) == nil && wrapper.Error.Message != "" {
		msg = wrapper.Error.Type + ": " + wrapper.Error.Message
	}
	e := &llm.Error{Message: msg, Status: status}
	switch {
	case status == http.StatusTooManyRequests:
		e.Code, e.Retryable = llm.ErrCodeRateLimited, true
	case status == 529:
		e.Code, e.Retryable = llm.ErrCodeOverloaded, true
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		e.Code = llm.ErrCodeAuth
	case status == http.StatusBadRequest && strings.Contains(strings.ToLower(msg), "too long"):
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
