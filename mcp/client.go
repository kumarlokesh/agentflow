package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
)

// --- MCP protocol types ---

type initializeParams struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    clientCapabilities `json:"capabilities"`
	ClientInfo      implementation     `json:"clientInfo"`
}

type clientCapabilities struct{}

type implementation struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type initializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    serverCapabilities `json:"capabilities"`
	ServerInfo      implementation     `json:"serverInfo"`
}

type serverCapabilities struct {
	Tools *struct{} `json:"tools,omitempty"`
}

type toolsListResult struct {
	Tools []mcpToolDef `json:"tools"`
}

type mcpToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Annotations *mcpAnnotations `json:"annotations,omitempty"`
}

type mcpAnnotations struct {
	ReadOnlyHint    *bool `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool `json:"destructiveHint,omitempty"`
	IdempotentHint  *bool `json:"idempotentHint,omitempty"`
}

type toolsCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type toolsCallResult struct {
	Content           []json.RawMessage `json:"content"`
	StructuredContent json.RawMessage   `json:"structuredContent,omitempty"`
	IsError           bool              `json:"isError,omitempty"`
}

type toolContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"` // base64 (image, audio)
	MimeType string `json:"mimeType,omitempty"`
	URI      string `json:"uri,omitempty"` // resource_link
	Resource *struct {
		URI      string `json:"uri"`
		MimeType string `json:"mimeType,omitempty"`
		Text     string `json:"text,omitempty"`
		Blob     string `json:"blob,omitempty"` // base64
	} `json:"resource,omitempty"`
}

// Client is an MCP client that discovers tools from an MCP server and exposes
// them as agentflow.Tool implementations. Calls to them are effects like any
// other tool call: recorded, and replayable without the server.
//
// Typical usage:
//
//	transport, _ := mcp.NewStdioTransport("npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp")
//	client, _ := mcp.NewClient(transport)
//	defer client.Close()
//	agent, _ := agentflow.NewAgent(agentflow.AgentConfig{Tools: client.Tools(), ...})
type Client struct {
	transport Transport
	tools     []agentflow.Tool
}

// NewClient performs the MCP initialize handshake and discovers the
// server's tools.
func NewClient(transport Transport) (*Client, error) {
	c := &Client{transport: transport}
	if err := c.initialize(); err != nil {
		return nil, err
	}
	if err := c.discoverTools(); err != nil {
		return nil, err
	}
	return c, nil
}

// Tools returns the discovered tools, in the server's order.
func (c *Client) Tools() []agentflow.Tool {
	out := make([]agentflow.Tool, len(c.tools))
	copy(out, c.tools)
	return out
}

// Close shuts down the underlying transport.
func (c *Client) Close() error {
	return c.transport.Close()
}

func (c *Client) initialize() error {
	params, _ := json.Marshal(initializeParams{
		ProtocolVersion: "2024-11-05",
		Capabilities:    clientCapabilities{},
		ClientInfo:      implementation{Name: "agentflow", Version: "1.0.0"},
	})
	raw, err := c.transport.Send(context.Background(), "initialize", params)
	if err != nil {
		return fmt.Errorf("mcp: initialize: %w", err)
	}
	var result initializeResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return fmt.Errorf("mcp: decode initialize result: %w", err)
	}
	// The initialized notification gets no response. Best effort: some
	// servers don't require it.
	if st, ok := c.transport.(*StdioTransport); ok {
		notif, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", Method: "notifications/initialized"})
		st.mu.Lock()
		_, _ = st.stdin.Write(append(notif, '\n'))
		st.mu.Unlock()
	}
	return nil
}

func (c *Client) discoverTools() error {
	raw, err := c.transport.Send(context.Background(), "tools/list", nil)
	if err != nil {
		return fmt.Errorf("mcp: tools/list: %w", err)
	}
	var result toolsListResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return fmt.Errorf("mcp: decode tools/list: %w", err)
	}
	c.tools = make([]agentflow.Tool, 0, len(result.Tools))
	for _, def := range result.Tools {
		schema := def.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		c.tools = append(c.tools, &mcpTool{
			transport: c.transport,
			spec: llm.ToolSpec{
				Name: def.Name, Description: def.Description, InputSchema: schema,
				Annotations: annotations(def.Annotations),
			},
		})
	}
	return nil
}

// annotations maps MCP tool hints. Absent annotations stay unknown (nil).
// Per the MCP spec, destructiveHint defaults to true for tools that aren't
// read-only.
func annotations(a *mcpAnnotations) *llm.ToolAnnotations {
	if a == nil {
		return nil
	}
	out := &llm.ToolAnnotations{}
	if a.ReadOnlyHint != nil {
		out.ReadOnly = *a.ReadOnlyHint
	}
	if a.DestructiveHint != nil {
		out.Destructive = *a.DestructiveHint && !out.ReadOnly
	} else {
		out.Destructive = !out.ReadOnly
	}
	if a.IdempotentHint != nil {
		out.Idempotent = *a.IdempotentHint
	}
	return out
}

// mcpTool adapts one MCP tool to agentflow.Tool.
type mcpTool struct {
	transport Transport
	spec      llm.ToolSpec
}

var _ agentflow.Tool = (*mcpTool)(nil)

func (t *mcpTool) Spec() llm.ToolSpec { return t.spec }

// Call invokes tools/call. Transport failures are tool failures (a Go
// error); a result with isError is shown to the model as an error result.
func (t *mcpTool) Call(ctx context.Context, input json.RawMessage) (agentflow.ToolOutput, error) {
	if len(input) == 0 {
		input = json.RawMessage(`{}`)
	}
	params, err := json.Marshal(toolsCallParams{Name: t.spec.Name, Arguments: input})
	if err != nil {
		return agentflow.ToolOutput{}, fmt.Errorf("mcp: marshal tools/call params: %w", err)
	}
	raw, err := t.transport.Send(ctx, "tools/call", params)
	if err != nil {
		return agentflow.ToolOutput{}, fmt.Errorf("mcp: tools/call %s: %w", t.spec.Name, err)
	}
	var result toolsCallResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return agentflow.ToolOutput{}, fmt.Errorf("mcp: decode tool result: %w", err)
	}
	blocks, err := contentBlocks(result.Content)
	if err != nil {
		return agentflow.ToolOutput{}, err
	}
	if len(result.StructuredContent) > 0 && string(result.StructuredContent) != "null" {
		blocks = append(blocks, llm.Data{Value: result.StructuredContent})
	}
	if result.IsError && len(blocks) == 0 {
		blocks = llm.Blocks{llm.Text{Text: "tool reported an error"}}
	}
	return agentflow.ToolOutput{Content: blocks, IsError: result.IsError}, nil
}

// contentBlocks maps MCP content to llm blocks without dropping anything:
// content types agentflow doesn't model are kept as llm.Unknown.
func contentBlocks(items []json.RawMessage) (llm.Blocks, error) {
	out := llm.Blocks{}
	for _, raw := range items {
		var c toolContent
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("mcp: decode content: %w", err)
		}
		switch c.Type {
		case "text":
			out = append(out, llm.Text{Text: c.Text})
		case "image", "audio":
			data, err := base64.StdEncoding.DecodeString(c.Data)
			if err != nil {
				return nil, fmt.Errorf("mcp: %s content: %w", c.Type, err)
			}
			out = append(out, llm.Media{MediaType: c.MimeType, Data: data})
		case "resource":
			if c.Resource == nil {
				return nil, fmt.Errorf("mcp: resource content without resource")
			}
			if c.Resource.Blob != "" {
				data, err := base64.StdEncoding.DecodeString(c.Resource.Blob)
				if err != nil {
					return nil, fmt.Errorf("mcp: resource %s: %w", c.Resource.URI, err)
				}
				out = append(out, llm.Media{MediaType: c.Resource.MimeType, Data: data, URL: c.Resource.URI})
			} else {
				out = append(out, llm.Text{Text: c.Resource.Text})
			}
		case "resource_link":
			out = append(out, llm.Text{Text: c.URI})
		default:
			out = append(out, llm.Unknown{Type: llm.BlockType("mcp." + c.Type), Raw: append(json.RawMessage(nil), raw...)})
		}
	}
	return out, nil
}
