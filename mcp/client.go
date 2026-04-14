package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/kumarlokesh/agentflow"
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
}

type toolsCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type toolsCallResult struct {
	Content []toolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

type toolContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Client is an MCP client that discovers tools from an MCP server and exposes
// them as agentflow.Tool implementations. The discovered tools flow through
// the normal agentflow event-sourced pipeline and are fully replayable.
//
// Typical usage:
//
//	transport, _ := mcp.NewStdioTransport("npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp")
//	client, _ := mcp.NewClient(transport)
//	defer client.Close()
//
//	agent, _ := agentflow.NewAgent(agentflow.AgentConfig{
//	    Tools: client.Tools(),
//	    ...
//	})
type Client struct {
	transport Transport
	tools     []agentflow.Tool
}

// NewClient performs the MCP initialize handshake and discovers available
// tools. Returns an error if the server does not support tools or the
// handshake fails.
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

// Tools returns the agentflow.Tool wrappers for all tools discovered from the
// MCP server. The slice is ordered as returned by the server's tools/list.
func (c *Client) Tools() []agentflow.Tool {
	out := make([]agentflow.Tool, len(c.tools))
	copy(out, c.tools)
	return out
}

// Close shuts down the underlying transport.
func (c *Client) Close() error {
	return c.transport.Close()
}

// initialize performs the MCP initialize + initialized handshake.
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

	// Send the initialized notification (no response expected).
	notif := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}
	notifBytes, _ := json.Marshal(notif)
	notifBytes = append(notifBytes, '\n')

	// Best-effort - some servers don't require it and don't fail without it.
	// We write directly if the transport is a StdioTransport; otherwise skip.
	if st, ok := c.transport.(*StdioTransport); ok {
		st.mu.Lock()
		st.stdin.Write(notifBytes) //nolint:errcheck
		st.mu.Unlock()
	}

	return nil
}

// discoverTools calls tools/list and builds the agentflow.Tool adapters.
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
			schema: agentflow.ToolSchema{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  schema,
			},
		})
	}
	return nil
}

// --- mcpTool: agentflow.Tool adapter for a single MCP tool ---

type mcpTool struct {
	transport Transport
	schema    agentflow.ToolSchema
}

var _ agentflow.Tool = (*mcpTool)(nil)

func (t *mcpTool) Schema() agentflow.ToolSchema { return t.schema }

func (t *mcpTool) Execute(ctx context.Context, params json.RawMessage) (*agentflow.ToolResult, error) {
	if len(params) == 0 {
		params = json.RawMessage(`{}`)
	}

	callParams, err := json.Marshal(toolsCallParams{
		Name:      t.schema.Name,
		Arguments: params,
	})
	if err != nil {
		return nil, fmt.Errorf("mcp: marshal tool call params: %w", err)
	}

	raw, err := t.transport.Send(ctx, "tools/call", callParams)
	if err != nil {
		return &agentflow.ToolResult{Error: err.Error()}, nil
	}

	var result toolsCallResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("mcp: decode tool result: %w", err)
	}

	if result.IsError {
		errMsg := extractText(result.Content)
		if errMsg == "" {
			errMsg = "tool reported an error"
		}
		return &agentflow.ToolResult{Error: errMsg}, nil
	}

	return &agentflow.ToolResult{Output: extractText(result.Content)}, nil
}

// extractText concatenates all text content blocks in an MCP result.
func extractText(blocks []toolContent) string {
	out := ""
	for _, b := range blocks {
		if b.Type == "text" {
			out += b.Text
		}
	}
	return out
}
