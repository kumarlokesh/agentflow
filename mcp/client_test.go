package mcp_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/mcp"
)

// newMCPServer returns a Transport backed by an in-process MCP server that
// responds to initialize and tools/list with the given tool definitions.
func newMCPServer(t *testing.T, tools []map[string]any) mcp.Transport {
	t.Helper()
	return newPipedTransport(t, func(s *mockServer) {
		s.handle("initialize", func(_ json.RawMessage) (any, error) {
			return map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test-server", "version": "1.0"},
			}, nil
		})
		s.handle("tools/list", func(_ json.RawMessage) (any, error) {
			return map[string]any{"tools": tools}, nil
		})
		s.handle("tools/call", func(params json.RawMessage) (any, error) {
			var p struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			json.Unmarshal(params, &p)
			return map[string]any{
				"content": []map[string]any{
					{"type": "text", "text": fmt.Sprintf("called %s", p.Name)},
				},
				"isError": false,
			}, nil
		})
	})
}

func TestClient_ToolDiscovery(t *testing.T) {
	transport := newMCPServer(t, []map[string]any{
		{
			"name":        "read_file",
			"description": "Reads a file",
			"inputSchema": map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":   []string{"path"},
			},
		},
		{
			"name":        "write_file",
			"description": "Writes a file",
			"inputSchema": map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}},
				"required":   []string{"path", "content"},
			},
		},
	})

	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	tools := client.Tools()
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Schema().Name] = true
	}
	if !names["read_file"] || !names["write_file"] {
		t.Errorf("tools = %v, want read_file and write_file", names)
	}
}

func TestClient_ToolSchemaPreserved(t *testing.T) {
	inputSchema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"path": map[string]any{"type": "string"}},
		"required":   []string{"path"},
	}
	transport := newMCPServer(t, []map[string]any{
		{"name": "read_file", "description": "Reads a file", "inputSchema": inputSchema},
	})

	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	tool := client.Tools()[0]
	schema := tool.Schema()

	if schema.Name != "read_file" {
		t.Errorf("Name = %q, want read_file", schema.Name)
	}
	if schema.Description != "Reads a file" {
		t.Errorf("Description = %q, want 'Reads a file'", schema.Description)
	}
	// Schema must be valid JSON.
	var parsed map[string]any
	if err := json.Unmarshal(schema.Parameters, &parsed); err != nil {
		t.Errorf("Parameters is not valid JSON: %v", err)
	}
}

func TestClient_ToolSatisfiesAgentflowInterface(t *testing.T) {
	transport := newMCPServer(t, []map[string]any{
		{"name": "calculator", "description": "Does math", "inputSchema": map[string]any{"type": "object"}},
	})
	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	tools := client.Tools()
	// Verify the tool implements the agentflow.Tool interface at runtime.
	var _ agentflow.Tool = tools[0]
}

func TestClient_ToolExecute(t *testing.T) {
	transport := newMCPServer(t, []map[string]any{
		{"name": "greet", "description": "Greets", "inputSchema": map[string]any{"type": "object"}},
	})
	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	tool := client.Tools()[0]
	params, _ := json.Marshal(map[string]string{"name": "world"})
	result, err := tool.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.Error != "" {
		t.Errorf("unexpected tool error: %s", result.Error)
	}
	if result.Output == "" {
		t.Error("expected non-empty output")
	}
}

func TestClient_ToolExecuteError(t *testing.T) {
	transport := newPipedTransport(t, func(s *mockServer) {
		s.handle("initialize", func(_ json.RawMessage) (any, error) {
			return map[string]any{"protocolVersion": "2024-11-05", "capabilities": map[string]any{}, "serverInfo": map[string]any{"name": "s", "version": "1"}}, nil
		})
		s.handle("tools/list", func(_ json.RawMessage) (any, error) {
			return map[string]any{"tools": []map[string]any{
				{"name": "fail_tool", "description": "Always fails", "inputSchema": map[string]any{"type": "object"}},
			}}, nil
		})
		s.handle("tools/call", func(_ json.RawMessage) (any, error) {
			return map[string]any{
				"content": []map[string]any{{"type": "text", "text": "permission denied"}},
				"isError": true,
			}, nil
		})
	})

	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	result, err := client.Tools()[0].Execute(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Error == "" {
		t.Error("expected tool error to be propagated")
	}
	if result.Error != "permission denied" {
		t.Errorf("error = %q, want 'permission denied'", result.Error)
	}
}

func TestClient_EmptyToolList(t *testing.T) {
	transport := newPipedTransport(t, func(s *mockServer) {
		s.handle("initialize", func(_ json.RawMessage) (any, error) {
			return map[string]any{"protocolVersion": "2024-11-05", "capabilities": map[string]any{}, "serverInfo": map[string]any{"name": "s", "version": "1"}}, nil
		})
		s.handle("tools/list", func(_ json.RawMessage) (any, error) {
			return map[string]any{"tools": []any{}}, nil
		})
	})

	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	if len(client.Tools()) != 0 {
		t.Errorf("expected 0 tools, got %d", len(client.Tools()))
	}
}

func TestClient_InitializeFailure(t *testing.T) {
	// Server that returns an error on initialize.
	transport := newPipedTransport(t, func(s *mockServer) {
		s.handle("initialize", func(_ json.RawMessage) (any, error) {
			return nil, fmt.Errorf("unsupported protocol version")
		})
	})

	_, err := mcp.NewClient(transport)
	if err == nil {
		t.Fatal("expected error on initialize failure")
	}
}

func TestClient_Tools_ReturnsCopy(t *testing.T) {
	transport := newMCPServer(t, []map[string]any{
		{"name": "t1", "description": "tool 1", "inputSchema": map[string]any{"type": "object"}},
	})
	client, err := mcp.NewClient(transport)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	t1 := client.Tools()
	t2 := client.Tools()
	// Modifying one slice must not affect the other.
	t1[0] = nil
	if t2[0] == nil {
		t.Error("Tools() returned the same underlying slice — must return a copy")
	}
}

// Satisfy the io.WriteCloser interface for io.PipeWriter in newPipedTransport.
var _ io.WriteCloser = (*io.PipeWriter)(nil)
