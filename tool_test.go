package agentflow

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"testing"
)

// simpleTool is a minimal Tool implementation for tests.
type simpleTool struct {
	name   string
	output string
}

func (t *simpleTool) Schema() ToolSchema {
	return ToolSchema{
		Name:        t.name,
		Description: "test tool: " + t.name,
		Parameters:  json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}}}`),
	}
}

func (t *simpleTool) Execute(_ context.Context, _ json.RawMessage) (*ToolResult, error) {
	return &ToolResult{Output: t.output}, nil
}

func TestToolRegistry_Register(t *testing.T) {
	reg := NewToolRegistry()

	tool := &simpleTool{name: "calc", output: "42"}
	if err := reg.Register(tool); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if reg.Len() != 1 {
		t.Errorf("Len() = %d, want 1", reg.Len())
	}
}

func TestToolRegistry_RegisterDuplicate(t *testing.T) {
	reg := NewToolRegistry()

	tool := &simpleTool{name: "calc", output: "42"}
	if err := reg.Register(tool); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}

	err := reg.Register(tool)
	if err == nil {
		t.Fatal("expected error for duplicate registration")
	}
}

func TestToolRegistry_RegisterEmptyName(t *testing.T) {
	reg := NewToolRegistry()
	tool := &simpleTool{name: "", output: "x"}
	err := reg.Register(tool)
	if err == nil {
		t.Fatal("expected error for empty tool name")
	}
}

func TestToolRegistry_Get(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(&simpleTool{name: "calc", output: "42"})

	got := reg.Get("calc")
	if got == nil {
		t.Fatal("Get() returned nil for registered tool")
	}

	notFound := reg.Get("nonexistent")
	if notFound != nil {
		t.Fatal("Get() returned non-nil for unregistered tool")
	}
}

func TestToolRegistry_Schemas(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(&simpleTool{name: "tool_a", output: "a"})
	reg.Register(&simpleTool{name: "tool_b", output: "b"})

	schemas := reg.Schemas()
	if len(schemas) != 2 {
		t.Fatalf("Schemas() len = %d, want 2", len(schemas))
	}

	// Sort for deterministic comparison.
	sort.Slice(schemas, func(i, j int) bool {
		return schemas[i].Name < schemas[j].Name
	})
	if schemas[0].Name != "tool_a" || schemas[1].Name != "tool_b" {
		t.Errorf("Schemas() names = [%s, %s], want [tool_a, tool_b]", schemas[0].Name, schemas[1].Name)
	}
}

func TestToolRegistry_Names(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(&simpleTool{name: "z_tool", output: ""})
	reg.Register(&simpleTool{name: "a_tool", output: ""})

	names := reg.Names()
	if len(names) != 2 {
		t.Fatalf("Names() len = %d, want 2", len(names))
	}
}

func TestToolRegistry_ConcurrentAccess(t *testing.T) {
	reg := NewToolRegistry()
	var wg sync.WaitGroup

	// Register tools concurrently.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			name := fmt.Sprintf("tool_%d", i)
			reg.Register(&simpleTool{name: name, output: "ok"})
		}(i)
	}

	// Read concurrently while writing.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reg.Schemas()
			reg.Len()
			reg.Names()
		}()
	}

	wg.Wait()

	if reg.Len() != 50 {
		t.Errorf("Len() = %d, want 50", reg.Len())
	}
}

func TestToolResult_JSON(t *testing.T) {
	result := ToolResult{Output: "42", Error: ""}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}

	var decoded ToolResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if decoded.Output != result.Output {
		t.Errorf("Output = %q, want %q", decoded.Output, result.Output)
	}
}
