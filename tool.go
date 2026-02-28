package agentflow

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// ToolSchema describes a tool's interface for LLM function-calling.
type ToolSchema struct {
	// Name is the unique identifier for the tool.
	Name string `json:"name"`
	// Description explains what the tool does (shown to the LLM).
	Description string `json:"description"`
	// Parameters is the JSON Schema describing the tool's input parameters.
	Parameters json.RawMessage `json:"parameters"`
}

// ToolResult is the output of a tool execution.
type ToolResult struct {
	// Output is the tool's string output on success.
	Output string `json:"output"`
	// Error is the error message if the tool failed.
	Error string `json:"error,omitempty"`
}

// Tool represents an executable capability available to the agent.
// Implementations must be safe for concurrent use.
type Tool interface {
	// Schema returns the tool's metadata and parameter schema.
	Schema() ToolSchema
	// Execute runs the tool with the given JSON parameters.
	Execute(ctx context.Context, params json.RawMessage) (*ToolResult, error)
}

// ToolRegistry is a thread-safe registry of named tools.
type ToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewToolRegistry creates an empty ToolRegistry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry. Returns an error if a tool with the
// same name is already registered.
func (r *ToolRegistry) Register(t Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := t.Schema().Name
	if name == "" {
		return fmt.Errorf("agentflow: tool name must not be empty")
	}
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("agentflow: tool %q already registered", name)
	}
	r.tools[name] = t
	return nil
}

// Get retrieves a tool by name. Returns nil if not found.
func (r *ToolRegistry) Get(name string) Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tools[name]
}

// Schemas returns the ToolSchema for every registered tool.
func (r *ToolRegistry) Schemas() []ToolSchema {
	r.mu.RLock()
	defer r.mu.RUnlock()

	schemas := make([]ToolSchema, 0, len(r.tools))
	for _, t := range r.tools {
		schemas = append(schemas, t.Schema())
	}
	return schemas
}

// Names returns the names of all registered tools in sorted order.
func (r *ToolRegistry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

// Len returns the number of registered tools.
func (r *ToolRegistry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}
