package trace

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace/canonical"
)

// Fingerprint returns "sha256:<hex>" of the canonical JSON (RFC 8785) of v.
func Fingerprint(v any) (string, error) {
	c, err := canonical.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("trace: fingerprint: %w", err)
	}
	sum := sha256.Sum256(c)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// fpTool is the part of a tool spec that is sent to the model. Annotations
// are not sent, so they are not fingerprinted.
type fpTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type fpRequest struct {
	Model      string          `json:"model"`
	System     llm.Blocks      `json:"system"`
	Messages   []llm.Message   `json:"messages"`
	Tools      []fpTool        `json:"tools"`
	ToolChoice *llm.ToolChoice `json:"tool_choice"`
	Params     llm.Params      `json:"params"`
}

// RequestFingerprint fingerprints an LLM request exactly as it is sent
// (RFC-0001 §9): model, system, messages (ephemerals included), tools sorted
// by name without annotations, tool choice, and params.
func RequestFingerprint(req *llm.Request) (string, error) {
	tools := make([]fpTool, 0, len(req.Tools))
	for _, t := range req.Tools {
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage("null")
		}
		tools = append(tools, fpTool{Name: t.Name, Description: t.Description, InputSchema: schema})
	}
	sort.SliceStable(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	msgs := req.Messages
	if msgs == nil {
		msgs = []llm.Message{}
	}
	return Fingerprint(fpRequest{
		Model:      req.Model,
		System:     req.System,
		Messages:   msgs,
		Tools:      tools,
		ToolChoice: req.ToolChoice,
		Params:     req.Params,
	})
}

func orEmptyObject(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	return raw
}

// ToolFingerprint fingerprints a tool call: {tool, input}.
func ToolFingerprint(tool string, input json.RawMessage) (string, error) {
	return Fingerprint(struct {
		Tool  string          `json:"tool"`
		Input json.RawMessage `json:"input"`
	}{tool, orEmptyObject(input)})
}

// EffectFingerprint fingerprints a custom effect: {name, input}.
func EffectFingerprint(name string, input json.RawMessage) (string, error) {
	if len(input) == 0 {
		input = json.RawMessage("null")
	}
	return Fingerprint(struct {
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	}{name, input})
}

// SpawnFingerprint fingerprints a child-run request: {agent, task, key}. The
// child's own config is deliberately excluded (RFC-0003 §4).
func SpawnFingerprint(agent string, task llm.Blocks, key string) (string, error) {
	return Fingerprint(struct {
		Agent string     `json:"agent"`
		Task  llm.Blocks `json:"task"`
		Key   string     `json:"key,omitempty"`
	}{agent, task, key})
}

// ConfigFingerprint fingerprints an agent's config (run.started config_fp).
func ConfigFingerprint(c *Config) (string, error) {
	if c == nil {
		return "", nil
	}
	return Fingerprint(c)
}
