// Package llm defines the provider-neutral language-model types used by
// agentflow: messages made of content blocks, requests, responses, and the
// Model interface that providers implement.
//
// The package depends only on the standard library so that providers can be
// used without the agentflow runtime.
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Role is the author of a message. Tool results are ToolResult blocks inside
// a user message; there is no separate tool role.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn of a conversation.
type Message struct {
	Role   Role   `json:"role"`
	Blocks Blocks `json:"blocks"`
}

// Ext carries provider-opaque data keyed by provider name, for example
// {"anthropic": {"signature": "..."}}. It is preserved byte-for-byte and sent
// back to the provider unchanged.
type Ext map[string]json.RawMessage

// BlockType identifies a content block's kind in JSON ("type" field).
type BlockType string

const (
	TypeText       BlockType = "text"
	TypeMedia      BlockType = "media"
	TypeToolUse    BlockType = "tool_use"
	TypeToolResult BlockType = "tool_result"
	TypeReasoning  BlockType = "reasoning"
	TypeData       BlockType = "data"
)

// Block is one piece of message content. It is a sealed interface: the
// concrete types are Text, Media, ToolUse, ToolResult, Reasoning, Data, and
// Unknown (a block type this version doesn't know, preserved as-is).
type Block interface {
	BlockType() BlockType
	isBlock()
}

// Text is plain text.
type Text struct {
	Text string `json:"text"`
	Ext  Ext    `json:"ext,omitempty"`
}

// Media is binary content such as an image, PDF, or audio clip. Exactly one
// of Data or URL is normally set.
type Media struct {
	MediaType string `json:"media_type"`
	Data      []byte `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	Ext       Ext    `json:"ext,omitempty"`
}

// ToolUse is a tool call requested by the model.
type ToolUse struct {
	CallID string          `json:"call_id"`
	Name   string          `json:"name"`
	Input  json.RawMessage `json:"input"`
	Ext    Ext             `json:"ext,omitempty"`
}

// ToolResult is the result of a tool call, sent back to the model.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content Blocks `json:"content"`
	IsError bool   `json:"is_error,omitempty"`
	Ext     Ext    `json:"ext,omitempty"`
}

// Reasoning is model reasoning ("thinking"). Providers keep signatures and
// encrypted content in Ext; such blocks must never be edited.
type Reasoning struct {
	Text     string `json:"text,omitempty"`
	Redacted bool   `json:"redacted,omitempty"`
	Ext      Ext    `json:"ext,omitempty"`
}

// Data is structured content, such as structured output or MCP structuredContent.
type Data struct {
	Value json.RawMessage `json:"value"`
	Ext   Ext             `json:"ext,omitempty"`
}

// Unknown is a block whose type this version of agentflow doesn't know. It is
// kept verbatim so that reading and re-writing a trace never loses data.
type Unknown struct {
	Type BlockType
	Raw  json.RawMessage
}

func (Text) BlockType() BlockType       { return TypeText }
func (Media) BlockType() BlockType      { return TypeMedia }
func (ToolUse) BlockType() BlockType    { return TypeToolUse }
func (ToolResult) BlockType() BlockType { return TypeToolResult }
func (Reasoning) BlockType() BlockType  { return TypeReasoning }
func (Data) BlockType() BlockType       { return TypeData }
func (u Unknown) BlockType() BlockType  { return u.Type }

func (Text) isBlock()       {}
func (Media) isBlock()      {}
func (ToolUse) isBlock()    {}
func (ToolResult) isBlock() {}
func (Reasoning) isBlock()  {}
func (Data) isBlock()       {}
func (Unknown) isBlock()    {}

// typed prepends the "type" discriminator to a block's JSON object.
func typed(t BlockType, v any) ([]byte, error) {
	body, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	head := fmt.Sprintf(`{"type":%q`, t)
	if bytes.Equal(body, []byte("{}")) {
		return []byte(head + "}"), nil
	}
	return append([]byte(head+","), body[1:]...), nil
}

func (b Text) MarshalJSON() ([]byte, error) {
	type plain Text
	return typed(TypeText, plain(b))
}

func (b Media) MarshalJSON() ([]byte, error) {
	type plain Media
	return typed(TypeMedia, plain(b))
}

func (b ToolUse) MarshalJSON() ([]byte, error) {
	type plain ToolUse
	if len(b.Input) == 0 {
		b.Input = json.RawMessage("{}")
	}
	return typed(TypeToolUse, plain(b))
}

func (b ToolResult) MarshalJSON() ([]byte, error) {
	type plain ToolResult
	if b.Content == nil {
		b.Content = Blocks{}
	}
	return typed(TypeToolResult, plain(b))
}

func (b Reasoning) MarshalJSON() ([]byte, error) {
	type plain Reasoning
	return typed(TypeReasoning, plain(b))
}

func (b Data) MarshalJSON() ([]byte, error) {
	type plain Data
	if len(b.Value) == 0 {
		b.Value = json.RawMessage("null")
	}
	return typed(TypeData, plain(b))
}

func (u Unknown) MarshalJSON() ([]byte, error) {
	if len(u.Raw) == 0 {
		return nil, fmt.Errorf("llm: unknown block %q has no raw content", u.Type)
	}
	return u.Raw, nil
}

// Blocks is a list of content blocks with polymorphic JSON decoding.
// A nil Blocks encodes as [] rather than null.
type Blocks []Block

// MarshalJSON encodes the blocks; nil encodes as an empty array.
func (bs Blocks) MarshalJSON() ([]byte, error) {
	if bs == nil {
		return []byte("[]"), nil
	}
	return json.Marshal([]Block(bs))
}

// UnmarshalJSON decodes each element by its "type" field.
func (bs *Blocks) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return fmt.Errorf("llm: blocks: %w", err)
	}
	if raws == nil {
		*bs = nil
		return nil
	}
	out := make(Blocks, 0, len(raws))
	for i, raw := range raws {
		b, err := UnmarshalBlock(raw)
		if err != nil {
			return fmt.Errorf("llm: block %d: %w", i, err)
		}
		out = append(out, b)
	}
	*bs = out
	return nil
}

// UnmarshalBlock decodes one block by its "type" field. Unrecognized types
// decode to Unknown.
func UnmarshalBlock(raw json.RawMessage) (Block, error) {
	var head struct {
		Type BlockType `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, err
	}
	switch head.Type {
	case TypeText:
		var b Text
		return b, json.Unmarshal(raw, &b)
	case TypeMedia:
		var b Media
		return b, json.Unmarshal(raw, &b)
	case TypeToolUse:
		var b ToolUse
		return b, json.Unmarshal(raw, &b)
	case TypeToolResult:
		var b ToolResult
		return b, json.Unmarshal(raw, &b)
	case TypeReasoning:
		var b Reasoning
		return b, json.Unmarshal(raw, &b)
	case TypeData:
		var b Data
		return b, json.Unmarshal(raw, &b)
	case "":
		return nil, fmt.Errorf("missing block type")
	default:
		return Unknown{Type: head.Type, Raw: append(json.RawMessage(nil), raw...)}, nil
	}
}

// TextOf concatenates the text of all Text blocks, separated by newlines.
func TextOf(bs Blocks) string {
	var buf bytes.Buffer
	for _, b := range bs {
		if t, ok := b.(Text); ok {
			if buf.Len() > 0 {
				buf.WriteByte('\n')
			}
			buf.WriteString(t.Text)
		}
	}
	return buf.String()
}
