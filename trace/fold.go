package trace

import (
	"fmt"
	"sort"

	"github.com/kumarlokesh/agentflow/llm"
)

// Folder replays a run's events in seq order and tracks the state needed to
// reconstruct requests: the conversation, the effective config, and the
// messages of llm.completed events (RFC-0001 §5).
//
// Feed it events with Apply, in seq order, starting at seq 0.
type Folder struct {
	next      uint64
	conv      []llm.Message
	config    *Config            // effective config after the latest config event
	configs   map[uint64]*Config // effective config at each config event seq
	completed map[uint64]llm.Message
}

// NewFolder returns an empty Folder.
func NewFolder() *Folder {
	return &Folder{configs: map[uint64]*Config{}, completed: map[uint64]llm.Message{}}
}

// Apply folds one event. Events must arrive in seq order with no gaps.
func (f *Folder) Apply(e Event) error {
	if e.Seq != f.next {
		return fmt.Errorf("trace: fold: expected seq %d, got %d", f.next, e.Seq)
	}
	f.next++
	switch b := e.Body.(type) {
	case *RunStarted:
		if b.Config != nil {
			c := cloneConfig(b.Config)
			f.config = c
			f.configs[e.Seq] = c
		}
	case *ConfigChanged:
		if f.config == nil {
			return fmt.Errorf("trace: fold: config.changed at seq %d in a run without config", e.Seq)
		}
		c := cloneConfig(f.config)
		if b.Model != nil {
			c.Model = *b.Model
		}
		if b.System != nil {
			c.System = *b.System
		}
		if b.Tools != nil {
			c.Tools = append([]llm.ToolSpec(nil), (*b.Tools)...)
			sortTools(c.Tools)
		}
		if b.Params != nil {
			c.Params = *b.Params
		}
		f.config = c
		f.configs[e.Seq] = c
	case *LLMCompleted:
		f.completed[e.Seq] = b.Message
	case *MessageAppended:
		switch {
		case b.Message != nil && b.SourceSeq != nil:
			return fmt.Errorf("trace: fold: message.appended at seq %d has both message and source_seq", e.Seq)
		case b.Message != nil:
			f.conv = append(f.conv, *b.Message)
		case b.SourceSeq != nil:
			m, ok := f.completed[*b.SourceSeq]
			if !ok {
				return fmt.Errorf("trace: fold: message.appended at seq %d: source_seq %d is not an earlier llm.completed", e.Seq, *b.SourceSeq)
			}
			f.conv = append(f.conv, m)
		default:
			return fmt.Errorf("trace: fold: message.appended at seq %d has neither message nor source_seq", e.Seq)
		}
	case *ConversationReplaced:
		f.conv = append([]llm.Message(nil), b.Messages...)
	}
	return nil
}

// Conversation returns a copy of the conversation as of the events applied so far.
func (f *Folder) Conversation() []llm.Message {
	return append([]llm.Message(nil), f.conv...)
}

// Config returns the effective config as of the events applied so far (nil
// for workflow runs).
func (f *Folder) Config() *Config { return f.config }

// Request reconstructs the exact request of an llm.started body, given that
// every event before it has been applied.
func (f *Folder) Request(s *LLMStarted) (*llm.Request, error) {
	cfg, ok := f.configs[s.Context.ConfigSeq]
	if !ok {
		return nil, fmt.Errorf("trace: request: config_seq %d is not a run.started or config.changed event", s.Context.ConfigSeq)
	}
	if s.Context.Messages < 0 || s.Context.Messages > len(f.conv) {
		return nil, fmt.Errorf("trace: request: context uses %d messages, conversation has %d", s.Context.Messages, len(f.conv))
	}
	msgs := append([]llm.Message(nil), f.conv[:s.Context.Messages]...)
	for _, eph := range s.Ephemeral {
		if eph.Index < 0 || eph.Index > len(msgs) {
			return nil, fmt.Errorf("trace: request: ephemeral index %d out of range (0..%d)", eph.Index, len(msgs))
		}
		msgs = append(msgs, llm.Message{})
		copy(msgs[eph.Index+1:], msgs[eph.Index:])
		msgs[eph.Index] = eph.Message
	}
	tools := append([]llm.ToolSpec(nil), cfg.Tools...)
	sortTools(tools)
	return &llm.Request{
		Model:      cfg.Model,
		System:     cfg.System,
		Messages:   msgs,
		Tools:      tools,
		ToolChoice: cfg.ToolChoice,
		Params:     cfg.Params,
	}, nil
}

// Conversation folds a run's events (seq order, starting at 0) and returns
// the final conversation.
func Conversation(events []Event) ([]llm.Message, error) {
	f := NewFolder()
	for _, e := range events {
		if err := f.Apply(e); err != nil {
			return nil, err
		}
	}
	return f.Conversation(), nil
}

// Request reconstructs the exact request recorded by the llm.started event
// at seq, from a run's events (seq order, starting at 0).
func Request(events []Event, seq uint64) (*llm.Request, error) {
	f := NewFolder()
	for _, e := range events {
		if e.Seq == seq {
			s, ok := e.Body.(*LLMStarted)
			if !ok {
				return nil, fmt.Errorf("trace: request: seq %d is %s, not llm.started", seq, e.Kind())
			}
			return f.Request(s)
		}
		if err := f.Apply(e); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("trace: request: seq %d not found", seq)
}

func cloneConfig(c *Config) *Config {
	cp := *c
	cp.Tools = append([]llm.ToolSpec(nil), c.Tools...)
	return &cp
}

func sortTools(ts []llm.ToolSpec) {
	sort.SliceStable(ts, func(i, j int) bool { return ts[i].Name < ts[j].Name })
}
