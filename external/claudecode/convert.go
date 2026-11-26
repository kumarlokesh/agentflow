// Package claudecode records Claude Code sessions as agentflow traces.
//
// A session's transcript (~/.claude/projects/<project>/<session>.jsonl,
// with its sub-agents under <session>/subagents/) becomes one trace tree:
// the main conversation is the root run, and each sub-agent a child run
// linked through the Agent tool call that started it. The traces are
// record-only: agentflow doesn't run Claude Code's loop, so they can be
// shown, diffed, blamed and exported, but not replayed or forked.
//
// Import is incremental and idempotent: run it from Claude Code's hooks
// (agentflow hook claude-code) as a session goes, and once more when it
// ends (Final).
package claudecode

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// Producer is the origin.producer of traces recorded from Claude Code.
const Producer = "claude-code"

// Options configure a conversion.
type Options struct {
	// Final ends the session: the root run is finished, and calls the
	// session never answered are answered "cancelled". Without it, an
	// incomplete tail (a response still streaming, a tool batch waiting
	// for results, a running sub-agent) is held back, so that a later
	// conversion of the longer transcript extends this one exactly.
	Final bool
}

// Session is a converted session: its runs, root first.
type Session struct {
	ID   string
	Runs []*Run
}

// Run is one run's events.
type Run struct {
	Path   string
	RunID  string
	Events []trace.Event
}

// Convert reads the transcript at path (a session's .jsonl file) and its
// sub-agents' transcripts, and converts them.
func Convert(path string, opts Options) (*Session, error) {
	lines, err := readLines(path)
	if err != nil {
		return nil, err
	}
	id := strings.TrimSuffix(filepath.Base(path), ".jsonl")
	for _, l := range lines {
		if l.SessionID != "" {
			id = l.SessionID
			break
		}
	}
	c := &converter{session: id, dir: filepath.Join(filepath.Dir(path), strings.TrimSuffix(filepath.Base(path), ".jsonl"), "subagents"), opts: opts}
	c.metaByTool = c.readMeta()
	s := &Session{ID: id}
	root, err := c.run(lines, "/", "", "", whenFinal)
	if err != nil {
		return nil, err
	}
	if root == nil {
		return s, nil // nothing complete yet
	}
	var walk func(r *runResult)
	walk = func(r *runResult) {
		s.Runs = append(s.Runs, &Run{Path: r.path, RunID: r.runID, Events: r.events})
		for _, ch := range r.children {
			walk(ch)
		}
	}
	walk(root)
	return s, nil
}

// --- transcript lines ---

type line struct {
	Type             string          `json:"type"`
	Subtype          string          `json:"subtype"`
	SessionID        string          `json:"sessionId"`
	Timestamp        time.Time       `json:"timestamp"`
	Version          string          `json:"version"`
	IsMeta           bool            `json:"isMeta"`
	IsSidechain      bool            `json:"isSidechain"`
	IsCompactSummary bool            `json:"isCompactSummary"`
	AgentID          string          `json:"agentId"`
	Message          *message        `json:"message"`
	ToolUseResult    json.RawMessage `json:"toolUseResult"`
	ToolDenialKind   json.RawMessage `json:"toolDenialKind"`
}

type message struct {
	ID         string          `json:"id"`
	Role       string          `json:"role"`
	Model      string          `json:"model"`
	Content    json.RawMessage `json:"content"`
	StopReason *string         `json:"stop_reason"`
	Usage      *usage          `json:"usage"`
}

type usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_input_tokens"`
	CacheWriteTokens int `json:"cache_creation_input_tokens"`
}

type block struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	Data      string          `json:"data"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

func readLines(path string) ([]line, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var out []line
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 256<<20)
	n := 0
	for sc.Scan() {
		n++
		b := sc.Bytes()
		if len(strings.TrimSpace(string(b))) == 0 {
			continue
		}
		var l line
		if err := json.Unmarshal(b, &l); err != nil {
			// A line cut off by a crash or still being written: stop
			// there; a later import reads it whole.
			break
		}
		out = append(out, l)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("claudecode: %s line %d: %w", path, n+1, err)
	}
	return out, nil
}

// blocks decodes message content: a string or a list of blocks.
func blocks(raw json.RawMessage) []block {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []block{{Type: "text", Text: s}}
	}
	var bs []block
	_ = json.Unmarshal(raw, &bs)
	return bs
}

// toLLM converts content blocks to the trace's blocks.
func toLLM(bs []block, raws []json.RawMessage) llm.Blocks {
	out := llm.Blocks{}
	for i, b := range bs {
		switch b.Type {
		case "text":
			out = append(out, llm.Text{Text: b.Text})
		case "thinking":
			ext, _ := json.Marshal(map[string]string{"signature": b.Signature})
			out = append(out, llm.Reasoning{Text: b.Thinking, Ext: llm.Ext{"anthropic": ext}})
		case "redacted_thinking":
			ext, _ := json.Marshal(map[string]string{"data": b.Data})
			out = append(out, llm.Reasoning{Redacted: true, Ext: llm.Ext{"anthropic": ext}})
		case "tool_use":
			in := b.Input
			if len(in) == 0 {
				in = json.RawMessage(`{}`)
			}
			out = append(out, llm.ToolUse{CallID: b.ID, Name: b.Name, Input: in})
		case "tool_result":
			out = append(out, llm.ToolResult{CallID: b.ToolUseID, Content: toLLM(blocks(b.Content), rawBlocks(b.Content)), IsError: b.IsError})
		default:
			var v json.RawMessage
			if i < len(raws) {
				v = raws[i]
			}
			out = append(out, llm.Data{Value: v})
		}
	}
	return out
}

func rawBlocks(raw json.RawMessage) []json.RawMessage {
	var rs []json.RawMessage
	_ = json.Unmarshal(raw, &rs)
	return rs
}

// DerivedID returns a stable UUID (version 8) for a session and a run path,
// so re-imports produce the same run IDs.
func DerivedID(session, path string) string {
	h := sha256.Sum256([]byte("agentflow/claude-code\x00" + session + "\x00" + path))
	h[6] = h[6]&0x0f | 0x80
	h[8] = h[8]&0x3f | 0x80
	x := hex.EncodeToString(h[:16])
	return x[0:8] + "-" + x[8:12] + "-" + x[12:16] + "-" + x[16:20] + "-" + x[20:32]
}

// errHeld means the run can't be converted yet (non-final import).
var errHeld = errors.New("held back")
