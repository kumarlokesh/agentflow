package claudecode

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// converter converts one session.
type converter struct {
	session    string
	dir        string // the session's subagents directory
	opts       Options
	metaByTool map[string]string // Agent tool_use ID → agent ID (from meta files)
}

// readMeta maps sub-agents' spawning tool calls to their agent IDs.
func (c *converter) readMeta() map[string]string {
	out := map[string]string{}
	metas, _ := filepath.Glob(filepath.Join(c.dir, "agent-*.meta.json"))
	for _, m := range metas {
		raw, err := os.ReadFile(m)
		if err != nil {
			continue
		}
		var v struct {
			ToolUseID string `json:"toolUseId"`
		}
		if json.Unmarshal(raw, &v) == nil && v.ToolUseID != "" {
			out[v.ToolUseID] = strings.TrimSuffix(strings.TrimPrefix(filepath.Base(m), "agent-"), ".meta.json")
		}
	}
	return out
}

// runResult is a converted run.
type runResult struct {
	path, runID string
	events      []trace.Event
	children    []*runResult
	finished    bool
	status      trace.RunStatus
	output      llm.Blocks
	usage       llm.Usage
}

func (r *runResult) treeUsage() llm.Usage {
	u := r.usage
	for _, c := range r.children {
		u = u.Add(c.treeUsage())
	}
	return u
}

// pending is a model response whose lines may continue.
type pending struct {
	id, model    string
	blocks       []block
	raws         []json.RawMessage
	stop         string
	usage        llm.Usage
	first, last  time.Time
	hasToolCalls bool
	retries      int // API errors retried before this response
}

type use struct {
	id, name string
	input    json.RawMessage
	seq      uint64    // its tool.started
	t        time.Time // when it started
	agent    string    // a delegation's agent type
	path     string    // a delegation's child path
}

type result struct {
	b block
	l line
}

// batch is the tool calls of one response, awaiting results.
type batch struct {
	uses    []use
	results map[string]result
	next    int // calls answered so far
	answers llm.Blocks
}

type builder struct {
	c        *converter
	r        *runResult
	parent   string
	convLen  int
	fp       *requestHasher
	held     []line // tool results that arrived while their response was still streaming
	model    string
	cfgSeq   uint64
	llmN     int
	retries  int
	childN   map[string]int
	resp     *pending
	batch    *batch
	lastText llm.Blocks
}

func (b *builder) emit(t time.Time, body trace.Body, cause *uint64) uint64 {
	seq := uint64(len(b.r.events))
	b.r.events = append(b.r.events, trace.Event{TraceID: b.c.session, RunID: b.r.runID, Seq: seq, Time: t.UTC(), Cause: cause, Body: body})
	return seq
}

// finishMode says when a run is over.
type finishMode int

const (
	whenFinal finishMode = iota // on a final import (the root)
	whenDone                    // when its transcript ends with an answer (a background sub-agent)
	nowFinish                   // now: its caller has its result, or the session moved on
)

// run converts the lines of one run.
func (c *converter) run(lines []line, path, parentRunID, agent string, mode finishMode) (*runResult, error) {
	root := path == "/"
	var kept []line
	for _, l := range lines {
		if root && l.IsSidechain {
			continue // an older transcript's inline sub-agent lines
		}
		kept = append(kept, l)
	}
	lines = kept
	// The task is the first user message; the model, the first response's.
	taskAt, model, version := -1, "", ""
	for i, l := range lines {
		if taskAt < 0 && l.Type == "user" && !l.IsMeta && l.Message != nil && userText(l) != nil {
			taskAt = i
		}
		if model == "" && l.Type == "assistant" && l.Message != nil {
			model = l.Message.Model
		}
		if version == "" {
			version = l.Version
		}
	}
	final := c.opts.Final || mode == nowFinish
	if taskAt < 0 || (model == "" && !final) {
		return nil, nil // nothing to record yet
	}
	if agent == "" {
		agent = Producer
	}
	r := &runResult{path: path, runID: DerivedID(c.session, path)}
	b := &builder{c: c, r: r, parent: parentRunID, model: model, childN: map[string]int{}, fp: newRequestHasher(model)}
	task := toLLM(blocks(lines[taskAt].Message.Content), rawBlocks(lines[taskAt].Message.Content))
	cfg := &trace.Config{Model: model, System: llm.Blocks{}, Tools: []llm.ToolSpec{}, Params: llm.Params{}}
	cfgFP, err := trace.ConfigFingerprint(cfg)
	if err != nil {
		return nil, err
	}
	rs := &trace.RunStarted{RunKind: trace.RunKindAgent, Path: path, Agent: trace.AgentInfo{Name: agent, Version: version}, Task: task, Config: cfg, ConfigFP: cfgFP}
	if parentRunID != "" {
		rs.Parent = &trace.ParentRef{RunID: parentRunID}
	} else {
		rs.Origin = &trace.Origin{Kind: trace.OriginLive, Producer: Producer}
	}
	t0 := lines[taskAt].Timestamp
	b.emit(t0, rs, nil)
	b.appendUser(t0, llm.Message{Role: llm.RoleUser, Blocks: task}, false)

	for _, l := range lines[taskAt+1:] {
		if err := b.line(l); err != nil {
			if err == errHeld {
				return r, nil
			}
			return nil, err
		}
	}
	// End of the transcript so far. A background sub-agent whose last
	// response is an answer (no tool calls) is done.
	if mode == whenDone && b.resp != nil && !b.resp.hasToolCalls && b.batch == nil {
		final = true
	}
	if b.resp != nil {
		if !final {
			return r, nil // the response may continue
		}
		if err := b.flushResp(); err != nil {
			if err == errHeld {
				return r, nil
			}
			return nil, err
		}
	}
	if b.batch != nil {
		if err := b.advance(final); err != nil {
			if err == errHeld {
				return r, nil
			}
			return nil, err
		}
		if b.batch != nil {
			if !final {
				return r, nil
			}
			if err := b.closeBatch(); err != nil {
				return nil, err
			}
		}
	}
	if !final {
		return r, nil
	}
	end := t0
	if n := len(lines); n > 0 {
		end = lines[n-1].Timestamp
	}
	output := b.lastText
	if output == nil {
		output = llm.Blocks{}
	}
	r.finished, r.status, r.output = true, trace.StatusCompleted, output
	b.emit(end, &trace.RunFinished{Status: trace.StatusCompleted, Output: output, Usage: r.usage, Steps: b.llmN}, nil)
	return r, nil
}

func userText(l line) llm.Blocks {
	bs := blocks(l.Message.Content)
	for _, x := range bs {
		if x.Type == "tool_result" {
			return nil
		}
	}
	if len(bs) == 0 {
		return nil
	}
	return toLLM(bs, rawBlocks(l.Message.Content))
}

// line processes one transcript line.
func (b *builder) line(l line) error {
	switch {
	case l.Type == "assistant" && l.Message != nil:
		m := l.Message
		if b.resp != nil && b.resp.id == m.ID && m.ID != "" {
			b.resp.add(l)
			return nil
		}
		if b.resp != nil {
			if err := b.flushResp(); err != nil {
				return err
			}
		}
		if b.batch != nil {
			if err := b.closeBatch(); err != nil {
				return err
			}
		}
		b.resp = &pending{id: m.ID, model: m.Model, first: l.Timestamp, retries: b.retries}
		b.retries = 0
		b.resp.add(l)
	case l.Type == "user" && l.Message != nil:
		bs := blocks(l.Message.Content)
		var results []block
		for _, x := range bs {
			if x.Type == "tool_result" {
				results = append(results, x)
			}
		}
		if len(results) > 0 {
			if b.resp != nil {
				// Claude Code runs a tool as soon as its block streams, so
				// results can arrive between the lines of one response:
				// they wait for the whole response.
				b.held = append(b.held, l)
				return nil
			}
			if b.batch == nil {
				return nil // results without calls (e.g. after compaction): nothing to answer
			}
			for _, x := range results {
				if _, ok := b.batch.results[x.ToolUseID]; !ok {
					b.batch.results[x.ToolUseID] = result{b: x, l: l}
				}
			}
			return b.advance(false)
		}
		if l.IsMeta {
			return nil
		}
		msg := userText(l)
		if msg == nil {
			return nil
		}
		if b.resp != nil {
			if err := b.flushResp(); err != nil {
				return err
			}
		}
		if b.batch != nil {
			if err := b.closeBatch(); err != nil {
				return err
			}
		}
		b.appendUser(l.Timestamp, llm.Message{Role: llm.RoleUser, Blocks: msg}, l.IsCompactSummary)
	case l.Type == "system" && l.Subtype == "api_error":
		b.retries++
	}
	return nil
}

func (p *pending) add(l line) {
	m := l.Message
	p.blocks = append(p.blocks, blocks(m.Content)...)
	p.raws = append(p.raws, rawBlocks(m.Content)...)
	if m.StopReason != nil && *m.StopReason != "" {
		p.stop = *m.StopReason
	}
	if u := m.Usage; u != nil {
		p.usage = llm.Usage{InputTokens: u.InputTokens, OutputTokens: u.OutputTokens, CacheReadTokens: u.CacheReadTokens, CacheWriteTokens: u.CacheWriteTokens}
	}
	if m.Model != "" {
		p.model = m.Model
	}
	p.last = l.Timestamp
	for _, x := range blocks(m.Content) {
		if x.Type == "tool_use" {
			p.hasToolCalls = true
		}
	}
}

func (b *builder) appendUser(t time.Time, m llm.Message, compact bool) {
	if compact {
		b.emit(t, &trace.ConversationReplaced{Messages: []llm.Message{m}}, nil)
		b.fp.reset()
		b.convLen = 0
	} else {
		b.emit(t, &trace.MessageAppended{Message: &m}, nil)
	}
	b.fp.add(m)
	b.convLen++
}

// flushResp records the pending response, then the results that arrived
// while it was streaming.
func (b *builder) flushResp() error {
	b.flush()
	held := b.held
	b.held = nil
	for _, l := range held {
		if err := b.line(l); err != nil {
			return err
		}
	}
	return nil
}

// flush records the pending response: the model call, and its tool calls'
// starts.
func (b *builder) flush() {
	p := b.resp
	b.resp = nil
	if p.model != "" && p.model != b.model {
		m := p.model
		b.cfgSeq = b.emit(p.first, &trace.ConfigChanged{Model: &m}, nil)
		b.model = m
		b.fp.setModel(m)
	}
	fp := b.fp.sum()
	step := b.llmN
	b.llmN++
	started := b.emit(p.first, &trace.LLMStarted{Step: step, Context: trace.ContextRef{Messages: b.convLen, ConfigSeq: b.cfgSeq}, Fingerprint: fp}, nil)
	msg := llm.Message{Role: llm.RoleAssistant, Blocks: toLLM(p.blocks, p.raws)}
	stop := llm.StopReason(p.stop)
	if stop == "" {
		stop = llm.StopEndTurn
		if p.hasToolCalls {
			stop = llm.StopToolUse
		}
	}
	done := b.emit(p.last, &trace.LLMCompleted{Step: step, Message: msg, StopReason: stop, Usage: p.usage, Model: p.model,
		Attempts: 1 + p.retries, Duration: p.last.Sub(p.first)}, trace.Cause(started))
	b.emit(p.last, &trace.MessageAppended{SourceSeq: trace.Cause(done)}, nil)
	b.fp.add(msg)
	b.convLen++
	b.r.usage = b.r.usage.Add(p.usage)
	var text llm.Blocks
	for _, x := range msg.Blocks {
		if t, ok := x.(llm.Text); ok {
			text = append(text, t)
		}
	}
	if text != nil {
		b.lastText = text
	}
	var uses []use
	for _, x := range msg.Blocks {
		u, ok := x.(llm.ToolUse)
		if !ok {
			continue
		}
		fp, _ := trace.ToolFingerprint(u.Name, u.Input)
		seq := b.emit(p.last, &trace.ToolStarted{Step: step, CallID: u.CallID, Tool: u.Name, Input: u.Input, Fingerprint: fp}, trace.Cause(done))
		x := use{id: u.CallID, name: u.Name, input: u.Input, seq: seq, t: p.last}
		if u.Name == "Agent" || u.Name == "Task" {
			var in struct {
				SubagentType string `json:"subagent_type"`
			}
			_ = json.Unmarshal(u.Input, &in)
			x.agent = agentName(in.SubagentType)
			x.path = joinPath(b.r.path, fmt.Sprintf("%s#%d", x.agent, b.childN[x.agent]))
			b.childN[x.agent]++
		}
		uses = append(uses, x)
	}
	if len(uses) > 0 {
		b.batch = &batch{uses: uses, results: map[string]result{}}
	}
}

func agentName(t string) string {
	if t == "" {
		t = "general-purpose"
	}
	return strings.NewReplacer("/", "_", "#", "_", "@", "_").Replace(t)
}

func joinPath(parent, seg string) string {
	if parent == "/" {
		return "/" + seg
	}
	return parent + "/" + seg
}

// advance answers the batch's calls in order, as far as their results
// allow; force answers a sub-agent that hasn't finished as cancelled.
func (b *builder) advance(force bool) error {
	bt := b.batch
	for bt.next < len(bt.uses) {
		u := bt.uses[bt.next]
		res, ok := bt.results[u.id]
		if !ok {
			return nil
		}
		if err := b.answer(u, &res, force); err != nil {
			return err
		}
		bt.next++
		bt.answers = append(bt.answers, toLLM([]block{res.b}, nil)...)
	}
	b.finishBatch(bt)
	return nil
}

// closeBatch answers every call of the batch: the session moved on, so a
// call without a result never got one.
func (b *builder) closeBatch() error {
	bt := b.batch
	for bt.next < len(bt.uses) {
		u := bt.uses[bt.next]
		if res, ok := bt.results[u.id]; ok {
			if err := b.answer(u, &res, true); err != nil {
				return err
			}
			bt.answers = append(bt.answers, toLLM([]block{res.b}, nil)...)
		} else {
			if u.path != "" {
				if err := b.delegation(u, nil, true); err != nil {
					return err
				}
			}
			b.emit(u.t, &trace.ToolFailed{CallID: u.id, Error: trace.Error{Code: trace.ErrCodeCancelled, Message: "the session recorded no result for this call"}}, trace.Cause(u.seq))
		}
		bt.next++
	}
	b.finishBatch(bt)
	return nil
}

func (b *builder) finishBatch(bt *batch) {
	if bt.next < len(bt.uses) {
		return
	}
	b.batch = nil
	if len(bt.answers) == 0 {
		return
	}
	t := bt.uses[len(bt.uses)-1].t
	for _, u := range bt.uses {
		if r, ok := bt.results[u.id]; ok && r.l.Timestamp.After(t) {
			t = r.l.Timestamp
		}
	}
	b.appendUser(t, llm.Message{Role: llm.RoleUser, Blocks: bt.answers}, false)
}

// answer records a call's answer (a delegation's child first).
func (b *builder) answer(u use, res *result, force bool) error {
	if u.path != "" {
		if err := b.delegation(u, res, force); err != nil {
			return err
		}
	}
	content := toLLM(blocks(res.b.Content), rawBlocks(res.b.Content))
	d := res.l.Timestamp.Sub(u.t)
	if denied(res.l) {
		b.emit(res.l.Timestamp, &trace.ToolFailed{CallID: u.id, Error: trace.Error{Code: trace.ErrCodeDenied, Message: llm.TextOf(content)}, Duration: d, Intercepted: true}, trace.Cause(u.seq))
		return nil
	}
	b.emit(res.l.Timestamp, &trace.ToolCompleted{CallID: u.id, Content: content, IsError: res.b.IsError, Duration: d}, trace.Cause(u.seq))
	return nil
}

func denied(l line) bool {
	s := strings.TrimSpace(string(l.ToolDenialKind))
	return s != "" && s != "null" && s != `""` && s != "false"
}

// subResult is the part of an Agent tool's toolUseResult the conversion
// reads.
type subResult struct {
	AgentID   string          `json:"agentId"`
	AgentType string          `json:"agentType"`
	Status    string          `json:"status"`
	Content   json.RawMessage `json:"content"`
	IsAsync   bool            `json:"isAsync"`
}

// delegation records a sub-agent: its child run, then run.spawned and
// run.joined in the caller. res is nil for a call that never got a result.
func (b *builder) delegation(u use, res *result, force bool) error {
	var sr subResult
	if res != nil {
		_ = json.Unmarshal(res.l.ToolUseResult, &sr)
	}
	agentID := sr.AgentID
	if agentID == "" {
		agentID = b.c.metaByTool[u.id]
	}
	var in struct {
		Prompt string `json:"prompt"`
	}
	_ = json.Unmarshal(u.input, &in)
	task := llm.Blocks{llm.Text{Text: in.Prompt}}
	childID := DerivedID(b.c.session, u.path)

	var child *runResult
	if agentID != "" {
		file := filepath.Join(b.c.dir, "agent-"+agentID+".jsonl")
		if lines, err := readLines(file); err == nil {
			async := sr.IsAsync || strings.HasPrefix(sr.Status, "async")
			// A sync sub-agent is done when its caller has the result;
			// a background one when its transcript ends with an answer.
			var err error
			mode := whenDone
			if (!async && res != nil) || force {
				mode = nowFinish
			}
			child, err = b.c.run(lines, u.path, b.r.runID, u.agent, mode)
			if err != nil {
				return err
			}
			if child != nil {
				b.r.children = append(b.r.children, child)
			}
			if async && (child == nil || !child.finished) && !force && !b.c.opts.Final {
				return errHeld
			}
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	fp, _ := trace.SpawnFingerprint(u.agent, task, "")
	t := u.t
	if res != nil {
		t = res.l.Timestamp
	}
	spawned := b.emit(t, &trace.RunSpawned{ChildRunID: childID, Path: u.path, Agent: u.agent, Task: task, Fingerprint: fp}, trace.Cause(u.seq))
	j := &trace.RunJoined{ChildRunID: childID, Status: trace.StatusCompleted, Output: llm.Blocks{}, Duration: t.Sub(u.t)}
	switch {
	case child != nil && child.finished:
		j.Status, j.Output, j.Usage = child.status, child.output, child.treeUsage()
	case child != nil:
		j.Status, j.Error = trace.StatusCancelled, &trace.Error{Code: trace.ErrCodeCancelled, Message: "the sub-agent hadn't finished when the session ended"}
	case res != nil:
		// No transcript for the sub-agent: its result is all there is.
		j.Output = toLLM(blocks(sr.Content), rawBlocks(sr.Content))
		if len(j.Output) == 0 {
			j.Output = toLLM(blocks(res.b.Content), rawBlocks(res.b.Content))
		}
		if res.b.IsError {
			j.Status = trace.StatusFailed
		}
	default:
		j.Status, j.Error = trace.StatusCancelled, &trace.Error{Code: trace.ErrCodeCancelled, Message: "the session recorded no result for this sub-agent"}
	}
	b.emit(t, j, trace.Cause(spawned))
	return nil
}
