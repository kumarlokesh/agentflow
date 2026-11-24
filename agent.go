package agentflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// DefaultMaxSteps is the default limit on Observe-Think-Act iterations.
const DefaultMaxSteps = 20

// AgentConfig configures an Agent.
type AgentConfig struct {
	// Name identifies the agent in traces. Required.
	Name string
	// Version is recorded in traces; optional.
	Version string

	// Model is the model backend. Required.
	Model llm.Model
	// ModelName is sent as llm.Request.Model.
	ModelName string
	// Instructions is the system prompt. System, if set, is used instead.
	Instructions string
	System       llm.Blocks
	Params       llm.Params
	ToolChoice   *llm.ToolChoice

	// Tools the model may call. Names must be unique.
	Tools []Tool
	// ValidateToolInput checks tool inputs against their JSON Schema before
	// calling the tool. Defaults to true.
	ValidateToolInput *bool
	// MaxParallelTools bounds how many tool calls from one model response
	// run at once. Default 1 (sequential); tools must be safe for concurrent
	// use to raise it.
	MaxParallelTools int
	// MaxSteps limits loop iterations. Zero means DefaultMaxSteps.
	MaxSteps int

	// Interceptors see every effect, outermost first.
	Interceptors []Interceptor
	// ContextProviders add ephemeral messages before each model call; each
	// call is recorded as a custom effect named "context.<name>".
	ContextProviders []ContextProvider
	// Timeouts bound effect durations.
	Timeouts Timeouts

	// Store receives the trace. Defaults to a new in-memory store per run.
	Store trace.Store
	// Observers are notified of every recorded event.
	Observers []trace.Observer
	// Logger receives debug logs. Defaults to a discarding logger.
	Logger *slog.Logger
	// Clock and NewID are injectable for deterministic tests. They only
	// affect event times and IDs, which are never compared.
	Clock func() time.Time
	NewID func() string
}

// ContextProvider adds ephemeral context (memory, retrieved documents, …)
// to each model request. Provide must not modify conv.
//
// Each call is recorded as the custom effect "context.<name>"; replay
// answers it from the recording unless it is verified
// (replay.Options.Verify "effect:context.<name>").
//
// Placement matters with reasoning models. Some providers bind each
// reasoning block to the exact messages before it, and reject a later
// request in which those changed. An ephemeral message that sits before an
// earlier assistant message must therefore appear in every later request at
// the same index with the same content: put stable context at Index 0, not
// at the end of the conversation. Context that changes per call is better
// offered as a tool, whose results become part of the conversation; or see
// the provider's option for dropping mismatched reasoning (Anthropic:
// WithDropMismatchedThinking).
type ContextProvider interface {
	Name() string
	Provide(ctx context.Context, conv []llm.Message) ([]trace.Ephemeral, error)
}

// Agent runs an Observe-Think-Act loop over a model and tools, recording
// every effect. It is safe to run concurrently.
type Agent struct {
	name, version string
	model         llm.Model
	modelName     string
	system        llm.Blocks
	params        llm.Params
	toolChoice    *llm.ToolChoice
	tools         map[string]Tool
	specs         []llm.ToolSpec
	validate      bool
	maxParallel   int
	maxSteps      int
	interceptors  []Interceptor
	providers     []ContextProvider
	timeouts      Timeouts
	store         trace.Store
	observers     []trace.Observer
	logger        *slog.Logger
	clock         func() time.Time
	newID         func() string
}

// NewAgent validates cfg and returns an Agent.
func NewAgent(cfg AgentConfig) (*Agent, error) {
	if cfg.Model == nil {
		return nil, ErrNoModel
	}
	if cfg.Name == "" {
		return nil, errors.New("agentflow: agent name is required")
	}
	a := &Agent{
		name: cfg.Name, version: cfg.Version,
		model: cfg.Model, modelName: cfg.ModelName,
		system: cfg.System, params: cfg.Params, toolChoice: cfg.ToolChoice,
		tools: map[string]Tool{}, validate: true,
		maxParallel: max(cfg.MaxParallelTools, 1), maxSteps: cfg.MaxSteps,
		interceptors: cfg.Interceptors, providers: cfg.ContextProviders, timeouts: cfg.Timeouts,
		store: cfg.Store, observers: cfg.Observers, logger: cfg.Logger, clock: cfg.Clock, newID: cfg.NewID,
	}
	if a.system == nil && cfg.Instructions != "" {
		a.system = llm.Blocks{llm.Text{Text: cfg.Instructions}}
	}
	if cfg.ValidateToolInput != nil {
		a.validate = *cfg.ValidateToolInput
	}
	if a.maxSteps <= 0 {
		a.maxSteps = DefaultMaxSteps
	}
	if a.logger == nil {
		a.logger = slog.New(slog.DiscardHandler)
	}
	if a.clock == nil {
		a.clock = time.Now
	}
	if a.newID == nil {
		a.newID = trace.NewID
	}
	for _, t := range cfg.Tools {
		spec := t.Spec()
		if spec.Name == "" {
			return nil, errors.New("agentflow: tool with empty name")
		}
		if _, dup := a.tools[spec.Name]; dup {
			return nil, fmt.Errorf("agentflow: duplicate tool %q", spec.Name)
		}
		a.tools[spec.Name] = t
		a.specs = append(a.specs, spec)
	}
	sort.Slice(a.specs, func(i, j int) bool { return a.specs[i].Name < a.specs[j].Name }) // R3
	seen := map[string]bool{}
	for _, p := range a.providers {
		if seen[p.Name()] {
			return nil, fmt.Errorf("agentflow: duplicate context provider %q", p.Name())
		}
		seen[p.Name()] = true
	}
	return a, nil
}

// Name returns the agent's name.
func (a *Agent) Name() string { return a.name }

func (a *Agent) isEntrypoint() {}

func (a *Agent) traceConfig() *trace.Config {
	return &trace.Config{
		Model: a.modelName, System: a.system, Tools: append([]llm.ToolSpec{}, a.specs...),
		ToolChoice: a.toolChoice, Params: a.params, MaxSteps: a.maxSteps,
	}
}

// Run executes the agent on task and records the run. It returns the result
// and, unless the run completed, a *RunError.
func (a *Agent) Run(ctx context.Context, task string, opts ...RunOption) (*RunResult, error) {
	var o runOptions
	for _, opt := range opts {
		opt(&o)
	}
	store := o.store
	if store == nil {
		store = a.store
	}
	if store == nil {
		store = memstore.New()
	}
	origin := o.origin
	if origin == nil {
		origin = &trace.Origin{Kind: trace.OriginLive}
	}
	taskBlocks := o.task
	if taskBlocks == nil {
		taskBlocks = llm.Blocks{llm.Text{Text: task}}
	}

	traceID, runID := a.newID(), a.newID()
	observers := append(append([]trace.Observer{}, a.observers...), o.observers...)
	rs := &runState{
		agent: a, traceID: traceID, runID: runID, path: "/",
		w:        trace.NewWriter(store, traceID, runID, trace.WriterOptions{Now: a.clock, Observers: observers}),
		replayer: o.replayer, mode: ModeLive, clock: a.clock, effectN: map[string]int{}, toolN: map[string]int{},
	}
	if rs.replayer != nil {
		rs.mode = ModeReplay
	}
	rs.runScope = &scope{rs: rs}
	rs.chain = rs.buildChain(append(append([]Interceptor{}, a.interceptors...), o.interceptors...), a.timeouts, o.noLive)

	start := a.clock()
	cfg := a.traceConfig()
	cfgFP, err := trace.ConfigFingerprint(cfg)
	if err != nil {
		return nil, err
	}
	if _, err := rs.w.Write(ctx, &trace.RunStarted{
		RunKind: trace.RunKindAgent, Path: rs.path, Agent: trace.AgentInfo{Name: a.name, Version: a.version},
		Task: taskBlocks, Config: cfg, ConfigFP: cfgFP, Origin: origin,
	}, nil); err != nil {
		return nil, err
	}
	a.logger.Debug("run started", "trace_id", traceID, "run_id", runID, "agent", a.name)

	status, output, steps, runErr := rs.loop(ctx, taskBlocks)

	if rs.replayer != nil && !errors.Is(runErr, ErrDiverged) {
		if ferr := rs.replayer.Finish(ctx, status, output); ferr != nil {
			if errors.Is(ferr, ErrDiverged) {
				status = trace.StatusDiverged
			}
			runErr = errors.Join(runErr, ferr)
		}
	}
	fin := &trace.RunFinished{Status: status, Output: output, Usage: rs.usage, Steps: steps}
	if runErr != nil {
		fin.Error = &trace.Error{Code: errorCode(status), Message: runErr.Error()}
	}
	if _, err := rs.w.Write(context.WithoutCancel(ctx), fin, nil); err != nil && runErr == nil {
		runErr, status = err, trace.StatusFailed
	}
	res := &RunResult{
		TraceID: traceID, RunID: runID, Path: rs.path, Status: status, Output: output,
		Usage: rs.usage, Steps: steps, Duration: a.clock().Sub(start), Store: store,
	}
	a.logger.Debug("run finished", "trace_id", traceID, "status", status, "steps", steps)
	if status != trace.StatusCompleted {
		return res, &RunError{Status: status, Err: runErr}
	}
	return res, nil
}

func errorCode(s trace.RunStatus) string {
	switch s {
	case trace.StatusCancelled:
		return trace.ErrCodeCancelled
	case trace.StatusDiverged:
		return string(trace.StatusDiverged)
	case trace.StatusMaxSteps:
		return trace.ErrCodeLimitExceeded
	}
	return trace.ErrCodeInternal
}

// endStatus classifies an error that ends the loop.
func endStatus(ctx context.Context, err error) trace.RunStatus {
	switch {
	case errors.Is(err, ErrDiverged):
		return trace.StatusDiverged
	case ctx.Err() != nil:
		return trace.StatusCancelled
	}
	return trace.StatusFailed
}

// loop is the Observe-Think-Act loop.
func (rs *runState) loop(ctx context.Context, task llm.Blocks) (trace.RunStatus, llm.Blocks, int, error) {
	a := rs.agent
	user := llm.Message{Role: llm.RoleUser, Blocks: task}
	if _, err := rs.w.Write(ctx, &trace.MessageAppended{Message: &user}, nil); err != nil {
		return trace.StatusFailed, nil, 0, err
	}
	conv := []llm.Message{user}
	steps := 0
	for step := 0; step < a.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return trace.StatusCancelled, nil, steps, err
		}
		// Observe: ephemeral context for this call.
		eph, err := rs.contextFor(ctx, conv)
		if err != nil {
			return endStatus(ctx, err), nil, steps, err
		}
		msgs, err := trace.ApplyEphemeral(conv, eph)
		if err != nil {
			return trace.StatusFailed, nil, steps, fmt.Errorf("agentflow: context providers: %w", err)
		}

		// Think: call the model.
		req := &llm.Request{Model: a.modelName, System: a.system, Messages: msgs, Tools: a.specs, ToolChoice: a.toolChoice, Params: a.params}
		fp, err := trace.RequestFingerprint(req)
		if err != nil {
			return trace.StatusFailed, nil, steps, err
		}
		rs.mu.Lock()
		key := fmt.Sprintf("llm#%d", rs.llmN)
		rs.llmN++
		rs.mu.Unlock()
		call := &Call{Kind: EffectLLM, Name: a.modelName, Request: req, Fingerprint: fp, Meta: CallMeta{Step: step, Key: key, Label: key}}
		body := &trace.LLMStarted{Step: step, Context: trace.ContextRef{Messages: len(conv)}, Ephemeral: eph, Fingerprint: fp}
		out, done, err := rs.invoke(ctx, rs.runScope, call, body, nil)
		if err != nil {
			return endStatus(ctx, err), nil, steps, err
		}
		steps = step + 1
		if out.Failure != nil {
			ferr := fmt.Errorf("model call failed (%s): %s", out.Failure.Code, out.Failure.Message)
			if out.Failure.Code == llm.ErrCodeCancelled || ctx.Err() != nil {
				return trace.StatusCancelled, nil, steps, ferr
			}
			return trace.StatusFailed, nil, steps, ferr
		}
		resp := out.Response.(*llm.Response)
		if _, err := rs.w.Write(ctx, &trace.MessageAppended{SourceSeq: trace.Cause(done.seq)}, nil); err != nil {
			return trace.StatusFailed, nil, steps, err
		}
		conv = append(conv, resp.Message)

		// Act: run requested tools, or finish.
		uses := toolUses(resp.Message.Blocks)
		if len(uses) == 0 {
			return trace.StatusCompleted, finalOutput(resp.Message.Blocks), steps, nil
		}
		results, err := rs.runTools(ctx, step, done, uses)
		if err != nil {
			return endStatus(ctx, err), nil, steps, err
		}
		toolMsg := llm.Message{Role: llm.RoleUser, Blocks: results}
		if _, err := rs.w.Write(ctx, &trace.MessageAppended{Message: &toolMsg}, nil); err != nil {
			return trace.StatusFailed, nil, steps, err
		}
		conv = append(conv, toolMsg)
	}
	return trace.StatusMaxSteps, nil, steps, fmt.Errorf("agentflow: exceeded %d steps", a.maxSteps)
}

// contextFor runs the context providers, each as a recorded effect.
func (rs *runState) contextFor(ctx context.Context, conv []llm.Message) ([]trace.Ephemeral, error) {
	if len(rs.agent.providers) == 0 {
		return nil, nil
	}
	convFP, err := trace.Fingerprint(conv)
	if err != nil {
		return nil, err
	}
	var all []trace.Ephemeral
	for _, p := range rs.agent.providers {
		p := p
		snapshot := append([]llm.Message(nil), conv...)
		// The input references the conversation instead of copying it, so
		// traces stay linear in size.
		input := map[string]any{"messages": len(conv), "conversation": convFP}
		out, err := rs.doEffect(ctx, rs.runScope, "context."+p.Name(), input, snapshot, func(c context.Context) (any, error) {
			eph, err := p.Provide(c, snapshot)
			if eph == nil {
				eph = []trace.Ephemeral{}
			}
			return eph, err
		})
		if err != nil {
			return nil, err
		}
		if out.Failure != nil {
			rs.agent.logger.Warn("context provider failed", "provider", p.Name(), "error", out.Failure.Message)
			continue
		}
		var eph []trace.Ephemeral
		if err := json.Unmarshal(out.Response.(json.RawMessage), &eph); err != nil {
			return nil, fmt.Errorf("agentflow: context provider %q output: %w", p.Name(), err)
		}
		all = append(all, eph...)
	}
	return all, nil
}

// runTools runs the tool calls of one model response as a batch: all tool.started events in response order, execution (concurrent up
// to MaxParallelTools), then completions in response order.
func (rs *runState) runTools(ctx context.Context, step int, cause ref, uses []llm.ToolUse) (llm.Blocks, error) {
	type item struct {
		s        *started
		out      *Outcome
		attempts int
		err      error
	}
	items := make([]item, len(uses))
	for i, u := range uses {
		input := u.Input
		if len(input) == 0 {
			input = json.RawMessage(`{}`)
		}
		fp, err := trace.ToolFingerprint(u.Name, input)
		if err != nil {
			return nil, err
		}
		key := ToolKey(u.CallID, rs.toolN[u.CallID])
		rs.toolN[u.CallID]++
		var ann *llm.ToolAnnotations
		if t, ok := rs.agent.tools[u.Name]; ok {
			ann = t.Spec().Annotations
		}
		call := &Call{Kind: EffectTool, Name: u.Name, Request: &ToolCall{CallID: u.CallID, Name: u.Name, Input: input, Annotations: ann}, Fingerprint: fp,
			Meta: CallMeta{Step: step, CallID: u.CallID, Key: key, Label: fmt.Sprintf("tool:%s(%s)", u.Name, u.CallID)}}
		body := &trace.ToolStarted{Step: step, CallID: u.CallID, Tool: u.Name, Input: input, Fingerprint: fp}
		s, err := rs.begin(ctx, rs.runScope, call, body, &cause)
		if err != nil {
			return nil, err
		}
		// The tool executes in its own scope: effects it starts are nested
		// and buffered until the batch completes (R7).
		s.sc = &scope{rs: rs, prefix: key + "/", nested: true, verified: call.Meta.Verify, parentStarted: s.ref.seq}
		items[i].s = s
	}

	run := func(i int) { items[i].out, items[i].attempts, items[i].err = rs.execute(ctx, items[i].s) }
	if rs.agent.maxParallel <= 1 || len(items) == 1 {
		for i := range items {
			run(i)
		}
	} else {
		sem := make(chan struct{}, rs.agent.maxParallel)
		var wg sync.WaitGroup
		for i := range items {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				run(i)
			}(i)
		}
		wg.Wait()
	}

	results := make(llm.Blocks, 0, len(items))
	for _, it := range items {
		if it.err != nil {
			return nil, it.err
		}
		cb, err := rs.completion(it.s, it.out, it.attempts)
		if err != nil {
			return nil, err
		}
		sw, err := rs.complete(ctx, it.s, it.out)
		if err != nil {
			return nil, err
		}
		pending := it.s.sc.drain(rs.w.Next())
		if sw != nil {
			pending = append(pending, trace.Pending{Body: sw, Cause: trace.Cause(it.s.ref.seq)})
		}
		pending = append(pending, trace.Pending{Body: cb, Cause: trace.Cause(it.s.ref.seq)})
		if _, err := rs.w.WriteBatch(ctx, pending); err != nil {
			return nil, err
		}
		results = append(results, toolResult(it.s.call.Meta.CallID, it.out))
	}
	return results, nil
}

// ToolKey is the replay key of the n-th (from 0) tool call with this call
// ID in a run. Call IDs are normally unique, but some providers reuse them
// across responses, so repeats get an occurrence suffix.
func ToolKey(callID string, n int) string {
	if n == 0 {
		return "tool:" + callID
	}
	return fmt.Sprintf("tool:%s#%d", callID, n)
}

func toolResult(callID string, out *Outcome) llm.ToolResult {
	if out.Failure != nil {
		return llm.ToolResult{CallID: callID, Content: llm.Blocks{llm.Text{Text: "Error: " + out.Failure.Message}}, IsError: true}
	}
	o := out.Response.(ToolOutput)
	content := o.Content
	if content == nil {
		content = llm.Blocks{}
	}
	return llm.ToolResult{CallID: callID, Content: content, IsError: o.IsError}
}

func toolUses(bs llm.Blocks) []llm.ToolUse {
	var out []llm.ToolUse
	for _, b := range bs {
		if u, ok := b.(llm.ToolUse); ok {
			out = append(out, u)
		}
	}
	return out
}

// finalOutput is the final answer: the response without reasoning or tool calls.
func finalOutput(bs llm.Blocks) llm.Blocks {
	out := llm.Blocks{}
	for _, b := range bs {
		switch b.(type) {
		case llm.Reasoning, llm.ToolUse:
			continue
		}
		out = append(out, b)
	}
	return out
}
