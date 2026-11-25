package agentflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/schema"
	"github.com/kumarlokesh/agentflow/trace"
)

// Error code for an error returned by a tool or a Do function.
const codeError = "error"

// runState is the machinery of one run: it records effects, consults the
// replayer, and runs the interceptor chain.
type runState struct {
	def      *runDef
	agent    *Agent // nil for workflow runs
	tree     *tree
	w        *trace.Writer
	traceID  string
	runID    string
	path     string
	depth    int
	chain    Invoker
	replayer Replayer
	mode     Mode
	clock    func() time.Time

	mu       sync.Mutex
	llmN     int
	effectN  map[string]int // scope prefix + effect name → next ordinal
	toolN    map[string]int // call ID → occurrences so far
	childN   map[string]int // agent name → children at run level or by delegation
	usage    llm.Usage
	childErr error // set when a child ended diverged
	runScope *scope
	cfg      runConfig // an agent run's effective model configuration

	slotMu     sync.Mutex
	holdsSlot  bool // a MaxConcurrentRuns slot
	gaveUpSlot bool // released while waiting for children
	waiting    int  // children being waited for
}

// runConfig is an agent run's effective model configuration: the agent's,
// changed by Patches.
type runConfig struct {
	model    string
	system   llm.Blocks
	params   llm.Params
	tools    []llm.ToolSpec
	disabled map[string]bool
	seq      uint64 // the run.started or config.changed that set it
}

// patch asks the replayer for a change to the effect with this key.
func (rs *runState) patch(ctx context.Context, key string) (*Patch, error) {
	p, ok := rs.replayer.(Patcher)
	if !ok {
		return nil, nil
	}
	return p.Patch(ctx, key)
}

// scope is where effects are recorded: the run itself, or one tool call in
// a batch, whose events are buffered until the batch completes (R7).
type scope struct {
	rs     *runState
	prefix string // "" for the run, "tool:<call_id>/" for a tool call
	nested bool
	// verified is set for the scope of a tool call that executes during
	// replay to be verified: its effects are matched against the recording.
	verified bool
	busy     atomic.Bool

	parentStarted uint64 // seq of the enclosing tool.started
	step, pos     int    // a tool call's model step and position in the response

	mu       sync.Mutex
	buf      []pendingEvent
	delegate string         // path segment reserved for a delegate tool's child
	childN   map[string]int // generated keys: per-agent ordinal within a tool call
	children []*RunResult   // children started here, in start order
}

// ref identifies an event that may not have a seq yet.
type ref struct {
	seq   uint64
	local int // index into a scope buffer, when buffered
	buf   bool
}

type pendingEvent struct {
	body  trace.Body
	cause *ref
}

type ctxKey int

const (
	scopeKey ctxKey = iota
	insideEffectKey
)

func withScope(ctx context.Context, sc *scope) context.Context {
	return context.WithValue(ctx, scopeKey, sc)
}

func scopeFrom(ctx context.Context) *scope {
	sc, _ := ctx.Value(scopeKey).(*scope)
	return sc
}

// insideEffect marks code that runs below the boundary (model providers,
// Do functions, context providers): nested Do calls there are plain calls,
// because the enclosing effect's recorded outcome already covers them.
func insideEffect(ctx context.Context) context.Context {
	return context.WithValue(ctx, insideEffectKey, true)
}

func isInsideEffect(ctx context.Context) bool {
	v, _ := ctx.Value(insideEffectKey).(bool)
	return v
}

// emit records an event in a scope: directly for the run, buffered for a
// tool call in a batch.
func (sc *scope) emit(ctx context.Context, body trace.Body, cause *ref) (ref, error) {
	if !sc.nested {
		var c *uint64
		if cause != nil {
			c = trace.Cause(cause.seq)
		}
		seq, err := sc.rs.w.Write(ctx, body, c)
		return ref{seq: seq}, err
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.buf = append(sc.buf, pendingEvent{body: body, cause: cause})
	return ref{local: len(sc.buf) - 1, buf: true}, nil
}

// drain returns the buffered events as Pending, resolving local causes
// against base, the seq the first of them will get.
func (sc *scope) drain(base uint64) []trace.Pending {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	out := make([]trace.Pending, len(sc.buf))
	for i, p := range sc.buf {
		out[i].Body = p.body
		if p.cause != nil {
			if p.cause.buf {
				out[i].Cause = trace.Cause(base + uint64(p.cause.local))
			} else {
				out[i].Cause = trace.Cause(p.cause.seq)
			}
		}
	}
	sc.buf = nil
	return out
}

// started is an effect whose *.started event has been recorded.
type started struct {
	call  *Call
	sc    *scope
	ref   ref
	begin time.Time
	// replayed is set when the replayer saw the call in Begin; it then
	// also sees it in Complete.
	replayed bool
}

// begin records an effect's start event and lets the replayer match it and
// decide the run's mode. Nested effects are matched only inside a call
// being verified or run transparently; otherwise their call's outcome
// covers them. Spawns always consult the replayer: a live parent can still
// start children that replay.
//
// If the start was recorded and begin then fails, it returns the started
// effect with the error; the caller must answer it (abandon).
func (rs *runState) begin(ctx context.Context, sc *scope, call *Call, body trace.Body, cause *ref) (*started, error) {
	if !sc.nested {
		if err := rs.stopped(); err != nil {
			return nil, err
		}
	}
	call.Meta.TraceID, call.Meta.RunID, call.Meta.Path = rs.traceID, rs.runID, rs.path
	call.Meta.Nested = sc.nested
	r, err := sc.emit(ctx, body, cause)
	if err != nil {
		return nil, err
	}
	if !r.buf {
		call.Meta.Seq = r.seq
	}
	rs.mu.Lock()
	call.Meta.Mode = rs.mode
	rs.mu.Unlock()
	replayed := rs.replayer != nil && (!sc.nested || sc.verified || call.Kind == EffectSpawn)
	s := &started{call: call, sc: sc, ref: r, begin: rs.clock(), replayed: replayed}
	if replayed {
		d, err := rs.replayer.Begin(ctx, call)
		if err != nil {
			s.replayed = false
			return s, err
		}
		call.Meta.Mode, call.Meta.Verify, call.Meta.Transparent = d.Mode, d.Verify, d.Transparent
		if sp, ok := call.Request.(*SpawnCall); ok {
			sp.child, sp.newRun = d.Child, d.NewRun
		}
		rs.mu.Lock()
		rs.mode = d.Mode
		rs.mu.Unlock()
		if d.Switch != nil {
			if _, err := sc.emit(ctx, d.Switch, &r); err != nil {
				return s, err
			}
		}
	}
	return s, nil
}

// execute runs the interceptor chain for a started effect.
func (rs *runState) execute(ctx context.Context, s *started) (out *Outcome, attempts int, err error) {
	if p := s.call.patched; p != nil {
		// Substituted by a Patch: not executed, and not an interceptor's
		// answer.
		s.call.reached.Store(true)
		return p, 1, nil
	}
	ctx, count := llm.WithAttemptCounter(withScope(ctx, s.sc))
	out, err = rs.chain(ctx, s.call)
	if err == nil && out == nil {
		err = fmt.Errorf("agentflow: interceptor returned neither an outcome nor an error for %s", s.call.Meta.Label)
	}
	return out, max(count(), 1), err
}

// completion builds the completion event for an effect.
func (rs *runState) completion(s *started, out *Outcome, attempts int) (trace.Body, error) {
	d := rs.clock().Sub(s.begin)
	call := s.call
	icp := !call.reached.Load()
	ovr := call.patched != nil
	switch call.Kind {
	case EffectLLM:
		step := call.Meta.Step
		if out.Failure != nil {
			return &trace.LLMFailed{Step: step, Error: *out.Failure, Attempts: attempts, Duration: d, Intercepted: icp}, nil
		}
		resp, ok := out.Response.(*llm.Response)
		if !ok || resp == nil {
			return nil, fmt.Errorf("agentflow: llm outcome is %T, want *llm.Response", out.Response)
		}
		rs.mu.Lock()
		rs.usage = rs.usage.Add(resp.Usage)
		rs.mu.Unlock()
		return &trace.LLMCompleted{Step: step, Message: resp.Message, StopReason: resp.StopReason, Usage: resp.Usage,
			Model: resp.Model, Attempts: attempts, Duration: d, Ext: resp.Ext, Override: ovr, Intercepted: icp}, nil
	case EffectTool:
		if out.Failure != nil {
			return &trace.ToolFailed{CallID: call.Meta.CallID, Error: *out.Failure, Duration: d, Intercepted: icp}, nil
		}
		o, ok := out.Response.(ToolOutput)
		if !ok {
			return nil, fmt.Errorf("agentflow: tool outcome is %T, want ToolOutput", out.Response)
		}
		return &trace.ToolCompleted{CallID: call.Meta.CallID, Content: o.Content, IsError: o.IsError, Duration: d, Override: ovr, Intercepted: icp}, nil
	case EffectCustom:
		if out.Failure != nil {
			return &trace.EffectFailed{Name: call.Name, Error: *out.Failure, Duration: d, Intercepted: icp}, nil
		}
		raw, ok := out.Response.(json.RawMessage)
		if !ok {
			return nil, fmt.Errorf("agentflow: custom effect outcome is %T, want json.RawMessage", out.Response)
		}
		return &trace.EffectCompleted{Name: call.Name, Output: raw, Duration: d, Override: ovr, Intercepted: icp}, nil
	case EffectSpawn:
		req := call.Request.(*SpawnCall)
		if out.Failure != nil {
			return &trace.RunJoined{ChildRunID: req.runID, Status: trace.StatusFailed, Output: llm.Blocks{}, Duration: d, Error: out.Failure, Intercepted: icp}, nil
		}
		res, ok := out.Response.(*RunResult)
		if !ok || res == nil {
			return nil, fmt.Errorf("agentflow: spawn outcome is %T, want *RunResult", out.Response)
		}
		output := res.Output
		if output == nil {
			output = llm.Blocks{}
		}
		return &trace.RunJoined{ChildRunID: req.runID, Status: res.Status, Output: output, Usage: res.TreeUsage(), Duration: d, Override: ovr, Intercepted: icp}, nil
	}
	return nil, fmt.Errorf("agentflow: unknown effect kind %q", call.Kind)
}

// abandon answers a started effect that won't complete (the run is ending:
// a divergence, an error), so that every start in a finished run has an
// answer: *.failed with code "cancelled", or run.joined with status
// "cancelled".
func (rs *runState) abandon(ctx context.Context, s *started, cause error) {
	msg := cause.Error()
	if i := strings.IndexByte(msg, '\n'); i >= 0 {
		msg = msg[:i]
	}
	f := trace.Error{Code: trace.ErrCodeCancelled, Message: "abandoned: " + msg}
	d := rs.clock().Sub(s.begin)
	call := s.call
	var body trace.Body
	switch call.Kind {
	case EffectLLM:
		body = &trace.LLMFailed{Step: call.Meta.Step, Error: f, Duration: d}
	case EffectTool:
		body = &trace.ToolFailed{CallID: call.Meta.CallID, Error: f, Duration: d}
	case EffectCustom:
		body = &trace.EffectFailed{Name: call.Name, Error: f, Duration: d}
	case EffectSpawn:
		body = &trace.RunJoined{ChildRunID: call.Request.(*SpawnCall).runID, Status: trace.StatusCancelled, Output: llm.Blocks{}, Duration: d, Error: &f}
	default:
		return
	}
	_, _ = s.sc.emit(context.WithoutCancel(ctx), body, &s.ref)
}

// abandonOrRecord answers an effect of an abandoned batch: with its real
// completion if it has an outcome, else as abandoned.
func (rs *runState) abandonOrRecord(ctx context.Context, s *started, out *Outcome, err error) {
	if out != nil && err == nil {
		if cb, cerr := rs.completion(s, out, 1); cerr == nil {
			_, _ = s.sc.emit(context.WithoutCancel(ctx), cb, &s.ref)
			return
		}
	}
	if err == nil {
		err = errors.New("the batch was abandoned")
	}
	rs.abandon(ctx, s, err)
}

// invoke runs one effect end to end: begin, execute, record completion.
func (rs *runState) invoke(ctx context.Context, sc *scope, call *Call, body trace.Body, cause *ref) (*Outcome, ref, error) {
	s, err := rs.begin(ctx, sc, call, body, cause)
	if err != nil {
		if s != nil {
			rs.abandon(ctx, s, err)
		}
		return nil, ref{}, err
	}
	out, attempts, err := rs.execute(ctx, s)
	if err != nil {
		rs.abandon(ctx, s, err)
		return nil, ref{}, err
	}
	return rs.finish(ctx, s, out, attempts)
}

// finish hands an executed effect to the replayer and records its
// completion (and a mode switch before it, if any). The completion is
// recorded even when the replayer then ends the run.
func (rs *runState) finish(ctx context.Context, s *started, out *Outcome, attempts int) (*Outcome, ref, error) {
	sw, cerr := rs.complete(ctx, s, out)
	if sw != nil {
		if _, err := s.sc.emit(ctx, sw, &s.ref); err != nil {
			return nil, ref{}, err
		}
	}
	cb, err := rs.completion(s, out, attempts)
	if err != nil {
		rs.abandon(ctx, s, err)
		return nil, ref{}, err
	}
	done, err := s.sc.emit(ctx, cb, &s.ref)
	if err == nil {
		err = cerr
	}
	return out, done, err
}

// complete hands an executed call's outcome to the replayer, which compares
// it with the recording when it didn't come from there, and returns a mode
// switch to record before the completion.
func (rs *runState) complete(ctx context.Context, s *started, out *Outcome) (*trace.RunModeChanged, error) {
	if !s.replayed {
		return nil, nil
	}
	sw, err := rs.replayer.Complete(ctx, s.call, out, !s.call.reached.Load())
	if sw != nil {
		rs.mu.Lock()
		rs.mode = ModeLive
		rs.mu.Unlock()
	}
	return sw, err
}

// doEffect runs a custom effect in scope sc (see Do). conv is set for
// context providers: the conversation the provider receives.
func (rs *runState) doEffect(ctx context.Context, sc *scope, name string, input any, conv []llm.Message, fn func(context.Context) (any, error)) (*Outcome, error) {
	if !sc.busy.CompareAndSwap(false, true) {
		return nil, fmt.Errorf("%w: effect %q started while another effect is in flight", ErrConcurrentEffect, name)
	}
	defer sc.busy.Store(false)

	in, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("agentflow: effect %q input: %w", name, err)
	}
	fp, err := trace.EffectFingerprint(name, in)
	if err != nil {
		return nil, err
	}
	rs.mu.Lock()
	n := rs.effectN[sc.prefix+name]
	rs.effectN[sc.prefix+name] = n + 1
	rs.mu.Unlock()
	key := fmt.Sprintf("%seffect:%s#%d", sc.prefix, name, n)
	call := &Call{Kind: EffectCustom, Name: name, Request: &CustomCall{Name: name, Input: in, fn: fn, conv: conv}, Fingerprint: fp,
		Meta: CallMeta{Key: key, Label: key}}
	p, err := rs.patch(ctx, key)
	if err != nil {
		return nil, err
	}
	if p != nil {
		call.patched = p.Outcome
	}
	var cause *ref
	if sc.nested {
		cause = &ref{seq: sc.parentStarted}
	}
	out, _, err := rs.invoke(ctx, sc, call, &trace.EffectStarted{Name: name, Input: in, Fingerprint: fp}, cause)
	return out, err
}

// --- chain ---

// buildChain composes user interceptors, the timeout layer, the replayer,
// and the terminal invoker.
func (rs *runState) buildChain(user []Interceptor, timeouts Timeouts, noLive bool) Invoker {
	var term Invoker = rs.terminal
	if noLive {
		// The network guard. Verified and transparent calls (never LLM
		// calls) run local code during replay, so they pass; a transparent
		// spawn's child is guarded by its own chain.
		term = func(ctx context.Context, call *Call) (*Outcome, error) {
			if (call.Meta.Verify || call.Meta.Transparent) && call.Kind != EffectLLM {
				return rs.terminal(ctx, call)
			}
			return nil, fmt.Errorf("%w: %s", ErrLiveCallInReplay, call.Meta.Label)
		}
	}
	next := term
	if rs.replayer != nil {
		r, inner := rs.replayer, next
		next = func(ctx context.Context, call *Call) (*Outcome, error) { return r.Intercept(ctx, call, inner) }
	}
	inner := next
	next = func(ctx context.Context, call *Call) (*Outcome, error) {
		call.reached.Store(true)
		return timeouts.intercept(ctx, call, inner)
	}
	for i := len(user) - 1; i >= 0; i-- {
		ic, inner := user[i], next
		next = func(ctx context.Context, call *Call) (*Outcome, error) { return ic.Intercept(ctx, call, inner) }
	}
	return next
}

// Timeouts bound how long effects may take. Zero means no limit.
type Timeouts struct {
	LLM     time.Duration
	Tool    time.Duration
	PerTool map[string]time.Duration // overrides Tool for named tools
	Effect  time.Duration            // custom effects
}

func (t Timeouts) forCall(call *Call) time.Duration {
	switch call.Kind {
	case EffectLLM:
		return t.LLM
	case EffectTool:
		if d, ok := t.PerTool[call.Name]; ok {
			return d
		}
		return t.Tool
	case EffectCustom:
		return t.Effect
	}
	return 0
}

func (t Timeouts) intercept(ctx context.Context, call *Call, next Invoker) (*Outcome, error) {
	d := t.forCall(call)
	if d <= 0 || (call.Meta.Mode == ModeReplay && !call.Meta.Verify && !call.Meta.Transparent) {
		return next(ctx, call)
	}
	tctx, cancel := context.WithTimeout(ctx, d)
	defer cancel()
	out, err := next(tctx, call)
	if errors.Is(tctx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
		return Fail(trace.ErrCodeTimeout, fmt.Sprintf("%s exceeded its %s timeout", call.Meta.Label, d)), nil
	}
	return out, err
}

// terminal performs the real effect.
func (rs *runState) terminal(ctx context.Context, call *Call) (out *Outcome, err error) {
	defer func() {
		if p := recover(); p != nil {
			code := trace.ErrCodePanic
			if call.Kind == EffectLLM {
				code = trace.ErrCodeInternal
			}
			out = &Outcome{Failure: &trace.Error{Code: code, Message: fmt.Sprint(p), Detail: truncate(string(debug.Stack()), 4096)}}
			err = nil
		}
	}()
	switch req := call.Request.(type) {
	case *llm.Request:
		resp, err := rs.agent.model.Generate(insideEffect(ctx), req)
		if err != nil {
			return &Outcome{Failure: llmFailure(ctx, err)}, nil
		}
		if resp == nil {
			return Fail(trace.ErrCodeInternal, "model returned no response"), nil
		}
		return &Outcome{Response: resp}, nil
	case *ToolCall:
		if rs.cfg.disabled[req.Name] {
			return Fail(trace.ErrCodeNotFound, fmt.Sprintf("tool %q not found", req.Name)), nil
		}
		return rs.agent.callTool(ctx, req), nil
	case *SpawnCall:
		return rs.startChild(ctx, req), nil
	case *CustomCall:
		v, err := req.fn(insideEffect(ctx))
		if err != nil {
			if ctx.Err() != nil {
				return Fail(trace.ErrCodeCancelled, err.Error()), nil
			}
			return Fail(codeError, err.Error()), nil
		}
		raw, err := json.Marshal(v)
		if err != nil {
			return Fail(trace.ErrCodeInternal, fmt.Sprintf("effect output: %v", err)), nil
		}
		return &Outcome{Response: json.RawMessage(raw)}, nil
	}
	return nil, fmt.Errorf("agentflow: unknown request type %T", call.Request)
}

func llmFailure(ctx context.Context, err error) *trace.Error {
	var le *llm.Error
	if errors.As(err, &le) {
		return &trace.Error{Code: le.Code, Message: le.Message, Retryable: le.Retryable}
	}
	if ctx.Err() != nil {
		return &trace.Error{Code: llm.ErrCodeCancelled, Message: err.Error()}
	}
	return &trace.Error{Code: trace.ErrCodeInternal, Message: err.Error()}
}

// callTool looks up, validates, and runs a tool.
func (a *Agent) callTool(ctx context.Context, req *ToolCall) *Outcome {
	tool, ok := a.tools[req.Name]
	if !ok {
		return Fail(trace.ErrCodeNotFound, fmt.Sprintf("tool %q not found", req.Name))
	}
	if a.validate {
		if s := tool.Spec().InputSchema; len(s) > 0 {
			input := req.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			if err := schema.Validate(s, input); err != nil {
				return Fail(trace.ErrCodeInvalidInput, err.Error())
			}
		}
	}
	out, err := tool.Call(ctx, req.Input)
	if err != nil {
		if ctx.Err() != nil {
			return Fail(trace.ErrCodeCancelled, err.Error())
		}
		return Fail(codeError, err.Error())
	}
	return &Outcome{Response: out}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
