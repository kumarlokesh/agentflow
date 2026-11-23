package agentflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"
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
	agent    *Agent
	w        *trace.Writer
	traceID  string
	runID    string
	path     string
	chain    Invoker
	replayer Replayer
	mode     Mode
	clock    func() time.Time

	mu       sync.Mutex
	llmN     int
	effectN  map[string]int // scope prefix + effect name → next ordinal
	toolN    map[string]int // call ID → occurrences so far
	usage    llm.Usage
	runScope *scope
}

// scope is where effects are recorded: the run itself, or one tool call in
// a batch, whose events are buffered until the batch completes (R7).
type scope struct {
	rs     *runState
	prefix string // "" for the run, "tool:<call_id>/" for a tool call
	nested bool
	busy   atomic.Bool

	parentStarted uint64 // seq of the enclosing tool.started
	mu            sync.Mutex
	buf           []pendingEvent
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
}

// begin records an effect's *.started event and, for non-nested effects,
// lets the replayer match it and decide the run's mode.
func (rs *runState) begin(ctx context.Context, sc *scope, call *Call, body trace.Body, cause *ref) (*started, error) {
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
	if rs.replayer != nil && !sc.nested {
		d, err := rs.replayer.Begin(ctx, call)
		if err != nil {
			return nil, err
		}
		call.Meta.Mode = d.Mode
		rs.mu.Lock()
		rs.mode = d.Mode
		rs.mu.Unlock()
		if d.Switch != nil {
			if _, err := sc.emit(ctx, d.Switch, &r); err != nil {
				return nil, err
			}
		}
	}
	return &started{call: call, sc: sc, ref: r, begin: rs.clock()}, nil
}

// execute runs the interceptor chain for a started effect.
func (rs *runState) execute(ctx context.Context, s *started) (out *Outcome, attempts int, err error) {
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
			Model: resp.Model, Attempts: attempts, Duration: d, Ext: resp.Ext, Intercepted: icp}, nil
	case EffectTool:
		if out.Failure != nil {
			return &trace.ToolFailed{CallID: call.Meta.CallID, Error: *out.Failure, Duration: d, Intercepted: icp}, nil
		}
		o, ok := out.Response.(ToolOutput)
		if !ok {
			return nil, fmt.Errorf("agentflow: tool outcome is %T, want ToolOutput", out.Response)
		}
		return &trace.ToolCompleted{CallID: call.Meta.CallID, Content: o.Content, IsError: o.IsError, Duration: d, Intercepted: icp}, nil
	case EffectCustom:
		if out.Failure != nil {
			return &trace.EffectFailed{Name: call.Name, Error: *out.Failure, Duration: d, Intercepted: icp}, nil
		}
		raw, ok := out.Response.(json.RawMessage)
		if !ok {
			return nil, fmt.Errorf("agentflow: custom effect outcome is %T, want json.RawMessage", out.Response)
		}
		return &trace.EffectCompleted{Name: call.Name, Output: raw, Duration: d, Intercepted: icp}, nil
	}
	return nil, fmt.Errorf("agentflow: unknown effect kind %q", call.Kind)
}

// invoke runs one effect end to end: begin, execute, record completion.
func (rs *runState) invoke(ctx context.Context, sc *scope, call *Call, body trace.Body, cause *ref) (*Outcome, ref, error) {
	s, err := rs.begin(ctx, sc, call, body, cause)
	if err != nil {
		return nil, ref{}, err
	}
	out, attempts, err := rs.execute(ctx, s)
	if err != nil {
		return nil, ref{}, err
	}
	if sw := rs.takeSwitch(s); sw != nil {
		if _, err := sc.emit(ctx, sw, &s.ref); err != nil {
			return nil, ref{}, err
		}
	}
	cb, err := rs.completion(s, out, attempts)
	if err != nil {
		return nil, ref{}, err
	}
	done, err := sc.emit(ctx, cb, &s.ref)
	return out, done, err
}

// takeSwitch returns a mode switch the replayer made while s executed.
func (rs *runState) takeSwitch(s *started) *trace.RunModeChanged {
	if rs.replayer == nil || s.call.Meta.Nested {
		return nil
	}
	sw := rs.replayer.TakeSwitch(s.call.Meta.Key)
	if sw != nil {
		rs.mu.Lock()
		rs.mode = ModeLive
		rs.mu.Unlock()
	}
	return sw
}

// doEffect runs a custom effect in scope sc (see Do).
func (rs *runState) doEffect(ctx context.Context, sc *scope, name string, input any, fn func(context.Context) (any, error)) (*Outcome, error) {
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
	call := &Call{Kind: EffectCustom, Name: name, Request: &CustomCall{Name: name, Input: in, fn: fn}, Fingerprint: fp,
		Meta: CallMeta{Key: key, Label: key}}
	var cause *ref
	if sc.nested {
		cause = &ref{seq: sc.parentStarted}
	}
	out, _, err := rs.invoke(ctx, sc, call, &trace.EffectStarted{Name: name, Input: in, Fingerprint: fp}, cause)
	return out, err
}

// --- chain ---

// buildChain composes user interceptors, the timeout layer, the replayer,
// and the terminal invoker (RFC-0002 §3).
func (rs *runState) buildChain(user []Interceptor, timeouts Timeouts, noLive bool) Invoker {
	var term Invoker = rs.terminal
	if noLive {
		term = func(_ context.Context, call *Call) (*Outcome, error) {
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
	if d <= 0 || call.Meta.Mode == ModeReplay {
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
		return rs.agent.callTool(ctx, req), nil
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
