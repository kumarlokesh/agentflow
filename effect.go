package agentflow

import (
	"context"
	"encoding/json"
	"sync/atomic"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// EffectKind is the kind of an effect: an interaction with the
// nondeterministic world.
type EffectKind string

const (
	EffectLLM    EffectKind = "llm"
	EffectTool   EffectKind = "tool"
	EffectCustom EffectKind = "custom"
)

// Mode says whether a run's effects are answered from a recording (replay)
// or by the real world (live).
type Mode string

const (
	ModeLive   Mode = "live"
	ModeReplay Mode = "replay"
)

// Call is one effect passing through the interceptor chain. Interceptors
// must treat it as read-only.
type Call struct {
	Kind EffectKind
	// Name is the model name (llm), tool name (tool), or effect name (custom).
	Name string
	// Request is *llm.Request, *ToolCall, or *CustomCall.
	Request     any
	Fingerprint string
	Meta        CallMeta

	// reached is set when the call passes the user interceptors; if it
	// never does, an interceptor produced the outcome.
	reached atomic.Bool
}

// CallMeta locates a call within its trace.
type CallMeta struct {
	TraceID string
	RunID   string
	Path    string
	// Seq is the seq of the call's *.started event. It is zero for effects
	// nested in a parallel batch, whose events are written when the batch
	// completes.
	Seq    uint64
	Step   int
	CallID string // tool calls
	// Key identifies the call for replay matching: "llm#0", "tool:toolu_01",
	// "effect:clock.now#0", or a scoped key such as
	// "tool:toolu_01/effect:clock.now#0" for nested effects.
	Key string
	// Label is a human-readable name, e.g. "tool:calculator(toolu_01)".
	Label string
	// Nested is true for effects started inside a tool call.
	Nested bool
	// Mode is the run's mode when the call started.
	Mode Mode
}

// ToolCall is the request of a tool effect.
type ToolCall struct {
	CallID string
	Name   string
	Input  json.RawMessage
}

// CustomCall is the request of a custom effect (see Do).
type CustomCall struct {
	Name  string
	Input json.RawMessage
	fn    func(ctx context.Context) (any, error)
}

// Outcome is what an effect produced. Response is *llm.Response (llm),
// ToolOutput (tool), or json.RawMessage (custom). Failure is set instead
// for a failure that is recorded and replayed: a rate-limited model, a tool
// timeout, a policy denial.
type Outcome struct {
	Response any
	Failure  *trace.Error
}

// Fail returns an Outcome carrying a recordable failure.
func Fail(code, message string) *Outcome {
	return &Outcome{Failure: &trace.Error{Code: code, Message: message}}
}

// Invoker calls the rest of the chain.
type Invoker func(ctx context.Context, call *Call) (*Outcome, error)

// Interceptor sees every effect. It may allow (call next),
// deny or otherwise fail the call (return an Outcome with Failure), delay,
// or observe, but must not modify the request. A non-nil error aborts the
// run; use it only for failures of the runtime itself.
//
// Interceptors run in live and replay mode alike, so they must be
// deterministic given the trace so far, or check call.Meta.Mode. Calls in
// one parallel tool batch go through the chain concurrently; an interceptor
// whose decisions depend on their order needs MaxParallelTools = 1.
type Interceptor interface {
	Intercept(ctx context.Context, call *Call, next Invoker) (*Outcome, error)
}

// InterceptorFunc adapts a function to Interceptor.
type InterceptorFunc func(ctx context.Context, call *Call, next Invoker) (*Outcome, error)

// Intercept implements Interceptor.
func (f InterceptorFunc) Intercept(ctx context.Context, call *Call, next Invoker) (*Outcome, error) {
	return f(ctx, call, next)
}

// Replayer answers effects from a recording. Package replay implements it;
// install one with WithReplayer.
type Replayer interface {
	// Begin is called for every non-nested effect right after its
	// *.started event is written, before user interceptors run. It matches
	// the effect against the recording and returns the run's mode for this
	// call. An error that wraps ErrDiverged ends the run with status
	// "diverged".
	Begin(ctx context.Context, call *Call) (Decision, error)
	// Interceptor answers calls in replay mode at the replayer slot of the
	// chain, and passes live calls to next.
	Interceptor
	// Finish is called when the run ends (except when Begin already ended
	// it). An error that wraps ErrDiverged changes the run's status to
	// "diverged".
	Finish(ctx context.Context, status trace.RunStatus, output llm.Blocks) error
	// TakeSwitch returns (once) a mode switch that happened while the call
	// with this key was executing, so the runtime can record it.
	TakeSwitch(key string) *trace.RunModeChanged
}

// Decision is a Replayer's verdict for one call.
type Decision struct {
	Mode Mode
	// Switch, if set, is written as a run.mode_changed event: the run
	// stops being answered from the recording at this call.
	Switch *trace.RunModeChanged
}
