// Package agentflow is a runtime for AI agents in which every run is
// recorded as a trace that can be replayed against new code, diffed, and
// (in later versions) forked.
//
// An Agent runs an Observe-Think-Act loop. Everything nondeterministic the
// loop does (model calls, tool calls, context providers, and effects
// declared with Do) crosses one effect boundary, where it is recorded in a
// trace.Store and passes through a chain of Interceptors. Package replay
// re-runs current code against a recorded trace and reports exactly where
// the new run diverges.
package agentflow

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

var (
	// ErrNoModel is returned by NewAgent when AgentConfig.Model is nil.
	ErrNoModel = errors.New("agentflow: no model configured")
	// ErrDiverged is wrapped by errors that end a run because it diverged
	// from the recording it was replayed against.
	ErrDiverged = errors.New("agentflow: run diverged from recording")
	// ErrLiveCallInReplay means an effect reached the real world during a
	// strict replay. It indicates a bug or a misconfigured replay.
	ErrLiveCallInReplay = errors.New("agentflow: live call during strict replay")
	// ErrConcurrentEffect is returned by Do when two effects are started
	// concurrently on the same run outside a runtime-managed batch
	// (RFC-0002 R7).
	ErrConcurrentEffect = errors.New("agentflow: concurrent effects on one run")
)

// Entrypoint is something that can be run as the root of a trace: an
// *Agent (and, from RFC-0003, a workflow). Replay and cassettes accept any
// Entrypoint.
type Entrypoint interface {
	Name() string
	Run(ctx context.Context, task string, opts ...RunOption) (*RunResult, error)
	isEntrypoint()
}

// RunResult describes a finished run.
type RunResult struct {
	TraceID  string
	RunID    string
	Path     string
	Status   trace.RunStatus
	Output   llm.Blocks
	Usage    llm.Usage
	Steps    int
	Duration time.Duration
	// Store holds the run's trace.
	Store trace.Store
}

// OutputText returns the text of the run's output.
func (r *RunResult) OutputText() string { return llm.TextOf(r.Output) }

// Events returns the run's recorded events.
func (r *RunResult) Events(ctx context.Context) ([]trace.Event, error) {
	return trace.Collect(r.Store.Events(ctx, r.RunID, 0))
}

// RunError is returned when a run ends with any status other than
// completed. The RunResult is returned alongside it.
type RunError struct {
	Status trace.RunStatus
	Err    error
}

func (e *RunError) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("agentflow: run %s", e.Status)
	}
	return fmt.Sprintf("agentflow: run %s: %v", e.Status, e.Err)
}

func (e *RunError) Unwrap() error { return e.Err }

// EffectError is returned by Do when the effect failed. During replay, a
// recorded failure is returned as the same EffectError.
type EffectError struct {
	Code    string
	Message string
}

func (e *EffectError) Error() string {
	return fmt.Sprintf("agentflow: effect failed (%s): %s", e.Code, e.Message)
}

// RunOption configures a single run.
type RunOption func(*runOptions)

type runOptions struct {
	replayer     Replayer
	origin       *trace.Origin
	store        trace.Store
	task         llm.Blocks
	noLive       bool
	interceptors []Interceptor
	observers    []trace.Observer
}

// WithReplayer installs a replayer at the replayer slot (RFC-0002 §3).
func WithReplayer(r Replayer) RunOption { return func(o *runOptions) { o.replayer = r } }

// WithOrigin stamps the root run's origin (default: live).
func WithOrigin(origin trace.Origin) RunOption {
	return func(o *runOptions) { o.origin = &origin }
}

// WithStore sets where the trace is written, overriding AgentConfig.Store.
func WithStore(s trace.Store) RunOption { return func(o *runOptions) { o.store = s } }

// WithTaskBlocks sets the task as content blocks, replacing the string task.
func WithTaskBlocks(task llm.Blocks) RunOption { return func(o *runOptions) { o.task = task } }

// WithLiveCallsDisabled makes any effect that would reach the real world
// fail the run with ErrLiveCallInReplay (the strict-replay network guard).
func WithLiveCallsDisabled() RunOption { return func(o *runOptions) { o.noLive = true } }

// WithInterceptors adds interceptors for this run, after the agent's own.
func WithInterceptors(is ...Interceptor) RunOption {
	return func(o *runOptions) { o.interceptors = append(o.interceptors, is...) }
}

// WithObservers adds observers for this run, after the agent's own.
func WithObservers(obs ...trace.Observer) RunOption {
	return func(o *runOptions) { o.observers = append(o.observers, obs...) }
}
