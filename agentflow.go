// Package agentflow is a runtime for AI agents in which every run is
// recorded as a trace that can be replayed against new code, diffed, and
// (in later versions) forked.
//
// An Agent runs an Observe-Think-Act loop. Agents can start other agents,
// as tools the model calls (Delegate) or from Go code (a Workflow, Spawn,
// SpawnAll); every run a top-level run causes lands in its trace as a tree. Everything nondeterministic the
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
	// ErrConcurrentEffect is returned by Do, Spawn and SpawnAll when two
	// effects are started concurrently on the same run outside a
	// runtime-managed batch.
	ErrConcurrentEffect = errors.New("agentflow: concurrent effects on one run")
	// ErrAgentNameConflict is returned when two different agent (or
	// workflow) definitions with the same name run in one trace: run paths
	// would be ambiguous.
	ErrAgentNameConflict = errors.New("agentflow: two different agents with one name in a trace")
	// ErrSpawnInsideEffect is returned when a child run is started inside an
	// effect function (a Do function, a context provider, a model): the
	// effect's recorded outcome would cover the child, so replay could never
	// run it. Start children from a tool or a workflow, or pass a context
	// that doesn't carry the run (e.g. context.Background()) to start a
	// separate trace.
	ErrSpawnInsideEffect = errors.New("agentflow: child run started inside an effect function")
)

// Entrypoint is something that can be run: an *Agent or a *Workflow.
// Replay and cassettes accept any Entrypoint. Run starts a new trace, or,
// when ctx belongs to a run in progress (inside a tool or a workflow),
// starts a child run in that run's tree, exactly like Spawn.
type Entrypoint interface {
	Name() string
	Run(ctx context.Context, task string, opts ...RunOption) (*RunResult, error)
	def() *runDef
}

// RunResult describes a finished run.
type RunResult struct {
	TraceID string
	RunID   string
	Path    string
	Kind    trace.RunKind
	Status  trace.RunStatus
	Output  llm.Blocks
	// Usage is this run's own usage; see TreeUsage for its children's too.
	Usage    llm.Usage
	Steps    int
	Duration time.Duration
	// Store holds the run's trace.
	Store trace.Store
	// Children are the runs this run started, in the order of their
	// run.spawned events. A child answered from a recording (an opaque
	// replay) has no children of its own, and its TraceID, RunID and Store
	// point at the recording.
	Children []*RunResult

	err error // the run's error, for a parent that joins it
}

// OutputText returns the text of the run's output.
func (r *RunResult) OutputText() string { return llm.TextOf(r.Output) }

// TreeUsage returns the usage of this run and all its descendants.
func (r *RunResult) TreeUsage() llm.Usage {
	u := r.Usage
	for _, c := range r.Children {
		u = u.Add(c.TreeUsage())
	}
	return u
}

// Find returns the run at path in this run's subtree, or nil.
func (r *RunResult) Find(path string) *RunResult {
	if r.Path == path {
		return r
	}
	for _, c := range r.Children {
		if f := c.Find(path); f != nil {
			return f
		}
	}
	return nil
}

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
	limits       *TreeLimits
}

// treeOptions reports whether any option other than the task is set; those
// configure a whole trace, so a child run can't take them.
func (o *runOptions) treeOptions() bool {
	return o.replayer != nil || o.origin != nil || o.store != nil || o.noLive ||
		len(o.interceptors) > 0 || len(o.observers) > 0 || o.limits != nil
}

// The options below configure the whole trace when given to a top-level
// run; a run started inside another run (a child) accepts only
// WithTaskBlocks.

// WithReplayer installs a replayer at the replayer slot.
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

// WithInterceptors adds interceptors to every run in the trace, after each
// agent's own (AgentConfig.Interceptors, which apply to that agent's runs
// only). Pass a policy.Budget here to share one budget across a team.
func WithInterceptors(is ...Interceptor) RunOption {
	return func(o *runOptions) { o.interceptors = append(o.interceptors, is...) }
}

// WithObservers adds observers to every run in the trace, after each
// agent's own (AgentConfig.Observers, which see that agent's runs only).
func WithObservers(obs ...trace.Observer) RunOption {
	return func(o *runOptions) { o.observers = append(o.observers, obs...) }
}

// TreeLimits bound a trace's run tree. Zero fields take the defaults.
type TreeLimits struct {
	// MaxDepth is the deepest a run may be; the root is depth 0. Default 5.
	MaxDepth int
	// MaxRuns is the most runs a trace may have, the root included.
	// Default 64. Under concurrency, which spawn reaches the limit depends
	// on scheduling, like a shared budget (see Interceptor).
	MaxRuns int
	// MaxConcurrentRuns is the most child runs executing at once in the
	// whole trace. A run waiting for its own children doesn't count.
	// Default 8.
	MaxConcurrentRuns int
}

// Default tree limits.
const (
	DefaultMaxDepth          = 5
	DefaultMaxRuns           = 64
	DefaultMaxConcurrentRuns = 8
)

// WithLimits sets the tree limits of the trace. A spawn beyond MaxDepth or
// MaxRuns fails with the recordable error code "limit_exceeded": a delegate
// tool returns it to the model as an error result, Spawn as a *RunError.
func WithLimits(l TreeLimits) RunOption { return func(o *runOptions) { o.limits = &l } }
