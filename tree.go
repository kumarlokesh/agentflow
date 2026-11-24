package agentflow

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// tree is what the runs of one trace share.
type tree struct {
	traceID      string
	store        trace.Store
	limits       TreeLimits
	interceptors []Interceptor    // WithInterceptors, for every run
	observers    []trace.Observer // WithObservers, for every run
	noLive       bool
	slots        chan struct{} // MaxConcurrentRuns

	mu    sync.Mutex
	runs  int
	names map[string]Entrypoint
	paths map[string]bool
}

func newTree(traceID string, store trace.Store, o *runOptions) *tree {
	l := TreeLimits{}
	if o.limits != nil {
		l = *o.limits
	}
	if l.MaxDepth <= 0 {
		l.MaxDepth = DefaultMaxDepth
	}
	if l.MaxRuns <= 0 {
		l.MaxRuns = DefaultMaxRuns
	}
	if l.MaxConcurrentRuns <= 0 {
		l.MaxConcurrentRuns = DefaultMaxConcurrentRuns
	}
	return &tree{
		traceID: traceID, store: store, limits: l,
		interceptors: o.interceptors, observers: o.observers, noLive: o.noLive,
		slots: make(chan struct{}, l.MaxConcurrentRuns),
		names: map[string]Entrypoint{}, paths: map[string]bool{},
	}
}

// register records that e runs in this trace under its name.
func (t *tree) register(e Entrypoint) error {
	name := e.Name()
	if name == "" {
		return errors.New("agentflow: agent name is required")
	}
	if strings.ContainsAny(name, "/#@") {
		return fmt.Errorf("agentflow: agent name %q: '/', '#' and '@' are reserved for run paths", name)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if prev, ok := t.names[name]; ok && prev != e {
		return fmt.Errorf("%w: %q", ErrAgentNameConflict, name)
	}
	t.names[name] = e
	return nil
}

// reservePath claims a child path; paths are unique within a trace.
func (t *tree) reservePath(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.paths[path] {
		return fmt.Errorf("agentflow: run path %s is already used in this trace (a repeated SpawnKey?)", path)
	}
	t.paths[path] = true
	return nil
}

// admit counts a new child run at depth, or refuses it.
func (t *tree) admit(depth int) *trace.Error {
	t.mu.Lock()
	defer t.mu.Unlock()
	switch {
	case depth > t.limits.MaxDepth:
		return &trace.Error{Code: trace.ErrCodeLimitExceeded, Message: fmt.Sprintf("run depth %d exceeds the limit of %d", depth, t.limits.MaxDepth)}
	case t.runs >= t.limits.MaxRuns:
		return &trace.Error{Code: trace.ErrCodeLimitExceeded, Message: fmt.Sprintf("the trace already has %d runs, the limit", t.runs)}
	}
	t.runs++
	return nil
}

func (t *tree) acquire(ctx context.Context) error {
	select {
	case t.slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (t *tree) release() { <-t.slots }

// runDef is what an Entrypoint contributes to a run.
type runDef struct {
	kind          trace.RunKind
	name, version string
	config        *trace.Config
	agent         *Agent // agent runs
	interceptors  []Interceptor
	observers     []trace.Observer
	timeouts      Timeouts
	store         trace.Store // default store for a top-level run
	logger        *slog.Logger
	clock         func() time.Time
	newID         func() string
	body          func(ctx context.Context, rs *runState, task llm.Blocks) (trace.RunStatus, llm.Blocks, int, error)
}

// runContext is where a run sits in its tree.
type runContext struct {
	tree      *tree
	runID     string
	path      string
	parent    string // parent run ID
	depth     int
	task      llm.Blocks
	replayer  Replayer
	origin    *trace.Origin // root only
	holdsSlot bool
}

// runEntrypoint implements Entrypoint.Run: a new trace, or a child run when
// ctx belongs to a run in progress.
func runEntrypoint(ctx context.Context, e Entrypoint, task string, opts []RunOption) (*RunResult, error) {
	var o runOptions
	for _, opt := range opts {
		opt(&o)
	}
	taskBlocks := o.task
	if taskBlocks == nil {
		taskBlocks = llm.Blocks{llm.Text{Text: task}}
	}
	if sc := scopeFrom(ctx); sc != nil {
		if o.treeOptions() {
			return nil, fmt.Errorf("agentflow: %s started inside a run joins that run's trace; only WithTaskBlocks applies to it", e.Name())
		}
		return sc.rs.spawnOne(ctx, sc, spawnReq{e: e, task: taskBlocks})
	}
	d := e.def()
	store := o.store
	if store == nil {
		store = d.store
	}
	if store == nil {
		store = memstore.New()
	}
	origin := o.origin
	if origin == nil {
		origin = &trace.Origin{Kind: trace.OriginLive}
	}
	t := newTree(d.newID(), store, &o)
	if err := t.register(e); err != nil {
		return nil, err
	}
	t.runs = 1
	return execRun(ctx, d, &runContext{tree: t, runID: d.newID(), path: "/", task: taskBlocks, replayer: o.replayer, origin: origin})
}

// execRun runs one run of a tree: it records run.started, runs the body,
// and records run.finished. It returns the result and, unless the run
// completed, a *RunError.
func execRun(ctx context.Context, d *runDef, rc *runContext) (*RunResult, error) {
	t := rc.tree
	observers := append(append([]trace.Observer{}, d.observers...), t.observers...)
	rs := &runState{
		def: d, agent: d.agent, tree: t, traceID: t.traceID, runID: rc.runID, path: rc.path, depth: rc.depth,
		w:        trace.NewWriter(t.store, t.traceID, rc.runID, trace.WriterOptions{Now: d.clock, Observers: observers}),
		replayer: rc.replayer, mode: ModeLive, clock: d.clock,
		effectN: map[string]int{}, toolN: map[string]int{}, childN: map[string]int{}, holdsSlot: rc.holdsSlot,
	}
	defer rs.releaseSlot()
	if rs.replayer != nil {
		rs.mode = ModeReplay
	}
	rs.runScope = &scope{rs: rs}
	rs.chain = rs.buildChain(append(append([]Interceptor{}, d.interceptors...), t.interceptors...), d.timeouts, t.noLive)
	ctx = withScope(context.WithValue(ctx, insideEffectKey, false), rs.runScope)

	start := d.clock()
	cfgFP, err := trace.ConfigFingerprint(d.config)
	if err != nil {
		return nil, err
	}
	var parent *trace.ParentRef
	if rc.parent != "" {
		parent = &trace.ParentRef{RunID: rc.parent}
	}
	if _, err := rs.w.Write(ctx, &trace.RunStarted{
		RunKind: d.kind, Path: rc.path, Parent: parent, Agent: trace.AgentInfo{Name: d.name, Version: d.version},
		Task: rc.task, Config: d.config, ConfigFP: cfgFP, Origin: rc.origin,
	}, nil); err != nil {
		return nil, err
	}
	d.logger.Debug("run started", "trace_id", t.traceID, "run_id", rc.runID, "path", rc.path, "agent", d.name)

	status, output, steps, runErr := d.body(ctx, rs, rc.task)
	if cerr := rs.stopped(); cerr != nil && !errors.Is(runErr, ErrDiverged) {
		// A child diverged and the body carried on (e.g. a workflow that
		// ignored Spawn's error): the run still ends diverged.
		status, runErr = trace.StatusDiverged, errors.Join(cerr, runErr)
	}

	if rs.replayer != nil && !errors.Is(runErr, ErrDiverged) {
		if ferr := rs.replayer.Finish(ctx, status, output); ferr != nil {
			if errors.Is(ferr, ErrDiverged) {
				status = trace.StatusDiverged
			}
			runErr = errors.Join(runErr, ferr)
		}
	}
	if output == nil {
		output = llm.Blocks{}
	}
	fin := &trace.RunFinished{Status: status, Output: output, Usage: rs.usageNow(), Steps: steps}
	if runErr != nil {
		fin.Error = &trace.Error{Code: errorCode(status), Message: runErr.Error()}
	}
	if _, err := rs.w.Write(context.WithoutCancel(ctx), fin, nil); err != nil && runErr == nil {
		runErr, status = err, trace.StatusFailed
	}
	res := &RunResult{
		TraceID: t.traceID, RunID: rc.runID, Path: rc.path, Kind: d.kind, Status: status, Output: output,
		Usage: rs.usageNow(), Steps: steps, Duration: d.clock().Sub(start), Store: t.store, Children: rs.runScope.joined(),
	}
	d.logger.Debug("run finished", "trace_id", t.traceID, "path", rc.path, "status", status, "steps", steps)
	if status != trace.StatusCompleted {
		res.err = &RunError{Status: status, Err: runErr}
		return res, res.err
	}
	return res, nil
}

func (rs *runState) usageNow() llm.Usage {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return rs.usage
}

// stopped returns the error of a child that ended diverged: its parent
// can't continue either.
func (rs *runState) stopped() error {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return rs.childErr
}

// --- concurrency slots (TreeLimits.MaxConcurrentRuns) ---
//
// A child run holds a slot while it executes. A run waiting for its own
// children gives its slot up until the last of them joins, so nested
// delegation never deadlocks however deep the tree is.

func (rs *runState) waitChildren() {
	rs.slotMu.Lock()
	defer rs.slotMu.Unlock()
	rs.waiting++
	if rs.waiting == 1 && rs.holdsSlot {
		rs.tree.release()
		rs.holdsSlot, rs.gaveUpSlot = false, true
	}
}

func (rs *runState) childrenJoined(ctx context.Context) {
	rs.slotMu.Lock()
	defer rs.slotMu.Unlock()
	rs.waiting--
	if rs.waiting == 0 && rs.gaveUpSlot {
		rs.gaveUpSlot = false
		if rs.tree.acquire(ctx) == nil {
			rs.holdsSlot = true
		}
	}
}

func (rs *runState) releaseSlot() {
	rs.slotMu.Lock()
	defer rs.slotMu.Unlock()
	if rs.holdsSlot {
		rs.tree.release()
		rs.holdsSlot = false
	}
}
