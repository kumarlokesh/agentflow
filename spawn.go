package agentflow

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// SpawnSpec is one child run of SpawnAll.
type SpawnSpec struct {
	Agent Entrypoint
	Task  string
	// Key, if set, names the child's path segment "<agent>@<key>" instead
	// of "<agent>#<n>" (see SpawnKey).
	Key string
}

// SpawnOption configures Spawn.
type SpawnOption func(*spawnReq)

// SpawnKey gives the child an explicit path segment, "<agent>@<key>",
// instead of "<agent>#<n>". Keys must be unique per agent and parent, and
// can't contain '/' or look like a generated key ("2.1.0").
func SpawnKey(key string) SpawnOption { return func(r *spawnReq) { r.key, r.hasKey = key, true } }

// SpawnCall is the request of a spawn effect: a child run.
type SpawnCall struct {
	// Agent is the child's agent (or workflow) name.
	Agent string
	Task  llm.Blocks
	// Key is the key of the child's path segment ("x" in "/a@x"), or "".
	Key string
	// Path is the child's run path.
	Path string

	e      Entrypoint
	runID  string
	child  Replayer // set by the replayer in Begin
	newRun bool     // set by the replayer in Begin
}

// spawnReq is a child run to start.
type spawnReq struct {
	e      Entrypoint
	task   llm.Blocks
	key    string
	hasKey bool
}

// Spawn starts a child run of e on task in the current run's tree and waits
// for it. If the child doesn't complete, Spawn returns its result with a
// *RunError; a child refused by an interceptor or a tree limit has no run,
// and its error wraps an *EffectError with the reason.
//
// The child is an effect of the current run: it passes through the
// interceptors, is recorded as run.spawned → run.joined, and is replayed
// with the run. Outside a run, Spawn runs e as the root of a new trace.
// Calling e.Run inside a run is the same as Spawn with no options.
func Spawn(ctx context.Context, e Entrypoint, task string, opts ...SpawnOption) (*RunResult, error) {
	r := spawnReq{e: e, task: llm.Blocks{llm.Text{Text: task}}}
	for _, o := range opts {
		o(&r)
	}
	sc := scopeFrom(ctx)
	if sc == nil {
		return e.Run(ctx, task)
	}
	return sc.rs.spawnOne(ctx, sc, r)
}

// SpawnAll starts child runs concurrently, up to the trace's
// MaxConcurrentRuns, and waits for all of them. Results are in spec order
// (a result is nil only if the batch couldn't run); the error joins the
// children's errors. Children get their paths in spec order, and are
// recorded in spec order whatever order they finish in.
func SpawnAll(ctx context.Context, specs []SpawnSpec) ([]*RunResult, error) {
	sc := scopeFrom(ctx)
	if sc == nil {
		out := make([]*RunResult, len(specs))
		var errs []error
		for i, s := range specs {
			res, err := s.Agent.Run(ctx, s.Task)
			out[i] = res
			errs = append(errs, err)
		}
		return out, errors.Join(errs...)
	}
	reqs := make([]spawnReq, len(specs))
	for i, s := range specs {
		reqs[i] = spawnReq{e: s.Agent, task: llm.Blocks{llm.Text{Text: s.Task}}, key: s.Key, hasKey: s.Key != ""}
	}
	return sc.rs.spawnBatch(ctx, sc, reqs)
}

func (rs *runState) spawnOne(ctx context.Context, sc *scope, r spawnReq) (*RunResult, error) {
	res, err := rs.spawnBatch(ctx, sc, []spawnReq{r})
	if res == nil {
		return nil, err
	}
	return res[0], err
}

// generatedKey is the shape of the keys of children started inside tools.
var generatedKey = regexp.MustCompile(`^\d+\.\d+\.\d+$`)

// childPath assigns the path of a child started in scope sc:
//
//   - "<agent>@<key>" for an explicit key;
//   - the segment a delegate tool call reserved ("<agent>#<n>", in response
//     order);
//   - "<agent>@<step>.<call>.<n>" inside any other tool call, whose calls
//     may run concurrently;
//   - "<agent>#<n>" at run level, in call order.
func (rs *runState) childPath(sc *scope, r spawnReq) (seg, key string, err error) {
	name := r.e.Name()
	switch {
	case r.hasKey:
		if r.key == "" || strings.ContainsAny(r.key, "/#@") || generatedKey.MatchString(r.key) {
			return "", "", fmt.Errorf("agentflow: SpawnKey %q: keys can't be empty, contain '/', '#' or '@', or have the generated shape <n>.<n>.<n>", r.key)
		}
		seg, key = name+"@"+r.key, r.key
	case sc.nested:
		sc.mu.Lock()
		if d := sc.delegate; d != "" && strings.HasPrefix(d, name+"#") {
			seg, sc.delegate = d, ""
		} else {
			if sc.childN == nil {
				sc.childN = map[string]int{}
			}
			key = fmt.Sprintf("%d.%d.%d", sc.step, sc.pos, sc.childN[name])
			sc.childN[name]++
			seg = name + "@" + key
		}
		sc.mu.Unlock()
	default:
		rs.mu.Lock()
		seg = fmt.Sprintf("%s#%d", name, rs.childN[name])
		rs.childN[name]++
		rs.mu.Unlock()
	}
	return seg, key, nil
}

func joinPath(parent, seg string) string {
	if parent == "/" {
		return "/" + seg
	}
	return parent + "/" + seg
}

// spawnCall prepares the spawn effect for one child: registration, path,
// fingerprint. Nothing is recorded yet.
func (rs *runState) spawnCall(ctx context.Context, sc *scope, r spawnReq) (*Call, error) {
	if err := rs.tree.register(r.e); err != nil {
		return nil, err
	}
	seg, key, err := rs.childPath(sc, r)
	if err != nil {
		return nil, err
	}
	path := joinPath(rs.path, seg)
	if err := rs.tree.reservePath(path); err != nil {
		return nil, err
	}
	name := r.e.Name()
	label := sc.prefix + "spawn:" + path
	p, err := rs.patch(ctx, label)
	if err != nil {
		return nil, err
	}
	task := r.task
	if p != nil && p.Task != nil {
		task = p.Task
	}
	fp, err := trace.SpawnFingerprint(name, task, key)
	if err != nil {
		return nil, err
	}
	req := &SpawnCall{Agent: name, Task: task, Key: key, Path: path, e: r.e, runID: r.e.def().newID()}
	call := &Call{Kind: EffectSpawn, Name: name, Fingerprint: fp, Request: req, Meta: CallMeta{Key: label, Label: label}}
	if p != nil && p.Outcome != nil {
		// The child doesn't run: its result is the substituted one.
		out := *p.Outcome
		if res, ok := out.Response.(*RunResult); ok {
			cp := *res
			cp.TraceID, cp.RunID, cp.Path, cp.Store = rs.traceID, req.runID, path, rs.tree.store
			out.Response = &cp
		}
		call.patched = &out
	}
	return call, nil
}

// spawnBatch starts children as one runtime-managed batch (R7): every
// run.spawned in order, the children concurrently, then every run.joined
// in order.
func (rs *runState) spawnBatch(ctx context.Context, sc *scope, reqs []spawnReq) ([]*RunResult, error) {
	if isInsideEffect(ctx) {
		return nil, ErrSpawnInsideEffect
	}
	if !sc.busy.CompareAndSwap(false, true) {
		return nil, fmt.Errorf("%w: a child run started while another effect is in flight", ErrConcurrentEffect)
	}
	defer sc.busy.Store(false)

	calls := make([]*Call, len(reqs))
	for i, r := range reqs {
		c, err := rs.spawnCall(ctx, sc, r)
		if err != nil {
			return nil, err
		}
		calls[i] = c
	}
	var cause *ref
	if sc.nested {
		cause = &ref{seq: sc.parentStarted}
	}
	type item struct {
		s   *started
		out *Outcome
		err error
	}
	items := make([]item, len(calls))
	abandon := func(from int, err error) error {
		for _, it := range items[from:] {
			if it.s != nil {
				rs.abandonOrRecord(ctx, it.s, it.out, it.err)
			}
		}
		return err
	}
	for i, call := range calls {
		req := call.Request.(*SpawnCall)
		s, err := rs.begin(ctx, sc, call, &trace.RunSpawned{ChildRunID: req.runID, Path: req.Path, Agent: req.Agent, Task: req.Task, Fingerprint: call.Fingerprint}, cause)
		items[i].s = s
		if err != nil {
			return nil, abandon(0, err)
		}
		if call.patched != nil {
			continue
		}
		if f := rs.tree.admit(rs.depth + 1); f != nil {
			items[i].out = &Outcome{Failure: f} // refused before the interceptors: intercepted
		}
	}

	run := func(i int) {
		if items[i].out == nil {
			items[i].out, _, items[i].err = rs.execute(ctx, items[i].s)
		}
	}
	if len(items) == 1 {
		run(0)
	} else {
		var wg sync.WaitGroup
		for i := range items {
			wg.Add(1)
			go func() { defer wg.Done(); run(i) }()
		}
		wg.Wait()
	}

	results := make([]*RunResult, len(items))
	var errs []error
	for i, it := range items {
		if it.err != nil {
			return nil, abandon(i, it.err)
		}
		if _, _, err := rs.finish(ctx, it.s, it.out, 1); err != nil {
			return nil, abandon(i+1, err)
		}
		res, err := rs.joinChild(sc, it.s.call, it.out)
		results[i] = res
		errs = append(errs, err)
	}
	return results, errors.Join(errs...)
}

// joinChild turns a spawn's outcome into the caller's result, and keeps the
// child in the scope's list of children.
func (rs *runState) joinChild(sc *scope, call *Call, out *Outcome) (*RunResult, error) {
	req := call.Request.(*SpawnCall)
	if out.Failure != nil {
		res := &RunResult{TraceID: rs.traceID, Path: req.Path, Status: trace.StatusFailed, Output: llm.Blocks{}, Store: rs.tree.store}
		return res, &RunError{Status: trace.StatusFailed, Err: &EffectError{Code: out.Failure.Code, Message: out.Failure.Message}}
	}
	res := out.Response.(*RunResult)
	sc.mu.Lock()
	sc.children = append(sc.children, res)
	sc.mu.Unlock()
	if res.Status == trace.StatusCompleted {
		return res, nil
	}
	err := res.err
	if err == nil {
		err = &RunError{Status: res.Status}
	}
	if res.Status == trace.StatusDiverged {
		rs.mu.Lock()
		if rs.childErr == nil {
			rs.childErr = err
		}
		rs.mu.Unlock()
	}
	return res, err
}

// joined returns the children started in a scope, moving them out of it.
func (sc *scope) joined() []*RunResult {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	out := sc.children
	sc.children = nil
	return out
}

// startChild runs a spawn's child (the spawn's terminal).
func (rs *runState) startChild(ctx context.Context, req *SpawnCall) *Outcome {
	rs.waitChildren()
	defer rs.childrenJoined(ctx)
	if err := rs.tree.acquire(ctx); err != nil {
		return Fail(trace.ErrCodeCancelled, err.Error())
	}
	rc := &runContext{tree: rs.tree, runID: req.runID, path: req.Path, parent: rs.runID, depth: rs.depth + 1,
		task: req.Task, replayer: req.child, holdsSlot: true, newRun: req.newRun || (rs.replayer != nil && req.child == nil)}
	res, err := execRun(ctx, req.e.def(), rc)
	if res == nil {
		return Fail(trace.ErrCodeInternal, err.Error())
	}
	return &Outcome{Response: res}
}
