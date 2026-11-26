package replay

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// Errors returned by Fork before any sample runs.
var (
	// ErrForkPointUnknown: the source has no such run, or no such effect in it.
	ErrForkPointUnknown = errors.New("replay: fork point not in the source trace")
	// ErrForkPointNotReached: the current code never reaches the fork point
	// (and nothing diverged on the way: the code path changed).
	ErrForkPointNotReached = errors.New("replay: the current code doesn't reach the fork point")
	// ErrDivergedBeforeForkPoint: the current code diverges from the
	// recording before the fork point. The error is a *EarlyDivergenceError.
	ErrDivergedBeforeForkPoint = errors.New("replay: diverged before the fork point")
	// ErrSampleBudget is a sample's error when it ran out of tokens.
	ErrSampleBudget = errors.New("replay: fork sample token budget exhausted")
)

// EarlyDivergenceError reports where the current code diverged before the
// fork point. It wraps ErrDivergedBeforeForkPoint.
type EarlyDivergenceError struct{ Divergence Divergence }

func (e *EarlyDivergenceError) Error() string {
	return "replay: diverged before the fork point:\n" + e.Divergence.Summary
}
func (e *EarlyDivergenceError) Unwrap() error { return ErrDivergedBeforeForkPoint }

// Point addresses an effect in a source trace: the fork point.
type Point struct {
	// Path is the run's path; "" means the root, "/".
	Path string
	// Effect is the effect's label or replay key: "llm#2", "tool:toolu_01"
	// or "tool:search(toolu_01)", "effect:clock.now#0", "spawn:/sql#0", or
	// a nested one, "tool:toolu_01/effect:x#0".
	Effect string
	// Seq, instead of Effect, is the source seq of the effect's start
	// event.
	Seq *uint64
}

func (p Point) String() string {
	path := p.Path
	if path == "" {
		path = "/"
	}
	if p.Effect == "" && p.Seq != nil {
		return fmt.Sprintf("%s seq %d", path, *p.Seq)
	}
	return path + " " + p.Effect
}

// AccessPolicy decides whether a tool call with undeclared accesses may be
// answered from the recording after a live tool may have written shared
// state.
type AccessPolicy int

const (
	// Optimistic answers it from the recording and lists an Assumption.
	Optimistic AccessPolicy = iota
	// Conservative sends the run live.
	Conservative
)

// Verdict is a fork sample's pass/fail judgment.
type Verdict struct {
	Pass bool
	Note string
}

// ForkOptions configure Fork.
type ForkOptions struct {
	// At is the fork point. Nil forks at the first natural divergence of
	// the current code: GoLive replay with lineage, samples and verdicts.
	At *Point
	// Overrides apply at At (which they require).
	Overrides []Override
	// Samples per arm; default 1.
	Samples int
	// Baseline also runs Samples forks at At with no overrides, to measure
	// how often the outcome changes through model variance alone.
	Baseline bool
	// Check judges a sample. Default: pass if the root run completed and no
	// expectation (agentflow.Expect) failed anywhere in its tree.
	Check func(*agentflow.RunResult) Verdict
	// UnknownAccess governs tools with undeclared accesses (default
	// Optimistic).
	UnknownAccess AccessPolicy
	// AllowTools lists destructive tools allowed to run live.
	AllowTools []string
	// StripReasoning removes reasoning from the target run's conversation
	// at the fork point (an LLM call); recorded as a StripReasoning
	// override.
	StripReasoning bool
	// AcceptEarlierDivergence forks at the first divergence before At,
	// instead of failing with ErrDivergedBeforeForkPoint.
	AcceptEarlierDivergence bool
	// MaxTokens bounds each sample's model tokens, over every run of its
	// tree; answers from the recording count. 0: twice the source's total,
	// at least 20,000. Negative: no limit.
	MaxTokens int
	// MaxConcurrentSamples runs samples concurrently; default 1.
	MaxConcurrentSamples int
	// Interceptors are installed on every run (see Options.Interceptors).
	Interceptors []agentflow.Interceptor
	// Dest receives the samples' traces. Defaults to the source store.
	Dest trace.Store
}

// ForkResult reports a fork.
type ForkResult struct {
	// Group is shared by all samples of the fork (their origin's group).
	Group string
	// SourceTraceID is the trace forked.
	SourceTraceID string
	// At is the requested fork point; ActualAt is set when
	// AcceptEarlierDivergence moved it.
	At, ActualAt *Point
	// Overrides as recorded in the samples' origin.
	Overrides []Override
	Treatment Arm
	Baseline  *Arm
}

// Arm is one arm of a fork: the treatment, or the baseline.
type Arm struct {
	Name                    string // "treatment" or "baseline"
	Samples                 []Sample
	Passed, Failed, Errored int
	// PassRate is Passed/(Passed+Failed), with a Wilson 95% interval.
	PassRate, Low95, High95 float64
}

// Sample is one fork run.
type Sample struct {
	Index   int
	TraceID string
	Run     *agentflow.RunResult
	// Verdict is nil when the sample errored.
	Verdict *Verdict
	// Divergences are where runs went live while the fork spread.
	Divergences []Divergence
	Reuse       ReuseStats
	Assumptions []Assumption
	SideEffects []SideEffect
	// Err is why the sample errored: it couldn't run, ran out of tokens
	// (ErrSampleBudget), or never reached the fork point.
	Err error

	arm string
}

// ReuseStats say how much of a sample came from the recording.
type ReuseStats struct {
	// Runs that replayed throughout, that went live, and that had no
	// recording (new).
	RunsReplayed, RunsLive, RunsNew int
	// Effects answered from the recording, and effects that ran live.
	EffectsReplayed, EffectsLive int
	// TokensSaved is the recorded usage of the model calls answered from
	// the recording.
	TokensSaved int
}

// Assumption is a tool call answered from the recording although its
// accesses are undeclared and a live tool may have changed shared state.
type Assumption struct {
	Path, Effect, Tool string
}

// SideEffect is a tool not annotated ReadOnly that ran live in a sample.
type SideEffect struct {
	Path, Effect, Tool string
}

// resolvedPoint is a fork point found in the source.
type resolvedPoint struct {
	path, key, label string
	seq              uint64
	kind             agentflow.EffectKind
}

// forkState is a fork's per-sample state, shared by the runs of its tree.
// Guarded by the team's mutex.
type forkState struct {
	point    *resolvedPoint
	switchAt bool // go live at the point (false for the probe)
	reached  bool
	before   []Divergence // in the point's run, before it was reached

	unknown   AccessPolicy
	allow     map[string]bool
	maxTokens int
	used      int
	budgetHit bool

	writes       map[string]bool
	unknownWrite bool
	assumptions  []Assumption
	sideEffects  []SideEffect

	served, live, saved int
}

// guard checks a fork's live call: the token budget, destructive tools; it
// notes live writes and side effects. A non-nil outcome answers the call.
func (f *forkState) guard(call *agentflow.Call, mu *sync.Mutex) *agentflow.Outcome {
	mu.Lock()
	defer mu.Unlock()
	switch call.Kind {
	case agentflow.EffectLLM:
		if f.maxTokens > 0 && f.used >= f.maxTokens {
			f.budgetHit = true
			return agentflow.Fail(trace.ErrCodeLimitExceeded, fmt.Sprintf("fork sample token budget exhausted (%d of %d used)", f.used, f.maxTokens))
		}
	case agentflow.EffectTool:
		tc, _ := call.Request.(*agentflow.ToolCall)
		if tc == nil || tc.Delegate != "" || call.Meta.Transparent {
			return nil
		}
		ann := tc.Annotations
		if ann != nil && ann.Destructive && !f.allow[call.Name] {
			return agentflow.Fail(trace.ErrCodeDenied, fmt.Sprintf("destructive tool %q disabled in fork (allow it with ForkOptions.AllowTools)", call.Name))
		}
		readOnly := ann != nil && ann.ReadOnly
		if !readOnly {
			f.sideEffects = append(f.sideEffects, SideEffect{Path: call.Meta.Path, Effect: call.Meta.Label, Tool: call.Name})
		}
		if tc.Accesses != nil {
			for _, a := range *tc.Accesses {
				if a.Mode == trace.AccessWrite {
					f.writes[a.Resource] = true
				}
			}
		} else if !readOnly {
			f.unknownWrite = true
		}
	}
	return nil
}

// counted counts an effect answered from the recording or run live.
func (f *forkState) counted(call *agentflow.Call, out *agentflow.Outcome, live bool, mu *sync.Mutex) {
	if call.Kind == agentflow.EffectSpawn || call.Meta.Transparent || delegateOf(call) != "" || out == nil {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	if live {
		f.live++
	} else {
		f.served++
	}
	if r, ok := out.Response.(*llm.Response); ok && out.Failure == nil {
		f.used += r.Usage.Total()
		if !live {
			f.saved += r.Usage.Total()
		}
	}
}

// taint decides whether a tool call about to be answered from the
// recording must run live instead, because a live tool may have changed
// what it touches. Called with the team's mutex held.
func (f *forkState) taint(call *agentflow.Call, path string) (string, bool) {
	if len(f.writes) == 0 && !f.unknownWrite {
		return "", false
	}
	tc := call.Request.(*agentflow.ToolCall)
	if tc.Accesses != nil {
		for _, a := range *tc.Accesses {
			if f.writes[a.Resource] {
				return a.Resource, true
			}
		}
		return "", false
	}
	if f.unknown == Conservative {
		return "(undeclared accesses)", true
	}
	f.assumptions = append(f.assumptions, Assumption{Path: path, Effect: call.Meta.Label, Tool: call.Name})
	return "", false
}

// Fork re-runs the trace traceID from src against e, the current code,
// changes it at a fork point, and continues live from there. Each run of
// the tree replays from the recording until its own requests change, so
// the change spreads exactly as far as it reaches. With Samples and
// Baseline it runs the fork several times, with and without the change,
// and reports pass rates.
//
// Before any sample, a strict replay of the current code (no external
// calls) checks that the fork point is reached the way it was recorded:
// see ErrForkPointNotReached and ErrDivergedBeforeForkPoint.
//
// Every sample is a trace whose origin records the source, the point, the
// overrides, and its group, sample and arm. Replaying a sample (Run)
// re-applies its overrides, so it replays with no external calls.
func Fork(ctx context.Context, e agentflow.Entrypoint, src trace.Store, traceID string, opts ForkOptions) (*ForkResult, error) {
	ovs := append([]Override(nil), opts.Overrides...)
	if opts.StripReasoning {
		ovs = append([]Override{StripReasoning{}}, ovs...)
	}
	if opts.At == nil && len(ovs) > 0 {
		return nil, fmt.Errorf("%w: overrides need a fork point (At)", ErrInvalidOverride)
	}
	if opts.Baseline && opts.At == nil {
		return nil, errors.New("replay: a baseline arm needs a fork point (At)")
	}
	samples := max(opts.Samples, 1)
	lineage, err := lineagePatches(ctx, src, traceID)
	if err != nil {
		return nil, err
	}
	res := &ForkResult{Group: trace.NewID(), SourceTraceID: traceID, At: opts.At, Overrides: ovs}

	var pt *resolvedPoint
	var patch *agentflow.Patch
	if opts.At != nil {
		if pt, err = resolvePoint(ctx, src, traceID, *opts.At); err != nil {
			return nil, err
		}
		if patch, err = toPatch(ovs, pt.kind); err != nil {
			return nil, err
		}
		actual, err := probe(ctx, e, src, traceID, lineage, pt, opts)
		if err != nil {
			var early *EarlyDivergenceError
			if !opts.AcceptEarlierDivergence || !errors.As(err, &early) {
				return nil, err
			}
			res.ActualAt = actual
		}
	}
	budget := opts.MaxTokens
	if budget == 0 {
		total, err := sourceTokens(ctx, src, traceID)
		if err != nil {
			return nil, err
		}
		budget = max(2*total, 20000)
	}
	overrides, err := MarshalOverrides(ovs)
	if err != nil {
		return nil, err
	}

	runArm := func(name string, p *agentflow.Patch, recorded []byte) Arm {
		arm := Arm{Name: name, Samples: make([]Sample, samples)}
		sem := make(chan struct{}, max(opts.MaxConcurrentSamples, 1))
		var wg sync.WaitGroup
		for i := range samples {
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				origin := trace.Origin{Kind: trace.OriginFork, SourceTraceID: traceID, Group: res.Group, Sample: &i, Arm: name}
				if pt != nil {
					seq := pt.seq
					origin.At = &trace.ForkPoint{Path: pt.path, Effect: pt.key, Seq: &seq}
					if len(recorded) > 2 { // not "[]"
						origin.Overrides = recorded
					}
				}
				if a := res.ActualAt; a != nil {
					origin.ActualAt = &trace.ForkPoint{Path: a.Path, Effect: a.Effect}
				}
				arm.Samples[i] = runSample(ctx, e, src, traceID, lineage, pt, p, origin, budget, opts)
				arm.Samples[i].Index, arm.Samples[i].arm = i, name
			}()
		}
		wg.Wait()
		arm.tally()
		return arm
	}
	res.Treatment = runArm("treatment", patch, overrides)
	if opts.Baseline {
		b := runArm("baseline", nil, nil)
		res.Baseline = &b
	}
	return res, nil
}

// runSample runs one fork sample.
func runSample(ctx context.Context, e agentflow.Entrypoint, src trace.Store, traceID string, lineage map[string]map[string]*agentflow.Patch,
	pt *resolvedPoint, p *agentflow.Patch, origin trace.Origin, budget int, opts ForkOptions) Sample {
	fs := &forkState{point: pt, switchAt: true, unknown: opts.UnknownAccess, allow: map[string]bool{}, writes: map[string]bool{}}
	if budget > 0 {
		fs.maxTokens = budget
	}
	for _, n := range opts.AllowTools {
		fs.allow[n] = true
	}
	patches := clonePatches(lineage)
	if pt != nil && p != nil {
		if patches[pt.path] == nil {
			patches[pt.path] = map[string]*agentflow.Patch{}
		}
		patches[pt.path][pt.key] = mergePatch(patches[pt.path][pt.key], p)
	}
	ropts := Options{OnDivergence: GoLive, Interceptors: opts.Interceptors, Dest: opts.Dest}
	result, tm, err := runTree(ctx, e, src, traceID, ropts, patches, &forkRun{state: fs, origin: origin})
	s := Sample{}
	if result != nil {
		s.TraceID, s.Run, s.Divergences = result.TraceID, result.Run, result.Divergences
	}
	if tm != nil {
		tm.mu.Lock()
		s.Reuse = ReuseStats{EffectsReplayed: fs.served, EffectsLive: fs.live, TokensSaved: fs.saved}
		for _, r := range tm.replayers {
			switch {
			case r.isNew:
				s.Reuse.RunsNew++
			case r.mode == agentflow.ModeLive:
				s.Reuse.RunsLive++
			default:
				s.Reuse.RunsReplayed++
			}
		}
		s.Assumptions = append(s.Assumptions, fs.assumptions...)
		s.SideEffects = append(s.SideEffects, fs.sideEffects...)
		hit, reached := fs.budgetHit, fs.reached
		tm.mu.Unlock()
		switch {
		case hit:
			s.Err = ErrSampleBudget
		case pt != nil && !reached && opts.AcceptEarlierDivergence:
			// The point moved earlier; the target may never be reached.
		case pt != nil && !reached:
			s.Err = ErrForkPointNotReached
		}
	}
	var re *agentflow.RunError
	if s.Err == nil && err != nil && !errors.As(err, &re) {
		s.Err = err
	}
	if s.Err == nil && s.Run != nil {
		var v Verdict
		if opts.Check != nil {
			v = opts.Check(s.Run)
		} else if v, err = defaultVerdict(ctx, s.Run); err != nil {
			s.Err = err
		}
		if s.Err == nil {
			s.Verdict = &v
		}
	}
	if s.Err == nil && s.Run == nil {
		s.Err = errors.New("replay: the sample didn't run")
	}
	return s
}

// defaultVerdict passes a sample whose root run completed and whose tree
// has no failed expectation (agentflow.Expect).
func defaultVerdict(ctx context.Context, res *agentflow.RunResult) (Verdict, error) {
	if res.Status != trace.StatusCompleted {
		return Verdict{Note: "the run ended " + string(res.Status)}, nil
	}
	var failed []string
	for info, err := range res.Store.Runs(ctx, trace.RunFilter{TraceID: res.TraceID}) {
		if err != nil {
			return Verdict{}, err
		}
		for e, err := range res.Store.Events(ctx, info.RunID, 0) {
			if err != nil {
				return Verdict{}, err
			}
			if a, ok := e.Body.(*trace.Annotation); ok {
				if x, ok := trace.ExpectationOf(a); ok && !x.Pass {
					failed = append(failed, fmt.Sprintf("%s %q", info.Path, x.Name))
				}
			}
		}
	}
	if len(failed) > 0 {
		return Verdict{Note: "failed expectations: " + strings.Join(failed, ", ")}, nil
	}
	return Verdict{Pass: true}, nil
}

func short(id string) string {
	if len(id) > 13 {
		return id[:13]
	}
	return id
}

// probe replays the current code strictly up to the fork point, into a
// throwaway store, and reports whether it gets there the recorded way. On
// an early divergence it also returns that divergence's point.
func probe(ctx context.Context, e agentflow.Entrypoint, src trace.Store, traceID string, lineage map[string]map[string]*agentflow.Patch,
	pt *resolvedPoint, opts ForkOptions) (*Point, error) {
	fs := &forkState{point: pt, writes: map[string]bool{}}
	ropts := Options{OnDivergence: Stop, Interceptors: opts.Interceptors, Dest: memstore.New()}
	origin := trace.Origin{Kind: trace.OriginReplay, SourceTraceID: traceID, Mode: "probe"}
	result, tm, err := runTree(ctx, e, src, traceID, ropts, clonePatches(lineage), &forkRun{state: fs, origin: origin})
	if tm == nil {
		return nil, err
	}
	tm.mu.Lock()
	reached, before := fs.reached, append([]Divergence(nil), fs.before...)
	tm.mu.Unlock()
	var first *Divergence
	switch {
	case len(before) > 0:
		first = &before[0]
	case !reached && result != nil && len(result.Divergences) > 0:
		first = &result.Divergences[0]
	case !reached:
		var de *DivergenceError
		if err != nil && !errors.As(err, &de) {
			var re *agentflow.RunError
			if !errors.As(err, &re) {
				return nil, err
			}
		}
		return nil, fmt.Errorf("%w: %s", ErrForkPointNotReached, Point{Path: pt.path, Effect: pt.label})
	default:
		return nil, nil
	}
	return &Point{Path: first.Path, Effect: first.Effect}, &EarlyDivergenceError{Divergence: *first}
}

// resolvePoint finds a fork point in the source.
func resolvePoint(ctx context.Context, src trace.Store, traceID string, p Point) (*resolvedPoint, error) {
	path := p.Path
	if path == "" {
		path = "/"
	}
	runs, err := sourceRuns(ctx, src, traceID)
	if err != nil {
		return nil, err
	}
	info, ok := runs[path]
	if !ok {
		return nil, fmt.Errorf("%w: no run at %s", ErrForkPointUnknown, path)
	}
	events, err := trace.Collect(src.Events(ctx, info.RunID, 0))
	if err != nil {
		return nil, err
	}
	tp, err := buildTape(events, src)
	if err != nil {
		return nil, err
	}
	for _, e := range tp.entries {
		if (p.Seq != nil && e.startedSeq == *p.Seq) || (p.Effect != "" && (e.key == p.Effect || e.label == p.Effect)) {
			return &resolvedPoint{path: path, key: e.key, label: e.label, seq: e.startedSeq, kind: e.kind}, nil
		}
	}
	return nil, fmt.Errorf("%w: %s", ErrForkPointUnknown, p)
}

// lineagePatches returns the overrides of the forks a trace descends from,
// as patches: following origin.source_trace_id while the sources are in
// src. The deepest source's come first; a later fork's override at the
// same point wins.
func lineagePatches(ctx context.Context, src trace.Store, traceID string) (map[string]map[string]*agentflow.Patch, error) {
	type step struct {
		at  *trace.ForkPoint
		ovs []Override
	}
	var chain []step
	seen := map[string]bool{}
	for id := traceID; id != "" && !seen[id]; {
		seen[id] = true
		runs, err := sourceRuns(ctx, src, id)
		if err != nil {
			if errors.Is(err, trace.ErrRunNotFound) && id != traceID {
				break // the source isn't in this store
			}
			return nil, err
		}
		o := runs["/"].Origin
		if o == nil || o.Kind != trace.OriginFork {
			break
		}
		if o.At != nil && len(o.Overrides) > 0 {
			ovs, err := ParseOverrides(o.Overrides)
			if err != nil {
				return nil, err
			}
			chain = append(chain, step{o.At, ovs})
		}
		id = o.SourceTraceID
	}
	out := map[string]map[string]*agentflow.Patch{}
	for i := len(chain) - 1; i >= 0; i-- {
		at := chain[i].at
		p, err := toPatch(chain[i].ovs, kindOfKey(at.Effect))
		if err != nil {
			return nil, err
		}
		path := at.Path
		if path == "" {
			path = "/"
		}
		if out[path] == nil {
			out[path] = map[string]*agentflow.Patch{}
		}
		out[path][at.Effect] = mergePatch(out[path][at.Effect], p)
	}
	return out, nil
}

func clonePatches(in map[string]map[string]*agentflow.Patch) map[string]map[string]*agentflow.Patch {
	out := make(map[string]map[string]*agentflow.Patch, len(in))
	for p, m := range in {
		out[p] = make(map[string]*agentflow.Patch, len(m))
		for k, v := range m {
			out[p][k] = v
		}
	}
	return out
}

// sourceTokens is the model usage of every run of a trace.
func sourceTokens(ctx context.Context, src trace.Store, traceID string) (int, error) {
	runs, err := sourceRuns(ctx, src, traceID)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, info := range runs {
		events, err := trace.Collect(src.Events(ctx, info.RunID, 0))
		if err != nil {
			return 0, err
		}
		for _, ev := range events {
			if f, ok := ev.Body.(*trace.RunFinished); ok {
				total += f.Usage.Total()
			}
		}
	}
	return total, nil
}

// tally counts verdicts and computes the pass rate and its interval.
func (a *Arm) tally() {
	a.Passed, a.Failed, a.Errored = 0, 0, 0
	for _, s := range a.Samples {
		switch {
		case s.Verdict == nil:
			a.Errored++
		case s.Verdict.Pass:
			a.Passed++
		default:
			a.Failed++
		}
	}
	a.PassRate, a.Low95, a.High95 = wilson(a.Passed, a.Passed+a.Failed)
}

// wilson returns the proportion k/n and its Wilson score 95% interval. With
// n = 0 it returns 0 and the uninformative interval [0, 1].
func wilson(k, n int) (p, low, high float64) {
	if n == 0 {
		return 0, 0, 1
	}
	const z = 1.959963984540054
	p = float64(k) / float64(n)
	nf := float64(n)
	den := 1 + z*z/nf
	center := (p + z*z/(2*nf)) / den
	half := z * math.Sqrt(p*(1-p)/nf+z*z/(4*nf*nf)) / den
	return p, math.Max(0, center-half), math.Min(1, center+half)
}

// String renders the fork's report for terminals.
func (r *ForkResult) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "FORK %s", short(r.Group))
	switch {
	case r.At != nil:
		fmt.Fprintf(&sb, "  at %s", r.At)
	default:
		sb.WriteString("  at the first divergence of the current code")
	}
	if len(r.Overrides) > 0 {
		var names []string
		for _, o := range r.Overrides {
			names = append(names, o.overrideType())
		}
		fmt.Fprintf(&sb, "   overrides: %s", strings.Join(names, ", "))
	}
	if r.ActualAt != nil {
		fmt.Fprintf(&sb, "\n  the current code diverges earlier, at %s: forked there", r.ActualAt)
	}
	arms := []*Arm{&r.Treatment}
	if r.Baseline != nil {
		arms = append(arms, r.Baseline)
	}
	small := false
	for _, a := range arms {
		n := a.Passed + a.Failed
		fmt.Fprintf(&sb, "\n  %-9s  %d/%d pass  %3.0f%%  [%2.0f%%, %3.0f%%]", a.Name, a.Passed, n, 100*a.PassRate, 100*a.Low95, 100*a.High95)
		if a.Errored > 0 {
			fmt.Fprintf(&sb, "  %d errored", a.Errored)
		}
		if n < 10 {
			small = true
		}
	}
	if small {
		sb.WriteString("\n  (small samples: the intervals are wide)")
	}
	var all []Sample
	for _, a := range arms {
		all = append(all, a.Samples...)
		if len(a.Samples) == 0 {
			continue
		}
		m := func(f func(ReuseStats) int) int {
			v := make([]int, len(a.Samples))
			for i, s := range a.Samples {
				v[i] = f(s.Reuse)
			}
			sort.Ints(v)
			return v[len(v)/2]
		}
		fmt.Fprintf(&sb, "\n  %-9s  reuse per sample (median): runs %d replayed / %d live / %d new · effects %d from recording / %d live · ~%d tokens saved",
			a.Name, m(func(s ReuseStats) int { return s.RunsReplayed }), m(func(s ReuseStats) int { return s.RunsLive }), m(func(s ReuseStats) int { return s.RunsNew }),
			m(func(s ReuseStats) int { return s.EffectsReplayed }), m(func(s ReuseStats) int { return s.EffectsLive }), m(func(s ReuseStats) int { return s.TokensSaved }))
	}
	seen := map[string]bool{}
	for _, s := range all {
		for _, a := range s.Assumptions {
			k := a.Path + " " + a.Effect
			if !seen[k] {
				seen[k] = true
				fmt.Fprintf(&sb, "\n  assumption: %s %s served from the recording; accesses undeclared", a.Path, a.Effect)
			}
		}
		if s.Err != nil {
			fmt.Fprintf(&sb, "\n  %s sample %d errored: %v", s.arm, s.Index, s.Err)
		}
	}
	return sb.String()
}
