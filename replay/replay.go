// Package replay runs current code against a recorded trace.
//
// Every effect the new run makes is matched against the recording: a match
// is answered from the recording without touching the world; a mismatch is
// a Divergence, reported down to the field that changed. After a divergence
// the run can stop (default), go live, or keep serving recorded outcomes.
package replay

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// OnDivergence is what a replay does after the first divergence.
type OnDivergence int

const (
	// Stop ends the run with status "diverged" (the default).
	Stop OnDivergence = iota
	// GoLive sends every later effect of the diverging run to the real
	// world. The agent needs working models and tools.
	GoLive
	// Ignore keeps answering from the recording even when requests differ.
	// It is unsound by design; every divergence is still reported.
	Ignore
)

func (o OnDivergence) String() string {
	switch o {
	case GoLive:
		return "golive"
	case Ignore:
		return "ignore"
	}
	return "strict"
}

// DivergenceKind classifies a divergence.
type DivergenceKind string

const (
	RequestMismatch  DivergenceKind = "request_mismatch"
	UnexpectedEffect DivergenceKind = "unexpected_effect"
	MissingEffect    DivergenceKind = "missing_effect"
	OutputMismatch   DivergenceKind = "output_mismatch"
)

// Divergence is one point where the new run differs from the recording.
type Divergence struct {
	Kind DivergenceKind
	// Path is the run path where it occurred ("/" for a single agent).
	Path string
	// Effect labels the effect: "llm#2", "tool:calculator(toolu_01)",
	// "effect:context.memory#0".
	Effect    string
	Step      int
	Seq       uint64 // in the new trace (0 if not applicable)
	SourceSeq uint64 // in the source trace (0 if not applicable)
	Diffs     []FieldDiff
	Summary   string
}

// DivergenceError ends a strict replay. It wraps agentflow.ErrDiverged.
type DivergenceError struct{ Divergence Divergence }

func (e *DivergenceError) Error() string { return e.Divergence.Summary }
func (e *DivergenceError) Unwrap() error { return agentflow.ErrDiverged }

// Options configure a replay.
type Options struct {
	OnDivergence OnDivergence
	// Ignore lists diff paths to disregard, e.g. "params.max_tokens" or
	// "system[*].text". "*" matches within one path segment; a pattern also
	// covers everything below it.
	Ignore []string
	// Task overrides the recorded task (expect a divergence at llm#0).
	Task *string
	// Dest receives the replay's trace. Defaults to the source store.
	Dest trace.Store
}

// Status summarizes a replay.
type Status string

const (
	Identical Status = "identical"
	Diverged  Status = "diverged"
)

// Result describes a replay.
type Result struct {
	TraceID     string // the new (replay) trace
	Status      Status
	Divergences []Divergence // sorted by (path, seq); the first is the earliest
	Run         *agentflow.RunResult
}

// Run replays the trace traceID from src against e, the current code.
//
// It returns an error wrapping a *DivergenceError when a strict (Stop)
// replay diverges, and other errors when the replay couldn't run. A replay
// that faithfully reproduces a recorded failure is identical and returns no
// error.
func Run(ctx context.Context, e agentflow.Entrypoint, src trace.Store, traceID string, opts Options) (*Result, error) {
	var root *trace.RunInfo
	for info, err := range src.Runs(ctx, trace.RunFilter{TraceID: traceID, RootsOnly: true}) {
		if err != nil {
			return nil, err
		}
		info := info
		root = &info
	}
	if root == nil {
		return nil, fmt.Errorf("replay: trace %s: %w", traceID, trace.ErrRunNotFound)
	}
	events, err := trace.Collect(src.Events(ctx, root.RunID, 0))
	if err != nil {
		return nil, err
	}
	tp, err := buildTape(events)
	if err != nil {
		return nil, err
	}
	ignore, err := compileIgnore(opts.Ignore)
	if err != nil {
		return nil, err
	}
	rp := &replayer{opts: opts, tape: tp, ignore: ignore, mode: agentflow.ModeReplay, answers: map[string]*entry{}, switches: map[string]*trace.RunModeChanged{}}

	dest := opts.Dest
	if dest == nil {
		dest = src
	}
	task := tp.task
	if opts.Task != nil {
		task = llm.Blocks{llm.Text{Text: *opts.Task}}
	}
	runOpts := []agentflow.RunOption{
		agentflow.WithReplayer(rp),
		agentflow.WithOrigin(trace.Origin{Kind: trace.OriginReplay, SourceTraceID: traceID, Mode: opts.OnDivergence.String()}),
		agentflow.WithStore(dest),
		agentflow.WithTaskBlocks(task),
	}
	if opts.OnDivergence != GoLive {
		runOpts = append(runOpts, agentflow.WithLiveCallsDisabled())
	}
	res, runErr := e.Run(ctx, llm.TextOf(task), runOpts...)
	if res == nil {
		return nil, runErr
	}

	result := &Result{TraceID: res.TraceID, Run: res, Divergences: rp.sorted(), Status: Identical}
	if len(result.Divergences) > 0 {
		result.Status = Diverged
	}
	var re *agentflow.RunError
	switch {
	case runErr == nil:
		return result, nil
	case errors.Is(runErr, agentflow.ErrDiverged):
		var de *DivergenceError
		if errors.As(runErr, &de) {
			return result, de
		}
		return result, runErr
	case errors.As(runErr, &re) && result.Status == Identical && tp.hasEnd && re.Status == tp.status:
		return result, nil // the recorded failure was reproduced exactly
	}
	return result, runErr
}

// RunBundle replays the single trace in a bundle (see trace.Export). The
// replay's trace goes to opts.Dest, or to an in-memory store.
func RunBundle(ctx context.Context, e agentflow.Entrypoint, bundle io.Reader, opts Options) (*Result, error) {
	src := memstore.New()
	traceID, err := trace.Import(ctx, bundle, src)
	if err != nil {
		return nil, err
	}
	return Run(ctx, e, src, traceID, opts)
}

// replayer implements agentflow.Replayer for one run.
type replayer struct {
	opts   Options
	tape   *tape
	ignore ignoreMatcher

	mu          sync.Mutex
	mode        agentflow.Mode
	answers     map[string]*entry
	switches    map[string]*trace.RunModeChanged
	divergences []Divergence
}

var _ agentflow.Replayer = (*replayer)(nil)

func (r *replayer) sorted() []Divergence {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := append([]Divergence(nil), r.divergences...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Seq < out[j].Seq
	})
	return out
}

// Begin implements agentflow.Replayer.
func (r *replayer) Begin(_ context.Context, call *agentflow.Call) (agentflow.Decision, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mode == agentflow.ModeLive {
		return agentflow.Decision{Mode: agentflow.ModeLive}, nil
	}
	e := r.tape.byKey[call.Meta.Key]
	switch {
	case e == nil || e.consumed:
		d := r.divergence(UnexpectedEffect, call, nil, nil)
		d.Summary = summarize(d, fmt.Sprintf("the recording has no %s; %s", call.Meta.Label, r.remaining()))
		return r.diverge(d, nil)
	case e.outcome == nil:
		d := r.divergence(UnexpectedEffect, call, e, nil)
		d.Summary = summarize(d, "the recording ends before this effect completed")
		e.consumed = true
		return r.diverge(d, nil)
	}
	e.consumed = true
	if e.fingerprint == call.Fingerprint {
		r.answers[call.Meta.Key] = e
		return agentflow.Decision{Mode: agentflow.ModeReplay}, nil
	}
	diffs, err := r.diffCall(e, call)
	if err != nil {
		return agentflow.Decision{}, err
	}
	diffs = r.ignore.filter(diffs)
	if len(diffs) == 0 && len(r.ignore) > 0 {
		r.answers[call.Meta.Key] = e // every difference is ignored
		return agentflow.Decision{Mode: agentflow.ModeReplay}, nil
	}
	d := r.divergence(RequestMismatch, call, e, diffs)
	d.Summary = summarize(d, "")
	return r.diverge(d, e)
}

// diverge applies OnDivergence. Called with r.mu held.
func (r *replayer) diverge(d Divergence, e *entry) (agentflow.Decision, error) {
	r.divergences = append(r.divergences, d)
	switch r.opts.OnDivergence {
	case GoLive:
		r.mode = agentflow.ModeLive
		detail, _ := json.Marshal(struct {
			Kind   DivergenceKind `json:"kind"`
			Effect string         `json:"effect"`
			Diffs  []string       `json:"diffs,omitempty"`
		}{d.Kind, d.Effect, diffPaths(d.Diffs)})
		return agentflow.Decision{Mode: agentflow.ModeLive, Switch: &trace.RunModeChanged{Mode: string(agentflow.ModeLive), Reason: trace.ReasonDivergence, Detail: detail}}, nil
	case Ignore:
		if e != nil && e.outcome != nil {
			r.answers[e.key] = e
			return agentflow.Decision{Mode: agentflow.ModeReplay}, nil
		}
	}
	return agentflow.Decision{}, &DivergenceError{Divergence: d}
}

func (r *replayer) divergence(kind DivergenceKind, call *agentflow.Call, e *entry, diffs []FieldDiff) Divergence {
	d := Divergence{Kind: kind, Path: r.tape.path, Effect: call.Meta.Label, Step: call.Meta.Step, Seq: call.Meta.Seq, Diffs: diffs}
	if e != nil {
		d.SourceSeq = e.startedSeq
		d.Effect = e.label
	}
	return d
}

// remaining describes what's left in the recording. Called with r.mu held.
func (r *replayer) remaining() string {
	n := 0
	for _, e := range r.tape.entries {
		if !e.consumed {
			n++
		}
	}
	if n == 0 {
		return "the recorded run had no more effects"
	}
	return fmt.Sprintf("%d recorded effect(s) remain unmatched", n)
}

func (r *replayer) diffCall(e *entry, call *agentflow.Call) ([]FieldDiff, error) {
	var current any
	switch req := call.Request.(type) {
	case *llm.Request:
		current = trace.FingerprintView(req)
	case *agentflow.ToolCall:
		current = map[string]any{"tool": req.Name, "input": req.Input}
	case *agentflow.CustomCall:
		current = map[string]any{"name": req.Name, "input": req.Input}
	default:
		return nil, fmt.Errorf("replay: unknown request type %T", call.Request)
	}
	a, err := json.Marshal(e.view)
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(current)
	if err != nil {
		return nil, err
	}
	return diffJSON(a, b)
}

// Intercept implements agentflow.Replayer: it answers replayed calls.
func (r *replayer) Intercept(ctx context.Context, call *agentflow.Call, next agentflow.Invoker) (*agentflow.Outcome, error) {
	if call.Meta.Mode == agentflow.ModeLive {
		return next(ctx, call)
	}
	r.mu.Lock()
	e := r.answers[call.Meta.Key]
	delete(r.answers, call.Meta.Key)
	if e == nil {
		r.mu.Unlock()
		return nil, fmt.Errorf("replay: no recorded outcome for %s", call.Meta.Label)
	}
	if e.intercepted {
		// In the recording an interceptor answered this call; now it
		// reaches the world, and the recording has no world outcome for it.
		d := r.divergence(UnexpectedEffect, call, e, nil)
		d.Summary = summarize(d, fmt.Sprintf("in the recording an interceptor answered this call (%s); now it reaches the world", outcomeText(e.outcome)))
		dec, err := r.diverge(d, nil)
		if err == nil && dec.Switch != nil {
			r.switches[call.Meta.Key] = dec.Switch
		}
		r.mu.Unlock()
		if err != nil {
			return nil, err
		}
		return next(ctx, call)
	}
	r.mu.Unlock()
	return e.outcome, nil
}

// TakeSwitch implements agentflow.Replayer.
func (r *replayer) TakeSwitch(key string) *trace.RunModeChanged {
	r.mu.Lock()
	defer r.mu.Unlock()
	sw := r.switches[key]
	delete(r.switches, key)
	return sw
}

func outcomeText(o *agentflow.Outcome) string {
	if o.Failure != nil {
		return fmt.Sprintf("%s: %s", o.Failure.Code, o.Failure.Message)
	}
	return "a synthesized result"
}

// Finish implements agentflow.Replayer.
func (r *replayer) Finish(_ context.Context, _ trace.RunStatus, output llm.Blocks) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mode == agentflow.ModeLive {
		return nil
	}
	var missing []*entry
	for _, e := range r.tape.entries {
		if !e.consumed {
			missing = append(missing, e)
		}
	}
	var d *Divergence
	if len(missing) > 0 {
		first := missing[0]
		d = &Divergence{Kind: MissingEffect, Path: r.tape.path, Effect: first.label, Step: first.step, SourceSeq: first.startedSeq}
		d.Summary = summarize(*d, fmt.Sprintf("the recorded run went on to %s; this run ended without it (%d recorded effect(s) unmatched)", first.label, len(missing)))
	} else if r.tape.hasEnd && len(r.divergences) == 0 {
		a, _ := json.Marshal(r.tape.output)
		b, _ := json.Marshal(output)
		if !bytes.Equal(a, b) {
			diffs, err := diffJSON(a, b)
			if err != nil {
				return err
			}
			d = &Divergence{Kind: OutputMismatch, Path: r.tape.path, Effect: "output", Diffs: prefix("output", diffs)}
			d.Summary = summarize(*d, "every effect matched, but the final output differs (output post-processing changed)")
		}
	}
	if d == nil {
		return nil
	}
	r.divergences = append(r.divergences, *d)
	if r.opts.OnDivergence == Stop {
		return &DivergenceError{Divergence: *d}
	}
	return nil
}

func prefix(p string, diffs []FieldDiff) []FieldDiff {
	for i := range diffs {
		diffs[i].Path = join(p, diffs[i].Path)
	}
	return diffs
}

func diffPaths(diffs []FieldDiff) []string {
	out := make([]string, len(diffs))
	for i, d := range diffs {
		out[i] = d.Path
	}
	return out
}

// summarize renders a divergence for humans.
func summarize(d Divergence, note string) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "DIVERGED in %s at %s (step %d) — %s", d.Path, d.Effect, d.Step, d.Kind)
	if note != "" {
		fmt.Fprintf(&sb, "\n  %s", note)
	}
	const maxDiffs = 10
	for i, f := range d.Diffs {
		if i == maxDiffs {
			fmt.Fprintf(&sb, "\n  … and %d more difference(s)", len(d.Diffs)-maxDiffs)
			break
		}
		switch {
		case f.TextDiff != "":
			fmt.Fprintf(&sb, "\n  %s %s:\n%s", f.Path, f.Op, indent(f.TextDiff, "    "))
		case f.Op == OpAdded:
			fmt.Fprintf(&sb, "\n  %s added: %s", f.Path, clip(f.Current))
		case f.Op == OpRemoved:
			fmt.Fprintf(&sb, "\n  %s removed: %s", f.Path, clip(f.Recorded))
		default:
			fmt.Fprintf(&sb, "\n  %s changed:\n    - %s\n    + %s", f.Path, clip(f.Recorded), clip(f.Current))
		}
	}
	return sb.String()
}

func indent(s, pad string) string {
	return pad + strings.ReplaceAll(s, "\n", "\n"+pad)
}

func clip(raw json.RawMessage) string {
	s := string(raw)
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}
