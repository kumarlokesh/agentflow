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
	// OutcomeMismatch: a call's outcome differs from the recording. The
	// call was verified (it ran), or an interceptor answered it.
	OutcomeMismatch DivergenceKind = "outcome_mismatch"
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
	// Verify lists effects to execute during replay and compare with the
	// recording, instead of answering them from it. That is how a replay
	// catches a change inside a tool or a context provider. Patterns:
	//
	//	"tool:<name>"    a tool, by name
	//	"effect:<name>"  a custom effect; context providers are "effect:context.<name>"
	//
	// "*" matches any characters in the name ("tool:*", "effect:context.*").
	// A wildcard selects only tools annotated ReadOnly; the others are
	// listed in Result.Unverified. Naming a tool exactly selects it whatever
	// its annotations. "effect:clock.now" and "effect:id.new" can't be
	// verified. Effects a verified tool starts (Now, NewID, Do) are answered
	// from the recording.
	Verify []string
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
	// Verified lists the effects that were executed and matched the
	// recording, e.g. "tool:list_orders(toolu_01)".
	Verified []string
	// Unverified names tools that a Verify wildcard selected but that were
	// not executed because they aren't annotated ReadOnly.
	Unverified []string
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
	verify, err := compileVerify(opts.Verify)
	if err != nil {
		return nil, err
	}
	rp := newReplayer(opts, tp, ignore, verify)

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
	result.Verified, result.Unverified = rp.verification()
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
	verify verifySet

	mu          sync.Mutex
	mode        agentflow.Mode
	answers     map[string]*entry // matched calls, answered at the replayer slot
	begun       map[string]*entry // matched calls, checked again in Complete
	seqs        map[string]uint64 // key → seq of the call's *.started
	switches    map[string]*trace.RunModeChanged
	divergences []Divergence
	stop        *DivergenceError // set once a Stop divergence ends the run
	verified    []string
	unverified  map[string]bool
}

var _ agentflow.Replayer = (*replayer)(nil)

func newReplayer(opts Options, tp *tape, ignore ignoreMatcher, verify verifySet) *replayer {
	return &replayer{
		opts: opts, tape: tp, ignore: ignore, verify: verify, mode: agentflow.ModeReplay,
		answers: map[string]*entry{}, begun: map[string]*entry{}, seqs: map[string]uint64{},
		switches: map[string]*trace.RunModeChanged{}, unverified: map[string]bool{},
	}
}

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

// verification returns the labels of verified effects and the names of
// tools a wildcard skipped.
func (r *replayer) verification() (verified, unverified []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for name := range r.unverified {
		unverified = append(unverified, name)
	}
	sort.Strings(unverified)
	return append([]string(nil), r.verified...), unverified
}

// Begin implements agentflow.Replayer.
func (r *replayer) Begin(_ context.Context, call *agentflow.Call) (agentflow.Decision, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stop != nil {
		return agentflow.Decision{}, r.stop
	}
	if r.mode == agentflow.ModeLive {
		return agentflow.Decision{Mode: agentflow.ModeLive}, nil
	}
	r.seqs[call.Meta.Key] = call.Meta.Seq
	e := r.tape.byKey[call.Meta.Key]
	switch {
	case e == nil || e.consumed:
		d := r.divergence(UnexpectedEffect, call, nil, nil)
		d.Summary = summarize(d, fmt.Sprintf("the recording has no %s; %s", call.Meta.Label, r.remaining()))
		return r.decide(call, d, nil)
	case e.outcome == nil:
		d := r.divergence(UnexpectedEffect, call, e, nil)
		d.Summary = summarize(d, "the recording ends before this effect completed")
		e.consumed = true
		return r.decide(call, d, nil)
	}
	e.consumed = true
	if e.fingerprint != call.Fingerprint {
		diffs, err := r.diffCall(e, call)
		if err != nil {
			return agentflow.Decision{}, err
		}
		diffs = r.ignore.filter(diffs)
		if len(diffs) > 0 || len(r.ignore) == 0 {
			d := r.divergence(RequestMismatch, call, e, diffs)
			d.Summary = r.summarizeRequest(d)
			return r.decide(call, d, e)
		}
		// every difference is ignored: a match
	}
	r.answers[call.Meta.Key] = e
	r.begun[call.Meta.Key] = e
	verify := !e.intercepted && r.verify.selects(call, r.unverified)
	return agentflow.Decision{Mode: agentflow.ModeReplay, Verify: verify}, nil
}

// decide applies OnDivergence to a divergence found in Begin. A nested
// effect's mode switch is recorded before its tool call's completion (trace
// spec §5), not among the nested events. Called with r.mu held.
func (r *replayer) decide(call *agentflow.Call, d Divergence, e *entry) (agentflow.Decision, error) {
	r.divergences = append(r.divergences, d)
	switch r.opts.OnDivergence {
	case GoLive:
		dec := agentflow.Decision{Mode: agentflow.ModeLive, Switch: r.switchTo(d)}
		if call.Meta.Nested {
			r.recordSwitch(call, dec.Switch)
			dec.Switch = nil
		}
		return dec, nil
	case Ignore:
		if e != nil && e.outcome != nil {
			r.answers[e.key] = e
			return agentflow.Decision{Mode: agentflow.ModeReplay}, nil
		}
	}
	r.stop = &DivergenceError{Divergence: d}
	return agentflow.Decision{}, r.stop
}

// after applies OnDivergence to a divergence found once a call has
// executed, when its outcome can no longer be replaced: GoLive switches the
// run, Ignore carries on, Stop ends the run. Called with r.mu held.
func (r *replayer) after(call *agentflow.Call, d Divergence) error {
	r.divergences = append(r.divergences, d)
	switch r.opts.OnDivergence {
	case GoLive:
		r.recordSwitch(call, r.switchTo(d))
		return nil
	case Ignore:
		return nil
	}
	r.stop = &DivergenceError{Divergence: d}
	return r.stop
}

// switchTo moves the run to live and returns the event recording it, or nil
// if the run is already live. Called with r.mu held.
func (r *replayer) switchTo(d Divergence) *trace.RunModeChanged {
	if r.mode == agentflow.ModeLive {
		return nil
	}
	r.mode = agentflow.ModeLive
	detail, _ := json.Marshal(struct {
		Kind   DivergenceKind `json:"kind"`
		Effect string         `json:"effect"`
		Diffs  []string       `json:"diffs,omitempty"`
	}{d.Kind, d.Effect, diffPaths(d.Diffs)})
	return &trace.RunModeChanged{Mode: string(agentflow.ModeLive), Reason: trace.ReasonDivergence, Detail: detail}
}

// recordSwitch keeps a mode switch for Complete to return: the call's own,
// or its tool call's for a nested effect. Called with r.mu held.
func (r *replayer) recordSwitch(call *agentflow.Call, sw *trace.RunModeChanged) {
	if sw == nil {
		return
	}
	key := call.Meta.Key
	if call.Meta.Nested {
		key = parentKey(key)
	}
	if _, ok := r.switches[key]; !ok {
		r.switches[key] = sw
	}
}

// parentKey returns the key of the tool call enclosing a nested effect.
func parentKey(key string) string {
	if i := strings.Index(key, "/"); i >= 0 {
		return key[:i]
	}
	return key
}

func (r *replayer) divergence(kind DivergenceKind, call *agentflow.Call, e *entry, diffs []FieldDiff) Divergence {
	d := Divergence{Kind: kind, Path: r.tape.path, Effect: call.Meta.Label, Step: call.Meta.Step, Seq: call.Meta.Seq, Diffs: diffs}
	if call.Meta.Nested {
		d.Seq = r.seqs[parentKey(call.Meta.Key)] // nested events have no seq yet
	}
	if e != nil {
		d.SourceSeq = e.startedSeq
		d.Effect = e.label
	}
	return d
}

// summarizeRequest summarizes a request mismatch. One that repeats an
// earlier divergence's differences exactly (typical in Ignore mode, where a
// changed system prompt differs at every call) refers to it instead of
// listing them again. Called with r.mu held.
func (r *replayer) summarizeRequest(d Divergence) string {
	for _, prev := range r.divergences {
		if prev.Kind == RequestMismatch && len(d.Diffs) > 0 && sameDiffs(prev.Diffs, d.Diffs) {
			short := d
			short.Diffs = nil
			return summarize(short, "same differences as "+prev.Effect)
		}
	}
	return summarize(d, "")
}

func sameDiffs(a, b []FieldDiff) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Path != b[i].Path || a[i].Op != b[i].Op || !bytes.Equal(a[i].Recorded, b[i].Recorded) || !bytes.Equal(a[i].Current, b[i].Current) {
			return false
		}
	}
	return true
}

// remaining describes what's left in the recording. Called with r.mu held.
func (r *replayer) remaining() string {
	n := 0
	for _, e := range r.tape.entries {
		if !e.consumed && e.parent == "" {
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
		input := any(req.Input)
		if conv := req.Conversation(); conv != nil {
			input = withConversation(req.Input, conv)
		}
		current = map[string]any{"name": req.Name, "input": input}
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

// Intercept implements agentflow.Replayer: it answers replayed calls, and
// executes and compares verified ones.
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
		dec, err := r.decide(call, d, nil)
		if err == nil {
			r.recordSwitch(call, dec.Switch)
		}
		r.mu.Unlock()
		if err != nil {
			return nil, err
		}
		return next(ctx, call)
	}
	if !call.Meta.Verify {
		r.mu.Unlock()
		return e.outcome, nil
	}
	r.mu.Unlock()

	out, err := next(ctx, call)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stop != nil {
		// Something the call did while running (a nested effect) already
		// diverged; that is the earliest difference.
		return nil, r.stop
	}
	if d := r.missingNested(call); d != nil {
		// Also earlier than the outcome, which it likely explains.
		if err := r.after(call, *d); err != nil {
			return nil, err
		}
	}
	diffs, err := outcomeDiffs(e.outcome, out)
	if err != nil {
		return nil, err
	}
	if len(diffs) == 0 {
		r.verified = append(r.verified, call.Meta.Label)
		return out, nil
	}
	d := r.divergence(OutcomeMismatch, call, e, diffs)
	d.Summary = summarize(d, "verified: the effect ran, and its outcome differs from the recording")
	if err := r.after(call, d); err != nil {
		return nil, err
	}
	if r.opts.OnDivergence == Ignore {
		return e.outcome, nil
	}
	return out, nil
}

// Complete implements agentflow.Replayer. It compares outcomes produced by
// an interceptor with the recording, and returns the mode switch to record.
func (r *replayer) Complete(_ context.Context, call *agentflow.Call, out *agentflow.Outcome, intercepted bool) (*trace.RunModeChanged, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := call.Meta.Key
	e := r.begun[key]
	delete(r.begun, key)
	err := r.complete(call, e, out, intercepted)
	sw := r.switches[key]
	delete(r.switches, key)
	if err == nil && r.stop != nil {
		err = r.stop // e.g. a nested effect diverged, and the tool swallowed the error
	}
	return sw, err
}

// complete does Complete's checks. Called with r.mu held.
func (r *replayer) complete(call *agentflow.Call, e *entry, out *agentflow.Outcome, intercepted bool) error {
	if e == nil || r.stop != nil || call.Meta.Mode != agentflow.ModeReplay {
		return nil
	}
	if intercepted {
		diffs, err := outcomeDiffs(e.outcome, out)
		if err != nil {
			return err
		}
		if e.intercepted && len(diffs) == 0 {
			return nil // the same interceptor decision as in the recording
		}
		was := "in the recording it reached the world"
		if e.intercepted {
			was = fmt.Sprintf("in the recording an interceptor answered it differently (%s)", outcomeText(e.outcome))
		}
		d := r.divergence(OutcomeMismatch, call, e, diffs)
		d.Summary = summarize(d, fmt.Sprintf("an interceptor answered this call (%s); %s", outcomeText(out), was))
		return r.after(call, d)
	}
	return nil
}

// missingNested reports recorded effects nested in a verified tool call
// that the call, executed now, did not make. Called with r.mu held.
func (r *replayer) missingNested(call *agentflow.Call) *Divergence {
	if call.Kind != agentflow.EffectTool || r.mode != agentflow.ModeReplay {
		return nil
	}
	var missing []*entry
	for _, n := range r.tape.entries {
		if n.parent == call.Meta.Key && !n.consumed {
			missing = append(missing, n)
			n.consumed = true
		}
	}
	if len(missing) == 0 {
		return nil
	}
	d := Divergence{Kind: MissingEffect, Path: r.tape.path, Effect: missing[0].label, Step: call.Meta.Step, Seq: call.Meta.Seq, SourceSeq: missing[0].startedSeq}
	d.Summary = summarize(d, fmt.Sprintf("in the recording %s went on to %s; verified, it completed without it (%d nested effect(s) unmatched)", call.Meta.Label, missing[0].label, len(missing)))
	return &d
}

func outcomeText(o *agentflow.Outcome) string {
	if o.Failure != nil {
		return fmt.Sprintf("%s: %s", o.Failure.Code, o.Failure.Message)
	}
	return "a synthesized result"
}

// compareView is the part of an outcome that verification compares (RFC
// §1.4): what is recorded, minus failure details such as stacks.
func compareView(o *agentflow.Outcome) any {
	if o == nil {
		return nil
	}
	if o.Failure != nil {
		return map[string]any{"error": map[string]any{"code": o.Failure.Code, "message": o.Failure.Message}}
	}
	switch v := o.Response.(type) {
	case agentflow.ToolOutput:
		content := v.Content
		if content == nil {
			content = llm.Blocks{}
		}
		return map[string]any{"content": content, "is_error": v.IsError}
	case json.RawMessage:
		return map[string]any{"output": v}
	case *llm.Response:
		return map[string]any{"message": v.Message, "stop_reason": v.StopReason}
	}
	return o.Response
}

// outcomeDiffs diffs two outcomes, with paths rooted at "outcome".
func outcomeDiffs(recorded, current *agentflow.Outcome) ([]FieldDiff, error) {
	a, err := json.Marshal(compareView(recorded))
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(compareView(current))
	if err != nil {
		return nil, err
	}
	diffs, err := diffJSON(a, b)
	if err != nil {
		return nil, err
	}
	return prefix("outcome", diffs), nil
}

// Finish implements agentflow.Replayer.
func (r *replayer) Finish(_ context.Context, status trace.RunStatus, output llm.Blocks) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stop != nil {
		return r.stop
	}
	if r.mode == agentflow.ModeLive {
		return nil
	}
	var missing []*entry
	for _, e := range r.tape.entries {
		if !e.consumed && e.parent == "" {
			missing = append(missing, e)
		}
	}
	var d *Divergence
	if len(missing) > 0 {
		first := missing[0]
		d = &Divergence{Kind: MissingEffect, Path: r.tape.path, Effect: first.label, Step: first.step, SourceSeq: first.startedSeq}
		d.Summary = summarize(*d, fmt.Sprintf("the recorded run went on to %s; this run ended (status %s) without it (%d recorded effect(s) unmatched)", first.label, status, len(missing)))
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
	fmt.Fprintf(&sb, "DIVERGED in %s at %s", d.Path, d.Effect)
	if !strings.HasPrefix(d.Effect, "effect:") && d.Effect != "output" {
		// Custom effects carry no step (trace spec §6), so none is shown.
		fmt.Fprintf(&sb, " (step %d)", d.Step)
	}
	fmt.Fprintf(&sb, " — %s", d.Kind)
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
