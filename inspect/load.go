// Package inspect reads recorded traces for people: a trace as a tree of
// runs and effects with its anomalies (Load), the root causes of a failure
// (Blame), a tree-aware comparison of two traces (Diff), a terminal tree
// view (RenderTree), and a self-contained HTML view (WriteHTML).
package inspect

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// Trace is a recorded trace as a tree of runs.
type Trace struct {
	ID     string        `json:"id"`
	Origin *trace.Origin `json:"origin,omitempty"`
	Root   *Run          `json:"-"`
	// Runs in path order ("/" first).
	Runs []*Run `json:"runs"`
	// Anomalies of every run, in time order.
	Anomalies []*Anomaly `json:"anomalies"`

	byPath map[string]*Run
}

// Run is one run of a trace.
type Run struct {
	Path     string          `json:"path"`
	RunID    string          `json:"run_id"`
	Kind     trace.RunKind   `json:"kind"`
	Agent    string          `json:"agent"`
	Status   trace.RunStatus `json:"status"`
	Error    *trace.Error    `json:"error,omitempty"`
	Task     llm.Blocks      `json:"task"`
	Output   llm.Blocks      `json:"output"`
	Usage    llm.Usage       `json:"usage"`
	Steps    int             `json:"steps"`
	Started  time.Time       `json:"started"`
	Finished time.Time       `json:"finished,omitzero"`
	// Conversation is the run's final conversation.
	Conversation []llm.Message `json:"conversation,omitempty"`
	Effects      []*Effect     `json:"effects"`
	// Modes are the run's mode changes (replay and fork traces).
	Modes        []ModeChange    `json:"modes,omitempty"`
	Expectations []ExpectationAt `json:"expectations,omitempty"`
	Anomalies    []*Anomaly      `json:"-"`
	Parent       *Run            `json:"-"`
	Children     []*Run          `json:"-"`
	ParentPath   string          `json:"parent,omitempty"`
	// Events are the run's recorded events.
	Events []trace.Event `json:"-"`

	spawnSeq, joinSeq uint64 // the run.spawned and run.joined in the parent
	spawned, joined   bool
}

// Effect is one effect of a run.
type Effect struct {
	// Label is the effect's label (its replay key), e.g. "llm#1",
	// "tool:toolu_01", "spawn:/sql#0".
	Label string `json:"label"`
	// Kind is "llm", "tool", "effect" or "spawn".
	Kind string `json:"kind"`
	// Name is the tool, effect or child agent name, or the model.
	Name     string `json:"name"`
	Parent   string `json:"parent,omitempty"`
	Step     int    `json:"step"`
	Position int    `json:"position"`
	Seq      uint64 `json:"seq"`
	// EndSeq is the seq of the answer; Done is false if there is none.
	EndSeq uint64    `json:"end_seq"`
	Done   bool      `json:"done"`
	Time   time.Time `json:"time"`
	// Child is a spawn's child path.
	Child string `json:"child,omitempty"`
	// Input is the tool input, effect input or child task.
	Input json.RawMessage `json:"input,omitempty"`
	// Output is the model's message, tool content, effect output or
	// child output.
	Output      json.RawMessage `json:"output,omitempty"`
	Failure     *trace.Error    `json:"failure,omitempty"`
	IsError     bool            `json:"is_error,omitempty"`
	Intercepted bool            `json:"intercepted,omitempty"`
	Override    bool            `json:"override,omitempty"`
	// Replayed is set in replay and fork traces for an effect answered
	// from the recording.
	Replayed bool            `json:"replayed,omitempty"`
	Accesses *[]trace.Access `json:"accesses,omitempty"`
	Usage    llm.Usage       `json:"usage,omitzero"`
}

// Short is the effect's label with the tool name, as in reports:
// "tool:search(toolu_01)".
func (e *Effect) Short() string {
	if e.Kind == "tool" && e.Parent == "" {
		return fmt.Sprintf("tool:%s(%s)", e.Name, strings.TrimPrefix(e.Label, "tool:"))
	}
	return e.Label
}

// ModeChange is a run switching to live.
type ModeChange struct {
	Seq uint64 `json:"seq"`
	// Cause is the start of the effect the run went live at, if any.
	Cause  *uint64         `json:"cause,omitempty"`
	Reason string          `json:"reason"`
	Detail json.RawMessage `json:"detail,omitempty"`
}

// ExpectationAt is an expectation recorded in a run.
type ExpectationAt struct {
	Seq uint64 `json:"seq"`
	trace.Expectation
}

// AnomalyKind classifies an anomaly.
type AnomalyKind string

// Anomaly kinds.
const (
	Failure     AnomalyKind = "failure"      // an effect failed
	Denial      AnomalyKind = "denial"       // an effect was denied or over a limit
	ErrorResult AnomalyKind = "error_result" // a tool returned an error result
	Divergence  AnomalyKind = "divergence"   // a run diverged from its recording
	Expectation AnomalyKind = "expectation"  // an expectation failed
	RunFailed   AnomalyKind = "run_failed"   // a run ended failed, max_steps or diverged
)

// Anomaly is something that went wrong, where it was recorded.
type Anomaly struct {
	Kind    AnomalyKind `json:"kind"`
	Path    string      `json:"path"`
	Seq     uint64      `json:"seq"`
	Time    time.Time   `json:"time"`
	Effect  string      `json:"effect,omitempty"`
	Summary string      `json:"summary"`
}

func (a *Anomaly) String() string {
	where := a.Path
	if a.Effect != "" {
		where += " " + a.Effect
	}
	return fmt.Sprintf("%s — %s: %s", where, a.Kind, a.Summary)
}

// Find returns the run at path, or nil.
func (t *Trace) Find(path string) *Run { return t.byPath[path] }

// Load reads a trace and every run in it.
func Load(ctx context.Context, s trace.Store, traceID string) (*Trace, error) {
	t := &Trace{ID: traceID, byPath: map[string]*Run{}}
	byID := map[string]*Run{}
	for info, err := range s.Runs(ctx, trace.RunFilter{TraceID: traceID}) {
		if err != nil {
			return nil, err
		}
		events, err := trace.Collect(s.Events(ctx, info.RunID, 0))
		if err != nil {
			return nil, err
		}
		r, err := loadRun(info, events)
		if err != nil {
			return nil, err
		}
		if _, dup := t.byPath[r.Path]; dup {
			return nil, fmt.Errorf("inspect: trace %s has two runs at %s", traceID, r.Path)
		}
		t.byPath[r.Path], byID[r.RunID] = r, r
		t.Runs = append(t.Runs, r)
		if info.ParentRunID == "" {
			t.Root, t.Origin = r, info.Origin
		}
	}
	if t.Root == nil {
		return nil, fmt.Errorf("inspect: trace %s: %w", traceID, trace.ErrRunNotFound)
	}
	sort.Slice(t.Runs, func(i, j int) bool { return t.Runs[i].Path < t.Runs[j].Path })
	// Link children to parents through their spawn events.
	for _, p := range t.Runs {
		for _, e := range p.Events {
			switch b := e.Body.(type) {
			case *trace.RunSpawned:
				if c := byID[b.ChildRunID]; c != nil {
					c.spawnSeq, c.spawned = e.Seq, true
					p.Children = append(p.Children, c)
				}
			case *trace.RunJoined:
				if c := byID[b.ChildRunID]; c != nil {
					c.joinSeq, c.joined = e.Seq, true
				}
			}
		}
	}
	for _, r := range t.Runs {
		if r.Parent == nil && r != t.Root {
			// A child the parent never linked (e.g. the parent crashed
			// before writing its batch): link it by path.
			if pp := parentPath(r.Path); t.byPath[pp] != nil {
				r.Parent = t.byPath[pp]
			}
		}
	}
	for _, p := range t.Runs {
		for _, c := range p.Children {
			c.Parent = p
		}
	}
	for _, r := range t.Runs {
		if r.Parent != nil {
			r.ParentPath = r.Parent.Path
			if !contains(r.Parent.Children, r) {
				r.Parent.Children = append(r.Parent.Children, r)
			}
		}
		t.Anomalies = append(t.Anomalies, r.Anomalies...)
	}
	sort.SliceStable(t.Anomalies, func(i, j int) bool { return t.Anomalies[i].Time.Before(t.Anomalies[j].Time) })
	return t, nil
}

func contains(rs []*Run, r *Run) bool {
	for _, x := range rs {
		if x == r {
			return true
		}
	}
	return false
}

func parentPath(p string) string {
	i := strings.LastIndex(p, "/")
	if i <= 0 {
		return "/"
	}
	return p[:i]
}

func loadRun(info trace.RunInfo, events []trace.Event) (*Run, error) {
	r := &Run{Path: info.Path, RunID: info.RunID, Kind: info.Kind, Agent: info.Agent, Status: info.Status,
		Started: info.Started, Finished: info.Finished, Events: events, Effects: []*Effect{}}
	f := trace.NewFolder()
	var firstSwitch *uint64
	switchCause := map[uint64]bool{}
	newRun := false
	for _, e := range events {
		switch b := e.Body.(type) {
		case *trace.RunStarted:
			r.Task = b.Task
		case *trace.RunFinished:
			r.Output, r.Usage, r.Steps, r.Error = b.Output, b.Usage, b.Steps, b.Error
			switch b.Status {
			case trace.StatusFailed, trace.StatusMaxSteps, trace.StatusDiverged:
				msg := string(b.Status)
				if b.Error != nil {
					msg = firstLine(b.Error.Message)
				}
				r.Anomalies = append(r.Anomalies, &Anomaly{Kind: RunFailed, Path: r.Path, Seq: e.Seq, Time: e.Time, Summary: "the run ended " + string(b.Status) + ": " + msg})
			}
		case *trace.RunModeChanged:
			r.Modes = append(r.Modes, ModeChange{Seq: e.Seq, Cause: e.Cause, Reason: b.Reason, Detail: b.Detail})
			if firstSwitch == nil {
				s := e.Seq
				firstSwitch = &s
			}
			if e.Cause != nil {
				switchCause[*e.Cause] = true
			}
			if b.Reason == trace.ReasonNewRun {
				newRun = true
			}
			if b.Reason == trace.ReasonDivergence {
				var d struct {
					Kind   string   `json:"kind"`
					Effect string   `json:"effect"`
					Diffs  []string `json:"diffs"`
				}
				_ = json.Unmarshal(b.Detail, &d)
				sum := d.Kind
				if len(d.Diffs) > 0 {
					sum += " at " + strings.Join(d.Diffs, ", ")
				}
				r.Anomalies = append(r.Anomalies, &Anomaly{Kind: Divergence, Path: r.Path, Seq: e.Seq, Time: e.Time, Effect: d.Effect, Summary: sum})
			}
		case *trace.Annotation:
			if x, ok := trace.ExpectationOf(b); ok {
				r.Expectations = append(r.Expectations, ExpectationAt{Seq: e.Seq, Expectation: x})
				if !x.Pass {
					sum := fmt.Sprintf("%q failed", x.Name)
					if x.Detail != "" {
						sum += ": " + x.Detail
					}
					r.Anomalies = append(r.Anomalies, &Anomaly{Kind: Expectation, Path: r.Path, Seq: e.Seq, Time: e.Time, Summary: sum})
				}
			}
		}
		if err := f.Apply(e); err != nil {
			return nil, fmt.Errorf("inspect: run %s: %w", r.Path, err)
		}
	}
	r.Conversation = f.Conversation()
	for _, rec := range trace.Effects(events) {
		eff := effectOf(rec)
		// Answered from the recording: in a run that switched to live,
		// what started before the switch, except the call that switched.
		if !newRun && firstSwitch != nil && eff.Seq < *firstSwitch && !switchCause[eff.Seq] {
			eff.Replayed = true
		}
		r.Effects = append(r.Effects, eff)
		if a := anomalyOf(r.Path, eff, rec); a != nil {
			r.Anomalies = append(r.Anomalies, a)
		}
	}
	if r.Status == trace.StatusIncomplete {
		// No run.finished yet: count what has been recorded.
		for _, e := range r.Effects {
			if e.Kind == "llm" && e.Done {
				r.Steps++
				r.Usage = r.Usage.Add(e.Usage)
			}
		}
	}
	sort.SliceStable(r.Anomalies, func(i, j int) bool { return r.Anomalies[i].Seq < r.Anomalies[j].Seq })
	return r, nil
}

func effectOf(rec trace.EffectRecord) *Effect {
	e := &Effect{Label: rec.Label, Kind: rec.Kind, Parent: rec.Parent, Step: rec.Step, Position: rec.Position, Seq: rec.Start.Seq, Time: rec.Start.Time}
	raw := func(v any) json.RawMessage {
		b, _ := json.Marshal(v)
		return b
	}
	switch b := rec.Start.Body.(type) {
	case *trace.ToolStarted:
		e.Name, e.Input, e.Accesses = b.Tool, b.Input, b.Accesses
	case *trace.EffectStarted:
		e.Name, e.Input = b.Name, b.Input
	case *trace.RunSpawned:
		e.Name, e.Child, e.Input = b.Agent, b.Path, raw(b.Task)
	}
	if rec.End == nil {
		return e
	}
	e.Done, e.EndSeq = true, rec.End.Seq
	switch b := rec.End.Body.(type) {
	case *trace.LLMCompleted:
		e.Name, e.Output, e.Usage, e.Override, e.Intercepted = b.Model, raw(b.Message), b.Usage, b.Override, b.Intercepted
	case *trace.LLMFailed:
		e.Failure, e.Intercepted = &b.Error, b.Intercepted
	case *trace.ToolCompleted:
		e.Output, e.IsError, e.Override, e.Intercepted = raw(b.Content), b.IsError, b.Override, b.Intercepted
	case *trace.ToolFailed:
		e.Failure, e.Intercepted = &b.Error, b.Intercepted
	case *trace.EffectCompleted:
		e.Output, e.Override, e.Intercepted = b.Output, b.Override, b.Intercepted
	case *trace.EffectFailed:
		e.Failure, e.Intercepted = &b.Error, b.Intercepted
	case *trace.RunJoined:
		e.Output, e.Usage, e.Failure, e.Override, e.Intercepted = raw(b.Output), b.Usage, b.Error, b.Override, b.Intercepted
	}
	return e
}

func anomalyOf(path string, e *Effect, rec trace.EffectRecord) *Anomaly {
	if rec.End == nil {
		return nil
	}
	a := &Anomaly{Path: path, Seq: rec.End.Seq, Time: rec.End.Time, Effect: e.Short()}
	switch {
	case e.Failure != nil && e.Failure.Code == trace.ErrCodeCancelled:
		return nil // abandoned: a consequence of something else
	case e.Failure != nil && (e.Failure.Code == trace.ErrCodeDenied || e.Failure.Code == trace.ErrCodeLimitExceeded):
		a.Kind, a.Summary = Denial, e.Failure.Code+": "+firstLine(e.Failure.Message)
	case e.Failure != nil:
		a.Kind, a.Summary = Failure, e.Failure.Code+": "+firstLine(e.Failure.Message)
	case e.IsError:
		var content llm.Blocks
		_ = json.Unmarshal(e.Output, &content)
		a.Kind, a.Summary = ErrorResult, fmt.Sprintf("%q", clip(llm.TextOf(content), 120))
	default:
		return nil
	}
	return a
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func clip(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

func utf8RuneStart(b byte) bool { return b&0xC0 != 0x80 }
