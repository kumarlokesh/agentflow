package inspect

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// BlameResult explains a symptom: the anomalies upstream of it that nothing
// else explains, and how each reached it.
type BlameResult struct {
	// Symptom is what is explained; nil if nothing in the trace failed.
	Symptom *Anomaly
	// RootCauses are anomalies upstream of the symptom with no anomaly
	// upstream of them, in time order. Empty if the symptom has no
	// recorded anomaly upstream (it originates in its own run).
	RootCauses []Cause
	// Consequences counts the other anomalies upstream of the symptom:
	// effects of the root causes.
	Consequences int
	// Anomalies are all of the trace's anomalies (for a trace where
	// nothing failed).
	Anomalies []*Anomaly
}

// Cause is a root cause and the runs that carried it to the symptom.
type Cause struct {
	Anomaly *Anomaly
	Chain   []Hop
	// SharedState names the resource when the link to the symptom goes
	// through declared shared-state accesses, ordered by recorded times (a
	// heuristic: the format defines no order across runs).
	SharedState string
}

// Hop is one step of a cause's way to the symptom.
type Hop struct {
	From, To string
	// Effect carried it: the run.joined of a child (going up), the
	// run.spawned of a child (its task, going down), or the shared-state
	// write and read.
	Effect string
	Kind   HopKind
}

// HopKind says how a hop carries a cause.
type HopKind string

const (
	HopUp     HopKind = "joined"       // a child's output, into its parent
	HopDown   HopKind = "task"         // a parent's task, into a child
	HopShared HopKind = "shared_state" // a declared write, read by another run
)

// point is an event's position in the tree.
type point struct {
	run  *Run
	seq  uint64
	time time.Time
}

func (t *Trace) pointOf(a *Anomaly) point {
	return point{run: t.byPath[a.Path], seq: a.Seq, time: a.Time}
}

// Blame finds the root causes of a symptom. at selects it: "" (the root's
// last failed expectation, else the root's failure, else the latest failed
// expectation or run failure anywhere), a run path (that run's), or
// "<path>@<seq>" (the anomaly recorded at that seq).
func Blame(t *Trace, at string) (*BlameResult, error) {
	res := &BlameResult{Anomalies: t.Anomalies}
	sym, err := t.symptom(at)
	if err != nil || sym == nil {
		return res, err
	}
	res.Symptom = sym
	sp := t.pointOf(sym)
	var cands []*Anomaly
	for _, a := range t.Anomalies {
		if a != sym && t.upstream(t.pointOf(a), sp) {
			cands = append(cands, a)
		}
	}
	for _, c := range cands {
		cp := t.pointOf(c)
		root := true
		for _, d := range cands {
			if d != c && t.upstream(t.pointOf(d), cp) {
				root = false
				break
			}
		}
		if root {
			res.RootCauses = append(res.RootCauses, t.cause(c, sp))
		} else {
			res.Consequences++
		}
	}
	sort.SliceStable(res.RootCauses, func(i, j int) bool { return res.RootCauses[i].Anomaly.Time.Before(res.RootCauses[j].Anomaly.Time) })
	return res, nil
}

func (t *Trace) symptom(at string) (*Anomaly, error) {
	symptomKind := func(a *Anomaly) bool { return a.Kind == Expectation || a.Kind == RunFailed }
	last := func(as []*Anomaly, ok func(*Anomaly) bool) *Anomaly {
		for i := len(as) - 1; i >= 0; i-- {
			if ok(as[i]) {
				return as[i]
			}
		}
		return nil
	}
	if at != "" {
		path, seqStr, hasSeq := strings.Cut(at, "@")
		r := t.byPath[path]
		if r == nil {
			return nil, fmt.Errorf("inspect: no run at %s", path)
		}
		if hasSeq {
			seq, err := strconv.ParseUint(seqStr, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("inspect: %q: want <path>@<seq>", at)
			}
			for _, a := range r.Anomalies {
				if a.Seq == seq {
					return a, nil
				}
			}
			return nil, fmt.Errorf("inspect: no anomaly at %s seq %d", path, seq)
		}
		if a := last(r.Anomalies, symptomKind); a != nil {
			return a, nil
		}
		if a := last(r.Anomalies, func(*Anomaly) bool { return true }); a != nil {
			return a, nil
		}
		return nil, fmt.Errorf("inspect: nothing failed in %s", path)
	}
	root := t.Root.Anomalies
	if a := last(root, func(a *Anomaly) bool { return a.Kind == Expectation }); a != nil {
		return a, nil
	}
	if a := last(root, func(a *Anomaly) bool { return a.Kind == RunFailed }); a != nil {
		return a, nil
	}
	return last(t.Anomalies, symptomKind), nil
}

// upstream reports whether u can have influenced v.
func (t *Trace) upstream(u, v point) bool {
	if structural(u, v) {
		return true
	}
	_, _, ok := t.sharedLink(u, v)
	return ok
}

// structural is the causal order the trace records: within a run, seq
// order; a child flows into its parent at its join; a parent flows into a
// child up to its spawn.
func structural(u, v point) bool {
	switch {
	case u.run == nil || v.run == nil:
		return false
	case u.run == v.run:
		return u.seq < v.seq
	case isAncestor(v.run, u.run):
		c := childToward(v.run, u.run)
		return c.joined && c.joinSeq <= v.seq
	case isAncestor(u.run, v.run):
		c := childToward(u.run, v.run)
		return c.spawned && u.seq < c.spawnSeq
	}
	l := lca(u.run, v.run)
	if l == nil {
		return false
	}
	cu, cv := childToward(l, u.run), childToward(l, v.run)
	return cu.joined && cv.spawned && cu.joinSeq < cv.spawnSeq
}

func upstreamOrSame(u, v point) bool {
	return (u.run == v.run && u.seq == v.seq) || structural(u, v)
}

// access is a declared access by a tool call.
type access struct {
	eff   *Effect
	run   *Run
	write bool
}

// sharedLink finds a declared write w, at or after u, and a later declared
// access r of the same resource in another run, at or before v.
func (t *Trace) sharedLink(u, v point) (w, r *access, ok bool) {
	byRes := map[string][]access{}
	for _, run := range t.Runs {
		for _, e := range run.Effects {
			if e.Accesses == nil || !e.Done {
				continue
			}
			for _, a := range *e.Accesses {
				byRes[a.Resource] = append(byRes[a.Resource], access{eff: e, run: run, write: a.Mode == "write"})
			}
		}
	}
	for _, accs := range byRes {
		for i := range accs {
			wa := &accs[i]
			if !wa.write {
				continue
			}
			wp := point{run: wa.run, seq: wa.eff.EndSeq}
			if !upstreamOrSame(u, wp) {
				continue
			}
			for j := range accs {
				ra := &accs[j]
				if ra.run == wa.run || !wa.eff.Time.Before(ra.eff.Time) {
					continue
				}
				if upstreamOrSame(point{run: ra.run, seq: ra.eff.Seq}, v) {
					return wa, ra, true
				}
			}
		}
	}
	return nil, nil, false
}

func (t *Trace) cause(a *Anomaly, sym point) Cause {
	c := Cause{Anomaly: a}
	ap := t.pointOf(a)
	if structural(ap, sym) {
		c.Chain = chain(ap.run, sym.run)
		return c
	}
	w, r, ok := t.sharedLink(ap, sym)
	if !ok {
		return c
	}
	for _, x := range *w.eff.Accesses {
		if x.Mode == "write" {
			c.SharedState = x.Resource
		}
	}
	c.Chain = append(c.Chain, chain(ap.run, w.run)...)
	c.Chain = append(c.Chain, Hop{From: w.run.Path, To: r.run.Path, Kind: HopShared,
		Effect: fmt.Sprintf("%s writes %s, read by %s", w.eff.Short(), c.SharedState, r.eff.Short())})
	c.Chain = append(c.Chain, chain(r.run, sym.run)...)
	return c
}

// chain is the way from run a to run b through the tree: up to their
// lowest common ancestor, then down.
func chain(a, b *Run) []Hop {
	l := lca(a, b)
	var hops []Hop
	for r := a; r != l && r.Parent != nil; r = r.Parent {
		hops = append(hops, Hop{From: r.Path, To: r.Parent.Path, Effect: spawnLabel(r.Parent, r), Kind: HopUp})
	}
	var down []Hop
	for r := b; r != l && r.Parent != nil; r = r.Parent {
		down = append(down, Hop{From: r.Parent.Path, To: r.Path, Effect: spawnLabel(r.Parent, r), Kind: HopDown})
	}
	for i := len(down) - 1; i >= 0; i-- {
		hops = append(hops, down[i])
	}
	return hops
}

func spawnLabel(parent, child *Run) string {
	for _, e := range parent.Effects {
		if e.Kind == "spawn" && e.Child == child.Path {
			return e.Label
		}
	}
	return "spawn:" + child.Path
}

func isAncestor(a, b *Run) bool {
	for r := b.Parent; r != nil; r = r.Parent {
		if r == a {
			return true
		}
	}
	return false
}

func childToward(a, b *Run) *Run {
	for r := b; r.Parent != nil; r = r.Parent {
		if r.Parent == a {
			return r
		}
	}
	return nil
}

func lca(a, b *Run) *Run {
	seen := map[*Run]bool{}
	for r := a; r != nil; r = r.Parent {
		seen[r] = true
	}
	for r := b; r != nil; r = r.Parent {
		if seen[r] {
			return r
		}
	}
	return nil
}

// String renders the result for terminals.
func (b *BlameResult) String() string {
	var sb strings.Builder
	if b.Symptom == nil {
		sb.WriteString("BLAME  nothing failed: no failed expectation or run")
		if len(b.Anomalies) > 0 {
			fmt.Fprintf(&sb, "\nanomalies in the trace:")
			for _, a := range b.Anomalies {
				fmt.Fprintf(&sb, "\n  %s", a)
			}
		}
		return sb.String()
	}
	fmt.Fprintf(&sb, "BLAME  symptom: %s", b.Symptom)
	if len(b.RootCauses) == 0 {
		sb.WriteString("\nno recorded anomaly upstream: the cause is in that run's own effects (see show -events -run " + b.Symptom.Path + ")")
	}
	for i, c := range b.RootCauses {
		fmt.Fprintf(&sb, "\nroot cause %d: %s", i+1, c.Anomaly)
		for _, h := range c.Chain {
			switch h.Kind {
			case HopUp:
				fmt.Fprintf(&sb, "\n  %-20s →  %s at %s", h.From, h.To, h.Effect)
			case HopDown:
				fmt.Fprintf(&sb, "\n  %-20s →  %s via %s (task)", h.From, h.To, h.Effect)
			case HopShared:
				fmt.Fprintf(&sb, "\n  %-20s →  %s: %s (shared state; ordered by recorded time)", h.From, h.To, h.Effect)
			}
		}
		if len(c.Chain) > 0 {
			fmt.Fprintf(&sb, "\n  %s is where the symptom was recorded", b.Symptom.Path)
		}
	}
	fmt.Fprintf(&sb, "\nother anomalies upstream (consequences): %d", b.Consequences)
	return sb.String()
}
