package cli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
)

// list is a comma-separated flag value.
type list []string

func (l *list) String() string { return strings.Join(*l, ",") }
func (l *list) Set(v string) error {
	for _, s := range strings.Split(v, ",") {
		if s = strings.TrimSpace(s); s != "" {
			*l = append(*l, s)
		}
	}
	return nil
}

// entrypoint finds the program's entrypoint for a trace, by the name of
// its root run's agent.
func (c *env) entrypoint(ctx context.Context, s trace.Store, traceID, cmd string) (agentflow.Entrypoint, error) {
	if len(c.cfg.Entrypoints) == 0 {
		return nil, usageErr("%s needs your agents' code: call cli.Main with Config.Entrypoints from your own program", cmd)
	}
	var name string
	for info, err := range s.Runs(ctx, trace.RunFilter{TraceID: traceID, RootsOnly: true}) {
		if err != nil {
			return nil, err
		}
		name = info.Agent
	}
	var names []string
	for _, e := range c.cfg.Entrypoints {
		if e.Name() == name {
			return e, nil
		}
		names = append(names, e.Name())
	}
	return nil, fmt.Errorf("trace %s was recorded by %q; this program has %s", traceID, name, strings.Join(names, ", "))
}

func (c *env) replay(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("replay", flag.ContinueOnError)
	on := fs.String("on", "stop", "after a divergence: stop, golive (needs working models and tools), or ignore")
	var verify, opaque, ignore list
	fs.Var(&verify, "verify", "effects to run and compare, e.g. tool:*,effect:context.*")
	fs.Var(&opaque, "opaque", "agents whose runs are answered from the recording, not re-run")
	fs.Var(&ignore, "ignore-paths", "diff paths to disregard, e.g. params.max_tokens")
	task := fs.String("task", "", "replace the recorded task")
	pos, err := parse(fs, args, 1, "replay <id> [-on stop|golive|ignore] [-verify p,…] [-opaque a,…] [-ignore-paths p,…] [-task t]")
	if err != nil {
		return err
	}
	opts := replay.Options{Verify: verify, Opaque: opaque, Ignore: ignore}
	switch *on {
	case "stop":
	case "golive":
		opts.OnDivergence = replay.GoLive
	case "ignore":
		opts.OnDivergence = replay.Ignore
	default:
		return usageErr("-on %q: want stop, golive or ignore", *on)
	}
	if *task != "" {
		opts.Task = task
	}
	s, traceID, err := c.resolve(ctx, pos[0])
	if err != nil {
		return err
	}
	e, err := c.entrypoint(ctx, s, traceID, "replay")
	if err != nil {
		return err
	}
	res, rerr := replay.Run(ctx, e, s, traceID, opts)
	if res == nil {
		return rerr
	}
	p := &printer{w: c.out}
	if res.Status == replay.Identical {
		p.printf("identical: replayed %s against the current code\n", shortID(traceID))
	} else {
		p.printf("diverged: %d divergence(s) replaying %s\n", len(res.Divergences), shortID(traceID))
		for _, d := range res.Divergences {
			p.printf("\n%s\n", d.Summary)
		}
	}
	if len(res.Verified) > 0 {
		p.printf("\nverified: %s\n", strings.Join(res.Verified, ", "))
	}
	if len(res.Unverified) > 0 {
		p.printf("not verified (not annotated ReadOnly; name them exactly to verify): %s\n", strings.Join(res.Unverified, ", "))
	}
	p.printf("\nreplay trace %s\n", res.TraceID)
	if p.err != nil {
		return p.err
	}
	if res.Status == replay.Diverged {
		return errDiverged
	}
	return rerr
}

func (c *env) fork(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("fork", flag.ContinueOnError)
	path := fs.String("path", "", "the fork point's run path (default /)")
	effect := fs.String("effect", "", "the fork point's effect label, e.g. llm#0, tool:toolu_01, spawn:/sql#0 (omit both to fork at the first divergence of the current code)")
	system := fs.String("system", "", "override: the system prompt (at llm#n)")
	systemFile := fs.String("system-file", "", "override: the system prompt, from a file")
	model := fs.String("model", "", "override: the model (at llm#n)")
	var disable list
	fs.Var(&disable, "disable-tools", "override: tools to remove (at llm#n)")
	inject := fs.String("inject", "", "override: a user message added before the model call (at llm#n)")
	task := fs.String("task", "", "override: a child's task (at spawn:<path>)")
	replace := fs.String("replace", "", "override: the effect's outcome, as text (the model's answer, the tool's or child's output)")
	overridesFile := fs.String("overrides", "", "overrides as JSON, in the form recorded in a fork's origin")
	samples := fs.Int("samples", 1, "samples per arm")
	baseline := fs.Bool("baseline", false, "also run the samples without the overrides")
	var allow list
	fs.Var(&allow, "allow-tools", "destructive tools allowed to run live")
	maxTokens := fs.Int("max-tokens", 0, "token budget per sample (0: twice the source's, at least 20000; -1: none)")
	conservative := fs.Bool("conservative", false, "send runs live when a tool with undeclared accesses could read state a live tool changed")
	acceptEarlier := fs.Bool("accept-earlier", false, "fork at an earlier divergence of the current code instead of failing")
	strip := fs.Bool("strip-reasoning", false, "remove reasoning from the conversation at the fork point")
	passIf := fs.String("pass-if", "", "a sample passes if the root's output contains this text (default: it completed and no expectation failed)")
	pos, err := parse(fs, args, 1, "fork <id> [-path p] [-effect e] [overrides] [-samples n] [-baseline] [flags]")
	if err != nil {
		return err
	}
	opts := replay.ForkOptions{Samples: *samples, Baseline: *baseline, AllowTools: allow, MaxTokens: *maxTokens,
		AcceptEarlierDivergence: *acceptEarlier, StripReasoning: *strip}
	if *conservative {
		opts.UnknownAccess = replay.Conservative
	}
	if *path != "" || *effect != "" {
		if *effect == "" {
			return usageErr("-path needs -effect")
		}
		opts.At = &replay.Point{Path: *path, Effect: *effect}
	}
	if *systemFile != "" {
		b, err := os.ReadFile(*systemFile)
		if err != nil {
			return err
		}
		*system = string(b)
	}
	cfg := replay.SetConfig{Model: *model, DisableTools: disable}
	if *system != "" {
		cfg.System = llm.Blocks{llm.Text{Text: *system}}
	}
	if cfg.System != nil || cfg.Model != "" || len(cfg.DisableTools) > 0 {
		opts.Overrides = append(opts.Overrides, cfg)
	}
	if *inject != "" {
		opts.Overrides = append(opts.Overrides, replay.InjectMessage{Message: llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: *inject}}}})
	}
	if *task != "" {
		opts.Overrides = append(opts.Overrides, replay.SetTask{Task: llm.Blocks{llm.Text{Text: *task}}})
	}
	if *replace != "" {
		if opts.At == nil {
			return usageErr("-replace needs a fork point (-effect)")
		}
		opts.Overrides = append(opts.Overrides, textOutcome(opts.At.Effect, *replace))
	}
	if *overridesFile != "" {
		b, err := os.ReadFile(*overridesFile)
		if err != nil {
			return err
		}
		ovs, err := replay.ParseOverrides(b)
		if err != nil {
			return err
		}
		opts.Overrides = append(opts.Overrides, ovs...)
	}
	if *passIf != "" {
		want := *passIf
		opts.Check = func(r *agentflow.RunResult) replay.Verdict {
			return replay.Verdict{Pass: strings.Contains(r.OutputText(), want)}
		}
	}
	s, traceID, err := c.resolve(ctx, pos[0])
	if err != nil {
		return err
	}
	e, err := c.entrypoint(ctx, s, traceID, "fork")
	if err != nil {
		return err
	}
	res, err := replay.Fork(ctx, e, s, traceID, opts)
	if err != nil {
		return err
	}
	p := &printer{w: c.out}
	p.printf("%s\n\n", res)
	arms := []*replay.Arm{&res.Treatment}
	if res.Baseline != nil {
		arms = append(arms, res.Baseline)
	}
	for _, a := range arms {
		for _, smp := range a.Samples {
			verdict := "errored"
			if smp.Verdict != nil {
				verdict = "fail"
				if smp.Verdict.Pass {
					verdict = "pass"
				}
				if smp.Verdict.Note != "" {
					verdict += " (" + smp.Verdict.Note + ")"
				}
			}
			p.printf("%-9s sample %d  %s  %s\n", a.Name, smp.Index, smp.TraceID, verdict)
		}
	}
	if p.err != nil {
		return p.err
	}
	if res.Treatment.Failed > 0 || res.Treatment.Errored > 0 {
		return errDiverged
	}
	return nil
}

// textOutcome is a ReplaceOutcome with text, fitting the effect's kind.
func textOutcome(effect, text string) replay.ReplaceOutcome {
	blocks := llm.Blocks{llm.Text{Text: text}}
	last := effect[strings.LastIndex(effect, "/")+1:]
	switch {
	case strings.Contains(effect, "spawn:"):
		return replay.ReplaceOutcome{ChildOutput: blocks}
	case strings.HasPrefix(last, "llm#"):
		return replay.ReplaceOutcome{Message: &llm.Message{Role: llm.RoleAssistant, Blocks: blocks}}
	case strings.HasPrefix(last, "tool:"):
		return replay.ReplaceOutcome{ToolOutput: &agentflow.ToolOutput{Content: blocks}}
	}
	raw, _ := json.Marshal(text)
	return replay.ReplaceOutcome{Output: raw}
}
