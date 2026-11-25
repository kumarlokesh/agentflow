// Command team shows a multi-agent run recorded as one trace tree, and
// replayed as a team, without an API key (it uses small offline models).
//
//	# Record a run (written to .agentflow/)
//	go run ./examples/team
//
//	# Replay the whole team against the same code: identical, no model calls
//	go run ./examples/team -replay <trace-id>
//
//	# Replay it after "changing" one sub-agent: the divergence is reported
//	# in that sub-agent's run, /metrics#0/sql#0
//	go run ./examples/team -replay <trace-id> -sql-instructions "Write efficient SQL."
//
//	# Fork it at the sql agent's first model call with a fixed prompt:
//	# 3 samples with the fix, 3 without, everything else from the recording
//	go run ./examples/team -fork <trace-id>
//
//	# Inspect every run of the tree
//	go run ./cmd/agentflow show <trace-id>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

// DefaultTask is the incident the example triages.
const DefaultTask = "checkout latency spike at 14:05 UTC"

func main() {
	replayID := flag.String("replay", "", "trace ID to replay instead of running live")
	forkID := flag.String("fork", "", "trace ID to fork at the sql agent's first model call, with the fixed prompt")
	dir := flag.String("store", ".agentflow", "trace store directory")
	sqlInstructions := flag.String("sql-instructions", DefaultSQLInstructions, "the sql agent's system prompt (change it to simulate a code change)")
	flag.Parse()
	if *forkID != "" {
		if err := fork(*forkID, *dir); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}
	if err := run(*replayID, *dir, *sqlInstructions); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(replayID, dir, sqlInstructions string) error {
	ctx := context.Background()
	store, err := filestore.Open(dir, filestore.Options{})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	if replayID == "" {
		team, err := NewTriage(sqlInstructions, nil)
		if err != nil {
			return err
		}
		res, err := team.Run(ctx, DefaultTask, agentflow.WithStore(store))
		if err != nil {
			return err
		}
		fmt.Printf("%s\n\nrecorded trace %s:\n", res.OutputText(), res.TraceID)
		printTree(res, "")
		fmt.Printf("\nreplay it:   go run ./examples/team -replay %s\ninspect it:  go run ./cmd/agentflow show %s\n", res.TraceID, res.TraceID)
		return nil
	}

	// During a strict replay no model may be called.
	noModel := func(agent string) llm.Model {
		return agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
			return nil, fmt.Errorf("the %s model was called during a strict replay", agent)
		})
	}
	team, err := NewTriage(sqlInstructions, noModel)
	if err != nil {
		return err
	}
	res, err := replay.Run(ctx, team, store, replayID, replay.Options{})
	var de *replay.DivergenceError
	switch {
	case errors.As(err, &de):
		fmt.Println(de.Divergence.Summary)
		return nil
	case err != nil:
		return err
	}
	fmt.Printf("identical: every run of the tree replayed from the recording, no model calls\nreplay trace %s:\n", res.TraceID)
	printTree(res.Run, "")
	return nil
}

func printTree(r *agentflow.RunResult, indent string) {
	name := r.Path
	if i := strings.LastIndex(name, "/"); i >= 0 && name != "/" {
		name = name[i+1:]
	}
	fmt.Printf("%-18s %-9s %s\n", indent+name, r.Kind, r.Status)
	for _, c := range r.Children {
		printTree(c, indent+"  ")
	}
}

// ForkOptions is the demo's fork: at the sql agent's first model call, with
// the fixed prompt, against a baseline without it. A sample passes if the
// team finds the cause.
func ForkOptions(samples int) replay.ForkOptions {
	return replay.ForkOptions{
		At:        &replay.Point{Path: "/metrics#0/sql#0", Effect: "llm#0"},
		Overrides: []replay.Override{replay.SetConfig{System: llm.Blocks{llm.Text{Text: FixedSQLInstructions}}}},
		Samples:   samples,
		Baseline:  true,
		Check: func(r *agentflow.RunResult) replay.Verdict {
			return replay.Verdict{Pass: strings.Contains(r.OutputText(), "payment-gateway")}
		},
	}
}

func fork(traceID, dir string) error {
	ctx := context.Background()
	store, err := filestore.Open(dir, filestore.Options{})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	team, err := NewTriage(DefaultSQLInstructions, nil)
	if err != nil {
		return err
	}
	res, err := replay.Fork(ctx, team, store, traceID, ForkOptions(3))
	if err != nil {
		return err
	}
	fmt.Println(res)
	s := res.Treatment.Samples[0]
	fmt.Printf("\nfirst treatment sample %s: %s\n", s.TraceID, s.Run.OutputText())
	return nil
}
