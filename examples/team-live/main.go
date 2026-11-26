// Command team-live runs the example incident-triage team (examples/team)
// on Claude. Its recorded run usually fails the team's own check, because
// of a bug in the sql agent's prompt; a fork with the fixed prompt tests
// the fix on fresh samples, with everything the fix doesn't reach answered
// from the recording.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/team-live                       # record a live run
//	go run ./examples/team-live blame <trace-id>      # the root cause
//	go run ./examples/team-live fork <trace-id> -path /metrics#0/sql#0 -effect llm#0 \
//	    -system "$(go run ./examples/team-live -print-fix)" -samples 5 -baseline
//	go run ./examples/team-live diff <baseline-sample> <treatment-sample>
//
// Every command but the first is the agentflow CLI's (package cli), with
// replay and fork for this team. Replay needs no key: it makes no calls.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/cli"
	"github.com/kumarlokesh/agentflow/examples/team/triage"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	fs := flag.NewFlagSet("team-live", flag.ExitOnError)
	dir := fs.String("store", ".agentflow", "trace store directory")
	model := fs.String("model", "claude-haiku-5-5", "the model every agent uses (fork and replay need the recording's)")
	sqlInstructions := fs.String("sql-instructions", triage.DefaultSQLInstructions, "the sql agent's system prompt")
	printFix := fs.Bool("print-fix", false, "print the fixed sql prompt, and exit")
	_ = fs.Parse(os.Args[1:])
	if *printFix {
		fmt.Println(triage.FixedSQLInstructions)
		return nil
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" && fs.NArg() == 0 {
		return errors.New("ANTHROPIC_API_KEY is not set")
	}
	client := retry.Wrap(anthropic.New(key, anthropic.WithModel(*model)))
	team, err := triage.NewTriage(*sqlInstructions, triage.Models{
		Name: *model,
		For:  func(string) llm.Model { return client },
	})
	if err != nil {
		return err
	}
	store, err := filestore.Open(*dir, filestore.Options{Redact: trace.SecretPatterns})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	if fs.NArg() > 0 {
		code := cli.Run(ctx, fs.Args(), os.Stdout, os.Stderr, cli.Config{Name: "team-live", Entrypoints: []agentflow.Entrypoint{team}, Store: store})
		_ = store.Close()
		os.Exit(code)
	}
	res, err := team.Run(ctx, triage.DefaultTask, agentflow.WithStore(store))
	if res == nil {
		return err
	}
	u := res.TreeUsage()
	fmt.Printf("%s\n\nrecorded trace %s (%d input / %d output tokens):\n", res.OutputText(), res.TraceID, u.InputTokens, u.OutputTokens)
	printTree(res, "")
	verdict := "the summary states the measured impact"
	if !triage.MeasuresImpact(res.OutputText()) {
		verdict = "FAILED: the summary doesn't state the measured impact"
	}
	fmt.Printf("\n%s\nnext:  go run ./examples/team-live blame %s\n", verdict, res.TraceID)
	return err
}

func printTree(r *agentflow.RunResult, indent string) {
	name := r.Path
	if i := strings.LastIndex(name, "/"); i >= 0 && name != "/" {
		name = name[i+1:]
	}
	fmt.Printf("%-18s %-9s %-10s %d steps\n", indent+name, r.Kind, r.Status, r.Steps)
	for _, c := range r.Children {
		printTree(c, indent+"  ")
	}
}
