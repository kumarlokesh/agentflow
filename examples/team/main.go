// Command team shows a multi-agent run recorded as one trace tree, inspected,
// replayed and forked, without an API key (it uses small offline models).
//
//	# Record a run (written to .agentflow/). It fails: the summary names no cause.
//	go run ./examples/team
//
//	# Inspect it: the run tree, and the root cause of the failure
//	go run ./examples/team show <trace-id>
//	go run ./examples/team blame <trace-id>
//	go run ./examples/team export <trace-id> -html -o trace.html
//
//	# Replay the whole team against the same code: identical, no model calls
//	go run ./examples/team replay <trace-id>
//
//	# Replay it after "changing" one sub-agent: the divergence is reported
//	# in that sub-agent's run, /metrics#0/sql#0
//	go run ./examples/team -sql-instructions "Write efficient SQL." replay <trace-id>
//
//	# Fork it at the sql agent's first model call with a fixed prompt:
//	# samples with the fix and without, everything else from the recording
//	go run ./examples/team fork <trace-id> -path /metrics#0/sql#0 -effect llm#0 \
//	    -system "$(go run ./examples/team -print-fix)" -samples 3 -baseline
//
// Every command but the first is cli.Main's: this program is the agentflow
// CLI plus replay and fork for its own team.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/cli"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

// DefaultTask is the incident the example triages.
const DefaultTask = "checkout latency spike at 14:05 UTC"

func main() {
	fs := flag.NewFlagSet("team", flag.ExitOnError)
	dir := fs.String("store", ".agentflow", "trace store directory")
	sqlInstructions := fs.String("sql-instructions", DefaultSQLInstructions, "the sql agent's system prompt (change it to simulate a code change)")
	printFix := fs.Bool("print-fix", false, "print the fixed sql prompt the fork demo uses, and exit")
	_ = fs.Parse(os.Args[1:])
	if *printFix {
		fmt.Println(FixedSQLInstructions)
		return
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	team, err := NewTriage(*sqlInstructions, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	store, err := filestore.Open(*dir, filestore.Options{})
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer func() { _ = store.Close() }()

	if fs.NArg() > 0 {
		code := cli.Run(ctx, fs.Args(), os.Stdout, os.Stderr, cli.Config{Name: "team", Entrypoints: []agentflow.Entrypoint{team}, Store: store})
		_ = store.Close()
		os.Exit(code)
	}
	res, err := team.Run(ctx, DefaultTask, agentflow.WithStore(store))
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Printf("%s\n\nrecorded trace %s:\n", res.OutputText(), res.TraceID)
	printTree(res, "")
	fmt.Printf("\nnext:  go run ./examples/team blame %s\n", res.TraceID)
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
