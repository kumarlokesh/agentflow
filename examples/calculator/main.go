// Command calculator shows agentflow's record → replay loop without an API
// key (it uses a small offline model).
//
//	# Record a run (written to .agentflow/)
//	go run ./examples/calculator
//
//	# Replay it against the same code: identical, no model calls
//	go run ./examples/calculator -replay <trace-id>
//
//	# Replay it against "changed code": the divergence is reported
//	go run ./examples/calculator -replay <trace-id> -instructions "Answer in Roman numerals."
//
//	# Inspect it
//	go run ./cmd/agentflow show <trace-id>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

func main() {
	replayID := flag.String("replay", "", "trace ID to replay instead of running live")
	dir := flag.String("store", ".agentflow", "trace store directory")
	task := flag.String("task", "What is (15 * 7) + (23 - 8)?", "task for the agent")
	instructions := flag.String("instructions", DefaultInstructions, "system prompt (change it to simulate a code change)")
	flag.Parse()

	if err := run(*replayID, *dir, *task, *instructions); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(replayID, dir, task, instructions string) error {
	ctx := context.Background()
	store, err := filestore.Open(dir, filestore.Options{})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	if replayID == "" {
		agent, err := NewCalculatorAgent(instructions, nil)
		if err != nil {
			return err
		}
		res, err := agent.Run(ctx, task, agentflow.WithStore(store))
		if err != nil {
			return err
		}
		fmt.Printf("%s\n\nrecorded trace %s (%d steps)\n\nreplay it:   go run ./examples/calculator -replay %s\ninspect it:  go run ./cmd/agentflow show %s\n",
			res.OutputText(), res.TraceID, res.Steps, res.TraceID, res.TraceID)
		return nil
	}

	// During a strict replay the model must never be called.
	noModel := agentflowtest.ModelFunc(func(context.Context, *llm.Request) (*llm.Response, error) {
		return nil, errors.New("the model was called during a strict replay")
	})
	agent, err := NewCalculatorAgent(instructions, noModel)
	if err != nil {
		return err
	}
	res, err := replay.Run(ctx, agent, store, replayID, replay.Options{})
	var de *replay.DivergenceError
	switch {
	case errors.As(err, &de):
		fmt.Println(de.Divergence.Summary)
		return nil
	case err != nil:
		return err
	}
	fmt.Printf("identical: %d steps replayed from the recording, no model or tool calls\noutput: %s\nreplay trace %s\n", res.Run.Steps, res.Run.OutputText(), res.TraceID)
	return nil
}
