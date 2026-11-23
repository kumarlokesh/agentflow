// Command openai-agent runs a small arithmetic agent on the OpenAI API, records the run,
// and can replay it strictly without any API calls.
//
//	export OPENAI_API_KEY=...
//	go run ./examples/openai-agent                     # live run, recorded to .agentflow/
//	go run ./examples/openai-agent -replay <trace-id>  # strict replay: no key, no network
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/providers/openai"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

const modelName = "gpt-6-luna"

func main() {
	replayID := flag.String("replay", "", "trace ID to replay strictly (no API key needed)")
	task := flag.String("task", "What is 47 * 89 + 12? Use the tools.", "task for the agent")
	flag.Parse()
	if err := run(*replayID, *task); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

type pair struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

var pairSchema = json.RawMessage(`{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}`)

func newAgent(model llm.Model) (*agentflow.Agent, error) {
	readOnly := &llm.ToolAnnotations{ReadOnly: true, Idempotent: true}
	add := agentflow.NewTool(llm.ToolSpec{Name: "add", Description: "Adds a and b.", InputSchema: pairSchema, Annotations: readOnly},
		func(_ context.Context, p pair) (agentflow.ToolOutput, error) {
			return agentflow.TextOutput(fmt.Sprint(p.A + p.B)), nil
		})
	mul := agentflow.NewTool(llm.ToolSpec{Name: "multiply", Description: "Multiplies a and b.", InputSchema: pairSchema, Annotations: readOnly},
		func(_ context.Context, p pair) (agentflow.ToolOutput, error) {
			return agentflow.TextOutput(fmt.Sprint(p.A * p.B)), nil
		})
	return agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "openai-agent",
		Model:        model,
		ModelName:    modelName,
		Instructions: "You are a precise assistant. Use the add and multiply tools for all arithmetic.",
		Params:       llm.Params{MaxTokens: 1024},
		Tools:        []agentflow.Tool{add, mul},
	})
}

func run(replayID, task string) error {
	ctx := context.Background()
	store, err := filestore.Open(".agentflow", filestore.Options{Redact: trace.SecretPatterns})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	if replayID != "" {
		agent, err := newAgent(openai.New("")) // never called in a strict replay
		if err != nil {
			return err
		}
		res, err := replay.Run(ctx, agent, store, replayID, replay.Options{})
		var de *replay.DivergenceError
		if errors.As(err, &de) {
			fmt.Println(de.Divergence.Summary)
			return nil
		}
		if err != nil {
			return err
		}
		fmt.Printf("identical replay (%d steps, no API calls): %s\n", res.Run.Steps, res.Run.OutputText())
		return nil
	}

	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return errors.New("OPENAI_API_KEY is not set")
	}
	agent, err := newAgent(retry.Wrap(openai.New(key, openai.WithModel(modelName))))
	if err != nil {
		return err
	}
	res, err := agent.Run(ctx, task, agentflow.WithStore(store))
	if res != nil {
		fmt.Printf("%s\n\ntrace %s: %s, %d steps, %d input / %d output tokens\nreplay:  go run ./examples/openai-agent -replay %s\n",
			res.OutputText(), res.TraceID, res.Status, res.Steps, res.Usage.InputTokens, res.Usage.OutputTokens, res.TraceID)
	}
	return err
}
