// Command mcp-agent runs an agent whose tools come from an MCP server (the
// reference filesystem server), records the run, and can replay it
// strictly: the replay needs neither the API key nor the MCP server.
//
// Prerequisites: Node.js 18+ (for npx) and ANTHROPIC_API_KEY.
//
//	go run ./examples/mcp-agent
//	go run ./examples/mcp-agent -replay <trace-id>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/mcp"
	"github.com/kumarlokesh/agentflow/policy"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
)

const modelName = "claude-haiku-5-5"

func main() {
	replayID := flag.String("replay", "", "trace ID to replay strictly (no API key or MCP server needed)")
	flag.Parse()
	if err := run(*replayID); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func newAgent(model llm.Model, tools []agentflow.Tool) (*agentflow.Agent, error) {
	return agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "mcp-files",
		Model:        model,
		ModelName:    modelName,
		Instructions: "You manage files in the allowed directory. Use the tools; keep answers short.",
		Params:       llm.Params{MaxTokens: 1024},
		Tools:        tools,
		// MCP tools come from another process; never let the agent move or
		// overwrite files in this demo.
		Interceptors: []agentflow.Interceptor{policy.Permissions{Deny: []string{"move_file"}}},
	})
}

func run(replayID string) error {
	ctx := context.Background()
	store, err := filestore.Open(".agentflow", filestore.Options{Redact: trace.SecretPatterns})
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	workDir, err := os.MkdirTemp("", "agentflow-mcp-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(workDir) }()

	if replayID != "" {
		// The recorded tool specs are part of every model request, so the
		// replay needs the same tools; recover them from the recording's
		// MCP server, or start one to list them. Here we start the server
		// only to discover the tool specs; no tool is called during replay.
		client, err := startServer(workDir)
		if err != nil {
			return err
		}
		defer func() { _ = client.Close() }()
		agent, err := newAgent(anthropic.New(""), client.Tools())
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
		fmt.Printf("identical replay (%d steps; no model or MCP tool calls): %s\n", res.Run.Steps, res.Run.OutputText())
		return nil
	}

	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return errors.New("ANTHROPIC_API_KEY is not set")
	}
	client, err := startServer(workDir)
	if err != nil {
		return err
	}
	defer func() { _ = client.Close() }()
	agent, err := newAgent(retry.Wrap(anthropic.New(key, anthropic.WithModel(modelName))), client.Tools())
	if err != nil {
		return err
	}
	task := fmt.Sprintf("In %s, create notes.txt containing three short facts about Go, then list the directory.", workDir)
	res, err := agent.Run(ctx, task, agentflow.WithStore(store))
	if res != nil {
		fmt.Printf("%s\n\ntrace %s: %s, %d steps\nreplay: go run ./examples/mcp-agent -replay %s\n", res.OutputText(), res.TraceID, res.Status, res.Steps, res.TraceID)
	}
	return err
}

func startServer(dir string) (*mcp.Client, error) {
	transport, err := mcp.NewStdioTransport("npx", "-y", "@modelcontextprotocol/server-filesystem", dir)
	if err != nil {
		return nil, fmt.Errorf("start MCP filesystem server (is Node.js installed?): %w", err)
	}
	return mcp.NewClient(transport)
}
