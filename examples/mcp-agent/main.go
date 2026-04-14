// Command mcp-agent demonstrates an agentflow agent that consumes tools from
// an MCP server. It spawns the reference MCP filesystem server and runs an
// agent that reads and writes files using MCP-discovered tools.
//
// Prerequisites:
//   - Node.js ≥ 18 (for npx)
//   - An LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)
//
// Usage:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./examples/mcp-agent
//
// The agent will:
//  1. Spawn the MCP filesystem server rooted at a temp directory
//  2. Discover available tools (read_file, write_file, list_directory, …)
//  3. Run a task that writes a file then reads it back
//  4. Print the result and verify replay determinism
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/mcp"
	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/providers/anthropic"
	"github.com/kumarlokesh/agentflow/providers/retry"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/store"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "error: ANTHROPIC_API_KEY is not set")
		os.Exit(1)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	// Create a temporary workspace directory for the MCP filesystem server.
	workDir, err := os.MkdirTemp("", "agentflow-mcp-*")
	if err != nil {
		logger.Error("failed to create temp dir", "error", err)
		os.Exit(1)
	}
	defer os.RemoveAll(workDir)
	logger.Info("MCP workspace", "dir", workDir)

	// Spawn the reference MCP filesystem server.
	// The server is given access only to workDir.
	transport, err := mcp.NewStdioTransport("npx", "-y", "@modelcontextprotocol/server-filesystem", workDir)
	if err != nil {
		logger.Error("failed to start MCP server — is Node.js installed?", "error", err)
		os.Exit(1)
	}

	// Discover tools from the MCP server.
	mcpClient, err := mcp.NewClient(transport)
	if err != nil {
		logger.Error("failed to connect to MCP server", "error", err)
		os.Exit(1)
	}
	defer mcpClient.Close()

	tools := mcpClient.Tools()
	toolNames := make([]string, len(tools))
	for i, t := range tools {
		toolNames[i] = t.Schema().Name
	}
	logger.Info("discovered MCP tools", "count", len(tools), "names", toolNames)

	// Build the agentflow event store.
	fileStore, err := store.NewFile(".agentflow/runs")
	if err != nil {
		logger.Error("failed to create store", "error", err)
		os.Exit(1)
	}

	// Wire observability.
	metrics := observe.NewMetrics()
	hook := observe.NewMultiHook(
		observe.NewMetricsHook(metrics),
		observe.NewTracingHook(),
	)

	// Wire LLM provider.
	llm := retry.Wrap(
		anthropic.New(apiKey, anthropic.WithModel("claude-haiku-4-5-20251001")),
		retry.WithLogger(logger),
	)

	// The task path is inside the workspace the MCP server has access to.
	notePath := filepath.Join(workDir, "note.txt")
	task := fmt.Sprintf(
		`Write the text "agentflow + MCP works!" to the file %s using the write_file tool, `+
			`then read it back with read_file and confirm the content matches.`,
		notePath,
	)

	agent, err := agentflow.NewAgent(agentflow.AgentConfig{
		Name: "mcp-filesystem-agent",
		Instructions: "You are a helpful assistant with access to filesystem tools via MCP. " +
			"Use the available tools to complete the task precisely.",
		LLM:      llm,
		Tools:    tools,
		MaxSteps: 10,
		Store:    fileStore,
		Logger:   logger,
		Hook:     hook,
	})
	if err != nil {
		logger.Error("failed to create agent", "error", err)
		os.Exit(1)
	}

	fmt.Printf("Task: %s\n\n", task)

	ctx := context.Background()
	result, err := agent.Run(ctx, task)
	if err != nil {
		logger.Error("agent run failed", "error", err)
		os.Exit(1)
	}

	snap := metrics.Snapshot()
	fmt.Printf("Answer:  %s\n", result.Output)
	fmt.Printf("RunID:   %s\n", result.RunID)
	fmt.Printf("Steps:   %d\n", result.Steps)
	fmt.Printf("Tokens:  %d (prompt: %d, completion: %d)\n",
		snap.TotalTokens, snap.TotalPromptTokens, snap.TotalCompletionTokens)

	// Verify replay.
	fmt.Println("\nVerifying replay...")
	engine := replay.NewEngine(fileStore, logger)
	replayResult, err := engine.Replay(ctx, result.RunID)
	if err != nil {
		logger.Error("replay failed", "error", err)
		return
	}
	if replayResult.Match {
		fmt.Println("Deterministic replay verified — identical output!")
	} else {
		fmt.Println("WARNING: Replay produced different output!")
	}
}
