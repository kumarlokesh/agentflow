# agentflow

Deterministic, observable runtime for building production-grade AI agents.

[![CI](https://github.com/kumarlokesh/agentflow/actions/workflows/ci.yaml/badge.svg)](https://github.com/kumarlokesh/agentflow/actions/workflows/ci.yaml)
[![Go Reference](https://pkg.go.dev/badge/github.com/kumarlokesh/agentflow.svg)](https://pkg.go.dev/github.com/kumarlokesh/agentflow)

## Why agentflow?

Most agent frameworks treat execution as a black box. You call `agent.run()`, get a result, and hope for the best. When something goes wrong - a hallucinated tool call, an infinite loop, unexpected cost - you're left guessing.

**agentflow takes a different approach.** Every action the agent takes is recorded as an immutable event. This gives you:

- **Deterministic replay** - re-execute any run without external API calls
- **Full observability** - inspect every LLM call, tool execution, and decision
- **Run diffing** - compare two runs to find exactly where behavior diverged
- **Debuggability** - no more "what did the agent do?" guesswork

## Quick Start

### Installation

```bash
go get github.com/kumarlokesh/agentflow@latest
```

### Basic Agent

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "github.com/kumarlokesh/agentflow"
    "github.com/kumarlokesh/agentflow/store"
)

func main() {
    // Create a persistent event store.
    eventStore, _ := store.NewFile(".agentflow/runs")

    // Create the agent.
    agent, err := agentflow.NewAgent(agentflow.AgentConfig{
        Name:         "my-agent",
        Instructions: "You are a helpful assistant.",
        LLM:          myLLMClient{},      // implement agentflow.LLM
        Tools:        []agentflow.Tool{},  // register tools here
        MaxSteps:     10,
        Store:        eventStore,
    })
    if err != nil {
        log.Fatal(err)
    }

    // Run the agent.
    result, err := agent.Run(context.Background(), "What is 2+2?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Answer: %s\n", result.Output)
    fmt.Printf("Run ID: %s (use this to replay)\n", result.RunID)
}
```

### Implementing a Tool

```go
type calculatorTool struct{}

func (c *calculatorTool) Schema() agentflow.ToolSchema {
    return agentflow.ToolSchema{
        Name:        "calculator",
        Description: "Evaluates a math expression",
        Parameters:  json.RawMessage(`{
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }`),
    }
}

func (c *calculatorTool) Execute(ctx context.Context, params json.RawMessage) (*agentflow.ToolResult, error) {
    var input struct{ Expression string `json:"expression"` }
    json.Unmarshal(params, &input)
    // ... evaluate expression ...
    return &agentflow.ToolResult{Output: "4"}, nil
}
```

### Implementing an LLM Client

```go
type myLLMClient struct{}

func (m myLLMClient) ChatCompletion(ctx context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
    // Call OpenAI, Anthropic, or any other LLM provider.
    // Convert req.Messages and req.Tools to the provider's format.
    // Return the response with optional tool calls.
    return &agentflow.LLMResponse{
        Content: "The answer is 4.",
    }, nil
}
```

### Replaying a Run

```go
import "github.com/kumarlokesh/agentflow/replay"

engine := replay.NewEngine(eventStore, logger)
result, err := engine.Replay(ctx, "run-id-from-previous-execution")
if result.Match {
    fmt.Println("Deterministic replay verified!")
}
```

### Comparing Two Runs

```go
diff, err := replay.Diff(ctx, eventStore, "run-id-a", "run-id-b")
fmt.Print(diff.Summary)
// Output:
//   Diff: run-id-a vs run-id-b
//   Result: 2 difference(s) found
//   [1] step=0 type=llm_response field=content
//     A: "Let me calculate..."
//     B: "I'll use the calculator..."
```

## CLI

```bash
# Build the CLI
make build

# List recorded runs
./bin/agentflow runs

# Replay a recorded run
./bin/agentflow replay -run <run-id>

# Diff two runs
./bin/agentflow diff -a <run-id-1> -b <run-id-2>

# Version info
./bin/agentflow version
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions.

### Package Structure

| Package | Purpose |
| --- | --- |
| `agentflow` (root) | Core interfaces and agent runtime |
| `store/` | EventStore implementations (memory, file) |
| `replay/` | Deterministic replay engine and run diffing |
| `schema/` | JSON Schema validation for tool parameters |
| `policy/` | Guardrails: cost tracking, rate limiting, permissions, timeouts |
| `observe/` | Observability: tracing spans, metrics, hooks |
| `memory/` | Agent memory: vector store, search, budget, eviction |
| `multi/` | Multi-agent: registry, mailbox, coordinator, fan-out |
| `cmd/agentflow/` | CLI binary |
| `examples/` | Runnable demo agents (calculator, multiagent) |

### Key Interfaces

```go
// LLM — language model abstraction
type LLM interface {
    ChatCompletion(ctx context.Context, req *LLMRequest) (*LLMResponse, error)
}

// Tool — executable capability
type Tool interface {
    Schema() ToolSchema
    Execute(ctx context.Context, params json.RawMessage) (*ToolResult, error)
}

// EventStore — append-only event persistence
type EventStore interface {
    Append(ctx context.Context, event Event) error
    LoadEvents(ctx context.Context, runID string) ([]Event, error)
    LoadEventsByType(ctx context.Context, runID string, eventType EventType) ([]Event, error)
    ListRuns(ctx context.Context) ([]string, error)
}
```

## Development

```bash
# Run tests with race detector
make test

# Run tests with coverage report
make cover

# Format code
make fmt

# Run linter
make lint

# Full check: fmt -> vet -> lint -> test
make check

# Build binary
make build

# Docker
make docker-build
make docker-run
```

## Event Types

| Event | When | Payload |
|-------|------|---------|
| `run_start` | Agent run begins | Task, instructions, tools, max steps |
| `step_start` | Loop iteration begins | Step index |
| `llm_request` | Before LLM call | Messages, tool schemas |
| `llm_response` | After LLM call | Content, tool calls, token usage |
| `tool_call` | Before tool execution | Tool name, call ID, input |
| `tool_result` | After tool execution | Output, error, duration |
| `step_end` | Loop iteration ends | Step index, duration |
| `run_end` | Agent run finishes | Status, output, error, total duration |
| `error` | Non-fatal error | Message, error code |

## Examples

### Calculator Agent

```bash
go run ./examples/calculator
# Output:
#   Task: What is (15 * 7) + (23 - 8)?
#   Answer: Let me break this down:
#   - 15 x 7 = 105
#   - 23 - 8 = 15
#   - 105 + 15 = 120
#   The answer is 120.
#   RunID: abc123-...

# Replay the run
go run ./examples/calculator -replay abc123-...
```

### Multi-Agent Orchestration

```bash
go run ./examples/multiagent
# Output:
#   === Multi-Agent Orchestration Demo ===
#   Step 1: Fan-out tasks to researcher and calculator...
#     [completed] researcher -> Tokyo population is ~14 million...
#     [completed] calculator -> 15% of 37 million is 5,550,000.
#   Step 2: Delegate summary to summarizer...
#   Step 3: Send message between agents...
#   === Metrics ===
#     Runs: 3, Steps: 4, LLM Calls: 4, Tool Calls: 1
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AGENTFLOW_STORE_DIR` | `.agentflow/runs` | Event store directory |
| `AGENTFLOW_LOG_LEVEL` | `info` | Log level: debug, info, warn, error |

## License

[Apache 2.0](LICENSE)
