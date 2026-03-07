# agentflow

Deterministic, observable runtime for building AI agents in Go.

## What it is

agentflow implements an explicit **Observe-Think-Act** loop where every action - every LLM call, tool execution, and state transition - is recorded as an immutable event. This append-only event log is the foundation for:

- **Deterministic replay** - re-execute any run with recorded LLM responses and tool outputs, no external calls required
- **Run diffing** - compare two runs event-by-event to identify where behavior diverged
- **Observability** - structured span traces, per-tool latency metrics, and token usage per step
- **Policy enforcement** - token budget limits, rate limiting, tool permissions, and execution timeouts

The single external dependency is `github.com/google/uuid`. Everything else is standard library.

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
    "fmt"
    "log"

    "github.com/kumarlokesh/agentflow"
    "github.com/kumarlokesh/agentflow/store"
)

func main() {
    // Persist events to disk for replay.
    eventStore, err := store.NewFile(".agentflow/runs")
    if err != nil {
        log.Fatal(err)
    }

    agent, err := agentflow.NewAgent(agentflow.AgentConfig{
        Name:         "my-agent",
        Instructions: "You are a helpful assistant.",
        LLM:          myLLMClient{},     // implements agentflow.LLM
        Tools:        []agentflow.Tool{}, // register tools here
        MaxSteps:     10,
        Store:        eventStore,
    })
    if err != nil {
        log.Fatal(err)
    }

    result, err := agent.Run(context.Background(), "What is 2+2?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Output: %s\n", result.Output)
    fmt.Printf("RunID:  %s\n", result.RunID)
    fmt.Printf("Steps:  %d\n", result.Steps)
}
```

### Implementing a Tool

```go
type calculatorTool struct{}

func (c *calculatorTool) Schema() agentflow.ToolSchema {
    return agentflow.ToolSchema{
        Name:        "calculator",
        Description: "Evaluates a math expression. Returns a numeric result.",
        Parameters: json.RawMessage(`{
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2 + 3 * 4'"}
            },
            "required": ["expression"]
        }`),
    }
}

func (c *calculatorTool) Execute(ctx context.Context, params json.RawMessage) (*agentflow.ToolResult, error) {
    var input struct{ Expression string `json:"expression"` }
    if err := json.Unmarshal(params, &input); err != nil {
        return &agentflow.ToolResult{Error: err.Error()}, nil
    }
    // ... evaluate expression ...
    return &agentflow.ToolResult{Output: "120"}, nil
}
```

### Implementing an LLM Client

```go
type myLLMClient struct{}

func (m myLLMClient) ChatCompletion(ctx context.Context, req *agentflow.LLMRequest) (*agentflow.LLMResponse, error) {
    // req.Messages contains the full conversation history.
    // req.Tools contains the JSON Schema for each registered tool.
    // Convert to your provider's format (OpenAI, Anthropic, etc.) and return.
    return &agentflow.LLMResponse{
        Content: "The answer is 4.",
        Usage: &agentflow.TokenUsage{
            PromptTokens: 20, CompletionTokens: 10, TotalTokens: 30,
        },
    }, nil
}
```

### Replay a Run

```go
import "github.com/kumarlokesh/agentflow/replay"

engine := replay.NewEngine(eventStore, logger)
result, err := engine.Replay(ctx, "run-id-from-previous-execution")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Match: %v\n", result.Match)  // true if output is identical
```

The replay engine substitutes the real LLM and tools with implementations that return the recorded responses. The agent code path is identical in live and replay modes — only the dependencies differ.

### Diff Two Runs

```go
diff, err := replay.Diff(ctx, eventStore, "run-id-a", "run-id-b")
fmt.Print(diff.Summary)
// Diff: run-id-a vs run-id-b
// Result: 2 difference(s) found
//
// [1] step=0 type=llm_response field=content
//   A: "Let me calculate..."
//   B: "I'll use the calculator tool..."
```

### Policy: Token Budget + Rate Limiting

```go
import (
    "github.com/kumarlokesh/agentflow"
    "github.com/kumarlokesh/agentflow/observe"
    "github.com/kumarlokesh/agentflow/policy"
)

tracker := policy.NewCostTracker(policy.CostTrackerConfig{MaxTotalTokens: 10_000})
rateLimiter := policy.NewRateLimiter(20, time.Minute)

// CostHook feeds LLM token usage back into the tracker after each call.
// Without it, the tracker always sees zero tokens.
costHook := agentflow.NewCostHook(tracker)

agent, _ := agentflow.NewAgent(agentflow.AgentConfig{
    LLM:    myLLMClient{},
    Policy: policy.NewChain(tracker, rateLimiter),
    Hook:   observe.NewMultiHook(costHook, observe.NewMetricsHook(metrics)),
    // ...
})
```

### Memory Injection

```go
import "github.com/kumarlokesh/agentflow/memory"

memStore := memory.NewInMemory()
memStore.Add(ctx, memory.Entry{
    ID:      "fact-1",
    Content: "The user's name is Alice and they prefer concise answers.",
})

agent, _ := agentflow.NewAgent(agentflow.AgentConfig{
    Memory:     memory.AsProvider(memStore),
    MemoryTopK: 3, // retrieve top 3 relevant entries per step
    // ...
})
```

Before each LLM call, the agent queries the memory store and prepends the top-K results to the system prompt.

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

## Package Structure

| Package | Purpose |
| --- | --- |
| `agentflow` (root) | Core interfaces (`LLM`, `Tool`, `EventStore`, `MemoryProvider`) and agent runtime |
| `store/` | `EventStore` implementations: `Memory` (in-process) and `File` (JSONL) |
| `replay/` | Replay engine and run diff utility |
| `schema/` | JSON Schema validation for tool parameters |
| `policy/` | `CostTracker`, `RateLimiter`, `PermissionChecker`, `TimeoutEnforcer`, policy `Chain` |
| `observe/` | `Tracer`, `Metrics`, `Hook` interface, `MetricsHook`, `TracingHook`, `MultiHook` |
| `memory/` | `Store` interface, `InMemory` vector store, `BudgetEnforcer`, `AsProvider()` adapter |
| `multi/` | `Registry`, `Mailbox`, `Coordinator`, `FanOut` for multi-agent orchestration |
| `cmd/agentflow/` | CLI binary |
| `examples/` | Runnable demos: `calculator`, `multiagent` |

## Key Interfaces

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

// MemoryProvider — context injection before each LLM call
type MemoryProvider interface {
    Recall(ctx context.Context, query string, topK int) ([]string, error)
}
```

## Event Log

Every agent action emits a structured event with a type, run ID, step index, timestamp, and a JSON payload:

| Event | When | Payload |
|-------|------|---------|
| `run_start` | Agent run begins | task, instructions, tools, max steps |
| `step_start` | Loop iteration begins | step index |
| `llm_request` | Before LLM call | messages (including injected memory), tool schemas |
| `llm_response` | After LLM call | content, tool calls, token usage |
| `tool_call` | Before tool execution | tool name, call ID, input |
| `tool_result` | After tool execution | output, error, duration |
| `step_end` | Loop iteration ends | step index, duration |
| `run_end` | Agent run finishes | status, output, error, total duration |
| `error` | Non-fatal error | message, error code |

Events are stored in JSONL format (one JSON object per line). The `schema_version` field in every event enables forward-compatible replay across version upgrades.

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

# Full check: fmt → vet → lint → test
make check

# Build binary
make build
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTFLOW_STORE_DIR` | `.agentflow/runs` | Event store directory for CLI commands |
| `AGENTFLOW_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warn`, `error` |

## Examples

```bash
# Calculator agent with replay
go run ./examples/calculator
go run ./examples/calculator -replay <run-id>

# Multi-agent orchestration
go run ./examples/multiagent
```

## License

[Apache 2.0](LICENSE)
