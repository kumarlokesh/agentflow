# Architecture

Design decisions, data flow, and package structure for agentflow.

## Overview

agentflow is a deterministic, observable runtime for AI agents. Its core design is that every agent action - every LLM call, tool execution, and state transition - is recorded as an immutable event. This event log is the basis for:

1. **Deterministic replay** - re-execute any run without external calls
2. **Full observability** - inspect every decision the agent made
3. **Run diffing** - compare two runs event-by-event to locate divergences
4. **Auditing** - prove exactly what an agent did and why

## Core Concepts

### The Observe-Think-Act Loop

Every agent run follows an explicit loop:

```
┌─────────────────────────────────────────────────────────┐
│                      Agent.Run()                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐                                           │
│  │ RunStart │ → emit event, invoke hook.OnRunStart      │
│  └────┬─────┘                                           │
│       │                                                 │
│       ▼  (loop, up to MaxSteps)                         │
│  ┌──────────┐                                           │
│  │StepStart │ → emit event, invoke hook.OnStepStart     │
│  └────┬─────┘                                           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────┐                               │
│  │  Memory Recall       │ → query MemoryProvider,       │
│  │  (if configured)     │   prepend results to messages │
│  └────┬─────────────────┘                               │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │  THINK               │───▶│ LLM Call (recorded)  │   │
│  │  Policy.Check()      │    │ emit llm_request     │   │
│  │  hook.OnLLMCall()    │    │ emit llm_response    │   │
│  └────┬─────────────────┘    └──────────────────────┘   │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │  ACT                 │───▶│ Tool Calls (recorded)│   │
│  │  Policy.Check()      │    │ emit tool_call       │   │
│  │  hook.OnToolCall()   │    │ emit tool_result     │   │
│  └────┬─────────────────┘    └──────────────────────┘   │
│       │                                                 │
│       ▼                                                 │
│  StepEnd → emit event, invoke hook.OnStepEnd            │
│       │                                                 │
│  Tool calls? ──yes──▶ loop back to StepStart            │
│       │                                                 │
│       no                                                │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────┐                                           │
│  │  RunEnd  │ → emit event, invoke hook.OnRunEnd        │
│  └──────────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Event Sourcing

Every action is an `Event` with:

- **Type** - what happened (`llm_request`, `tool_call`, `run_start`, etc.)
- **RunID** - which run it belongs to
- **StepIndex** - which loop iteration
- **Data** - type-specific JSON payload (`json.RawMessage`)
- **SchemaVersion** - integer field enabling forward-compatible decoding

Events are append-only and never modified. The full event log for a run can be replayed to reconstruct exactly what happened.

### Deterministic Replay

The replay engine substitutes external dependencies with mocks backed by the event log:

```
Original Run:
  Agent → Real LLM   → Real Tools   → Events stored in EventStore

Replay:
  Agent → Mock LLM   (returns recorded llm_response payloads)
        → Mock Tools  (returns recorded tool_result payloads)
        → New events  (compared against originals for divergence)
```

The agent code path is **identical** between live and replay modes. No replay-mode flag in `Agent`; the substitution happens entirely at the dependency injection boundary (`AgentConfig.LLM` and `AgentConfig.Tools`).

### Memory Injection

Before each LLM call, the agent queries its `MemoryProvider` and prepends the top-K results to the system prompt:

```
Per-step memory injection:
  agent.memory.Recall(ctx, task, topK)
    → []string of relevant context snippets
    → prepended as a "system" message before the full conversation history
    → LLM sees: [memory block] [system instructions] [conversation history]
```

The `MemoryProvider` interface is defined in the root package (where it is consumed). The `memory.AsProvider()` adapter bridges `memory.Store` to this interface. Agents without memory configured skip the recall step entirely.

### Policy Enforcement

Before each LLM call and before each tool execution, the agent calls `Policy.Check()`. The policy chain is a slice of `Checker` implementations evaluated in order:

```
Policy.Check() call order:
  CostTracker.Check()       → reject if token budget exceeded
  RateLimiter.Check()       → reject if call rate exceeded
  PermissionChecker.Check() → reject if tool not in allow-list
  TimeoutEnforcer.Check()   → reject if deadline exceeded
```

A `Checker` returns an error to block the action. The agent records the rejection and either halts or continues based on the error type.

**Budget accumulation**: `CostTracker.Check()` only reads the current budget. Token usage is fed back via `CostHook.OnLLMCall()`, which calls `CostTracker.Record()` after each LLM response. Without `CostHook` wired in, the tracker always sees zero tokens used.

## Package Structure

```
agentflow/
├── agent.go          # Agent struct, NewAgent(), Run(), AgentConfig
├── event.go          # Event type, EventType constants, typed payload structs
├── eventstore.go     # EventStore interface (defined where consumed)
├── llm.go            # LLM interface, Message, LLMRequest/Response, TokenUsage
├── tool.go           # Tool interface, ToolRegistry, ToolSchema
├── hooks.go          # CostHook — bridges policy.CostTracker to observe.Hook
├── errors.go         # Sentinel errors, typed error wrappers (ToolError, LLMError)
├── uuid.go           # Replaceable newUUID/nowUTC vars for deterministic tests
│
├── store/            # EventStore implementations
│   ├── memory.go     # In-memory store (tests, short-lived agents)
│   └── file.go       # JSONL file-based persistent store
│
├── replay/           # Deterministic replay engine
│   ├── engine.go     # ReplayEngine: loads events, injects mocks, re-runs agent
│   └── diff.go       # Run comparison: finds field-level divergences between runs
│
├── schema/           # JSON Schema validation for tool parameters
│   └── validator.go  # Subset validator: object, string, number, array, enum, required
│
├── policy/           # Guardrails and execution policy
│   └── policy.go     # Checker interface, CostTracker, RateLimiter,
│                     # PermissionChecker, TimeoutEnforcer, Chain
│
├── observe/          # Observability and tracing
│   └── observe.go    # Hook interface, Span, Tracer, Metrics,
│                     # MetricsHook, TracingHook, MultiHook
│
├── memory/           # Agent memory system
│   └── memory.go     # Store interface, InMemory vector store,
│                     # BudgetEnforcer, eviction strategies, AsProvider()
│
├── multi/            # Multi-agent orchestration
│   └── multi.go      # Registry, Mailbox, Coordinator, FanOut
│
├── cmd/agentflow/    # CLI binary
│   └── main.go       # Subcommands: runs, replay, diff, version
│
├── examples/         # Runnable examples
│   ├── calculator/   # Single-agent math solver with calculator tool
│   └── multiagent/   # Multi-agent coordination demo
│
└── internal/
    └── testutil/     # Shared test helpers and mock constructors
```

### Dependency Flow

```
cmd/agentflow ──▶ agentflow (root)
                  ├──▶ store/
                  ├──▶ replay/
                  └──▶ schema/

agentflow (root) ──▶ observe/   (Hook interface consumed here)
                 ──▶ policy/    (Checker interface consumed here)
                 ──▶ schema/    (tool parameter validation)
                 [MemoryProvider interface defined here, implemented in memory/]

replay/ ──▶ agentflow (root)
store/  ──▶ agentflow (root)

policy/  ──▶ (stdlib only)
observe/ ──▶ (stdlib only)
memory/  ──▶ (stdlib only)
multi/   ──▶ (stdlib only)
schema/  ──▶ (stdlib only)
```

**Circular dependency prevention**: interfaces are defined in the package that consumes them (the root package), not in the package that implements them. `MemoryProvider` is in the root package; `memory.StoreProvider` implements it without importing the root. `CostHook` in the root package bridges `policy.CostTracker` and `observe.Hook` - both are already imported by the root, so no new dependency is introduced.

## Key Types

### CostHook

`CostHook` (in `hooks.go`) is the bridge between the observability layer and the policy layer. Neither `policy` nor `observe` imports the other; the root package imports both and provides the connection:

```go
// CostHook implements observe.Hook. It feeds LLM token usage back into
// a CostTracker after each LLM call. Without it, CostTracker.Check() always
// sees zero tokens used and never enforces a budget.
type CostHook struct {
    tracker *policy.CostTracker
}

func (h *CostHook) OnLLMCall(ctx context.Context, runID string, step int,
    promptTokens, completionTokens, totalTokens int,
    duration time.Duration, err error) {
    _ = h.tracker.Record(policy.UsageReport{...})
}
```

### MemoryProvider

`MemoryProvider` is defined in the root package and consumed by `Agent`. It is intentionally minimal — a single method that returns a slice of strings:

```go
type MemoryProvider interface {
    Recall(ctx context.Context, query string, topK int) ([]string, error)
}
```

`memory.AsProvider(store)` returns a `*StoreProvider` that wraps any `memory.Store` and implements this interface via keyword or vector search.

### Hook

`observe.Hook` is the primary extension point for observability. All methods are called synchronously in the agent loop; implementations must not block. `observe.MultiHook` composes multiple hooks:

```go
hook := observe.NewMultiHook(
    observe.NewTracingHook(tracer),
    observe.NewMetricsHook(metrics),
    agentflow.NewCostHook(costTracker),
)
```

### Policy Chain

`policy.NewChain(checkers...)` composes multiple `Checker` implementations. The chain stops at the first rejection:

```go
policy := policy.NewChain(
    policy.NewCostTracker(policy.CostTrackerConfig{MaxTotalTokens: 10_000}),
    policy.NewRateLimiter(20, time.Minute),
    policy.NewPermissionChecker(allowedTools),
    policy.NewTimeoutEnforcer(30 * time.Second),
)
```

## Design Decisions

### 1. Why event sourcing instead of structured logging?

Structured logging (`log.Info("tool called", "name", x)`) is write-only. You cannot replay a run from logs because the LLM responses and tool outputs are not recorded. Event sourcing records the full input/output at every step, enabling exact re-execution and field-level diff between runs.

### 2. Why JSONL for persistence?

JSONL (one JSON object per line) is:

- **Append-only** - a single `fprintf` adds an event; no locking needed across processes
- **Streamable** - a reader can process events without loading the entire file into memory
- **Human-readable** - `cat run.jsonl | jq .` shows the full history

The alternative (SQLite, embedded KV) would require a database dependency and make raw inspection harder.

### 3. Why a single external dependency?

`github.com/google/uuid` is the only external import. Every other component - HTTP, JSON, concurrency, sorting - uses the standard library. This reduces supply chain risk, speeds up compilation, and makes the module easy to audit and vendor.

### 4. Why hand-written mocks?

Test mocks in `internal/testutil/` are written by hand rather than generated with mockgen or similar tools because:

- No code generation step in the build
- Mocks are simple structs with behavior controlled by fields (e.g., `responses []LLMResponse`, `callCount int`)
- Easy to extend with test-specific behavior without regenerating

### 5. Why `newUUID` and `nowUTC` as package-level vars?

`uuid.go` exposes these as `var` instead of inline calls. Tests that need deterministic event IDs or timestamps replace them via `t.Cleanup()`. This avoids introducing an interface for event creation and keeps the hot path allocation-free.

### 6. Why in-house JSON Schema validation?

`schema/validator.go` implements the subset of JSON Schema needed for LLM tool-calling: `object`, `string`, `number`, `boolean`, `array`, `enum`, `required`. A full JSON Schema library would be ~10× the code for features (JSON Pointer, `$ref` resolution, format validators) that tool parameters never use.

### 7. Why is `MemoryProvider` in the root package, not `memory/`?

Go interfaces are defined where they are consumed, not where they are implemented. The root package consumes `MemoryProvider` in `agent.go`; `memory.StoreProvider` implements it in `memory/memory.go` without any import of the root package. If `MemoryProvider` were defined in `memory/`, the root package would import `memory/`, and `memory/` would be a mandatory transitive dependency — defeating the point of a pluggable interface.

## Event Schema Versioning

Every event carries a `schema_version` field (currently `1`). When the event model changes in a backwards-incompatible way:

1. Bump the `SchemaVersion` constant in `event.go`
2. Add a migration in the (planned) `migration/` package
3. Old JSONL files remain decodeable - the version field determines which decoder to use

This is a prerequisite for M7 (Event Schema Migration Tooling): existing replays must continue to work after version upgrades.

## Error Philosophy

Errors fall into three categories:

- **Fatal** - agent cannot continue; returned from `Run()` with no further steps (`ErrNoLLM`, context cancelled)
- **Non-fatal** - recorded as `error` events; agent may continue (`tool not found`, schema validation failure)
- **Wrapped** - typed wrappers (`ToolError`, `LLMError`, `StoreError`) carry the original error and an error code for structured handling

The agent loop never calls `panic`. All errors surface as either a returned error or a recorded event.

## Testing Strategy

- **Table-driven tests** - all core logic uses `[]struct{ name, input, want }` patterns
- **Race detector** - `go test -race` enabled for all packages; concurrent types (`InMemory`, `ToolRegistry`, `Mailbox`) have dedicated concurrency tests
- **Replay round-trip tests** - record a run, replay it, assert the event log matches exactly
- **Coverage target** - ≥ 80% for core packages (`agentflow`, `replay`, `store`, `policy`, `memory`)
- **Interface mocks** - hand-written in `internal/testutil/`; no code generation
