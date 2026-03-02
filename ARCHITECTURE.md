# Architecture

> Design decisions, data flow, and package structure for agentflow.

## Overview

agentflow is a **deterministic, observable runtime** for AI agents. Its core insight is that every agent action - every LLM call, tool execution, and state transition - should be recorded as an immutable event. This event log enables:

1. **Deterministic replay** - re-execute any run without external calls
2. **Full observability** - inspect every decision the agent made
3. **Debugging** - compare runs to find where behavior diverged
4. **Auditing** - prove exactly what an agent did and why

## Core Concepts

### The Observe-Think-Act Loop

Every agent run follows an explicit loop:

```
┌─────────────────────────────────────────┐
│               Agent.Run()               │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐                           │
│  │ RunStart │ ← emit event              │
│  └────┬─────┘                           │
│       │                                 │
│       ▼                                 │
│  ┌──────────┐                           │
│  │StepStart │ ← emit event              │
│  └────┬─────┘                           │
│       │                                 │
│       ▼                                 │
│  ┌──────────┐    ┌───────────┐          │
│  │  THINK   │───▶│ LLM Call  │          │
│  │          │    │ (recorded)│          │
│  └────┬─────┘    └───────────┘          │
│       │                                 │
│       ▼                                 │
│  ┌──────────┐    ┌───────────┐          │
│  │   ACT    │───▶│Tool Calls │          │
│  │          │    │ (recorded)│          │
│  └────┬─────┘    └───────────┘          │
│       │                                 │
│       ▼                                 │
│  Tool calls? ──yes──▶ loop back         │
│       │                                 │
│       no                                │
│       │                                 │
│       ▼                                 │
│  ┌──────────┐                           │
│  │  RunEnd  │ ← emit event              │
│  └──────────┘                           │
│                                         │
└─────────────────────────────────────────┘
```

### Event Sourcing

Every action is an `Event` with:

- **Type** - what happened (`llm_request`, `tool_call`, `run_start`, etc.)
- **RunID** - which run it belongs to
- **StepIndex** - which loop iteration
- **Data** - type-specific JSON payload
- **SchemaVersion** - for forward compatibility

Events are append-only. They are never modified or deleted. This gives us an immutable, replayable record.

### Deterministic Replay

The replay engine works by substituting external dependencies:

```
Original Run:
  Agent → Real LLM → Real Tools → Events stored

Replay:
  Agent → Mock LLM (returns recorded responses)
        → Mock Tools (returns recorded results)
        → New events (should match originals)
```

This design keeps the agent code path **identical** between live and replay modes. No special "replay mode" in the agent - just different LLM and Tool implementations.

## Package Structure

```
agentflow/
├── agent.go          # Agent struct, NewAgent(), Run() - the core runtime
├── event.go          # Event type, EventType constants, typed payloads
├── eventstore.go     # EventStore interface (defined where consumed)
├── llm.go            # LLM interface, Message, LLMRequest/Response
├── tool.go           # Tool interface, ToolRegistry, ToolSchema
├── errors.go         # Sentinel errors, typed error wrappers
├── uuid.go           # Testable UUID/time generation
│
├── store/            # EventStore implementations
│   ├── memory.go     # In-memory store (for tests, short-lived agents)
│   └── file.go       # JSONL file-based persistent store
│
├── replay/           # Deterministic replay engine
│   ├── engine.go     # ReplayEngine - loads events, creates mocks, re-runs
│   └── diff.go       # Run comparison - finds divergences between runs
│
├── schema/           # JSON Schema validation for tool parameters
│   └── validator.go  # Lightweight validator (object, string, number, etc.)
│
├── policy/           # Guardrails and enforcement
│   └── policy.go     # CostTracker, RateLimiter, PermissionChecker, TimeoutEnforcer
│
├── observe/          # Observability and tracing
│   └── observe.go    # Span, Tracer, Metrics, Hook interface, MetricsHook, TracingHook
│
├── memory/           # Agent memory system
│   └── memory.go     # Store interface, InMemory vector store, BudgetEnforcer, eviction
│
├── multi/            # Multi-agent orchestration
│   └── multi.go      # Registry, Mailbox, Coordinator, FanOut, task delegation
│
├── cmd/agentflow/    # CLI binary
│   └── main.go       # Subcommands: runs, replay, diff, version
│
├── examples/         # Runnable examples
│   ├── calculator/   # Math agent with calculator tool
│   └── multiagent/   # Multi-agent coordination demo
│
└── internal/
    └── testutil/     # Shared test mocks and helpers
```

### Dependency Flow

```
cmd/agentflow ──▶ agentflow (root)
                  ├──▶ store/
                  ├──▶ replay/
                  └──▶ schema/

agentflow (root) ──▶ observe/   (Hook interface)
                 ──▶ policy/    (Checker interface)
                 ──▶ schema/    (tool param validation)

replay/ ──▶ agentflow (root)

store/ ──▶ agentflow (root)

policy/  ──▶ (no internal deps)
observe/ ──▶ (no internal deps)
memory/  ──▶ (no internal deps)
multi/   ──▶ (no internal deps)
schema/  ──▶ (no internal deps)
```

Key principle: **interfaces are defined where consumed** (root package), implementations live in subpackages. This prevents circular dependencies and keeps the API clean. The `policy/`, `observe/`, `memory/`, and `multi/` packages are self-contained with no internal dependencies, making them easy to use independently.

## Design Decisions

### 1. Why Event Sourcing?

Most agent frameworks use procedural logging - `print("called tool X")`. This is:

- Not structured (can't query/filter)
- Not replayable (no recorded inputs/outputs)
- Not diffable (no way to compare runs)

Event sourcing gives us all three. The trade-off is slightly more complex code, but the debugging and testing benefits are enormous.

### 2. Why JSONL for Persistence?

The file store uses JSON Lines (one JSON object per line):

- **Append-only** - perfect for event sourcing
- **Human-readable** - `cat` a file to see what happened
- **Streamable** - can process events without loading entire file
- **Simple** - no database dependency

### 3. Why Minimal Dependencies?

The entire framework has **one external dependency**: `github.com/google/uuid`. This is intentional:

- Fewer supply chain risks
- Faster compilation
- Easier to audit
- Standard library is stable and well-tested

### 4. Why Hand-Written Test Mocks?

We use hand-written mocks instead of codegen (mockgen, etc.) because:

- No build-time dependency
- Mocks are simple enough to read
- Test helpers provide ergonomic constructors
- No generated code to maintain

### 5. Why Package-Level Vars for UUID/Time?

`uuid.go` exposes `newUUID` and `nowUTC` as replaceable functions. This enables:

- Deterministic tests (pin time, pin UUIDs)
- No interface overhead for every event creation
- Clean test cleanup via `t.Cleanup()`

### 6. Why Schema Validation In-House?

The `schema/` package implements a focused JSON Schema subset instead of using a full library:

- Covers the practical needs of LLM tool-calling
- ~200 lines vs pulling in a large dependency
- Well-tested with table-driven tests
- Easy to extend as needed

## Event Schema Versioning

Every event carries a `schema_version` field (currently `1`). When the Event struct changes in a backwards-incompatible way:

1. Bump `SchemaVersion` constant
2. Add migration logic in the replay engine
3. Old events remain readable (the version tells us how to decode them)

This ensures replays continue to work across upgrades - a critical requirement for the determinism guarantee.

## Error Philosophy

Errors are categorized:

- **Fatal** - agent cannot continue (`ErrNoLLM`, context cancelled)
- **Non-fatal** - recorded as events, agent continues (`tool not found`, validation failure)
- **Wrapped** - typed wrappers (`ToolError`, `LLMError`, `StoreError`) preserve context

The agent loop **never panics**. All errors are either returned or recorded as events.

## Testing Strategy

- **Table-driven tests** for all core logic
- **Race detector** enabled by default (`-race` flag)
- **Golden file tests** for replay determinism (record → replay → assert identical)
- **Concurrent access tests** for thread-safety verification
