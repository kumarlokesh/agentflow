# agentflow

A Go runtime for single- and multi-agent systems in which every run is recorded as a causal trace that can be replayed against new code, diffed, and forked at any step.

> **Status: pre-release.** No version has been published yet; the API and trace format may still change before v0.1.

## Why

When an agent, or a team of agents, fails, two questions are hard to answer:

1. **Which agent's action caused it?**
2. **Does my fix actually fix *that* failure?**

agentflow records every model call, tool call, and side effect in a trace. You can then:

- **Replay** a recorded run against your *current* code. Replay stops at the first request that differs from the recording and reports exactly which field changed.
- **Test** for regressions by using recorded runs as `go test` fixtures, with no network calls.
- **Diff** two runs effect by effect.
- **Fork** a run at any step with a changed prompt or model, and compare the outcomes. *(planned)*
- **Trace teams** of agents as one causal tree, to find which sub-agent caused a failure. *(planned)*

## Current state

| Area | State |
|---|---|
| Agent runtime with tools, context providers, and interceptors (permissions, token budgets, rate limits) | Working |
| Anthropic and OpenAI providers (including reasoning blocks), retries, MCP tools | Working |
| Trace format: JSONL file store, deduplicated blobs, opt-in secret redaction, portable bundles | Working |
| Replay against current code, with field-level divergence reports; cassettes for `go test` | Working |
| CLI (`runs`, `show`, `diff`, `export`); OpenTelemetry traces and metrics | Working |
| Multi-agent trace tree, fork, visual trace viewer | Planned |

## Try it

Requires Go 1.24+. The calculator example uses an offline model, so it needs no API key.

```bash
go run ./examples/calculator                     # records a run to .agentflow/
go run ./examples/calculator -replay <trace-id>  # identical: no model or tool calls
go run ./examples/calculator -replay <trace-id> -instructions "Answer in Roman numerals."
go run ./cmd/agentflow show <trace-id>           # every recorded event
```

The third command simulates a code change. The replay stops at the first model call and shows what changed:

```
DIVERGED in / at llm#0 (step 0) — request_mismatch
  system[0].text changed:
    - You are a careful calculator. Use the calculator tool for arithmetic and show the steps.
    + Answer in Roman numerals.
```

Examples that call real providers are in `examples/anthropic-agent`, `examples/openai-agent`, `examples/mcp-agent`, and `examples/otel-agent`.

## In code

```go
agent, _ := agentflow.NewAgent(agentflow.AgentConfig{
    Name:         "calculator",
    Model:        anthropic.New(os.Getenv("ANTHROPIC_API_KEY")),
    ModelName:    "claude-opus-5-5",
    Instructions: "Use the calculator tool for arithmetic.",
    Tools:        []agentflow.Tool{calculator},
    Store:        store, // e.g. filestore.Open(".agentflow", filestore.Options{})
})
res, err := agent.Run(ctx, "What is 6*7?")

// Later, after changing the code: replay the recorded run.
rep, err := replay.Run(ctx, agent, store, res.TraceID, replay.Options{})
```

In tests, a recorded run becomes a regression test that fails with a divergence report when behavior changes:

```go
func TestCalculator(t *testing.T) {
    agentflowtest.Cassette(t, "testdata/calculator.trace.jsonl", newAgent(), "What is 6*7?")
}
```

How it works: [ARCHITECTURE.md](ARCHITECTURE.md). The trace format is specified in [docs/spec/trace-v1.md](docs/spec/trace-v1.md), with a [JSON Schema](docs/spec/trace-v1.schema.json).

## Development

```bash
make test    # tests with the race detector
make check   # fmt, vet, lint, test, trace schema conformance
```

## License

[Apache 2.0](LICENSE)
