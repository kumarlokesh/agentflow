# agentflow

A Go runtime for single- and multi-agent systems in which every run is recorded as a causal trace that can be replayed against new code, diffed, and forked at any step.

> **Status: pre-release.** No version has been published yet; the API and trace format may still change before v0.1.

## Why

When an agent, or a team of agents, fails, two questions are hard to answer:

1. **Which agent's action caused it?**
2. **Does my fix actually fix *that* failure?**

agentflow records every model call, tool call, and side effect in a trace. You can then:

- **Replay** a recorded run against your *current* code. Replay stops at the first request that differs from the recording and reports exactly which field changed. With `Verify`, it also runs the tools and context providers you name and compares their results, so a change inside a tool is caught too.
- **Test** for regressions by using recorded runs as `go test` fixtures, with no network calls.
- **Diff** two runs effect by effect.
- **Fork** a recorded run at any step of any agent with a changed prompt, model, message or result, and continue live from there. Everything the change doesn't reach is answered from the recording. Run several samples with and without the change to see whether your fix actually fixes that failure.
- **Trace teams** of agents as one tree: agents delegate to agents as tools, or a workflow runs them in sequence or in parallel. Replaying a team reports a change in the sub-agent where it was made.

## Current state

| Area | State |
|---|---|
| Agent runtime with tools, context providers, and interceptors (permissions, token budgets, rate limits) | Working |
| Anthropic and OpenAI providers (including reasoning blocks), retries, MCP tools | Working |
| Trace format: JSONL file store, deduplicated blobs, opt-in secret redaction, portable bundles | Working |
| Replay against current code, with field-level divergence reports; cassettes for `go test` | Working |
| CLI (`runs`, `show`, `diff`, `export`); OpenTelemetry traces and metrics | Working |
| Multi-agent run trees: delegation, workflows, parallel children; team replay | Working |
| Fork with overrides, team-aware spreading, samples, baseline arm and pass rates | Working |
| Visual trace viewer, `blame` | Planned |

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

A team works the same way. `examples/team` runs a workflow whose agents delegate to each other. Changing one sub-agent's prompt is reported in that sub-agent's run:

```bash
go run ./examples/team                                        # prints the run tree
go run ./examples/team -replay <trace-id> -sql-instructions "Write efficient SQL."
```

```
DIVERGED in /metrics#0/sql#0 at llm#0 (step 0) — request_mismatch
  system[0].text changed:
    - "You write one SQL query per question and report the rows."
    + "Write efficient SQL."
```

The recorded run is a failure: the summary says "likely cause: unknown", because the `sql` agent's query misses the spike. Fork it at that agent's first model call with a fixed prompt:

```bash
go run ./examples/team -fork <trace-id>
```

```
FORK 01a2147b-89fa  at /metrics#0/sql#0 llm#0   overrides: set_config
  treatment  3/3 pass  100%  [44%, 100%]
  baseline   0/3 pass    0%  [ 0%,  56%]
  (small samples: the intervals are wide)
  treatment  reuse per sample (median): runs 1 replayed / 3 live / 1 new · effects 4 from recording / 5 live · ~45 tokens saved
  baseline   reuse per sample (median): runs 4 replayed / 1 live / 0 new · effects 6 from recording / 3 live · ~75 tokens saved
```

The `logs` agent, which the fix doesn't reach, is answered entirely from the recording in every sample.

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
    agentflowtest.Cassette(t, "testdata/calculator.trace.jsonl", newAgent(), "What is 6*7?",
        // Also run read-only tools and context providers, and compare their
        // results with the recording, to catch changes inside them.
        agentflowtest.Verify("tool:*", "effect:context.*"))
}
```

Without `Verify`, tool and context-provider results come from the recording: replay checks what your code asks for and what it does with the answers, not the code behind the effect. Verify only what is deterministic in tests; a tool that reads a live database will differ for reasons unrelated to your code.

Teams are built from the same agents. A delegate tool lets one agent's model start another agent; a workflow starts agents from Go code. Every run lands in one trace, and `replay.Run` and cassettes accept a workflow as well as an agent:

```go
sql, _ := agentflow.NewAgent(agentflow.AgentConfig{Name: "sql" /* … */})
metrics, _ := agentflow.NewAgent(agentflow.AgentConfig{Name: "metrics",
    Tools: []agentflow.Tool{agentflow.Delegate(sql, agentflow.DelegateOptions{ToolName: "query_db"})} /* … */})

triage := agentflow.NewWorkflow("incident-triage", func(ctx context.Context, incident string) (string, error) {
    found, err := agentflow.SpawnAll(ctx, []agentflow.SpawnSpec{   // in parallel
        {Agent: logs, Task: "Find errors related to: " + incident},
        {Agent: metrics, Task: "Find anomalies related to: " + incident},
    })
    if err != nil {
        return "", err
    }
    sum, err := agentflow.Spawn(ctx, summarizer, found[0].OutputText()+"\n"+found[1].OutputText())
    if err != nil {
        return "", err
    }
    return sum.OutputText(), nil
})
res, err := triage.Run(ctx, "checkout latency spike at 14:05 UTC", agentflow.WithStore(store))
```

A fork takes a fork point, overrides, and a check:

```go
res, err := replay.Fork(ctx, triage, store, traceID, replay.ForkOptions{
    At:        &replay.Point{Path: "/metrics#0/sql#0", Effect: "llm#0"},
    Overrides: []replay.Override{replay.SetConfig{System: llm.Blocks{llm.Text{Text: fixedPrompt}}}},
    Samples:   5,
    Baseline:  true, // also 5 samples without the change
    Check: func(r *agentflow.RunResult) replay.Verdict {
        return replay.Verdict{Pass: strings.Contains(r.OutputText(), "payment-gateway")}
    },
})
fmt.Println(res) // pass rates with 95% intervals, reuse per arm
```

How it works: [ARCHITECTURE.md](ARCHITECTURE.md). The trace format is specified in [docs/spec/trace-v1.md](docs/spec/trace-v1.md), with a [JSON Schema](docs/spec/trace-v1.schema.json).

## Development

```bash
make test    # tests with the race detector
make check   # fmt, vet, lint, test, trace schema conformance
```

## License

[Apache 2.0](LICENSE)
