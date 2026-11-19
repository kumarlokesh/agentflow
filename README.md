# agentflow

A Go runtime for single- and multi-agent systems in which every run is recorded as a causal trace that can be replayed against new code, diffed, and forked at any step.

> **Status: pre-release, being redesigned.** No version has been published, and the API and trace format will change.

## Why

When an agent, or a team of agents, fails, two questions are hard to answer:

1. **Which agent's action caused it?**
2. **Does my fix actually fix *that* failure?**

agentflow records every LLM call, tool call, and side effect as an append-only event log. You can then:

- **Replay** a recorded run against your *current* code. Replay stops at the first request that differs from the recording and reports exactly which field changed.
- **Test** for regressions by using recorded runs as `go test` fixtures, with no network calls.
- **Diff** two runs to see where their behavior split.
- **Fork** a run at any step with a changed prompt or model, and compare the outcomes. *(planned)*
- **Trace teams** of agents as one causal tree, to find which sub-agent caused a failure. *(planned)*

## Current state

| Area | State |
|---|---|
| Agent loop, tools, Anthropic and OpenAI providers, MCP tools | Working prototype |
| Event log (JSONL file store), run diff, OpenTelemetry export | Working prototype; trace format being redesigned |
| Replay | Prototype only: it re-feeds recorded responses and cannot yet detect changes. Being redesigned. |
| Multi-agent trace tree, fork, inspection CLI | Planned |

## Try the prototype

Requires Go 1.24+. The calculator example uses a scripted model, so it needs no API key.

```bash
go run ./examples/calculator                  # records a run to .agentflow/runs
go run ./examples/calculator -replay <run-id> # replays it
```

Examples that call real providers live in `examples/anthropic-agent` and `examples/openai-agent`. They need `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` respectively.

## Development

```bash
make test    # tests with the race detector
make check   # fmt, vet, lint, test
```

## License

[Apache 2.0](LICENSE)
