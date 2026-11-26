// Package triage is the example incident-triage team: four agents under one
// workflow, with a bug in one sub-agent's prompt that makes the team fail.
// examples/team runs it with small offline models; examples/team-live runs
// it on Claude.
package triage

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/replay"
)

// DefaultTask is the incident the team triages.
const DefaultTask = "checkout latency spike at 14:05 UTC"

// DefaultSQLInstructions is the sql agent's system prompt. Changing it
// simulates a change to one sub-agent. It has a bug: its schema is stale.
// The latency table now has a row per service, so a query that doesn't
// filter by service scans every service and times out, and the team can't
// measure the incident.
const DefaultSQLInstructions = "You write one SQL query per question against the metrics database and report the rows. " +
	"Schema: latency(minute TEXT 'HH:MM' UTC, p99_ms INTEGER), the site-wide p99 latency per minute."

// FixedSQLInstructions is the fix tried by the fork demo: the current
// schema, and a rule to filter by service.
const FixedSQLInstructions = "You write one SQL query per question against the metrics database and report the rows. " +
	"Schema: latency(minute TEXT 'HH:MM' UTC, service TEXT, p99_ms INTEGER), one row per service per minute. " +
	"Always filter by the service in question."

// Models supplies the agents' models. The zero value uses offline models
// that stand in for real ones, so the team runs without an API key.
type Models struct {
	// Name is the model name recorded in each agent's config.
	Name string
	// For returns the model for an agent, by agent name.
	For func(agent string) llm.Model
}

// NewTriage builds the team:
//
//	/                incident-triage   workflow
//	├── /logs#0      logs              searches logs
//	├── /metrics#0   metrics           delegates queries to sql
//	│   └── /metrics#0/sql#0   sql     runs queries
//	└── /summarizer#0 summarizer       combines the findings
//
// logs and metrics run in parallel; summarizer runs once both are done.
func NewTriage(sqlInstructions string, models Models) (*agentflow.Workflow, error) {
	if models.For == nil {
		models = Models{Name: "offline", For: offline}
	}
	searchLogs := agentflow.NewTool(llm.ToolSpec{
		Name: "search_logs", Description: "Searches the service logs for lines containing the query string.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`),
		Annotations: &llm.ToolAnnotations{ReadOnly: true},
	}, func(_ context.Context, in struct{ Query string }) (agentflow.ToolOutput, error) {
		var hits []string
		for _, l := range logLines {
			if strings.Contains(strings.ToLower(l), strings.ToLower(in.Query)) {
				hits = append(hits, l)
			}
		}
		if len(hits) == 0 {
			return agentflow.TextOutput("no matching lines"), nil
		}
		return agentflow.TextOutput(strings.Join(hits, "\n")), nil
	})
	runQuery := agentflow.NewTool(llm.ToolSpec{
		Name: "run_query", Description: "Runs a read-only SQL query against the metrics database.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}`),
		Annotations: &llm.ToolAnnotations{ReadOnly: true},
	}, func(_ context.Context, in struct{ SQL string }) (agentflow.ToolOutput, error) {
		switch {
		case filtersCheckout.MatchString(in.SQL):
			return agentflow.TextOutput("minute | p99_ms\n14:00 | 180\n14:05 | 2400\n14:10 | 2300"), nil
		case readsLatency.MatchString(in.SQL):
			// The table has a row per service now: unfiltered, it's too big.
			return agentflow.ErrorOutput("query timed out after 30s"), nil
		case !strings.Contains(strings.ToLower(in.SQL), "from"):
			return agentflow.TextOutput("1 row"), nil
		}
		return agentflow.ErrorOutput("no such table"), nil
	})

	sql, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "sql", ModelName: models.Name, Model: models.For("sql"),
		Instructions: sqlInstructions, Tools: []agentflow.Tool{runQuery}})
	if err != nil {
		return nil, err
	}
	logs, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "logs", ModelName: models.Name, Model: models.For("logs"),
		Instructions: "You find errors in the logs that match an incident. Report what you find in two sentences.",
		Tools:        []agentflow.Tool{searchLogs}})
	if err != nil {
		return nil, err
	}
	metrics, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "metrics", ModelName: models.Name, Model: models.For("metrics"),
		Instructions: "You measure the metrics around an incident. Ask the sql agent for data; you can't query the database yourself. " +
			"Report the numbers you got in two sentences.",
		Tools: []agentflow.Tool{agentflow.Delegate(sql, agentflow.DelegateOptions{
			ToolName: "query_db", Description: "Ask the sql agent a question about the metrics database.",
		})}})
	if err != nil {
		return nil, err
	}
	summarizer, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "summarizer", ModelName: models.Name, Model: models.For("summarizer"),
		Instructions: "You combine findings into a one-paragraph incident summary that states the measured impact and the likely cause. " +
			"Use only numbers from the findings."})
	if err != nil {
		return nil, err
	}

	return agentflow.NewWorkflow("incident-triage", func(ctx context.Context, incident string) (string, error) {
		found, err := agentflow.SpawnAll(ctx, []agentflow.SpawnSpec{
			{Agent: logs, Task: "Find errors related to: " + incident},
			{Agent: metrics, Task: "Measure the latency of the affected service around: " + incident},
		})
		if err != nil {
			return "", err
		}
		sum, err := agentflow.Spawn(ctx, summarizer,
			"Logs: "+found[0].OutputText()+"\nMetrics: "+found[1].OutputText())
		if err != nil {
			return "", err
		}
		// The team's own check: a summary must state the measured impact,
		// the p99 latency at the spike.
		agentflow.Expect(ctx, "impact measured", MeasuresImpact(sum.OutputText()), sum.OutputText())
		return sum.OutputText(), nil
	}), nil
}

// filtersCheckout matches a query that filters by the checkout service:
// service = 'checkout', service IN ('checkout'), service LIKE 'checkout%'.
var filtersCheckout = regexp.MustCompile(`(?is)\bservice\b\s*(=|in\s*\(|like)\s*'checkout`)

var readsLatency = regexp.MustCompile(`(?i)\b(from|join)\s+latency\b`)

var spikeP99 = regexp.MustCompile(`(?i)\b2,?400\s*ms|\b2\.4\s*s(ec|\b)`)

// MeasuresImpact reports whether a summary states the p99 latency at the
// spike, which only the metrics database has.
func MeasuresImpact(summary string) bool { return spikeP99.MatchString(summary) }

var logLines = []string{
	"14:04:58 checkout INFO request ok",
	"14:05:02 checkout ERROR payment-gateway timeout after 2000ms",
	"14:05:07 checkout ERROR payment-gateway timeout after 2000ms",
	"14:05:09 search INFO request ok",
}

// offline stands in for real models: each agent calls its tool once, then
// answers from the result.
func offline(agent string) llm.Model {
	return agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]
		var result string
		for _, b := range last.Blocks {
			if r, ok := b.(llm.ToolResult); ok {
				result = llm.TextOf(r.Content)
			}
		}
		first := len(req.Messages) == 1
		switch agent {
		case "logs":
			if first {
				return agentflowtest.CallTools(agentflowtest.Use("logs_1", "search_logs", map[string]string{"query": "checkout ERROR"})), nil
			}
			return agentflowtest.Reply(fmt.Sprintf("%d checkout errors, all payment-gateway timeouts, from 14:05.", strings.Count(result, "\n")+1)), nil
		case "metrics":
			if first {
				return agentflowtest.CallTools(agentflowtest.Use("metrics_1", "query_db", map[string]string{"task": "p99 latency of checkout from 14:00 to 14:10"})), nil
			}
			if strings.Contains(result, "2400") {
				return agentflowtest.Reply("Checkout p99 jumped from 180ms to 2400ms at 14:05."), nil
			}
			return agentflowtest.Reply("No latency data: " + result), nil
		case "sql":
			if first {
				// The query depends on the instructions, as a real model's would.
				q := "SELECT minute, p99_ms FROM latency WHERE minute BETWEEN '14:00' AND '14:10'"
				if strings.Contains(llm.TextOf(req.System), "filter by the service") {
					q += " AND service='checkout'"
				}
				return agentflowtest.CallTools(agentflowtest.Use("sql_1", "run_query", map[string]string{"sql": q})), nil
			}
			if strings.Contains(result, "timed out") {
				return agentflowtest.Reply("No rows: the query failed (" + result + ")."), nil
			}
			return agentflowtest.Reply(strings.ReplaceAll(result, "\n", "; ")), nil
		}
		task := llm.TextOf(last.Blocks)
		impact := "impact not measured"
		if strings.Contains(task, "2400ms") {
			impact = "p99 rose from 180ms to 2400ms"
		}
		return agentflowtest.Reply("Checkout latency spiked at 14:05 (" + impact + "); likely cause: payment-gateway timeouts."), nil
	})
}

// ForkOptions is the demo's fork: at the sql agent's first model call, with
// the fixed prompt, against a baseline without it. A sample passes by
// default if it completes and no expectation fails: here, "impact
// measured".
func ForkOptions(samples int) replay.ForkOptions {
	return replay.ForkOptions{
		At:        &replay.Point{Path: "/metrics#0/sql#0", Effect: "llm#0"},
		Overrides: []replay.Override{replay.SetConfig{System: llm.Blocks{llm.Text{Text: FixedSQLInstructions}}}},
		Samples:   samples,
		Baseline:  true,
	}
}
