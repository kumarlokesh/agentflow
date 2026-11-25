package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
)

// DefaultSQLInstructions is the sql agent's system prompt. Changing it
// simulates a change to one sub-agent. It has a bug: without being told to
// filter by service, the sql agent averages over every service and misses
// the checkout spike, so the team can't find the cause.
const DefaultSQLInstructions = "You write one SQL query per question and report the rows."

// FixedSQLInstructions is the fix tried by the fork demo.
const FixedSQLInstructions = DefaultSQLInstructions + " Always filter by the service in question."

// Models supplies each agent's model by agent name; nil uses the offline
// models below.
type Models func(agent string) llm.Model

// NewTriage builds the example team:
//
//	/                incident-triage   workflow
//	├── /logs#0      logs              searches logs
//	├── /metrics#0   metrics           delegates queries to sql
//	│   └── /metrics#0/sql#0   sql     runs queries
//	└── /summarizer#0 summarizer       combines the findings
//
// logs and metrics run in parallel; summarizer runs once both are done.
func NewTriage(sqlInstructions string, models Models) (*agentflow.Workflow, error) {
	if models == nil {
		models = offline
	}
	searchLogs := agentflow.NewTool(llm.ToolSpec{
		Name: "search_logs", Description: "Searches the service logs.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`),
		Annotations: &llm.ToolAnnotations{ReadOnly: true},
	}, func(_ context.Context, in struct{ Query string }) (agentflow.ToolOutput, error) {
		var hits []string
		for _, l := range logLines {
			if strings.Contains(l, in.Query) {
				hits = append(hits, l)
			}
		}
		return agentflow.TextOutput(strings.Join(hits, "\n")), nil
	})
	runQuery := agentflow.NewTool(llm.ToolSpec{
		Name: "run_query", Description: "Runs a read-only SQL query against the metrics database.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]}`),
		Annotations: &llm.ToolAnnotations{ReadOnly: true},
	}, func(_ context.Context, in struct{ SQL string }) (agentflow.ToolOutput, error) {
		if strings.Contains(in.SQL, "service='checkout'") {
			return agentflow.TextOutput("14:00 p99=180ms\n14:05 p99=2400ms\n14:10 p99=2300ms"), nil
		}
		return agentflow.TextOutput("14:00 p99=190ms\n14:05 p99=205ms\n14:10 p99=200ms"), nil // all services
	})

	sql, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "sql", ModelName: "offline", Model: models("sql"),
		Instructions: sqlInstructions, Tools: []agentflow.Tool{runQuery}})
	if err != nil {
		return nil, err
	}
	logs, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "logs", ModelName: "offline", Model: models("logs"),
		Instructions: "You find errors in the logs that match an incident.", Tools: []agentflow.Tool{searchLogs}})
	if err != nil {
		return nil, err
	}
	metrics, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "metrics", ModelName: "offline", Model: models("metrics"),
		Instructions: "You find metric anomalies around an incident. Ask the sql agent for data.",
		Tools: []agentflow.Tool{agentflow.Delegate(sql, agentflow.DelegateOptions{
			ToolName: "query_db", Description: "Ask the sql agent a question about the metrics database.",
		})}})
	if err != nil {
		return nil, err
	}
	summarizer, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "summarizer", ModelName: "offline", Model: models("summarizer"),
		Instructions: "You combine findings into a one-paragraph incident summary."})
	if err != nil {
		return nil, err
	}

	return agentflow.NewWorkflow("incident-triage", func(ctx context.Context, incident string) (string, error) {
		found, err := agentflow.SpawnAll(ctx, []agentflow.SpawnSpec{
			{Agent: logs, Task: "Find errors related to: " + incident},
			{Agent: metrics, Task: "Find anomalies related to: " + incident},
		})
		if err != nil {
			return "", err
		}
		sum, err := agentflow.Spawn(ctx, summarizer,
			"Logs: "+found[0].OutputText()+"\nMetrics: "+found[1].OutputText())
		if err != nil {
			return "", err
		}
		return sum.OutputText(), nil
	}), nil
}

var logLines = []string{
	"14:04:58 checkout INFO request ok",
	"14:05:02 checkout ERROR payment-gateway timeout after 2000ms",
	"14:05:07 checkout ERROR payment-gateway timeout after 2000ms",
	"14:05:09 search INFO request ok",
}

// offline stands in for real models so the example runs without an API
// key: each agent calls its tool once, then answers from the result.
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
			if strings.Contains(result, "2400ms") {
				return agentflowtest.Reply("Checkout p99 jumped from 180ms to 2400ms at 14:05 (" + result + ")."), nil
			}
			return agentflowtest.Reply("No latency anomaly found (" + result + ")."), nil
		case "sql":
			if first {
				// The query depends on the instructions, as a real model's would.
				q := "SELECT minute, p99 FROM latency"
				if strings.Contains(llm.TextOf(req.System), "filter by the service") {
					q += " WHERE service='checkout'"
				}
				return agentflowtest.CallTools(agentflowtest.Use("sql_1", "run_query", map[string]string{"sql": q})), nil
			}
			return agentflowtest.Reply(strings.ReplaceAll(result, "\n", ", ")), nil
		}
		task := llm.TextOf(last.Blocks)
		cause := "unknown"
		if strings.Contains(task, "payment-gateway") && strings.Contains(task, "2400ms") {
			cause = "payment-gateway timeouts"
		}
		return agentflowtest.Reply("Checkout latency spiked at 14:05; likely cause: " + cause + "."), nil
	})
}
