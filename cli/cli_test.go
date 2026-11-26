package cli_test

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/cli"
	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

// team is a workflow whose sql agent's query fails unless its prompt says
// to filter; the workflow expects a result.
func team(t *testing.T, prompt string) agentflow.Entrypoint {
	t.Helper()
	query := agentflow.NewTool(llm.ToolSpec{Name: "query"}, func(_ context.Context, in struct{ SQL string }) (agentflow.ToolOutput, error) {
		if !strings.Contains(in.SQL, "WHERE") {
			return agentflow.ErrorOutput("timed out"), nil
		}
		return agentflow.TextOutput("42 rows"), nil
	})
	sql, err := agentflow.NewAgent(agentflow.AgentConfig{Name: "sql", ModelName: "m", Instructions: prompt, Tools: []agentflow.Tool{query},
		Model: agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
			last := req.Messages[len(req.Messages)-1]
			if r, ok := last.Blocks[0].(llm.ToolResult); ok {
				return agentflowtest.Reply("result: " + llm.TextOf(r.Content)), nil
			}
			q := "SELECT *"
			if strings.Contains(llm.TextOf(req.System), "filter") {
				q += " WHERE ok"
			}
			return agentflowtest.CallTools(agentflowtest.Use("q1", "query", map[string]string{"SQL": q})), nil
		})})
	if err != nil {
		t.Fatal(err)
	}
	return agentflow.NewWorkflow("report", func(ctx context.Context, task string) (string, error) {
		res, err := agentflow.Spawn(ctx, sql, task)
		if err != nil {
			return "", err
		}
		agentflow.Expect(ctx, "has rows", strings.Contains(res.OutputText(), "rows"), res.OutputText())
		return res.OutputText(), nil
	})
}

type harness struct {
	t     *testing.T
	store *memstore.Store
	cfg   cli.Config
}

func (h *harness) run(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := cli.Run(context.Background(), args, &out, &errb, h.cfg)
	return code, out.String(), errb.String()
}

func setup(t *testing.T) (*harness, string) {
	store := memstore.New()
	res, err := team(t, "Query.").Run(context.Background(), "sales", agentflow.WithStore(store))
	if err != nil {
		t.Fatal(err)
	}
	return &harness{t: t, store: store, cfg: cli.Config{Name: "tool", Store: store, Entrypoints: []agentflow.Entrypoint{team(t, "Query.")}}}, res.TraceID
}

func TestInspectionCommands(t *testing.T) {
	h, id := setup(t)
	suffix := id[len(id)-8:]

	code, out, _ := h.run("runs")
	if code != 0 || !strings.Contains(out, id) {
		t.Fatalf("runs: %d %s", code, out)
	}
	code, out, _ = h.run("show", suffix)
	if code != 0 || !strings.Contains(out, "└─ /sql#0") || !strings.Contains(out, "tool:query!") || !strings.Contains(out, `"has rows" failed`) {
		t.Fatalf("show: %d\n%s", code, out)
	}
	code, out, _ = h.run("show", suffix, "-events", "-run", "/sql#0")
	if code != 0 || !strings.Contains(out, "run /sql#0") || strings.Contains(out, "run / ") || !strings.Contains(out, "tool.completed") {
		t.Fatalf("show -events: %d\n%s", code, out)
	}
	code, out, _ = h.run("blame", suffix)
	if code != 0 || !strings.Contains(out, "root cause 1: /sql#0 tool:query(q1) — error_result") {
		t.Fatalf("blame: %d\n%s", code, out)
	}
	code, out, _ = h.run("diff", suffix, suffix)
	if code != 0 || !strings.Contains(out, "identical") {
		t.Fatalf("diff: %d %s", code, out)
	}
	dir := t.TempDir()
	html := filepath.Join(dir, "t.html")
	if code, _, errs := h.run("export", suffix, "-html", "-o", html); code != 0 {
		t.Fatalf("export -html: %d %s", code, errs)
	}
	b, err := os.ReadFile(html)
	if err != nil || !bytes.Contains(b, []byte("<!doctype html>")) || !bytes.Contains(b, []byte(`"root_causes"`)) {
		t.Fatalf("html: %v", err)
	}
	code, out, _ = h.run("export", suffix)
	if code != 0 || !strings.HasPrefix(out, `{"bundle":"agentflow.trace"`) {
		t.Fatalf("export: %d %.80s", code, out)
	}
}

func TestReplayAndFork(t *testing.T) {
	h, id := setup(t)
	code, out, _ := h.run("replay", id)
	if code != cli.ExitOK || !strings.Contains(out, "identical") {
		t.Fatalf("replay: %d\n%s", code, out)
	}
	changed := *h
	changed.cfg.Entrypoints = []agentflow.Entrypoint{team(t, "Query fast.")}
	code, out, _ = changed.run("replay", id)
	if code != cli.ExitDiverged || !strings.Contains(out, "DIVERGED in /sql#0 at llm#0") {
		t.Fatalf("replay changed: %d\n%s", code, out)
	}

	code, out, _ = h.run("fork", id, "-path", "/sql#0", "-effect", "llm#0", "-system", "Query; filter.", "-samples", "2", "-baseline")
	if code != cli.ExitOK || !strings.Contains(out, "treatment  2/2 pass") || !strings.Contains(out, "baseline   0/2 pass") ||
		!regexp.MustCompile(`baseline  sample 1  \S+  fail \(failed expectations: / "has rows"\)`).MatchString(out) {
		t.Fatalf("fork: %d\n%s", code, out)
	}
	code, out, _ = h.run("fork", id, "-path", "/sql#0", "-effect", "tool:q1", "-replace", "7 rows")
	if code != cli.ExitOK || !strings.Contains(out, "1/1 pass") {
		t.Fatalf("fork -replace: %d\n%s", code, out)
	}
	// The baseline alone fails: exit code 3.
	code, _, _ = h.run("fork", id, "-path", "/sql#0", "-effect", "llm#0", "-inject", "hurry")
	if code != cli.ExitDiverged {
		t.Fatalf("failing fork: %d", code)
	}
}

func TestUsageErrors(t *testing.T) {
	h, id := setup(t)
	for _, c := range []struct {
		args []string
		code int
		msg  string
	}{
		{nil, cli.ExitUsage, "Usage:"},
		{[]string{"nope"}, cli.ExitUsage, `unknown command "nope"`},
		{[]string{"show"}, cli.ExitUsage, "usage: show <id>"},
		{[]string{"show", "abc"}, cli.ExitUsage, "too short"},
		{[]string{"show", "zzzzzzzz"}, cli.ExitError, "no trace or run matches"},
		{[]string{"replay", id, "-on", "maybe"}, cli.ExitUsage, `-on "maybe"`},
		{[]string{"fork", id, "-path", "/sql#0"}, cli.ExitUsage, "-path needs -effect"},
		{[]string{"blame", id, "-at", "/nobody#0"}, cli.ExitError, "no run at /nobody#0"},
	} {
		code, out, errs := h.run(c.args...)
		if code != c.code || !strings.Contains(out+errs, c.msg) {
			t.Errorf("%v: code %d, output %q %q", c.args, code, out, errs)
		}
	}
	// Without entrypoints (the agentflow command), replay and fork explain.
	bare := cli.Config{Name: "agentflow", Store: h.store}
	var out, errs bytes.Buffer
	if code := cli.Run(context.Background(), []string{"replay", id}, &out, &errs, bare); code != cli.ExitUsage || !strings.Contains(errs.String(), "needs your agents' code") {
		t.Fatalf("bare replay: %d %s", code, errs.String())
	}
	out.Reset()
	cli.Run(context.Background(), []string{"help"}, &out, &errs, bare)
	if strings.Contains(out.String(), "replay <id>") {
		t.Fatal("bare usage lists replay")
	}
}
