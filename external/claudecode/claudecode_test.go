package claudecode_test

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow/external/claudecode"
	"github.com/kumarlokesh/agentflow/inspect"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

const session = "5e55a0b1-0000-4000-8000-000000000001"

// fixture writes a session in Claude Code's transcript shape: the main
// transcript and three sub-agents' (one missing on purpose).
type fixture struct {
	t   time.Time
	out []string
}

func (f *fixture) add(v map[string]any) {
	f.t = f.t.Add(time.Second)
	v["timestamp"] = f.t.Format(time.RFC3339Nano)
	v["sessionId"] = session
	v["version"] = "2.9.0"
	b, _ := json.Marshal(v)
	f.out = append(f.out, string(b))
}

func user(content any, extra ...map[string]any) map[string]any {
	v := map[string]any{"type": "user", "message": map[string]any{"role": "user", "content": content}}
	for _, e := range extra {
		for k, x := range e {
			v[k] = x
		}
	}
	return v
}

func asst(id, model string, stop any, blocks ...map[string]any) map[string]any {
	return map[string]any{"type": "assistant", "message": map[string]any{"id": id, "role": "assistant", "model": model, "content": blocks,
		"stop_reason": stop, "usage": map[string]any{"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 50}}}
}

func text(s string) map[string]any { return map[string]any{"type": "text", "text": s} }
func toolUse(id, name string, input any) map[string]any {
	return map[string]any{"type": "tool_use", "id": id, "name": name, "input": input}
}
func result(id string, content any, isErr bool) map[string]any {
	return map[string]any{"type": "tool_result", "tool_use_id": id, "content": content, "is_error": isErr}
}

func mainTranscript(secret string) []string {
	f := &fixture{t: time.Date(2026, 11, 26, 10, 0, 0, 0, time.UTC)}
	const m1, m2 = "claude-opus-5-5", "claude-sonnet-5-5"
	f.add(map[string]any{"type": "file-history-snapshot", "snapshot": map[string]any{}})
	f.add(user("Fix the flaky test. My key is " + secret))
	// One response, split over four lines: thinking, text, two tool calls.
	// Claude Code runs Read as soon as it streams: its result arrives
	// between the lines of the response.
	f.add(asst("msg_1", m1, nil, map[string]any{"type": "thinking", "thinking": "Let me look.", "signature": "sig1"}))
	f.add(asst("msg_1", m1, nil, text("Looking at the test.")))
	f.add(asst("msg_1", m1, nil, toolUse("t2", "Read", map[string]any{"file_path": "x_test.go"})))
	f.add(user([]any{result("t2", "package x", false)}, map[string]any{"toolUseResult": map[string]any{"type": "text"}}))
	f.add(asst("msg_1", m1, "tool_use", toolUse("t1", "Bash", map[string]any{"command": "go test ./..."})))
	// The test run fails.
	f.add(user([]any{result("t1", "FAIL: TestX (timeout)", true)}))
	f.add(map[string]any{"type": "assistant", "isSidechain": true, "message": map[string]any{"id": "old", "content": []any{text("inline sidechain, ignored")}}})
	f.add(map[string]any{"type": "system", "subtype": "api_error", "content": "overloaded"})
	f.add(asst("msg_2", m1, "tool_use", toolUse("t3", "Write", map[string]any{"file_path": "/etc/hosts"})))
	f.add(user([]any{result("t3", "Permission to use Write has been denied.", true)}, map[string]any{"toolDenialKind": "user"}))
	// Sub-agents: a sync Explore (which has its own sub-agent), a background
	// one, and one whose transcript is missing.
	f.add(asst("msg_3", m1, "tool_use",
		toolUse("t4", "Agent", map[string]any{"description": "find flake", "prompt": "Find why TestX times out", "subagent_type": "Explore"}),
		toolUse("t5", "Agent", map[string]any{"description": "bg", "prompt": "Scan CI logs", "run_in_background": true}),
		toolUse("t6", "Agent", map[string]any{"description": "plan", "prompt": "Plan the fix", "subagent_type": "Plan"})))
	f.add(user([]any{result("t5", []any{text("Async agent launched.")}, false)}, map[string]any{"toolUseResult": map[string]any{"agentId": "ccc", "isAsync": true, "status": "async_launched"}}))
	f.add(user([]any{result("t4", []any{text("TestX waits on a real network call.")}, false)}, map[string]any{"toolUseResult": map[string]any{"agentId": "aaa", "agentType": "Explore", "status": "completed"}}))
	f.add(user([]any{result("t6", []any{text("1. mock the client")}, false)}, map[string]any{"toolUseResult": map[string]any{"agentId": "ddd", "agentType": "Plan", "status": "completed", "content": []any{text("1. mock the client")}}}))
	f.add(map[string]any{"type": "system", "subtype": "compact_boundary"})
	f.add(user("Summary: TestX makes a real network call; plan is to mock it.", map[string]any{"isCompactSummary": true}))
	f.add(map[string]any{"type": "attachment", "attachment": map[string]any{}})
	f.add(user("<command>ignored</command>", map[string]any{"isMeta": true}))
	f.add(user("Go ahead and mock it."))
	f.add(asst("msg_4", m2, "end_turn", text("Done: the client is mocked.")))
	return f.out
}

func subTranscript(id string, lines ...map[string]any) (string, []string) {
	f := &fixture{t: time.Date(2026, 11, 26, 10, 0, 30, 0, time.UTC)}
	for _, l := range lines {
		l["isSidechain"] = true
		l["agentId"] = id
		f.add(l)
	}
	return id, f.out
}

func writeSession(t *testing.T, dir string, main []string) string {
	t.Helper()
	path := filepath.Join(dir, session+".jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(main, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, session, "subagents")
	if err := os.MkdirAll(sub, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, s := range [][]map[string]any{
		{user("Find why TestX times out"),
			asst("s1", "claude-haiku-5-5", "tool_use", toolUse("g1", "Grep", map[string]any{"pattern": "http.Get"})),
			user([]any{result("g1", "x_test.go:12: http.Get(url)", false)}),
			asst("s2", "claude-haiku-5-5", "tool_use", toolUse("g2", "Agent", map[string]any{"prompt": "Check the URL's host"})),
			user([]any{result("g2", []any{text("it's a public host")}, false)}, map[string]any{"toolUseResult": map[string]any{"agentId": "bbb", "status": "completed"}}),
			asst("s3", "claude-haiku-5-5", "end_turn", text("TestX waits on a real network call."))},
	} {
		id, lines := subTranscript("aaa", s...)
		write(t, filepath.Join(sub, "agent-"+id+".jsonl"), lines)
	}
	_, nested := subTranscript("bbb", user("Check the URL's host"), asst("n1", "claude-haiku-5-5", "end_turn", text("it's a public host")))
	write(t, filepath.Join(sub, "agent-bbb.jsonl"), nested)
	_, bg := subTranscript("ccc", user("Scan CI logs"), asst("c1", "claude-haiku-5-5", nil, text("CI shows the same timeout since Monday.")))
	write(t, filepath.Join(sub, "agent-ccc.jsonl"), bg)
	write(t, filepath.Join(sub, "agent-ccc.meta.json"), []string{`{"agentType":"general-purpose","toolUseId":"t5","spawnDepth":1}`})
	return path
}

func write(t *testing.T, path string, lines []string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func byPath(s *claudecode.Session) map[string]*claudecode.Run {
	out := map[string]*claudecode.Run{}
	for _, r := range s.Runs {
		out[r.Path] = r
	}
	return out
}

func kinds(evs []trace.Event) string {
	var ks []string
	for _, e := range evs {
		ks = append(ks, string(e.Kind()))
	}
	return strings.Join(ks, " ")
}

// checkRules checks the spec's ordering rules (§5) on one run.
func checkRules(t *testing.T, path string, evs []trace.Event) {
	t.Helper()
	open := map[uint64]bool{}
	for i, e := range evs {
		if e.Seq != uint64(i) {
			t.Fatalf("%s: seq %d at %d", path, e.Seq, i)
		}
		if _, err := trace.EncodeEvent(e, trace.EncodeOptions{}); err != nil {
			t.Fatalf("%s seq %d: %v", path, i, err)
		}
		switch e.Kind() {
		case trace.KindLLMStarted, trace.KindToolStarted, trace.KindEffectStarted, trace.KindRunSpawned:
			open[e.Seq] = true
		case trace.KindLLMCompleted, trace.KindLLMFailed, trace.KindToolCompleted, trace.KindToolFailed, trace.KindRunJoined:
			if e.Cause == nil || !open[*e.Cause] {
				t.Fatalf("%s seq %d: %s answers nothing open", path, i, e.Kind())
			}
			delete(open, *e.Cause)
		case trace.KindRunFinished:
			if len(open) > 0 || i != len(evs)-1 {
				t.Fatalf("%s: finished with open %v", path, open)
			}
		}
	}
	if _, err := trace.Conversation(evs); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func TestConvertSession(t *testing.T) {
	path := writeSession(t, t.TempDir(), mainTranscript("sk-ant-api03-"+strings.Repeat("x", 40)))
	s, err := claudecode.Convert(path, claudecode.Options{Final: true})
	if err != nil {
		t.Fatal(err)
	}
	runs := byPath(s)
	if s.ID != session || len(s.Runs) != 4 || runs["/"] == nil || runs["/Explore#0"] == nil || runs["/Explore#0/general-purpose#0"] == nil || runs["/general-purpose#0"] == nil {
		var ps []string
		for _, r := range s.Runs {
			ps = append(ps, r.Path)
		}
		t.Fatalf("runs %v", ps)
	}
	for _, r := range s.Runs {
		checkRules(t, r.Path, r.Events)
	}
	root := runs["/"].Events
	rs := root[0].Body.(*trace.RunStarted)
	if rs.Origin == nil || rs.Origin.Producer != "claude-code" || rs.Agent.Name != "claude-code" || rs.Agent.Version != "2.9.0" || rs.Config.Model != "claude-opus-5-5" {
		t.Fatalf("run.started %+v", rs)
	}
	want := "run.started message.appended " +
		"llm.started llm.completed message.appended tool.started tool.started tool.completed tool.completed message.appended " +
		"llm.started llm.completed message.appended tool.started tool.failed message.appended " +
		"llm.started llm.completed message.appended tool.started tool.started tool.started " +
		"run.spawned run.joined tool.completed run.spawned run.joined tool.completed run.spawned run.joined tool.completed message.appended " +
		"conversation.replaced message.appended config.changed llm.started llm.completed message.appended run.finished"
	if got := kinds(root); got != want {
		t.Fatalf("root events:\n%s\nwant\n%s", got, want)
	}
	var calls []string
	for _, e := range root {
		switch b := e.Body.(type) {
		case *trace.LLMCompleted:
			if b.Step == 0 {
				if len(b.Message.Blocks) != 4 || b.Usage.CacheReadTokens != 50 {
					t.Fatalf("merged response %+v", b)
				}
			}
			if b.Step == 1 && b.Attempts != 2 {
				t.Fatalf("retried response attempts %d", b.Attempts)
			}
		case *trace.ToolCompleted:
			calls = append(calls, fmt.Sprintf("%s:%v", b.CallID, b.IsError))
		case *trace.ToolFailed:
			if b.CallID != "t3" || b.Error.Code != trace.ErrCodeDenied || !b.Intercepted {
				t.Fatalf("denial %+v", b)
			}
		case *trace.RunJoined:
			calls = append(calls, fmt.Sprintf("joined:%s", b.Status))
		case *trace.RunSpawned:
			calls = append(calls, b.Path)
		}
	}
	wantCalls := "t2:false t1:true /Explore#0 joined:completed t4:false /general-purpose#0 joined:completed t5:false /Plan#0 joined:completed t6:false"
	if got := strings.Join(calls, " "); got != wantCalls {
		t.Fatalf("calls %s\nwant  %s", got, wantCalls)
	}
	// The background agent's join carries its final answer, not "launched".
	explore := runs["/Explore#0"].Events
	if explore[0].Body.(*trace.RunStarted).Parent.RunID != runs["/"].RunID || explore[len(explore)-1].Body.(*trace.RunFinished).Status != trace.StatusCompleted {
		t.Fatal("Explore run not linked or not finished")
	}
	for _, e := range root {
		if j, ok := e.Body.(*trace.RunJoined); ok && j.ChildRunID == runs["/general-purpose#0"].RunID && !strings.Contains(fmt.Sprint(j.Output), "same timeout") {
			t.Fatalf("background join %+v", j)
		}
	}
	if runs["/"].RunID != claudecode.DerivedID(session, "/") {
		t.Fatal("run IDs aren't derived")
	}
}

func TestIncrementalImportEqualsOneImport(t *testing.T) {
	ctx := context.Background()
	lines := mainTranscript("nothing secret")
	dir := t.TempDir()
	path := writeSession(t, dir, lines)
	once := memstore.New()
	if _, err := claudecode.Import(ctx, once, path, claudecode.ImportOptions{Options: claudecode.Options{Final: true}}); err != nil {
		t.Fatal(err)
	}
	step := memstore.New()
	for k := 1; k <= len(lines); k++ {
		write(t, path, lines[:k])
		if _, err := claudecode.Import(ctx, step, path, claudecode.ImportOptions{LockDir: filepath.Join(dir, "locks")}); err != nil {
			t.Fatalf("prefix %d: %v", k, err)
		}
	}
	if _, err := claudecode.Import(ctx, step, path, claudecode.ImportOptions{Options: claudecode.Options{Final: true}}); err != nil {
		t.Fatal(err)
	}
	a, b := dump(t, once), dump(t, step)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("incremental import differs:\n%v\nvs\n%v", a, b)
	}
	// Importing again changes nothing.
	if _, err := claudecode.Import(ctx, step, path, claudecode.ImportOptions{Options: claudecode.Options{Final: true}}); err != nil || !reflect.DeepEqual(dump(t, step), a) {
		t.Fatalf("re-import: %v", err)
	}
	// A rewritten transcript is refused.
	rewritten := append([]string(nil), lines...)
	rewritten[1] = strings.Replace(rewritten[1], "Fix the flaky test", "Delete the flaky test", 1)
	rewritten = append(rewritten[:3], rewritten[5:]...)
	write(t, path, rewritten)
	if _, err := claudecode.Import(ctx, step, path, claudecode.ImportOptions{Options: claudecode.Options{Final: true}}); !errors.Is(err, claudecode.ErrRewritten) {
		t.Fatalf("rewritten: %v", err)
	}
}

func dump(t *testing.T, s trace.Store) map[string][]string {
	t.Helper()
	out := map[string][]string{}
	for info, err := range s.Runs(context.Background(), trace.RunFilter{}) {
		if err != nil {
			t.Fatal(err)
		}
		evs, _ := trace.Collect(s.Events(context.Background(), info.RunID, 0))
		for _, e := range evs {
			b, _ := trace.EncodeEvent(e, trace.EncodeOptions{})
			out[info.Path] = append(out[info.Path], string(b))
		}
	}
	return out
}

func TestImportedTracesAreForReading(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	key := "sk-ant-api03-" + strings.Repeat("Q", 40)
	path := writeSession(t, dir, mainTranscript(key))
	store, err := filestore.Open(filepath.Join(dir, "store"), filestore.Options{Redact: trace.SecretPatterns})
	if err != nil {
		t.Fatal(err)
	}
	id, err := claudecode.Import(ctx, store, path, claudecode.ImportOptions{Options: claudecode.Options{Final: true}})
	if err != nil || id != session {
		t.Fatalf("%v %s", err, id)
	}
	_ = store.Close()
	// The pasted key never reaches the files.
	err = filepath.Walk(filepath.Join(dir, "store"), func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		f, _ := os.Open(p)
		defer func() { _ = f.Close() }()
		sc := bufio.NewScanner(f)
		sc.Buffer(nil, 1<<24)
		for sc.Scan() {
			if strings.Contains(sc.Text(), key) {
				t.Errorf("%s holds the key", p)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	store, _ = filestore.Open(filepath.Join(dir, "store"), filestore.Options{})
	defer func() { _ = store.Close() }()

	// Replay and fork refuse.
	if _, err := replay.Run(ctx, nil, store, id, replay.Options{}); !errors.Is(err, replay.ErrExternalTrace) {
		t.Fatalf("replay: %v", err)
	}
	if _, err := replay.Fork(ctx, nil, store, id, replay.ForkOptions{}); !errors.Is(err, replay.ErrExternalTrace) {
		t.Fatalf("fork: %v", err)
	}

	// Inspection works: the tree, blame at the failed test run, HTML.
	tr, err := inspect.Load(ctx, store, id)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	if err := inspect.RenderTree(&sb, tr); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"claude-code", "├─ /Explore#0", "│  └─ /Explore#0/general-purpose#0", "tool:Bash!", "tool:Write✗", "tool:Agent→Explore#0", "tool:Agent→Plan#0"} {
		if !strings.Contains(sb.String(), want) {
			t.Fatalf("tree lacks %q:\n%s", want, sb.String())
		}
	}
	var denial *inspect.Anomaly
	for _, a := range tr.Anomalies {
		if a.Kind == inspect.Denial {
			denial = a
		}
	}
	if denial == nil {
		t.Fatal("no denial anomaly")
	}
	b, err := inspect.Blame(tr, fmt.Sprintf("/@%d", denial.Seq))
	if err != nil || len(b.RootCauses) != 1 || b.RootCauses[0].Anomaly.Effect != "tool:Bash(t1)" {
		t.Fatalf("%v %s", err, b)
	}
	if err := inspect.WriteHTML(&sb, tr, b); err != nil {
		t.Fatal(err)
	}
}

func TestDiffTwoSessions(t *testing.T) {
	ctx := context.Background()
	store := memstore.New()
	for i, task := range []string{"Fix the flaky test.", "Fix the flaky test, please."} {
		dir := t.TempDir()
		lines := mainTranscript("x")
		lines[1] = strings.Replace(lines[1], "Fix the flaky test.", task, 1)
		path := writeSession(t, dir, lines)
		// Different sessions: give the second its own ID.
		if i == 1 {
			data, _ := os.ReadFile(path)
			other := strings.ReplaceAll(string(data), session, "5e55a0b1-0000-4000-8000-000000000002")
			path = filepath.Join(dir, "5e55a0b1-0000-4000-8000-000000000002.jsonl")
			_ = os.WriteFile(path, []byte(other), 0o600)
			_ = os.Rename(filepath.Join(dir, session), filepath.Join(dir, "5e55a0b1-0000-4000-8000-000000000002"))
		}
		if _, err := claudecode.Import(ctx, store, path, claudecode.ImportOptions{Options: claudecode.Options{Final: true}}); err != nil {
			t.Fatal(err)
		}
	}
	d, err := inspect.Diff(ctx, store, session, "5e55a0b1-0000-4000-8000-000000000002", inspect.DiffOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if d.Identical() || d.Runs[0].Path != "/" || !strings.Contains(d.String(), "Fix the flaky test, please.") {
		t.Fatalf("%s", d)
	}
	for _, r := range d.Runs {
		if r.Path != "/" {
			t.Fatalf("only the root differs: %s", d)
		}
	}
}

// The linear-time fingerprints equal trace.RequestFingerprint of the
// request rebuilt from the trace (spec §9, §10), across a compaction and a
// model change.
func TestFingerprintsMatchTheSpec(t *testing.T) {
	path := writeSession(t, t.TempDir(), mainTranscript("x"))
	s, err := claudecode.Convert(path, claudecode.Options{Final: true})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, r := range s.Runs {
		f := trace.NewFolder()
		for _, e := range r.Events {
			if st, ok := e.Body.(*trace.LLMStarted); ok {
				req, err := f.Request(st)
				if err != nil {
					t.Fatal(err)
				}
				want, _ := trace.RequestFingerprint(req)
				if st.Fingerprint != want {
					t.Fatalf("%s seq %d: fingerprint %s, want %s", r.Path, e.Seq, st.Fingerprint, want)
				}
				n++
			}
			if err := f.Apply(e); err != nil {
				t.Fatal(err)
			}
		}
	}
	if n < 8 {
		t.Fatalf("checked %d calls", n)
	}
}
