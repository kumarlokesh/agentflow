package trace_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/filestore"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

var update = flag.Bool("update", false, "rewrite golden files")

const (
	traceID = "0193c4a0-0000-7000-8000-000000000001"
	rootID  = "0193c4a0-0000-7000-8000-000000000002"
	childID = "0193c4a0-0000-7000-8000-000000000003"
)

var t0 = time.Date(2026, 11, 22, 10, 0, 0, 0, time.UTC)

func u64(v uint64) *uint64 { return &v }
func intp(v int) *int      { return &v }
func f64(v float64) *float64 {
	return &v
}

// sampleEvents has one event of every kind, with every optional field set
// somewhere, in a valid seq order for one run.
func sampleEvents() []trace.Event {
	calcTool := llm.ToolSpec{
		Name:        "calculator",
		Description: "Evaluates an arithmetic expression.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
		Annotations: &llm.ToolAnnotations{ReadOnly: true, Idempotent: true},
	}
	cfg := &trace.Config{
		Model:      "claude-opus-5-5",
		System:     llm.Blocks{llm.Text{Text: "You are a calculator. Use <tools> & show work."}},
		Tools:      []llm.ToolSpec{calcTool},
		ToolChoice: &llm.ToolChoice{Mode: llm.ToolChoiceAuto},
		Params:     llm.Params{MaxTokens: 1024, Temperature: f64(0.2)},
		MaxSteps:   10,
	}
	empty := []trace.Access{}
	accesses := []trace.Access{{Resource: "db:orders/42", Mode: trace.AccessRead}}
	model := "claude-sonnet-5-5"
	bodies := []trace.Body{
		&trace.RunStarted{
			RunKind: trace.RunKindAgent, Path: "/", Agent: trace.AgentInfo{Name: "calculator", Version: "1.2.0"},
			Task: llm.Blocks{llm.Text{Text: "What is 6*7?"}}, Config: cfg, ConfigFP: "sha256:9c1e",
			Origin: &trace.Origin{Kind: trace.OriginFork, SourceTraceID: "0193b000-0000-7000-8000-000000000009",
				At:        &trace.ForkPoint{Path: "/", Effect: "llm#0", Seq: u64(2)},
				Overrides: json.RawMessage(`[{"type":"set_config","model":"claude-sonnet-5-5"}]`),
				Group:     "0193d1aa-0000-7000-8000-000000000000", Sample: intp(0), Arm: "treatment"},
		},
		&trace.MessageAppended{Message: &llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "What is 6*7?"}}}},
		&trace.LLMStarted{Step: 0, Context: trace.ContextRef{Messages: 1, ConfigSeq: 0},
			Ephemeral:   []trace.Ephemeral{{Index: 0, Message: llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "Memory: the user likes short answers."}}}}},
			Fingerprint: "sha256:41aa"},
		&trace.LLMCompleted{Step: 0, Message: llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{
			llm.Reasoning{Text: "Multiply.", Ext: llm.Ext{"anthropic": json.RawMessage(`{"signature":"EqQBCkYIBxgC"}`)}},
			llm.ToolUse{CallID: "toolu_01", Name: "calculator", Input: json.RawMessage(`{"expression":"6*7"}`)},
		}}, StopReason: llm.StopToolUse, Usage: llm.Usage{InputTokens: 112, OutputTokens: 21, CacheReadTokens: 64},
			Model: "claude-opus-5-5", Attempts: 2, Duration: 812 * time.Millisecond, Ext: llm.Ext{"anthropic": json.RawMessage(`{"request_id":"req_1"}`)}},
		&trace.MessageAppended{SourceSeq: u64(3)},
		&trace.ToolStarted{Step: 0, CallID: "toolu_01", Tool: "calculator", Input: json.RawMessage(`{"expression":"6*7"}`), Fingerprint: "sha256:07be", Accesses: &accesses},
		&trace.ToolCompleted{CallID: "toolu_01", Content: llm.Blocks{llm.Text{Text: "42"}, llm.Data{Value: json.RawMessage(`{"value":42}`)}}, Duration: 41 * time.Microsecond},
		&trace.ToolStarted{Step: 0, CallID: "toolu_02", Tool: "web_fetch", Input: json.RawMessage(`{"url":"https://example.com"}`), Fingerprint: "sha256:5a5a", Accesses: &empty},
		&trace.ToolFailed{CallID: "toolu_02", Error: trace.Error{Code: trace.ErrCodeDenied, Message: "destructive tool disabled in fork"}, Duration: time.Millisecond},
		&trace.MessageAppended{Message: &llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{
			llm.ToolResult{CallID: "toolu_01", Content: llm.Blocks{llm.Text{Text: "42"}}},
			llm.ToolResult{CallID: "toolu_02", Content: llm.Blocks{llm.Text{Text: "denied"}}, IsError: true},
		}}},
		&trace.EffectStarted{Name: "context.memory", Input: json.RawMessage(`{"query":"preferences"}`), Fingerprint: "sha256:e1"},
		&trace.EffectCompleted{Name: "context.memory", Output: json.RawMessage(`["likes short answers"]`), Duration: 3 * time.Millisecond, Override: true},
		&trace.EffectStarted{Name: "clock.now", Input: json.RawMessage(`null`), Fingerprint: "sha256:e2"},
		&trace.EffectFailed{Name: "clock.now", Error: trace.Error{Code: trace.ErrCodePanic, Message: "boom", Detail: "goroutine 1 [running]"}, Duration: time.Microsecond},
		&trace.ConfigChanged{Model: &model, Params: &llm.Params{MaxTokens: 2048, Reasoning: &llm.ReasoningParams{Effort: "high"}}},
		&trace.LLMStarted{Step: 1, Context: trace.ContextRef{Messages: 3, ConfigSeq: 14}, Fingerprint: "sha256:d2f0"},
		&trace.LLMFailed{Step: 1, Error: trace.Error{Code: llm.ErrCodeRateLimited, Message: "slow down", Retryable: true}, Attempts: 3, Duration: 2 * time.Second},
		&trace.RunSpawned{ChildRunID: childID, Path: "/researcher#0", Agent: "researcher", Task: llm.Blocks{llm.Text{Text: "Find the 2025 revenue"}}, Fingerprint: "sha256:5p"},
		&trace.RunJoined{ChildRunID: childID, Status: trace.StatusCompleted, Output: llm.Blocks{llm.Text{Text: "$4.2B"}}, Usage: llm.Usage{InputTokens: 900, OutputTokens: 80}, Duration: 5 * time.Second},
		&trace.RunModeChanged{Mode: "live", Reason: trace.ReasonDivergence, Detail: json.RawMessage(`{"kind":"request_mismatch","effect":"llm#2"}`)},
		&trace.ConversationReplaced{Messages: []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "Summary so far: 6*7=42."}}}}},
		&trace.Annotation{Key: "agentflow.verdict", Value: json.RawMessage(`{"pass":true,"note":"correct"}`)},
		&trace.RunFinished{Status: trace.StatusCompleted, Output: llm.Blocks{llm.Text{Text: "6 × 7 = 42"}}, Usage: llm.Usage{InputTokens: 252, OutputTokens: 30}, Steps: 2},
	}
	events := make([]trace.Event, len(bodies))
	for i, b := range bodies {
		events[i] = trace.Event{TraceID: traceID, RunID: rootID, Seq: uint64(i), Time: t0.Add(time.Duration(i) * time.Millisecond), Body: b}
	}
	events[3].Cause = u64(2)
	events[5].Cause = u64(3)
	events[6].Cause = u64(5)
	return events
}

// TestGoldenEvents pins the exact JSON encoding of every event kind. Run with
// -update to rewrite testdata/golden.jsonl after an intentional change.
func TestGoldenEvents(t *testing.T) {
	events := sampleEvents()
	var buf bytes.Buffer
	kinds := map[trace.Kind]bool{}
	for _, e := range events {
		line, err := trace.EncodeEvent(e, trace.EncodeOptions{})
		if err != nil {
			t.Fatal(err)
		}
		buf.Write(line)
		buf.WriteByte('\n')
		kinds[e.Kind()] = true
	}
	all := []trace.Kind{
		trace.KindRunStarted, trace.KindRunFinished, trace.KindRunSpawned, trace.KindRunJoined, trace.KindRunModeChanged,
		trace.KindMessageAppended, trace.KindConversationReplaced, trace.KindConfigChanged,
		trace.KindLLMStarted, trace.KindLLMCompleted, trace.KindLLMFailed,
		trace.KindToolStarted, trace.KindToolCompleted, trace.KindToolFailed,
		trace.KindEffectStarted, trace.KindEffectCompleted, trace.KindEffectFailed, trace.KindAnnotation,
	}
	for _, k := range all {
		if !kinds[k] {
			t.Errorf("sampleEvents has no %s event", k)
		}
	}

	golden := filepath.Join("testdata", "golden.jsonl")
	if *update {
		if err := os.WriteFile(golden, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("%v (run with -update to create it)", err)
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("encoding changed; diff against %s (run with -update if intended)\ngot:\n%s", golden, buf.String())
	}

	// decode → encode is byte-identical.
	for i, line := range bytes.Split(bytes.TrimSpace(want), []byte("\n")) {
		e, err := trace.DecodeEvent(line, nil)
		if err != nil {
			t.Fatalf("line %d: %v", i, err)
		}
		again, err := trace.EncodeEvent(e, trace.EncodeOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(line, again) {
			t.Errorf("line %d not stable\nin  %s\nout %s", i, line, again)
		}
	}
}

func TestEncodingIsReadable(t *testing.T) {
	line, err := trace.EncodeEvent(sampleEvents()[0], trace.EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(line, []byte("<tools> & show work")) {
		t.Errorf("HTML characters should not be escaped in trace files:\n%s", line)
	}
	if !bytes.Contains(line, []byte(`"time":"2026-11-22T10:00:00.000000Z"`)) {
		t.Errorf("time should be UTC with microseconds:\n%s", line)
	}
}

func TestDecodeErrors(t *testing.T) {
	prototype := `{"id":"x","schema_version":1,"type":"run_start","run_id":"r","step_index":0,"timestamp":"2026-03-01T00:00:00Z","data":{}}`
	if _, err := trace.DecodeEvent([]byte(prototype), nil); !errors.Is(err, trace.ErrPrototypeTrace) {
		t.Errorf("prototype event: got %v, want ErrPrototypeTrace", err)
	}
	future := `{"v":2,"trace_id":"t","run_id":"r","seq":0,"time":"2026-11-22T10:00:00.000000Z","kind":"run.started","body":{}}`
	if _, err := trace.DecodeEvent([]byte(future), nil); !errors.Is(err, trace.ErrUnsupportedVersion) {
		t.Errorf("v2 event: got %v, want ErrUnsupportedVersion", err)
	}
	missingBlob := `{"v":1,"trace_id":"t","run_id":"r","seq":1,"time":"2026-11-22T10:00:00.000000Z","kind":"annotation","body":{"key":"k","value":{"$blob":"sha256:00","size":3}}}`
	if _, err := trace.DecodeEvent([]byte(missingBlob), nil); !errors.Is(err, trace.ErrBlobNotFound) {
		t.Errorf("missing blob: got %v, want ErrBlobNotFound", err)
	}
}

func TestUnknownKindPreserved(t *testing.T) {
	line := `{"v":1,"trace_id":"t","run_id":"r","seq":7,"time":"2026-11-22T10:00:00.000000Z","kind":"hologram.projected","cause":6,"body":{"frames":[1,2,3]}}`
	e, err := trace.DecodeEvent([]byte(line), nil)
	if err != nil {
		t.Fatal(err)
	}
	if u, ok := e.Body.(trace.UnknownBody); !ok || u.Kind() != "hologram.projected" {
		t.Fatalf("body = %#v", e.Body)
	}
	out, err := trace.EncodeEvent(e, trace.EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != line {
		t.Errorf("unknown kind not preserved\nin  %s\nout %s", line, out)
	}
}

func TestRedactedMarkerDecodesToPlaceholder(t *testing.T) {
	line := `{"v":1,"trace_id":"t","run_id":"r","seq":1,"time":"2026-11-22T10:00:00.000000Z","kind":"message.appended","body":{"message":{"role":"user","blocks":[{"type":"text","text":{"$redacted":"api-key"}}]}}}`
	e, err := trace.DecodeEvent([]byte(line), nil)
	if err != nil {
		t.Fatal(err)
	}
	got := llm.TextOf(e.Body.(*trace.MessageAppended).Message.Blocks)
	if got != "[redacted: api-key]" {
		t.Errorf("text = %q", got)
	}
}

func TestBlobThresholdAndEventSize(t *testing.T) {
	blobs := map[trace.BlobRef][]byte{}
	put := func(b []byte) (trace.BlobRef, error) { r := trace.BlobRefOf(b); blobs[r] = b; return r, nil }
	get := func(r trace.BlobRef) ([]byte, error) { return blobs[r], nil }
	at := strings.Repeat("a", 100)
	over := strings.Repeat("b", 101)
	e := trace.Event{TraceID: "t", RunID: "r", Seq: 1, Time: t0, Body: &trace.Annotation{Key: "k",
		Value: json.RawMessage(`{"at":"` + at + `","over":"` + over + `"}`)}}
	line, err := trace.EncodeEvent(e, trace.EncodeOptions{BlobThreshold: 100, PutBlob: put})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(line, []byte(at)) || bytes.Contains(line, []byte(over)) || len(blobs) != 1 {
		t.Errorf("threshold should keep 100-byte strings inline and move 101-byte ones:\n%s", line)
	}
	back, err := trace.DecodeEvent(line, get)
	if err != nil {
		t.Fatal(err)
	}
	if string(back.Body.(*trace.Annotation).Value) != string(e.Body.(*trace.Annotation).Value) {
		t.Errorf("round trip lost data: %s", back.Body.(*trace.Annotation).Value)
	}

	huge := trace.Event{TraceID: "t", RunID: "r", Seq: 1, Time: t0, Body: &trace.Annotation{Key: "k",
		Value: json.RawMessage(`"` + strings.Repeat("z", trace.MaxEventSize) + `"`)}}
	if _, err := trace.EncodeEvent(huge, trace.EncodeOptions{}); !errors.Is(err, trace.ErrEventTooLarge) {
		t.Errorf("oversized event: got %v, want ErrEventTooLarge", err)
	}
	if _, err := trace.EncodeEvent(huge, trace.EncodeOptions{BlobThreshold: 1024, PutBlob: put}); err != nil {
		t.Errorf("oversized string should fit once externalized: %v", err)
	}
}

// calculatorRun records RFC-0001 §13's example through a Writer and returns
// the events plus the two requests that were "sent".
func calculatorRun(t *testing.T, s trace.Store) ([]trace.Event, []*llm.Request) {
	t.Helper()
	ctx := context.Background()
	tool := llm.ToolSpec{Name: "calculator", Description: "Evaluates arithmetic.", InputSchema: json.RawMessage(`{"type":"object"}`)}
	cfg := &trace.Config{Model: "claude-opus-5-5", System: llm.Blocks{llm.Text{Text: "You are a calculator."}},
		Tools: []llm.ToolSpec{tool}, Params: llm.Params{MaxTokens: 1024}, MaxSteps: 10}
	fp, err := trace.ConfigFingerprint(cfg)
	if err != nil {
		t.Fatal(err)
	}
	clock := t0
	w := trace.NewWriter(s, traceID, rootID, trace.WriterOptions{Now: func() time.Time { clock = clock.Add(time.Millisecond); return clock }})
	must := func(seq uint64, err error) uint64 {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return seq
	}
	task := llm.Blocks{llm.Text{Text: "What is 6*7?"}}
	must(w.Write(ctx, &trace.RunStarted{RunKind: trace.RunKindAgent, Path: "/", Agent: trace.AgentInfo{Name: "calculator"}, Task: task, Config: cfg, ConfigFP: fp, Origin: &trace.Origin{Kind: trace.OriginLive}}, nil))
	user := llm.Message{Role: llm.RoleUser, Blocks: task}
	must(w.Write(ctx, &trace.MessageAppended{Message: &user}, nil))

	memory := llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "Memory: user prefers digits."}}}
	req1 := &llm.Request{Model: cfg.Model, System: cfg.System, Messages: []llm.Message{memory, user}, Tools: cfg.Tools, Params: cfg.Params}
	fp1, err := trace.RequestFingerprint(req1)
	if err != nil {
		t.Fatal(err)
	}
	s1 := must(w.Write(ctx, &trace.LLMStarted{Step: 0, Context: trace.ContextRef{Messages: 1, ConfigSeq: 0},
		Ephemeral: []trace.Ephemeral{{Index: 0, Message: memory}}, Fingerprint: fp1}, nil))
	call := llm.ToolUse{CallID: "toolu_01", Name: "calculator", Input: json.RawMessage(`{"expression":"6*7"}`)}
	assistant1 := llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{call}}
	c1 := must(w.Write(ctx, &trace.LLMCompleted{Step: 0, Message: assistant1, StopReason: llm.StopToolUse}, trace.Cause(s1)))
	must(w.Write(ctx, &trace.MessageAppended{SourceSeq: trace.Cause(c1)}, nil))
	tfp, _ := trace.ToolFingerprint(call.Name, call.Input)
	ts := must(w.Write(ctx, &trace.ToolStarted{Step: 0, CallID: call.CallID, Tool: call.Name, Input: call.Input, Fingerprint: tfp}, trace.Cause(c1)))
	result := llm.Blocks{llm.Text{Text: "42"}}
	must(w.Write(ctx, &trace.ToolCompleted{CallID: call.CallID, Content: result}, trace.Cause(ts)))
	toolMsg := llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.ToolResult{CallID: call.CallID, Content: result}}}
	must(w.Write(ctx, &trace.MessageAppended{Message: &toolMsg}, nil))

	req2 := &llm.Request{Model: cfg.Model, System: cfg.System, Messages: []llm.Message{user, assistant1, toolMsg}, Tools: cfg.Tools, Params: cfg.Params}
	fp2, _ := trace.RequestFingerprint(req2)
	s2 := must(w.Write(ctx, &trace.LLMStarted{Step: 1, Context: trace.ContextRef{Messages: 3, ConfigSeq: 0}, Fingerprint: fp2}, nil))
	final := llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{llm.Text{Text: "6 × 7 = 42"}}}
	c2 := must(w.Write(ctx, &trace.LLMCompleted{Step: 1, Message: final, StopReason: llm.StopEndTurn}, trace.Cause(s2)))
	must(w.Write(ctx, &trace.MessageAppended{SourceSeq: trace.Cause(c2)}, nil))
	must(w.Write(ctx, &trace.RunFinished{Status: trace.StatusCompleted, Output: final.Blocks, Steps: 2}, nil))

	events, err := trace.Collect(s.Events(ctx, rootID, 0))
	if err != nil {
		t.Fatal(err)
	}
	return events, []*llm.Request{req1, req2}
}

func TestFoldReconstructsExactRequests(t *testing.T) {
	for name, s := range map[string]trace.Store{"memstore": memstore.New(), "filestore": newFileStore(t, 8)} {
		t.Run(name, func(t *testing.T) {
			events, sent := calculatorRun(t, s)
			var starts []uint64
			for _, e := range events {
				if _, ok := e.Body.(*trace.LLMStarted); ok {
					starts = append(starts, e.Seq)
				}
			}
			if len(starts) != 2 {
				t.Fatalf("got %d llm.started events", len(starts))
			}
			for i, seq := range starts {
				got, err := trace.Request(events, seq)
				if err != nil {
					t.Fatal(err)
				}
				wantJSON, _ := json.Marshal(sent[i])
				gotJSON, _ := json.Marshal(got)
				if string(wantJSON) != string(gotJSON) {
					t.Fatalf("request %d differs\nsent  %s\nfolded %s", i, wantJSON, gotJSON)
				}
				fp, err := trace.RequestFingerprint(got)
				if err != nil {
					t.Fatal(err)
				}
				if recorded := events[seq].Body.(*trace.LLMStarted).Fingerprint; fp != recorded {
					t.Errorf("request %d: fingerprint of folded request %s != recorded %s", i, fp, recorded)
				}
			}
			conv, err := trace.Conversation(events)
			if err != nil {
				t.Fatal(err)
			}
			if len(conv) != 4 || llm.TextOf(conv[3].Blocks) != "6 × 7 = 42" {
				t.Errorf("final conversation has %d messages, last %q", len(conv), llm.TextOf(conv[len(conv)-1].Blocks))
			}
		})
	}
}

func TestFoldConfigChangesAndReplacement(t *testing.T) {
	events := sampleEvents()
	// The second llm.started (seq 15) references config_seq 14 (config.changed).
	req, err := trace.Request(events, 15)
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "claude-sonnet-5-5" || req.Params.MaxTokens != 2048 || req.Params.Reasoning == nil {
		t.Errorf("config.changed not applied: model %s, params %+v", req.Model, req.Params)
	}
	if llm.TextOf(req.System) != "You are a calculator. Use <tools> & show work." || len(req.Tools) != 1 {
		t.Errorf("unchanged config fields lost: system %q, %d tools", llm.TextOf(req.System), len(req.Tools))
	}
	if len(req.Messages) != 3 {
		t.Errorf("got %d messages, want 3", len(req.Messages))
	}
	conv, err := trace.Conversation(events)
	if err != nil {
		t.Fatal(err)
	}
	if len(conv) != 1 || llm.TextOf(conv[0].Blocks) != "Summary so far: 6*7=42." {
		t.Errorf("conversation.replaced not applied: %d messages", len(conv))
	}
}

func TestFoldErrors(t *testing.T) {
	events := sampleEvents()
	if _, err := trace.Request(events, 1); err == nil {
		t.Error("Request at a non-llm.started seq should fail")
	}
	bad := append([]trace.Event(nil), events[:3]...)
	bad[2] = trace.Event{Seq: 2, Body: &trace.LLMStarted{Context: trace.ContextRef{Messages: 5}}}
	if _, err := trace.Request(bad, 2); err == nil {
		t.Error("context larger than the conversation should fail")
	}
	gap := []trace.Event{events[0], events[2]}
	if _, err := trace.Conversation(gap); err == nil {
		t.Error("a seq gap should fail")
	}
	dangling := []trace.Event{events[0], {Seq: 1, Body: &trace.MessageAppended{SourceSeq: u64(0)}}}
	if _, err := trace.Conversation(dangling); err == nil {
		t.Error("source_seq not pointing at llm.completed should fail")
	}
}

func TestRequestFingerprintProperties(t *testing.T) {
	a := llm.ToolSpec{Name: "a", Description: "A", InputSchema: json.RawMessage(`{"type":"object"}`)}
	b := llm.ToolSpec{Name: "b", Description: "B", InputSchema: json.RawMessage(`{"type": "object"}`)}
	base := &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "hi"}}}}, Tools: []llm.ToolSpec{a, b}}
	fp := func(r *llm.Request) string {
		t.Helper()
		s, err := trace.RequestFingerprint(r)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	ref := fp(base)
	if !strings.HasPrefix(ref, "sha256:") || len(ref) != len("sha256:")+64 {
		t.Fatalf("bad fingerprint %s", ref)
	}

	reordered := *base
	reordered.Tools = []llm.ToolSpec{b, a}
	if fp(&reordered) != ref {
		t.Error("tool order must not change the fingerprint")
	}
	annotated := *base
	ac := a
	ac.Annotations = &llm.ToolAnnotations{Destructive: true}
	annotated.Tools = []llm.ToolSpec{ac, b}
	if fp(&annotated) != ref {
		t.Error("annotations are not sent to the model and must not change the fingerprint")
	}
	changed := *base
	changed.Params.Temperature = f64(0)
	if fp(&changed) == ref {
		t.Error("temperature 0 vs unset must differ")
	}
	text := *base
	text.Messages = []llm.Message{{Role: llm.RoleUser, Blocks: llm.Blocks{llm.Text{Text: "hi!"}}}}
	if fp(&text) == ref {
		t.Error("message text must change the fingerprint")
	}

	t1, _ := trace.ToolFingerprint("calc", json.RawMessage(`{"x":1,"y":2}`))
	t2, _ := trace.ToolFingerprint("calc", json.RawMessage(`{"y":2, "x":1.0}`))
	if t1 != t2 {
		t.Error("tool input fingerprint must be canonical")
	}
	s1, _ := trace.SpawnFingerprint("researcher", llm.Blocks{llm.Text{Text: "go"}}, "")
	s2, _ := trace.SpawnFingerprint("researcher", llm.Blocks{llm.Text{Text: "go"}}, "eu")
	if s1 == s2 {
		t.Error("spawn key must change the fingerprint")
	}
}

func TestWriterAssignsSeqAndNotifiesObservers(t *testing.T) {
	s := memstore.New()
	var seen []uint64
	obs := trace.ObserverFunc(func(_ context.Context, e trace.Event) { seen = append(seen, e.Seq) })
	w := trace.NewWriter(s, traceID, rootID, trace.WriterOptions{Observers: []trace.Observer{obs}})
	ctx := context.Background()
	if _, err := w.Write(ctx, &trace.RunStarted{RunKind: trace.RunKindWorkflow, Path: "/", Agent: trace.AgentInfo{Name: "wf"}}, nil); err != nil {
		t.Fatal(err)
	}
	seqs, err := w.WriteBatch(ctx, []trace.Pending{
		{Body: &trace.Annotation{Key: "a", Value: json.RawMessage(`1`)}},
		{Body: &trace.Annotation{Key: "b", Value: json.RawMessage(`2`)}, Cause: trace.Cause(1)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(seqs, []uint64{1, 2}) || w.Next() != 3 {
		t.Errorf("seqs %v, next %d", seqs, w.Next())
	}
	if !reflect.DeepEqual(seen, []uint64{0, 1, 2}) {
		t.Errorf("observers saw %v", seen)
	}
	// A failed append doesn't advance seq or notify.
	failing := trace.NewWriter(s, traceID, rootID, trace.WriterOptions{Observers: []trace.Observer{obs}})
	if _, err := failing.Write(ctx, &trace.Annotation{Key: "x", Value: json.RawMessage(`0`)}, nil); err == nil {
		t.Fatal("expected seq conflict (seq 0 already exists)")
	}
	if failing.Next() != 0 || len(seen) != 3 {
		t.Errorf("failed write advanced state: next %d, seen %v", failing.Next(), seen)
	}
}

func newFileStore(t *testing.T, threshold int) *filestore.Store {
	t.Helper()
	s, err := filestore.Open(t.TempDir(), filestore.Options{BlobThreshold: threshold})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestBundleRoundTrip(t *testing.T) {
	ctx := context.Background()
	src := newFileStore(t, 16)
	events, _ := calculatorRun(t, src)
	// Add a child run so the bundle holds a tree.
	child := trace.NewWriter(src, traceID, childID, trace.WriterOptions{Now: func() time.Time { return t0 }})
	if _, err := child.Write(ctx, &trace.RunStarted{RunKind: trace.RunKindAgent, Path: "/helper#0", Parent: &trace.ParentRef{RunID: rootID},
		Agent: trace.AgentInfo{Name: "helper"}, Task: llm.Blocks{llm.Text{Text: strings.Repeat("long task ", 50)}}}, nil); err != nil {
		t.Fatal(err)
	}

	var bundle bytes.Buffer
	if err := trace.Export(ctx, &bundle, src, traceID, trace.BundleOptions{BlobThreshold: 64, Now: func() time.Time { return t0 }}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(bundle.Bytes(), []byte(`{"blob":"sha256:`)) {
		t.Fatal("bundle should contain blob lines")
	}
	dst := memstore.New()
	id, err := trace.Import(ctx, bytes.NewReader(bundle.Bytes()), dst)
	if err != nil {
		t.Fatal(err)
	}
	if id != traceID {
		t.Errorf("imported trace %s, want %s", id, traceID)
	}
	got, err := trace.Collect(dst.Events(ctx, rootID, 0))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(events) {
		t.Fatalf("imported %d events, want %d", len(got), len(events))
	}
	for i := range events {
		a, _ := trace.EncodeEvent(events[i], trace.EncodeOptions{})
		b, _ := trace.EncodeEvent(got[i], trace.EncodeOptions{})
		if !bytes.Equal(a, b) {
			t.Fatalf("event %d differs after export/import\n%s\n%s", i, a, b)
		}
	}
	if kids, err := trace.Collect(dst.Events(ctx, childID, 0)); err != nil || len(kids) != 1 {
		t.Fatalf("child run not imported: %d events, %v", len(kids), err)
	}

	// Tampering with a blob is detected.
	tampered := bytes.Replace(bundle.Bytes(), []byte(`"data":"`), []byte(`"data":"AAAA`), 1)
	if _, err := trace.Import(ctx, bytes.NewReader(tampered), memstore.New()); err == nil {
		t.Error("tampered blob was accepted")
	}
	if _, err := trace.Import(ctx, strings.NewReader(`{"hello":"world"}`+"\n"), memstore.New()); err == nil {
		t.Error("non-bundle input was accepted")
	}
}

// TestStorageGrowsLinearly is the M1 exit criterion: with references into
// the conversation instead of full request copies, trace size grows linearly
// with the number of steps.
func TestStorageGrowsLinearly(t *testing.T) {
	if testing.Short() {
		t.Skip("writes ~15 MB")
	}
	size := func(steps int) int64 {
		dir := t.TempDir()
		s, err := filestore.Open(dir, filestore.Options{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = s.Close() }()
		ctx := context.Background()
		w := trace.NewWriter(s, trace.NewID(), trace.NewID(), trace.WriterOptions{})
		if _, err := w.Write(ctx, &trace.RunStarted{RunKind: trace.RunKindAgent, Path: "/", Agent: trace.AgentInfo{Name: "a"},
			Config: &trace.Config{Model: "m", MaxSteps: steps}}, nil); err != nil {
			t.Fatal(err)
		}
		msgs := 0
		for i := 0; i < steps; i++ {
			out := strings.Repeat(string(rune('a'+i%26)), 20<<10) + string(rune(i)) // unique 20 KiB output
			call := llm.ToolUse{CallID: "c", Name: "t", Input: json.RawMessage(`{}`)}
			batch := []trace.Pending{
				{Body: &trace.LLMStarted{Step: i, Context: trace.ContextRef{Messages: msgs}, Fingerprint: "sha256:f"}},
				{Body: &trace.LLMCompleted{Step: i, Message: llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{call}}}},
				{Body: &trace.ToolCompleted{CallID: "c", Content: llm.Blocks{llm.Text{Text: out}}}},
				{Body: &trace.MessageAppended{Message: &llm.Message{Role: llm.RoleUser, Blocks: llm.Blocks{llm.ToolResult{CallID: "c", Content: llm.Blocks{llm.Text{Text: out}}}}}}},
			}
			if _, err := w.WriteBatch(ctx, batch); err != nil {
				t.Fatal(err)
			}
			msgs++
		}
		var total int64
		err = filepath.WalkDir(dir, func(_ string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return err
			}
			info, err := d.Info()
			if err == nil {
				total += info.Size()
			}
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		return total
	}
	small, large := size(250), size(500)
	ratio := float64(large) / float64(small)
	t.Logf("250 steps: %d bytes, 500 steps: %d bytes, ratio %.3f", small, large, ratio)
	if ratio > 2.1 {
		t.Errorf("doubling the steps grew storage %.2fx; expected ~2x (linear)", ratio)
	}
}
