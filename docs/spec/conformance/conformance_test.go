package conformance

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

const (
	schemaFile = "../trace-v1.schema.json"
	schemaURL  = "https://github.com/kumarlokesh/agentflow/docs/spec/trace-v1.schema.json"
	repoRoot   = "../../.."
)

// schemas holds the compiled entry points of one variant of the schema.
type schemas struct {
	event, header, blob *jsonschema.Schema
}

// load compiles the schema. In strict mode every object schema that lists
// its properties also forbids unlisted ones. The published schema allows
// unknown fields (readers must ignore them, for forward compatibility);
// strict mode makes sure everything agentflow writes is specified.
func load(t *testing.T, strict bool) schemas {
	t.Helper()
	f, err := os.Open(schemaFile)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	doc, err := jsonschema.UnmarshalJSON(f)
	if err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	if strict {
		forbidExtra(doc, true)
	}
	c := jsonschema.NewCompiler()
	if err := c.AddResource(schemaURL, doc); err != nil {
		t.Fatal(err)
	}
	compile := func(loc string) *jsonschema.Schema {
		s, err := c.Compile(loc)
		if err != nil {
			t.Fatalf("compile %s: %v", loc, err)
		}
		return s
	}
	return schemas{
		event:  compile(schemaURL),
		header: compile(schemaURL + "#/$defs/bundle_header"),
		blob:   compile(schemaURL + "#/$defs/bundle_blob"),
	}
}

// forbidExtra adds "additionalProperties": false to every schema with
// "properties", except the conditions and branches of if/then (they
// constrain one property of a larger object).
func forbidExtra(n any, self bool) {
	m, ok := n.(map[string]any)
	if !ok {
		return
	}
	if _, has := m["properties"]; has && self {
		if _, set := m["additionalProperties"]; !set {
			m["additionalProperties"] = false
		}
	}
	for k, v := range m {
		switch k {
		case "properties", "$defs":
			for _, s := range v.(map[string]any) {
				forbidExtra(s, true)
			}
		case "if", "then", "else":
			forbidExtra(v, false)
		case "allOf", "anyOf", "oneOf":
			for _, s := range v.([]any) {
				forbidExtra(s, true)
			}
		case "items", "not":
			forbidExtra(v, true)
		}
	}
}

func validate(s *jsonschema.Schema, line []byte) error {
	v, err := jsonschema.UnmarshalJSON(bytes.NewReader(line))
	if err != nil {
		return err
	}
	return s.Validate(v)
}

// validateFile validates a run file or a bundle, line by line.
func validateFile(t *testing.T, s schemas, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(nil, trace.MaxEventSize+1)
	for n := 1; sc.Scan(); n++ {
		line := sc.Bytes()
		target := s.event
		switch {
		case n == 1 && bytes.HasPrefix(line, []byte(`{"bundle":`)):
			target = s.header
		case bytes.HasPrefix(line, []byte(`{"blob":`)):
			target = s.blob
		}
		if err := validate(target, line); err != nil {
			t.Errorf("%s:%d: %v", path, n, err)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
}

func fixtures(t *testing.T) []string {
	t.Helper()
	var out []string
	for _, pattern := range []string{"trace/testdata/*.jsonl", "examples/*/testdata/*.jsonl"} {
		m, err := filepath.Glob(filepath.Join(repoRoot, pattern))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, m...)
	}
	if len(out) < 2 {
		t.Fatalf("expected the golden fixture and at least one cassette, found %v", out)
	}
	return out
}

func TestFixtures(t *testing.T) {
	s := load(t, true)
	for _, path := range fixtures(t) {
		t.Run(filepath.Base(path), func(t *testing.T) { validateFile(t, s, path) })
	}
}

// allBodies has one zero body of every kind agentflow defines.
func allBodies() []trace.Body {
	return []trace.Body{
		&trace.RunStarted{}, &trace.RunFinished{}, &trace.RunSpawned{}, &trace.RunJoined{},
		&trace.RunModeChanged{}, &trace.MessageAppended{}, &trace.ConversationReplaced{},
		&trace.ConfigChanged{}, &trace.LLMStarted{}, &trace.LLMCompleted{}, &trace.LLMFailed{},
		&trace.ToolStarted{}, &trace.ToolCompleted{}, &trace.ToolFailed{}, &trace.EffectStarted{},
		&trace.EffectCompleted{}, &trace.EffectFailed{}, &trace.Annotation{},
	}
}

// TestKindsAgree checks that the Go bodies, the schema, and the golden
// fixture cover the same set of event kinds.
func TestKindsAgree(t *testing.T) {
	var goKinds []string
	for _, b := range allBodies() {
		goKinds = append(goKinds, string(b.Kind()))
	}

	raw, err := os.ReadFile(schemaFile)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Defs struct {
			Event struct {
				AllOf []struct {
					If struct {
						Properties struct {
							Kind struct {
								Const string `json:"const"`
							} `json:"kind"`
						} `json:"properties"`
					} `json:"if"`
				} `json:"allOf"`
			} `json:"event"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var schemaKinds []string
	for _, c := range doc.Defs.Event.AllOf {
		schemaKinds = append(schemaKinds, c.If.Properties.Kind.Const)
	}

	golden, err := os.ReadFile(filepath.Join(repoRoot, "trace/testdata/golden.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var goldenKinds []string
	for _, line := range bytes.Split(bytes.TrimSpace(golden), []byte("\n")) {
		var e struct {
			Kind string `json:"kind"`
		}
		if err := json.Unmarshal(line, &e); err != nil {
			t.Fatal(err)
		}
		if !seen[e.Kind] {
			seen[e.Kind] = true
			goldenKinds = append(goldenKinds, e.Kind)
		}
	}

	sort.Strings(goKinds)
	sort.Strings(schemaKinds)
	sort.Strings(goldenKinds)
	if !reflect.DeepEqual(goKinds, schemaKinds) {
		t.Errorf("Go kinds and schema kinds differ:\n go:     %v\n schema: %v", goKinds, schemaKinds)
	}
	if !reflect.DeepEqual(goKinds, goldenKinds) {
		t.Errorf("golden fixture doesn't cover every kind:\n go:     %v\n golden: %v", goKinds, goldenKinds)
	}
}

// TestEveryField fills every field of every body (and of every content
// block) with a valid non-zero value, encodes it with the real encoder, and
// validates it in strict mode. A field added to the Go types but not to the
// schema fails here.
func TestEveryField(t *testing.T) {
	s := load(t, true)
	check := func(t *testing.T, b trace.Body) {
		t.Helper()
		line := encode(t, b, trace.EncodeOptions{})
		if err := validate(s.event, line); err != nil {
			t.Errorf("%s\n%s", err, line)
		}
	}
	for _, b := range allBodies() {
		t.Run(string(b.Kind()), func(t *testing.T) {
			fill(reflect.ValueOf(b).Elem(), "")
			if m, ok := b.(*trace.MessageAppended); ok {
				// message and source_seq are exclusive: check each form.
				src := m.SourceSeq
				m.SourceSeq = nil
				check(t, m)
				m.Message, m.SourceSeq = nil, src
			}
			check(t, b)
		})
	}
}

// TestReferences re-encodes the golden events with blob externalization
// and redaction forced onto every string, and with application data that
// uses reserved keys, and validates the result.
func TestReferences(t *testing.T) {
	s := load(t, true)
	events := goldenEvents(t)
	put := func(data []byte) (trace.BlobRef, error) { return trace.BlobRefOf(data), nil }
	cases := map[string]trace.EncodeOptions{
		"blobs":  {BlobThreshold: 8, PutBlob: put},
		"redact": {Redact: func(trace.Kind, string, string) (string, bool) { return "all", true }},
	}
	for name, opts := range cases {
		t.Run(name, func(t *testing.T) {
			marker := []byte(`"$blob"`)
			if opts.Redact != nil {
				marker = []byte(`"$redacted"`)
			}
			found := false
			for _, e := range events {
				line, err := trace.EncodeEvent(e, opts)
				if err != nil {
					t.Fatal(err)
				}
				found = found || bytes.Contains(line, marker)
				if err := validate(s.event, line); err != nil {
					t.Errorf("seq %d: %v\n%s", e.Seq, err, line)
				}
			}
			if !found {
				t.Fatalf("no %s reference was produced", marker)
			}
		})
	}
	t.Run("lit", func(t *testing.T) {
		line := encode(t, &trace.Annotation{Key: "app.note", Value: json.RawMessage(`{"$blob":"not a reference"}`)}, trace.EncodeOptions{})
		if !bytes.Contains(line, []byte(`"$lit"`)) {
			t.Fatalf("expected $lit escaping: %s", line)
		}
		if err := validate(s.event, line); err != nil {
			t.Error(err)
		}
	})
}

// TestRejects makes sure the schema is not vacuous: each mutation of a
// valid event must fail validation. Mutations marked strictOnly are
// unknown fields, which the published (lax) schema accepts on purpose.
func TestRejects(t *testing.T) {
	lax, strict := load(t, false), load(t, true)
	byKind := map[string]map[string]any{}
	for _, e := range goldenEvents(t) {
		line := encode(t, e.Body, trace.EncodeOptions{})
		var m map[string]any
		if err := json.Unmarshal(line, &m); err != nil {
			t.Fatal(err)
		}
		if _, ok := byKind[string(e.Kind())]; !ok {
			byKind[string(e.Kind())] = m
		}
	}
	body := func(m map[string]any) map[string]any { return m["body"].(map[string]any) }
	cases := []struct {
		name       string
		kind       string
		mutate     func(m map[string]any)
		strictOnly bool
	}{
		{"wrong version", "annotation", func(m map[string]any) { m["v"] = 2 }, false},
		{"negative seq", "annotation", func(m map[string]any) { m["seq"] = -1 }, false},
		{"missing body", "annotation", func(m map[string]any) { delete(m, "body") }, false},
		{"bad time", "annotation", func(m map[string]any) { m["time"] = "yesterday" }, false},
		{"missing fingerprint", "llm.started", func(m map[string]any) { delete(body(m), "fingerprint") }, false},
		{"short fingerprint", "tool.started", func(m map[string]any) { body(m)["fingerprint"] = "sha256:abc" }, false},
		{"both message forms", "message.appended", func(m map[string]any) { body(m)["source_seq"] = 3 }, false},
		{"neither message form", "message.appended", func(m map[string]any) { delete(body(m), "message") }, false},
		{"incomplete is never written", "run.finished", func(m map[string]any) { body(m)["status"] = "incomplete" }, false},
		{"unknown role", "message.appended", func(m map[string]any) {
			body(m)["message"].(map[string]any)["role"] = "system"
		}, false},
		{"text block without text", "message.appended", func(m map[string]any) {
			blocks := body(m)["message"].(map[string]any)["blocks"].([]any)
			delete(blocks[0].(map[string]any), "text")
		}, false},
		{"malformed blob ref", "annotation", func(m map[string]any) { body(m)["key"] = map[string]any{"$blob": "md5:00", "size": 1} }, false},
		{"empty config change", "config.changed", func(m map[string]any) { m["body"] = map[string]any{} }, false},
		{"unknown body field", "tool.completed", func(m map[string]any) { body(m)["surprise"] = true }, true},
		{"unknown block field", "message.appended", func(m map[string]any) {
			blocks := body(m)["message"].(map[string]any)["blocks"].([]any)
			blocks[0].(map[string]any)["surprise"] = true
		}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base, ok := byKind[tc.kind]
			if !ok {
				t.Fatalf("golden fixture has no %s event", tc.kind)
			}
			m := deepCopy(t, base)
			tc.mutate(m)
			line, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			if validate(strict.event, line) == nil {
				t.Errorf("strict schema accepted %s", line)
			}
			laxErr := validate(lax.event, line)
			if tc.strictOnly && laxErr != nil {
				t.Errorf("published schema must accept unknown fields: %v", laxErr)
			}
			if !tc.strictOnly && laxErr == nil {
				t.Errorf("published schema accepted %s", line)
			}
		})
	}
	t.Run("blob line with bad hash", func(t *testing.T) {
		if validate(lax.blob, []byte(`{"blob":"sha256:xyz","data":"aGk="}`)) == nil {
			t.Error("accepted a malformed blob line")
		}
	})
	t.Run("unknown kind", func(t *testing.T) {
		line := []byte(`{"v":1,"trace_id":"t","run_id":"r","seq":3,"time":"2026-11-23T10:00:00.000000Z","kind":"future.thing","body":{"anything":[1,2]}}`)
		if err := validate(strict.event, line); err != nil {
			t.Errorf("unknown kinds must be accepted: %v", err)
		}
	})
}

// --- helpers ---

func encode(t *testing.T, b trace.Body, opts trace.EncodeOptions) []byte {
	t.Helper()
	line, err := trace.EncodeEvent(trace.Event{
		TraceID: "0193c4a0-0000-7000-8000-000000000001",
		RunID:   "0193c4a0-0000-7000-8000-000000000002",
		Seq:     1,
		Time:    time.Date(2026, 11, 23, 10, 0, 0, 0, time.UTC),
		Cause:   trace.Cause(0),
		Body:    b,
	}, opts)
	if err != nil {
		t.Fatal(err)
	}
	return line
}

func goldenEvents(t *testing.T) []trace.Event {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(repoRoot, "trace/testdata/golden.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var out []trace.Event
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		e, err := trace.DecodeEvent(line, nil)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, e)
	}
	return out
}

func deepCopy(t *testing.T, m map[string]any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// enumValues gives a valid value for each string type with a fixed vocabulary.
var enumValues = map[reflect.Type]string{
	reflect.TypeOf(llm.Role("")):           string(llm.RoleAssistant),
	reflect.TypeOf(llm.StopReason("")):     string(llm.StopEndTurn),
	reflect.TypeOf(llm.ToolChoiceMode("")): string(llm.ToolChoiceTool),
	reflect.TypeOf(trace.RunKind("")):      string(trace.RunKindAgent),
	reflect.TypeOf(trace.RunStatus("")):    string(trace.StatusCompleted),
	reflect.TypeOf(trace.OriginKind("")):   string(trace.OriginFork),
	reflect.TypeOf(trace.AccessMode("")):   string(trace.AccessWrite),
}

// stringFor gives a valid value for a plain string field, by JSON name.
func stringFor(name string) string {
	switch name {
	case "fingerprint", "config_fp":
		return "sha256:" + strings.Repeat("ab", 32)
	case "mode":
		return "live"
	case "reason":
		return trace.ReasonDivergence
	}
	return "value of " + name
}

var (
	rawType    = reflect.TypeOf(json.RawMessage(nil))
	blocksType = reflect.TypeOf(llm.Blocks(nil))
	extType    = reflect.TypeOf(llm.Ext(nil))
	bytesType  = reflect.TypeOf([]byte(nil))
)

// blockDepth stops recursion through tool_result content.
var blockDepth int

// fill sets every exported field reachable from v to a valid non-zero value.
func fill(v reflect.Value, name string) {
	switch v.Type() {
	case rawType:
		v.Set(reflect.ValueOf(json.RawMessage(`{"k":[1,"v",null]}`)))
		return
	case extType:
		v.Set(reflect.ValueOf(llm.Ext{"provider": json.RawMessage(`{"signature":"c2ln"}`)}))
		return
	case bytesType:
		v.SetBytes([]byte("media bytes"))
		return
	case blocksType:
		v.Set(reflect.ValueOf(everyBlock()))
		return
	}
	if s, ok := enumValues[v.Type()]; ok {
		v.SetString(s)
		return
	}
	switch v.Kind() {
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			tag, _, _ := strings.Cut(f.Tag.Get("json"), ",")
			if tag == "-" {
				continue
			}
			fill(v.Field(i), tag)
		}
	case reflect.Pointer:
		v.Set(reflect.New(v.Type().Elem()))
		fill(v.Elem(), name)
	case reflect.Slice:
		s := reflect.MakeSlice(v.Type(), 1, 1)
		fill(s.Index(0), name)
		v.Set(s)
	case reflect.Map:
		m := reflect.MakeMap(v.Type())
		val := reflect.New(v.Type().Elem()).Elem()
		fill(val, name)
		m.SetMapIndex(reflect.ValueOf("key").Convert(v.Type().Key()), val)
		v.Set(m)
	case reflect.String:
		v.SetString(stringFor(name))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(1)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		v.SetUint(1)
	case reflect.Float32, reflect.Float64:
		v.SetFloat(0.5)
	case reflect.Bool:
		v.SetBool(true)
	default:
		panic("conformance: fill: unhandled type " + v.Type().String())
	}
}

// everyBlock returns one fully populated block of every known type, plus a
// block of a type this version doesn't know.
func everyBlock() llm.Blocks {
	if blockDepth > 0 {
		return llm.Blocks{llm.Text{Text: "nested"}}
	}
	blockDepth++
	defer func() { blockDepth-- }()
	var out llm.Blocks
	for _, b := range []llm.Block{llm.Text{}, llm.Media{}, llm.ToolUse{}, llm.ToolResult{}, llm.Reasoning{}, llm.Data{}} {
		p := reflect.New(reflect.TypeOf(b))
		fill(p.Elem(), "")
		out = append(out, p.Elem().Interface().(llm.Block))
	}
	return append(out, llm.Unknown{Type: "x.future", Raw: json.RawMessage(`{"type":"x.future","n":1}`)})
}

// TestSpecExample checks the example in trace-v1.md §15: every line is a
// valid event, and the request reconstruction described under it holds.
func TestSpecExample(t *testing.T) {
	doc, err := os.ReadFile("../trace-v1.md")
	if err != nil {
		t.Fatal(err)
	}
	_, after, ok := strings.Cut(string(doc), "## 15. Example")
	if !ok {
		t.Fatal("trace-v1.md has no example section")
	}
	_, block, ok := strings.Cut(after, "```json\n")
	if !ok {
		t.Fatal("example section has no json block")
	}
	block, _, _ = strings.Cut(block, "```")
	// The example shortens fingerprints for readability ("sha256:3f…").
	block = shortFP.ReplaceAllString(block, `"sha256:`+strings.Repeat("ab", 32)+`"`)

	s := load(t, true)
	var events []trace.Event
	for i, line := range strings.Split(strings.TrimSpace(block), "\n") {
		if err := validate(s.event, []byte(line)); err != nil {
			t.Fatalf("example line %d: %v", i+1, err)
		}
		e, err := trace.DecodeEvent([]byte(line), nil)
		if err != nil {
			t.Fatalf("example line %d: %v", i+1, err)
		}
		if e.Seq != uint64(i) {
			t.Fatalf("example line %d has seq %d", i+1, e.Seq)
		}
		events = append(events, e)
	}
	req, err := trace.Request(events, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 3 || req.Messages[1].Role != llm.RoleAssistant || req.Messages[2].Role != llm.RoleUser {
		t.Fatalf("example request at seq 8 is not [task, tool call, tool result]: %+v", req.Messages)
	}
	if _, ok := req.Messages[2].Blocks[0].(llm.ToolResult); !ok {
		t.Fatalf("third message should be the tool result, got %T", req.Messages[2].Blocks[0])
	}
}

var shortFP = regexp.MustCompile(`"sha256:[0-9a-f]+…"`)
