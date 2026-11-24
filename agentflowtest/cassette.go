package agentflowtest

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
)

var record = flag.Bool("agentflow.record", false, "agentflowtest: record missing cassettes by running agents live")

// RecordEnv enables recording when set to "1" (an alternative to the
// -agentflow.record flag).
const RecordEnv = "AGENTFLOW_RECORD"

func recording() bool { return *record || os.Getenv(RecordEnv) == "1" }

// CassetteOption configures Cassette.
type CassetteOption func(*replay.Options)

// IgnorePaths makes the cassette disregard differences at these diff paths
// (see replay.Options.Ignore).
func IgnorePaths(paths ...string) CassetteOption {
	return func(o *replay.Options) { o.Ignore = append(o.Ignore, paths...) }
}

// Verify executes these effects during the replay and compares their
// outcomes with the recording (see replay.Options.Verify), so the cassette
// also catches changes inside tools and context providers, e.g.
// Verify("tool:*", "effect:context.*"). Verified tools really run: name only
// tools that are safe to call in tests. A wildcard selects only tools
// annotated ReadOnly.
func Verify(patterns ...string) CassetteOption {
	return func(o *replay.Options) { o.Verify = append(o.Verify, patterns...) }
}

// Cassette replays the trace bundle at path strictly against e and returns
// the replayed run. On any divergence the test fails with the divergence
// report. The replay never reaches real models, and reaches tools only when
// they are selected with Verify.
//
// If the file doesn't exist and recording is enabled (-agentflow.record or
// AGENTFLOW_RECORD=1), e runs live on task and the bundle is written. A
// missing file without recording enabled fails the test, so CI never makes
// live calls by accident. To re-record, delete the file.
func Cassette(t testing.TB, path string, e agentflow.Entrypoint, task string, opts ...CassetteOption) *agentflow.RunResult {
	t.Helper()
	ctx := context.Background()
	data, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist) && recording():
		return recordCassette(t, ctx, path, e, task)
	case errors.Is(err, os.ErrNotExist):
		t.Fatalf("cassette %s does not exist; record it with -agentflow.record or %s=1 (this runs the agent live)", path, RecordEnv)
	case err != nil:
		t.Fatalf("cassette %s: %v", path, err)
	}
	var o replay.Options
	for _, opt := range opts {
		opt(&o)
	}
	res, err := replay.RunBundle(ctx, e, bytes.NewReader(data), o)
	if err != nil {
		var de *replay.DivergenceError
		if errors.As(err, &de) {
			t.Fatalf("cassette %s: the agent no longer matches its recording.\n%s\n\nIf this change is intended, delete the cassette and re-record it.", path, de.Divergence.Summary)
		}
		t.Fatalf("cassette %s: %v", path, err)
	}
	return res.Run
}

func recordCassette(t testing.TB, ctx context.Context, path string, e agentflow.Entrypoint, task string) *agentflow.RunResult {
	t.Helper()
	store := memstore.New()
	res, err := e.Run(ctx, task, agentflow.WithStore(store))
	if res == nil {
		t.Fatalf("cassette %s: recording run failed: %v", path, err)
	}
	if err != nil {
		t.Logf("cassette %s: recorded run ended with %v (recorded anyway)", path, err)
	}
	var buf bytes.Buffer
	if err := trace.Export(ctx, &buf, store, res.TraceID, trace.BundleOptions{}); err != nil {
		t.Fatalf("cassette %s: export: %v", path, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("cassette %s: %v", path, err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("cassette %s: %v", path, err)
	}
	t.Logf("cassette %s: recorded trace %s (traces may contain prompts and tool output; review before committing)", path, res.TraceID)
	return res
}
