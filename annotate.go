package agentflow

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/kumarlokesh/agentflow/trace"
)

// Annotate records an annotation in the current run: a label, a verdict, a
// note for whoever reads the trace. Keys starting with "agentflow." are
// reserved. Inside a tool call it is recorded with the call's nested events,
// in batch order. It does nothing outside a run, and inside an effect
// function (a Do function, a context provider, a model), whose code a replay
// doesn't run.
//
// Annotations aren't effects: replays and forks record them again, and
// never match them against the recording.
func Annotate(ctx context.Context, key string, value any) error {
	if strings.HasPrefix(key, "agentflow.") {
		return fmt.Errorf("agentflow: annotation key %q is reserved", key)
	}
	return annotate(ctx, key, value)
}

// Expect records a check of the application's own: whether something the
// run produced is what it should be ("the summary names a cause"). A failed
// expectation is a symptom: inspect.Blame starts from it to find the
// anomalies upstream, and a fork sample with one fails by default. It is
// recorded like Annotate, under the key trace.AnnotationExpectation.
func Expect(ctx context.Context, name string, pass bool, detail string) {
	_ = annotate(ctx, trace.AnnotationExpectation, trace.Expectation{Name: name, Pass: pass, Detail: detail})
}

func annotate(ctx context.Context, key string, value any) error {
	sc := scopeFrom(ctx)
	if sc == nil || isInsideEffect(ctx) {
		return nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("agentflow: annotation %q: %w", key, err)
	}
	_, err = sc.emit(ctx, &trace.Annotation{Key: key, Value: raw}, nil)
	return err
}
