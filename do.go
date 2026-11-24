package agentflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/kumarlokesh/agentflow/trace"
)

// Do runs fn as a recorded effect so that application code's
// own nondeterminism (reading a clock, calling an API, querying a database)
// is replayable:
//
//   - live, fn runs and its result is recorded;
//   - during replay, the input is matched against the recording and fn is
//     not called: the recorded result (or failure) is returned. If the
//     effect is selected for verification (replay.Options.Verify), fn is
//     called and its result compared with the recording.
//
// The result is always returned via a JSON round trip, so live and replayed
// runs see identical values. Outside a run, or inside another effect, Do
// simply calls fn.
func Do[In, Out any](ctx context.Context, name string, in In, fn func(context.Context, In) (Out, error)) (Out, error) {
	var zero Out
	sc := scopeFrom(ctx)
	if sc == nil || isInsideEffect(ctx) {
		return fn(ctx, in)
	}
	out, err := sc.rs.doEffect(ctx, sc, name, in, nil, func(c context.Context) (any, error) { return fn(c, in) })
	if err != nil {
		return zero, err
	}
	if out.Failure != nil {
		return zero, &EffectError{Code: out.Failure.Code, Message: out.Failure.Message}
	}
	var v Out
	if err := json.Unmarshal(out.Response.(json.RawMessage), &v); err != nil {
		return zero, fmt.Errorf("agentflow: effect %q output: %w", name, err)
	}
	return v, nil
}

// Now returns the current time as a recorded effect ("clock.now").
// It panics if called concurrently with another effect on the same run
// (ErrConcurrentEffect), since silently falling back would make the run
// nondeterministic.
func Now(ctx context.Context) time.Time {
	t, err := Do(ctx, "clock.now", struct{}{}, func(context.Context, struct{}) (time.Time, error) { return time.Now().UTC(), nil })
	if err != nil {
		mustNotRace(err)
		return time.Now().UTC()
	}
	return t
}

// NewID returns a new UUIDv7 as a recorded effect ("id.new"). Like Now, it
// panics on ErrConcurrentEffect.
func NewID(ctx context.Context) string {
	id, err := Do(ctx, "id.new", struct{}{}, func(context.Context, struct{}) (string, error) { return trace.NewID(), nil })
	if err != nil {
		mustNotRace(err)
		return trace.NewID()
	}
	return id
}

func mustNotRace(err error) {
	if errors.Is(err, ErrConcurrentEffect) {
		panic(err)
	}
}
