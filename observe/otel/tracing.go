// Package otel exports agentflow traces to OpenTelemetry. It is a separate
// Go module, so only programs that want OTel pay for its dependencies.
//
// Both exporters are agentflow trace.Observers. Pass them to a top-level
// run with agentflow.WithObservers to cover every run in its tree (child
// runs included), or add them to an agent's AgentConfig.Observers to see
// that agent's runs only. Spans and metrics are built from recorded events,
// so their timings are the recorded ones.
//
// Span hierarchy for each run:
//
//	agentflow.run                (one per run)
//	├── agentflow.llm            (one per model call; gen_ai.* attributes)
//	├── agentflow.tool.<name>    (one per tool call)
//	├── agentflow.effect.<name>  (one per custom effect, e.g. context providers)
//	└── agentflow.spawn.<agent>  (one per child run started)
//
// A child run's span is a child of its spawn span when the child is started
// from workflow code (Spawn, SpawnAll), and of its parent's run span when it
// is started inside a tool call (delegation), whose spawn is recorded only
// after the child ran.
package otel

import (
	"context"
	"sync"

	aftrace "github.com/kumarlokesh/agentflow/trace"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const instrumentation = "github.com/kumarlokesh/agentflow"

// Tracer turns recorded events into OpenTelemetry spans. It is safe for
// concurrent use.
type Tracer struct {
	tracer trace.Tracer

	mu     sync.Mutex
	runs   map[string]runSpan         // by run ID
	spans  map[spanKey]trace.Span     // open effect spans
	ctxs   map[string]context.Context // run span contexts
	spawns map[string]context.Context // spawn span contexts, by child run ID
}

type runSpan struct {
	span  trace.Span
	model string
}

type spanKey struct {
	run string
	seq uint64
}

var _ aftrace.Observer = (*Tracer)(nil)

// NewTracer returns a Tracer that creates spans with tp.
func NewTracer(tp trace.TracerProvider) *Tracer {
	return &Tracer{
		tracer: tp.Tracer(instrumentation),
		runs:   map[string]runSpan{},
		spans:  map[spanKey]trace.Span{},
		ctxs:   map[string]context.Context{},
		spawns: map[string]context.Context{},
	}
}

// Observe implements trace.Observer.
func (t *Tracer) Observe(_ context.Context, e aftrace.Event) {
	t.mu.Lock()
	defer t.mu.Unlock()
	switch b := e.Body.(type) {
	case *aftrace.RunStarted:
		attrs := []attribute.KeyValue{
			attribute.String("agentflow.trace_id", e.TraceID),
			attribute.String("agentflow.run_id", e.RunID),
			attribute.String("agentflow.path", b.Path),
			attribute.String("agentflow.agent", b.Agent.Name),
		}
		model := ""
		if b.Config != nil {
			model = b.Config.Model
			attrs = append(attrs, attribute.String("gen_ai.request.model", model))
		}
		if b.Origin != nil {
			attrs = append(attrs, attribute.String("agentflow.origin", string(b.Origin.Kind)))
		}
		parent := context.Background()
		if b.Parent != nil {
			attrs = append(attrs, attribute.String("agentflow.parent_run_id", b.Parent.RunID))
			if c, ok := t.spawns[e.RunID]; ok {
				parent = c
			} else if c, ok := t.ctxs[b.Parent.RunID]; ok {
				parent = c
			}
			delete(t.spawns, e.RunID)
		}
		ctx, span := t.tracer.Start(parent, "agentflow.run",
			trace.WithTimestamp(e.Time), trace.WithAttributes(attrs...))
		t.runs[e.RunID] = runSpan{span: span, model: model}
		t.ctxs[e.RunID] = ctx

	case *aftrace.LLMStarted:
		model := t.runs[e.RunID].model
		t.start(e, "agentflow.llm",
			attribute.String("gen_ai.operation.name", "chat"),
			attribute.String("gen_ai.request.model", model),
			attribute.Int("agentflow.step", b.Step))
	case *aftrace.LLMCompleted:
		t.end(e, nil, func(s trace.Span) {
			s.SetAttributes(
				attribute.Int("gen_ai.usage.input_tokens", b.Usage.InputTokens),
				attribute.Int("gen_ai.usage.output_tokens", b.Usage.OutputTokens),
				attribute.String("gen_ai.response.model", b.Model),
				attribute.StringSlice("gen_ai.response.finish_reasons", []string{string(b.StopReason)}),
				attribute.Int("agentflow.attempts", b.Attempts))
		})
	case *aftrace.LLMFailed:
		t.end(e, &b.Error, nil)

	case *aftrace.ToolStarted:
		t.start(e, "agentflow.tool."+b.Tool,
			attribute.String("gen_ai.operation.name", "execute_tool"),
			attribute.String("gen_ai.tool.name", b.Tool),
			attribute.String("gen_ai.tool.call.id", b.CallID),
			attribute.Int("agentflow.step", b.Step))
	case *aftrace.ToolCompleted:
		t.end(e, nil, func(s trace.Span) { s.SetAttributes(attribute.Bool("agentflow.tool.is_error", b.IsError)) })
	case *aftrace.ToolFailed:
		t.end(e, &b.Error, nil)

	case *aftrace.EffectStarted:
		t.start(e, "agentflow.effect."+b.Name, attribute.String("agentflow.effect", b.Name))
	case *aftrace.EffectCompleted:
		t.end(e, nil, nil)
	case *aftrace.EffectFailed:
		t.end(e, &b.Error, nil)

	case *aftrace.RunSpawned:
		if ctx := t.start(e, "agentflow.spawn."+b.Agent,
			attribute.String("agentflow.agent", b.Agent),
			attribute.String("agentflow.child_path", b.Path),
			attribute.String("agentflow.child_run_id", b.ChildRunID)); ctx != nil {
			t.spawns[b.ChildRunID] = ctx
		}
	case *aftrace.RunJoined:
		t.end(e, b.Error, func(s trace.Span) { s.SetAttributes(attribute.String("agentflow.status", string(b.Status))) })

	case *aftrace.RunModeChanged:
		if r, ok := t.runs[e.RunID]; ok {
			r.span.AddEvent("agentflow.mode_changed", trace.WithTimestamp(e.Time),
				trace.WithAttributes(attribute.String("agentflow.mode", b.Mode), attribute.String("agentflow.reason", b.Reason)))
		}

	case *aftrace.RunFinished:
		r, ok := t.runs[e.RunID]
		if !ok {
			return
		}
		r.span.SetAttributes(
			attribute.String("agentflow.status", string(b.Status)),
			attribute.Int("agentflow.steps", b.Steps),
			attribute.Int("gen_ai.usage.input_tokens", b.Usage.InputTokens),
			attribute.Int("gen_ai.usage.output_tokens", b.Usage.OutputTokens))
		if b.Status == aftrace.StatusCompleted {
			r.span.SetStatus(codes.Ok, "")
		} else {
			msg := string(b.Status)
			if b.Error != nil {
				msg = b.Error.Message
			}
			r.span.SetStatus(codes.Error, msg)
		}
		r.span.End(trace.WithTimestamp(e.Time))
		delete(t.runs, e.RunID)
		delete(t.ctxs, e.RunID)
	}
}

// start opens an effect span under its run span and returns its context.
// Called with t.mu held.
func (t *Tracer) start(e aftrace.Event, name string, attrs ...attribute.KeyValue) context.Context {
	parent, ok := t.ctxs[e.RunID]
	if !ok {
		return nil
	}
	ctx, span := t.tracer.Start(parent, name, trace.WithTimestamp(e.Time), trace.WithAttributes(attrs...))
	t.spans[spanKey{e.RunID, e.Seq}] = span
	return ctx
}

// end closes the span opened by the event's cause. Called with t.mu held.
func (t *Tracer) end(e aftrace.Event, failure *aftrace.Error, annotate func(trace.Span)) {
	if e.Cause == nil {
		return
	}
	k := spanKey{e.RunID, *e.Cause}
	span, ok := t.spans[k]
	if !ok {
		return
	}
	delete(t.spans, k)
	if annotate != nil {
		annotate(span)
	}
	if failure != nil {
		span.SetAttributes(attribute.String("error.type", failure.Code))
		span.SetStatus(codes.Error, failure.Message)
	} else {
		span.SetStatus(codes.Ok, "")
	}
	span.End(trace.WithTimestamp(e.Time))
}
