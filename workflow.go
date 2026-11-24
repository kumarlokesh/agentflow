package agentflow

import (
	"context"
	"fmt"
	"log/slog"
	"runtime/debug"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
)

// WorkflowFunc is the code of a workflow. Its return value is the run's
// output.
type WorkflowFunc func(ctx context.Context, task string) (string, error)

// Workflow orchestrates agents from Go code: it starts them with Spawn and
// SpawnAll (or their Run methods), in sequence or in parallel, and every
// run lands in one trace as a tree.
//
// A workflow's code runs again during replay, and only its effects come
// from the recording, so it must be deterministic given those effects:
// anything nondeterministic goes through Do, Now, NewID, Spawn or SpawnAll.
// To call a model, spawn an agent.
type Workflow struct {
	name string
	fn   WorkflowFunc
}

// NewWorkflow returns a workflow named name (its name in run paths and
// traces) that runs fn.
func NewWorkflow(name string, fn WorkflowFunc) *Workflow {
	return &Workflow{name: name, fn: fn}
}

// Name returns the workflow's name.
func (w *Workflow) Name() string { return w.name }

// Run runs the workflow on task. Like Agent.Run, it starts a new trace, or
// a child run when ctx belongs to a run in progress. It returns the result
// and, unless the run completed, a *RunError.
func (w *Workflow) Run(ctx context.Context, task string, opts ...RunOption) (*RunResult, error) {
	return runEntrypoint(ctx, w, task, opts)
}

func (w *Workflow) def() *runDef {
	return &runDef{
		kind: trace.RunKindWorkflow, name: w.name,
		logger: slog.New(slog.DiscardHandler), clock: time.Now, newID: trace.NewID,
		body: w.body,
	}
}

func (w *Workflow) body(ctx context.Context, _ *runState, task llm.Blocks) (trace.RunStatus, llm.Blocks, int, error) {
	out, err := w.call(ctx, llm.TextOf(task))
	if err != nil {
		return endStatus(ctx, err), nil, 0, err
	}
	return trace.StatusCompleted, llm.Blocks{llm.Text{Text: out}}, 0, nil
}

func (w *Workflow) call(ctx context.Context, task string) (out string, err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("workflow %s panicked: %v\n%s", w.name, p, truncate(string(debug.Stack()), 4096))
		}
	}()
	return w.fn(ctx, task)
}
