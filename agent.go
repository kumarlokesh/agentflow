package agentflow

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/kumarlokesh/agentflow/observe"
	"github.com/kumarlokesh/agentflow/policy"
	"github.com/kumarlokesh/agentflow/schema"
)

// DefaultMaxSteps is the default maximum number of agent loop iterations.
const DefaultMaxSteps = 20

// AgentConfig configures an Agent instance.
type AgentConfig struct {
	// Name identifies this agent (used in logs and events).
	Name string
	// Instructions is the system prompt prepended to every LLM call.
	Instructions string
	// LLM is the language model backend.
	LLM LLM
	// Tools are the capabilities available to the agent.
	Tools []Tool
	// MaxSteps limits the number of Observe-Think-Act iterations.
	// Zero means DefaultMaxSteps.
	MaxSteps int
	// Store persists events. If nil, events are not persisted.
	Store EventStore
	// Logger is the structured logger. If nil, slog.Default() is used.
	Logger *slog.Logger
	// ValidateToolParams enables JSON Schema validation of tool parameters.
	// Defaults to true.
	ValidateToolParams *bool
	// Hook receives callbacks at key points during agent execution.
	// If nil, no hooks are fired.
	Hook observe.Hook
	// Policy is evaluated before each tool call. If it returns Deny,
	// the tool call is skipped. If nil, all tool calls are allowed.
	Policy policy.Checker
	// TimeoutEnforcer wraps tool execution contexts with deadlines.
	// If nil, no timeouts are enforced.
	TimeoutEnforcer *policy.TimeoutEnforcer
}

// RunResult is the outcome of a single agent run.
type RunResult struct {
	// RunID uniquely identifies this run.
	RunID string
	// Output is the agent's final text response.
	Output string
	// Steps is the number of loop iterations executed.
	Steps int
	// Events is the ordered list of events produced during the run.
	Events []Event
	// Duration is the total wall-clock time of the run.
	Duration time.Duration
}

// Agent implements the Observe-Think-Act loop. It is the core runtime of
// agentflow. Each call to Run executes one complete agent session: the agent
// receives a task, reasons about it using the LLM, calls tools as needed,
// and returns a final answer.
//
// Every action is recorded as an Event, producing an append-only log that
// can be persisted via an EventStore and replayed deterministically.
type Agent struct {
	name            string
	instructions    string
	llm             LLM
	registry        *ToolRegistry
	maxSteps        int
	store           EventStore
	logger          *slog.Logger
	validateTool    bool
	hook            observe.Hook
	policy          policy.Checker
	timeoutEnforcer *policy.TimeoutEnforcer
}

// NewAgent creates an Agent from the provided configuration.
func NewAgent(cfg AgentConfig) (*Agent, error) {
	if cfg.LLM == nil {
		return nil, ErrNoLLM
	}

	maxSteps := cfg.MaxSteps
	if maxSteps <= 0 {
		maxSteps = DefaultMaxSteps
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	validateTool := true
	if cfg.ValidateToolParams != nil {
		validateTool = *cfg.ValidateToolParams
	}

	reg := NewToolRegistry()
	for _, t := range cfg.Tools {
		if err := reg.Register(t); err != nil {
			return nil, fmt.Errorf("agentflow: register tools: %w", err)
		}
	}

	return &Agent{
		name:            cfg.Name,
		instructions:    cfg.Instructions,
		llm:             cfg.LLM,
		registry:        reg,
		maxSteps:        maxSteps,
		store:           cfg.Store,
		logger:          logger,
		validateTool:    validateTool,
		hook:            cfg.Hook,
		policy:          cfg.Policy,
		timeoutEnforcer: cfg.TimeoutEnforcer,
	}, nil
}

// Run executes the agent loop for the given task. It blocks until the agent
// produces a final answer, exceeds MaxSteps, or the context is cancelled.
func (a *Agent) Run(ctx context.Context, task string) (*RunResult, error) {
	runID := newUUID()
	start := nowUTC()
	log := a.logger.With("run_id", runID, "agent", a.name)

	log.Info("run started", "task", task, "max_steps", a.maxSteps)

	if a.hook != nil {
		a.hook.OnRunStart(ctx, runID, task)
	}

	// Build initial messages.
	messages := make([]Message, 0, 4)
	if a.instructions != "" {
		messages = append(messages, Message{Role: "system", Content: a.instructions})
	}
	messages = append(messages, Message{Role: "user", Content: task})

	// Collect tool schemas for the LLM.
	toolSchemas := a.registry.Schemas()
	toolNames := a.registry.Names()

	// Emit run_start event.
	var allEvents []Event
	if err := a.emit(ctx, &allEvents, EventRunStart, runID, 0, RunStartData{
		Task:         task,
		Instructions: a.instructions,
		Tools:        toolNames,
		MaxSteps:     a.maxSteps,
	}); err != nil {
		return nil, err
	}

	var finalOutput string
	step := 0

	for step < a.maxSteps {
		// Check context before each step.
		if err := ctx.Err(); err != nil {
			a.emitRunEnd(ctx, &allEvents, runID, step, "cancelled", "", err.Error(), start)
			return nil, ErrRunCancelled
		}

		stepStart := nowUTC()
		log.Info("step started", "step", step)

		if a.hook != nil {
			a.hook.OnStepStart(ctx, runID, step)
		}

		if err := a.emit(ctx, &allEvents, EventStepStart, runID, step, StepStartData{
			StepIndex: step,
		}); err != nil {
			return nil, err
		}

		// --- THINK: Call the LLM ---
		llmReq := &LLMRequest{
			Messages: messages,
			Tools:    toolSchemas,
		}

		if err := a.emit(ctx, &allEvents, EventLLMRequest, runID, step, LLMRequestData{
			Messages: messages,
			Tools:    toolSchemas,
		}); err != nil {
			return nil, err
		}

		llmStart := nowUTC()
		llmResp, err := a.llm.ChatCompletion(ctx, llmReq)
		llmDur := nowUTC().Sub(llmStart)
		if err != nil {
			if a.hook != nil {
				a.hook.OnLLMCall(ctx, runID, step, 0, 0, 0, llmDur, err)
			}
			// Record LLM error as an event and fail the run.
			_ = a.emit(ctx, &allEvents, EventError, runID, step, ErrorData{
				Message: err.Error(),
				Code:    "llm_error",
			})
			a.emitRunEnd(ctx, &allEvents, runID, step, "failed", "", err.Error(), start)
			if a.hook != nil {
				a.hook.OnRunEnd(ctx, runID, step, nowUTC().Sub(start), err)
			}
			return nil, &LLMError{Err: err}
		}

		// Fire LLM hook with token usage.
		if a.hook != nil {
			var pt, ct, tt int
			if llmResp.Usage != nil {
				pt = llmResp.Usage.PromptTokens
				ct = llmResp.Usage.CompletionTokens
				tt = llmResp.Usage.TotalTokens
			}
			a.hook.OnLLMCall(ctx, runID, step, pt, ct, tt, llmDur, nil)
		}

		if err := a.emit(ctx, &allEvents, EventLLMResponse, runID, step, LLMResponseData{
			Content:   llmResp.Content,
			ToolCalls: llmResp.ToolCalls,
			Usage:     llmResp.Usage,
		}); err != nil {
			return nil, err
		}

		// --- ACT: Execute tool calls or finish ---
		if len(llmResp.ToolCalls) == 0 {
			// No tool calls — the LLM is giving a final answer.
			finalOutput = llmResp.Content
			log.Info("agent produced final answer", "step", step)

			stepDur := nowUTC().Sub(stepStart)
			if err := a.emit(ctx, &allEvents, EventStepEnd, runID, step, StepEndData{
				StepIndex: step,
				Duration:  stepDur,
			}); err != nil {
				return nil, err
			}
			if a.hook != nil {
				a.hook.OnStepEnd(ctx, runID, step, stepDur)
			}
			break
		}

		// Add assistant message with tool calls to conversation.
		messages = append(messages, Message{
			Role:      "assistant",
			ToolCalls: llmResp.ToolCalls,
		})

		// Execute each tool call.
		for _, tc := range llmResp.ToolCalls {
			toolResult, toolErr := a.executeTool(ctx, &allEvents, runID, step, tc)
			if toolErr != nil {
				var te *ToolError
				if errors.As(toolErr, &te) {
					// Tool execution error — record but continue (non-fatal).
					log.Warn("tool execution failed",
						"tool", tc.Name, "call_id", tc.ID, "error", toolErr)
				} else {
					// Non-tool errors (e.g. event persistence failures) are fatal.
					a.emitRunEnd(ctx, &allEvents, runID, step, "failed", "", toolErr.Error(), start)
					if a.hook != nil {
						a.hook.OnRunEnd(ctx, runID, step, nowUTC().Sub(start), toolErr)
					}
					return nil, toolErr
				}
			}

			// Add tool result to conversation history.
			resultContent := toolResult.Output
			if toolResult.Error != "" {
				resultContent = "Error: " + toolResult.Error
			}
			messages = append(messages, Message{
				Role:       "tool",
				Content:    resultContent,
				ToolCallID: tc.ID,
			})
		}

		stepDur := nowUTC().Sub(stepStart)
		if err := a.emit(ctx, &allEvents, EventStepEnd, runID, step, StepEndData{
			StepIndex: step,
			Duration:  stepDur,
		}); err != nil {
			return nil, err
		}
		if a.hook != nil {
			a.hook.OnStepEnd(ctx, runID, step, stepDur)
		}

		step++
	}

	// Determine final status.
	if step >= a.maxSteps && finalOutput == "" {
		a.emitRunEnd(ctx, &allEvents, runID, step, "failed", "", ErrMaxStepsExceeded.Error(), start)
		if a.hook != nil {
			a.hook.OnRunEnd(ctx, runID, step, nowUTC().Sub(start), ErrMaxStepsExceeded)
		}
		return nil, ErrMaxStepsExceeded
	}

	a.emitRunEnd(ctx, &allEvents, runID, step, "completed", finalOutput, "", start)

	totalDur := nowUTC().Sub(start)
	log.Info("run completed", "steps", step+1, "duration", totalDur)

	if a.hook != nil {
		a.hook.OnRunEnd(ctx, runID, step+1, totalDur, nil)
	}

	return &RunResult{
		RunID:    runID,
		Output:   finalOutput,
		Steps:    step + 1,
		Events:   allEvents,
		Duration: totalDur,
	}, nil
}

// executeTool runs a single tool call with optional schema validation.
func (a *Agent) executeTool(ctx context.Context, allEvents *[]Event, runID string, step int, tc ToolCallRequest) (*ToolResult, error) {
	log := a.logger.With("run_id", runID, "step", step, "tool", tc.Name, "call_id", tc.ID)

	if a.policy != nil {
		decision, err := a.policy.CheckTool(ctx, policy.ToolRequest{
			ToolName: tc.Name,
			CallID:   tc.ID,
			Step:     step,
		})
		if err != nil || decision == policy.Deny {
			msg := "tool call denied by policy"
			if err != nil {
				msg = fmt.Sprintf("tool call denied by policy: %v", err)
			}
			result := &ToolResult{Error: msg}
			if err := a.emit(ctx, allEvents, EventToolResult, runID, step, ToolResultData{
				ToolName: tc.Name,
				CallID:   tc.ID,
				Error:    msg,
			}); err != nil {
				return result, err
			}
			return result, &ToolError{ToolName: tc.Name, CallID: tc.ID, Err: fmt.Errorf("policy: %s", msg)}
		}
	}

	if err := a.emit(ctx, allEvents, EventToolCall, runID, step, ToolCallData{
		ToolName: tc.Name,
		CallID:   tc.ID,
		Input:    tc.Arguments,
	}); err != nil {
		return &ToolResult{Error: fmt.Sprintf("emit tool_call event: %v", err)}, err
	}

	tool := a.registry.Get(tc.Name)
	if tool == nil {
		result := &ToolResult{Error: fmt.Sprintf("tool %q not found", tc.Name)}
		if err := a.emit(ctx, allEvents, EventToolResult, runID, step, ToolResultData{
			ToolName: tc.Name,
			CallID:   tc.ID,
			Error:    result.Error,
		}); err != nil {
			return result, err
		}
		return result, &ToolError{ToolName: tc.Name, CallID: tc.ID, Err: ErrToolNotFound}
	}

	if a.validateTool {
		toolSchema := tool.Schema()
		if len(toolSchema.Parameters) > 0 && len(tc.Arguments) > 0 {
			if err := schema.Validate(toolSchema.Parameters, tc.Arguments); err != nil {
				result := &ToolResult{Error: fmt.Sprintf("invalid parameters: %v", err)}
				if err := a.emit(ctx, allEvents, EventToolResult, runID, step, ToolResultData{
					ToolName: tc.Name,
					CallID:   tc.ID,
					Error:    result.Error,
				}); err != nil {
					return result, err
				}
				return result, &ToolError{ToolName: tc.Name, CallID: tc.ID, Err: fmt.Errorf("%w: %v", ErrInvalidToolParams, err)}
			}
		}
	}

	// Execute the tool with optional timeout.
	log.Info("executing tool")
	execCtx := ctx
	var cancelTimeout context.CancelFunc
	if a.timeoutEnforcer != nil {
		execCtx, cancelTimeout = a.timeoutEnforcer.WrapContext(ctx, tc.Name)
	}
	toolStart := nowUTC()
	result, err := tool.Execute(execCtx, tc.Arguments)
	toolDur := nowUTC().Sub(toolStart)
	if cancelTimeout != nil {
		cancelTimeout()
	}

	if err != nil {
		errResult := &ToolResult{Error: err.Error()}
		if err := a.emit(ctx, allEvents, EventToolResult, runID, step, ToolResultData{
			ToolName: tc.Name,
			CallID:   tc.ID,
			Error:    err.Error(),
			Duration: toolDur,
		}); err != nil {
			return errResult, err
		}
		if a.hook != nil {
			a.hook.OnToolCall(ctx, runID, step, tc.Name, toolDur, err)
		}
		return errResult, &ToolError{ToolName: tc.Name, CallID: tc.ID, Err: err}
	}

	if err := a.emit(ctx, allEvents, EventToolResult, runID, step, ToolResultData{
		ToolName: tc.Name,
		CallID:   tc.ID,
		Output:   result.Output,
		Duration: toolDur,
	}); err != nil {
		return result, err
	}

	if a.hook != nil {
		a.hook.OnToolCall(ctx, runID, step, tc.Name, toolDur, nil)
	}

	log.Info("tool executed", "duration", toolDur)
	return result, nil
}

// emit creates and persists an event.
func (a *Agent) emit(ctx context.Context, allEvents *[]Event, eventType EventType, runID string, step int, data any) error {
	event, err := NewEvent(eventType, runID, step, data)
	if err != nil {
		return fmt.Errorf("agentflow: create event: %w", err)
	}

	*allEvents = append(*allEvents, event)

	if a.store != nil {
		if err := a.store.Append(ctx, event); err != nil {
			return &StoreError{Op: "append", Err: err}
		}
	}

	return nil
}

// emitRunEnd is a convenience for emitting the run_end event.
func (a *Agent) emitRunEnd(ctx context.Context, allEvents *[]Event, runID string, step int, status, output, errMsg string, start time.Time) {
	dur := nowUTC().Sub(start)
	// Best-effort - don't propagate errors from the final event.
	_ = a.emit(ctx, allEvents, EventRunEnd, runID, step, RunEndData{
		Status:   status,
		Output:   output,
		Error:    errMsg,
		Steps:    step,
		Duration: dur,
	})
}
