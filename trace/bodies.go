package trace

import (
	"encoding/json"
	"time"

	"github.com/kumarlokesh/agentflow/llm"
)

// RunKind is what executes a run.
type RunKind string

const (
	RunKindAgent    RunKind = "agent"
	RunKindWorkflow RunKind = "workflow"
)

// RunStatus is how a run ended. StatusIncomplete is never written; readers
// report it for runs without a run.finished event.
type RunStatus string

const (
	StatusCompleted  RunStatus = "completed"
	StatusFailed     RunStatus = "failed"
	StatusCancelled  RunStatus = "cancelled"
	StatusMaxSteps   RunStatus = "max_steps"
	StatusDiverged   RunStatus = "diverged"
	StatusIncomplete RunStatus = "incomplete"
)

// Error is a recorded failure (spec §7).
type Error struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// Error codes for tool and spawn failures. LLM codes are in package llm.
const (
	ErrCodeNotFound      = "not_found"
	ErrCodeInvalidInput  = "invalid_input"
	ErrCodeDenied        = "denied"
	ErrCodeTimeout       = "timeout"
	ErrCodePanic         = "panic"
	ErrCodeCancelled     = "cancelled"
	ErrCodeInternal      = "internal"
	ErrCodeLimitExceeded = "limit_exceeded"
)

// AgentInfo identifies the agent (or workflow) definition that ran.
type AgentInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
}

// ParentRef links a child run to its parent.
type ParentRef struct {
	RunID string `json:"run_id"`
}

// Config is an agent run's model configuration (spec §7). Tools are
// sorted by name.
type Config struct {
	Model      string          `json:"model"`
	System     llm.Blocks      `json:"system"`
	Tools      []llm.ToolSpec  `json:"tools"`
	ToolChoice *llm.ToolChoice `json:"tool_choice"`
	Params     llm.Params      `json:"params"`
	MaxSteps   int             `json:"max_steps"`
}

// OriginKind says how a trace came to exist.
type OriginKind string

const (
	OriginLive   OriginKind = "live"
	OriginReplay OriginKind = "replay"
	OriginFork   OriginKind = "fork"
)

// Origin is recorded on a trace's root run (spec §7).
type Origin struct {
	Kind          OriginKind      `json:"kind"`
	SourceTraceID string          `json:"source_trace_id,omitempty"`
	Mode          string          `json:"mode,omitempty"`      // replay mode
	At            *ForkPoint      `json:"at,omitempty"`        // fork point requested
	ActualAt      *ForkPoint      `json:"actual_at,omitempty"` // fork point used, if it moved
	Overrides     json.RawMessage `json:"overrides,omitempty"` // fork overrides, verbatim
	Group         string          `json:"group,omitempty"`
	Sample        *int            `json:"sample,omitempty"`
	Arm           string          `json:"arm,omitempty"`
}

// ForkPoint addresses an effect in a source trace.
type ForkPoint struct {
	Path   string  `json:"path"`
	Effect string  `json:"effect,omitempty"`
	Seq    *uint64 `json:"seq,omitempty"`
}

// RunStarted is always seq 0 of a run.
type RunStarted struct {
	RunKind  RunKind    `json:"kind"`
	Path     string     `json:"path"`
	Parent   *ParentRef `json:"parent"`
	Agent    AgentInfo  `json:"agent"`
	Task     llm.Blocks `json:"task"`
	Config   *Config    `json:"config"`
	ConfigFP string     `json:"config_fp,omitempty"`
	Origin   *Origin    `json:"origin,omitempty"`
}

// RunFinished ends a run.
type RunFinished struct {
	Status RunStatus  `json:"status"`
	Output llm.Blocks `json:"output"`
	Error  *Error     `json:"error,omitempty"`
	Usage  llm.Usage  `json:"usage"` // this run only
	Steps  int        `json:"steps"`
}

// RunSpawned is written in the parent when a child run starts.
type RunSpawned struct {
	ChildRunID  string     `json:"child_run_id"`
	Path        string     `json:"path"`
	Agent       string     `json:"agent"`
	Task        llm.Blocks `json:"task"`
	Fingerprint string     `json:"fingerprint"`
}

// RunJoined is written in the parent when that child run finishes.
type RunJoined struct {
	ChildRunID string        `json:"child_run_id"`
	Status     RunStatus     `json:"status"`
	Output     llm.Blocks    `json:"output"`
	Usage      llm.Usage     `json:"usage"` // the child's whole subtree
	Duration   time.Duration `json:"duration_ns"`
	Override   bool          `json:"override,omitempty"`
}

// RunModeChanged records a run switching from replaying to live.
type RunModeChanged struct {
	Mode   string          `json:"mode"`
	Reason string          `json:"reason"`
	Detail json.RawMessage `json:"detail,omitempty"`
}

// Mode-change reasons.
const (
	ReasonDivergence  = "divergence"
	ReasonForkPoint   = "fork_point"
	ReasonSharedState = "shared_state"
	ReasonNewRun      = "new_run"
)

// MessageAppended adds one message to the run's conversation. Exactly one of
// Message or SourceSeq is set; SourceSeq refers to an llm.completed event
// whose message is appended as-is.
type MessageAppended struct {
	Message   *llm.Message `json:"message,omitempty"`
	SourceSeq *uint64      `json:"source_seq,omitempty"`
}

// ConversationReplaced replaces the whole conversation (compaction).
type ConversationReplaced struct {
	Messages []llm.Message `json:"messages"`
}

// ConfigChanged changes part of the configuration mid-run. Nil fields are
// unchanged; Tools uses a pointer so that "remove all tools" is expressible.
type ConfigChanged struct {
	Model  *string         `json:"model,omitempty"`
	System *llm.Blocks     `json:"system,omitempty"`
	Tools  *[]llm.ToolSpec `json:"tools,omitempty"`
	Params *llm.Params     `json:"params,omitempty"`
}

// ContextRef says which conversation prefix and config an LLM call used.
type ContextRef struct {
	Messages  int    `json:"messages"`
	ConfigSeq uint64 `json:"config_seq"`
}

// Ephemeral is a message inserted into one request only.
type Ephemeral struct {
	Index   int         `json:"index"`
	Message llm.Message `json:"message"`
}

// LLMStarted records an LLM call as a reference into the conversation.
type LLMStarted struct {
	Step        int         `json:"step"`
	Context     ContextRef  `json:"context"`
	Ephemeral   []Ephemeral `json:"ephemeral,omitempty"`
	Fingerprint string      `json:"fingerprint"`
}

// LLMCompleted records a model response.
type LLMCompleted struct {
	Step       int            `json:"step"`
	Message    llm.Message    `json:"message"`
	StopReason llm.StopReason `json:"stop_reason"`
	Usage      llm.Usage      `json:"usage"`
	Model      string         `json:"model,omitempty"`
	Attempts   int            `json:"attempts,omitempty"`
	Duration   time.Duration  `json:"duration_ns"`
	Ext        llm.Ext        `json:"ext,omitempty"`
	Override   bool           `json:"override,omitempty"`
	// Intercepted: an interceptor produced this outcome; the call never
	// reached the world.
	Intercepted bool `json:"intercepted,omitempty"`
}

// LLMFailed records a failed model call.
type LLMFailed struct {
	Step        int           `json:"step"`
	Error       Error         `json:"error"`
	Attempts    int           `json:"attempts,omitempty"`
	Duration    time.Duration `json:"duration_ns"`
	Intercepted bool          `json:"intercepted,omitempty"`
}

// AccessMode is how a tool touches a shared resource.
type AccessMode string

const (
	AccessRead  AccessMode = "read"
	AccessWrite AccessMode = "write"
)

// Access is a declared touch of a shared resource.
type Access struct {
	Resource string     `json:"resource"`
	Mode     AccessMode `json:"mode"`
}

// ToolStarted records a tool call. Accesses is nil when unknown and points
// to an empty slice when the tool declared it touches nothing shared.
type ToolStarted struct {
	Step        int             `json:"step"`
	CallID      string          `json:"call_id"`
	Tool        string          `json:"tool"`
	Input       json.RawMessage `json:"input"`
	Fingerprint string          `json:"fingerprint"`
	Accesses    *[]Access       `json:"accesses,omitempty"`
}

// ToolCompleted records a tool result (IsError: the model sees an error).
type ToolCompleted struct {
	CallID      string        `json:"call_id"`
	Content     llm.Blocks    `json:"content"`
	IsError     bool          `json:"is_error"`
	Duration    time.Duration `json:"duration_ns"`
	Override    bool          `json:"override,omitempty"`
	Intercepted bool          `json:"intercepted,omitempty"`
}

// ToolFailed records a tool failure (not found, denied, timeout, panic, …).
type ToolFailed struct {
	CallID      string        `json:"call_id"`
	Error       Error         `json:"error"`
	Duration    time.Duration `json:"duration_ns"`
	Intercepted bool          `json:"intercepted,omitempty"`
}

// EffectStarted records a user-defined effect.
type EffectStarted struct {
	Name        string          `json:"name"`
	Input       json.RawMessage `json:"input"`
	Fingerprint string          `json:"fingerprint"`
}

// EffectCompleted records a user-defined effect's output.
type EffectCompleted struct {
	Name        string          `json:"name"`
	Output      json.RawMessage `json:"output"`
	Duration    time.Duration   `json:"duration_ns"`
	Override    bool            `json:"override,omitempty"`
	Intercepted bool            `json:"intercepted,omitempty"`
}

// EffectFailed records a user-defined effect's failure.
type EffectFailed struct {
	Name        string        `json:"name"`
	Error       Error         `json:"error"`
	Duration    time.Duration `json:"duration_ns"`
	Intercepted bool          `json:"intercepted,omitempty"`
}

// Annotation is a marker written by the application or the runtime. Keys
// starting with "agentflow." are reserved for the runtime.
type Annotation struct {
	Key   string          `json:"key"`
	Value json.RawMessage `json:"value"`
}

func (*RunStarted) Kind() Kind           { return KindRunStarted }
func (*RunFinished) Kind() Kind          { return KindRunFinished }
func (*RunSpawned) Kind() Kind           { return KindRunSpawned }
func (*RunJoined) Kind() Kind            { return KindRunJoined }
func (*RunModeChanged) Kind() Kind       { return KindRunModeChanged }
func (*MessageAppended) Kind() Kind      { return KindMessageAppended }
func (*ConversationReplaced) Kind() Kind { return KindConversationReplaced }
func (*ConfigChanged) Kind() Kind        { return KindConfigChanged }
func (*LLMStarted) Kind() Kind           { return KindLLMStarted }
func (*LLMCompleted) Kind() Kind         { return KindLLMCompleted }
func (*LLMFailed) Kind() Kind            { return KindLLMFailed }
func (*ToolStarted) Kind() Kind          { return KindToolStarted }
func (*ToolCompleted) Kind() Kind        { return KindToolCompleted }
func (*ToolFailed) Kind() Kind           { return KindToolFailed }
func (*EffectStarted) Kind() Kind        { return KindEffectStarted }
func (*EffectCompleted) Kind() Kind      { return KindEffectCompleted }
func (*EffectFailed) Kind() Kind         { return KindEffectFailed }
func (*Annotation) Kind() Kind           { return KindAnnotation }
