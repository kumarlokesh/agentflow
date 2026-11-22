// Package trace defines agentflow's trace format (RFC-0001): the event
// envelope and every event body, canonical fingerprints, the conversation
// fold that reconstructs exact LLM requests, the Store interfaces, the
// Writer used by the runtime, and portable bundles.
//
// A trace is a tree of runs. Each run is an append-only sequence of events
// numbered by seq (0, 1, 2, …). Events are encoded as one JSON object per
// line; large strings are moved to a content-addressed blob store.
package trace

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// Version is the trace format version written in every event's "v" field.
const Version = 1

// Kind identifies an event type.
type Kind string

// Event kinds (RFC-0001 §4).
const (
	KindRunStarted           Kind = "run.started"
	KindRunFinished          Kind = "run.finished"
	KindRunSpawned           Kind = "run.spawned"
	KindRunJoined            Kind = "run.joined"
	KindRunModeChanged       Kind = "run.mode_changed"
	KindMessageAppended      Kind = "message.appended"
	KindConversationReplaced Kind = "conversation.replaced"
	KindConfigChanged        Kind = "config.changed"
	KindLLMStarted           Kind = "llm.started"
	KindLLMCompleted         Kind = "llm.completed"
	KindLLMFailed            Kind = "llm.failed"
	KindToolStarted          Kind = "tool.started"
	KindToolCompleted        Kind = "tool.completed"
	KindToolFailed           Kind = "tool.failed"
	KindEffectStarted        Kind = "effect.started"
	KindEffectCompleted      Kind = "effect.completed"
	KindEffectFailed         Kind = "effect.failed"
	KindAnnotation           Kind = "annotation"
)

// Body is an event's kind-specific payload. Concrete bodies are the types in
// bodies.go; kinds this version doesn't know decode to UnknownBody.
type Body interface {
	Kind() Kind
}

// Event is one immutable record in a run, addressed by (RunID, Seq).
type Event struct {
	TraceID string
	RunID   string
	Seq     uint64
	Time    time.Time // wall clock; informational only, never compared
	Cause   *uint64   // seq of the event in the same run that this one answers
	Body    Body
}

// Kind returns the event's kind, taken from its body.
func (e Event) Kind() Kind {
	if e.Body == nil {
		return ""
	}
	return e.Body.Kind()
}

// UnknownBody holds the body of an event kind this version doesn't know.
// It is preserved verbatim so that traces written by newer versions can be
// read and copied without loss.
type UnknownBody struct {
	EventKind Kind
	Raw       json.RawMessage
}

// Kind implements Body.
func (u UnknownBody) Kind() Kind { return u.EventKind }

// MarshalJSON implements json.Marshaler.
func (u UnknownBody) MarshalJSON() ([]byte, error) {
	if len(u.Raw) == 0 {
		return []byte("{}"), nil
	}
	return u.Raw, nil
}

var (
	// ErrPrototypeTrace is returned when reading an event written by the
	// unreleased prototype (it has "schema_version" instead of "v").
	ErrPrototypeTrace = errors.New("trace: prototype trace, unsupported")
	// ErrUnsupportedVersion is returned for an unknown format version.
	ErrUnsupportedVersion = errors.New("trace: unsupported format version")
	// ErrSeqConflict is returned by Store.Append when seqs are not contiguous
	// with what the store already holds for the run.
	ErrSeqConflict = errors.New("trace: seq conflict")
	// ErrBlobNotFound is returned when a referenced blob is missing.
	ErrBlobNotFound = errors.New("trace: blob not found")
	// ErrEventTooLarge is returned when an encoded event exceeds MaxEventSize.
	ErrEventTooLarge = errors.New("trace: event too large")
	// ErrRunNotFound is returned when a run has no events in the store.
	ErrRunNotFound = errors.New("trace: run not found")
)

// timeLayout is RFC 3339 in UTC with microsecond precision.
const timeLayout = "2006-01-02T15:04:05.000000Z07:00"

// envelope is the JSON shape of an event (RFC-0001 §3).
type envelope struct {
	V             int             `json:"v"`
	SchemaVersion *int            `json:"schema_version,omitempty"` // prototype marker; read only
	TraceID       string          `json:"trace_id"`
	RunID         string          `json:"run_id"`
	Seq           uint64          `json:"seq"`
	Time          string          `json:"time"`
	Kind          Kind            `json:"kind"`
	Cause         *uint64         `json:"cause,omitempty"`
	Body          json.RawMessage `json:"body"`
}

// newBody returns a pointer to a zero body for a known kind.
func newBody(k Kind) (Body, bool) {
	switch k {
	case KindRunStarted:
		return &RunStarted{}, true
	case KindRunFinished:
		return &RunFinished{}, true
	case KindRunSpawned:
		return &RunSpawned{}, true
	case KindRunJoined:
		return &RunJoined{}, true
	case KindRunModeChanged:
		return &RunModeChanged{}, true
	case KindMessageAppended:
		return &MessageAppended{}, true
	case KindConversationReplaced:
		return &ConversationReplaced{}, true
	case KindConfigChanged:
		return &ConfigChanged{}, true
	case KindLLMStarted:
		return &LLMStarted{}, true
	case KindLLMCompleted:
		return &LLMCompleted{}, true
	case KindLLMFailed:
		return &LLMFailed{}, true
	case KindToolStarted:
		return &ToolStarted{}, true
	case KindToolCompleted:
		return &ToolCompleted{}, true
	case KindToolFailed:
		return &ToolFailed{}, true
	case KindEffectStarted:
		return &EffectStarted{}, true
	case KindEffectCompleted:
		return &EffectCompleted{}, true
	case KindEffectFailed:
		return &EffectFailed{}, true
	case KindAnnotation:
		return &Annotation{}, true
	}
	return nil, false
}

// decodeBody decodes raw JSON into the body type for kind k.
func decodeBody(k Kind, raw json.RawMessage) (Body, error) {
	b, ok := newBody(k)
	if !ok {
		return UnknownBody{EventKind: k, Raw: append(json.RawMessage(nil), raw...)}, nil
	}
	if err := json.Unmarshal(raw, b); err != nil {
		return nil, fmt.Errorf("trace: decode %s body: %w", k, err)
	}
	return b, nil
}

// NewID returns a new UUIDv7 string, used for trace and run IDs.
func NewID() string {
	return uuid.Must(uuid.NewV7()).String()
}

// Cause is a convenience for building an Event.Cause pointer.
func Cause(seq uint64) *uint64 { return &seq }
