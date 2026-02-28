package agentflow

import (
	"encoding/json"
	"testing"
	"time"
)

func TestNewEvent(t *testing.T) {
	// Pin UUID and time for deterministic tests.
	origUUID := newUUID
	origNow := nowUTC
	defer func() { newUUID = origUUID; nowUTC = origNow }()

	newUUID = func() string { return "test-uuid-001" }
	fixedTime := time.Date(2025, 6, 15, 12, 0, 0, 0, time.UTC)
	nowUTC = func() time.Time { return fixedTime }

	data := RunStartData{
		Task:     "solve math",
		MaxSteps: 10,
		Tools:    []string{"calculator"},
	}

	event, err := NewEvent(EventRunStart, "run-123", 0, data)
	if err != nil {
		t.Fatalf("NewEvent() error = %v", err)
	}

	if event.ID != "test-uuid-001" {
		t.Errorf("ID = %q, want %q", event.ID, "test-uuid-001")
	}
	if event.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", event.SchemaVersion, SchemaVersion)
	}
	if event.Type != EventRunStart {
		t.Errorf("Type = %q, want %q", event.Type, EventRunStart)
	}
	if event.RunID != "run-123" {
		t.Errorf("RunID = %q, want %q", event.RunID, "run-123")
	}
	if event.StepIndex != 0 {
		t.Errorf("StepIndex = %d, want 0", event.StepIndex)
	}
	if !event.Timestamp.Equal(fixedTime) {
		t.Errorf("Timestamp = %v, want %v", event.Timestamp, fixedTime)
	}
	if event.Data == nil {
		t.Fatal("Data is nil")
	}
}

func TestNewEvent_MarshalError(t *testing.T) {
	// Channels cannot be JSON-marshalled.
	_, err := NewEvent(EventError, "run-1", 0, make(chan int))
	if err == nil {
		t.Fatal("expected error for unmarshalable data")
	}
}

func TestEvent_DecodeData(t *testing.T) {
	original := ToolCallData{
		ToolName: "calculator",
		CallID:   "call-1",
		Input:    json.RawMessage(`{"expression":"2+2"}`),
	}

	event, err := NewEvent(EventToolCall, "run-1", 0, original)
	if err != nil {
		t.Fatalf("NewEvent() error = %v", err)
	}

	var decoded ToolCallData
	if err := event.DecodeData(&decoded); err != nil {
		t.Fatalf("DecodeData() error = %v", err)
	}

	if decoded.ToolName != original.ToolName {
		t.Errorf("ToolName = %q, want %q", decoded.ToolName, original.ToolName)
	}
	if decoded.CallID != original.CallID {
		t.Errorf("CallID = %q, want %q", decoded.CallID, original.CallID)
	}
}

func TestEvent_DecodeData_NilData(t *testing.T) {
	event := Event{ID: "test"}
	err := event.DecodeData(&struct{}{})
	if err == nil {
		t.Fatal("expected error for nil data")
	}
}

func TestEventTypes(t *testing.T) {
	// Verify all event type constants are distinct non-empty strings.
	types := []EventType{
		EventLLMRequest, EventLLMResponse,
		EventToolCall, EventToolResult,
		EventStepStart, EventStepEnd,
		EventRunStart, EventRunEnd,
		EventError,
	}

	seen := make(map[EventType]bool)
	for _, et := range types {
		if et == "" {
			t.Error("event type is empty string")
		}
		if seen[et] {
			t.Errorf("duplicate event type: %q", et)
		}
		seen[et] = true
	}
}

func TestTokenUsage_JSON(t *testing.T) {
	usage := TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}
	data, err := json.Marshal(usage)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}

	var decoded TokenUsage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if decoded != usage {
		t.Errorf("round-trip mismatch: got %+v, want %+v", decoded, usage)
	}
}
