package schema

import (
	"encoding/json"
	"testing"
)

func TestValidate_ObjectWithRequiredFields(t *testing.T) {
	schemaJSON := json.RawMessage(`{
		"type": "object",
		"properties": {
			"name":  {"type": "string"},
			"age":   {"type": "integer"},
			"email": {"type": "string"}
		},
		"required": ["name", "age"]
	}`)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{
			name:    "valid with all fields",
			value:   `{"name": "Alice", "age": 30, "email": "alice@example.com"}`,
			wantErr: false,
		},
		{
			name:    "valid with only required fields",
			value:   `{"name": "Bob", "age": 25}`,
			wantErr: false,
		},
		{
			name:    "missing required field name",
			value:   `{"age": 25}`,
			wantErr: true,
		},
		{
			name:    "missing required field age",
			value:   `{"name": "Alice"}`,
			wantErr: true,
		},
		{
			name:    "wrong type for name",
			value:   `{"name": 123, "age": 25}`,
			wantErr: true,
		},
		{
			name:    "wrong type for age (float instead of integer)",
			value:   `{"name": "Alice", "age": 25.5}`,
			wantErr: true,
		},
		{
			name:    "not an object",
			value:   `"just a string"`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_StringConstraints(t *testing.T) {
	minLen := 2
	maxLen := 10
	s := Schema{
		Type:      "string",
		MinLength: &minLen,
		MaxLength: &maxLen,
	}
	schemaJSON, _ := json.Marshal(s)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"valid", `"hello"`, false},
		{"too short", `"a"`, true},
		{"too long", `"this is way too long string"`, true},
		{"exact min", `"ab"`, false},
		{"wrong type", `42`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_NumberConstraints(t *testing.T) {
	min := 0.0
	max := 100.0
	s := Schema{
		Type:    "number",
		Minimum: &min,
		Maximum: &max,
	}
	schemaJSON, _ := json.Marshal(s)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"valid", `50`, false},
		{"min boundary", `0`, false},
		{"max boundary", `100`, false},
		{"below min", `-1`, true},
		{"above max", `101`, true},
		{"float valid", `50.5`, false},
		{"wrong type", `"not a number"`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_Array(t *testing.T) {
	schemaJSON := json.RawMessage(`{
		"type": "array",
		"items": {"type": "string"}
	}`)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"valid string array", `["a", "b", "c"]`, false},
		{"empty array", `[]`, false},
		{"invalid item type", `["a", 1, "c"]`, true},
		{"not an array", `"string"`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_Enum(t *testing.T) {
	schemaJSON := json.RawMessage(`{
		"type": "string",
		"enum": ["red", "green", "blue"]
	}`)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"valid enum value", `"red"`, false},
		{"invalid enum value", `"yellow"`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_Boolean(t *testing.T) {
	schemaJSON := json.RawMessage(`{"type": "boolean"}`)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{"true", `true`, false},
		{"false", `false`, false},
		{"not boolean", `"true"`, true},
		{"number", `1`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_Null(t *testing.T) {
	schemaJSON := json.RawMessage(`{"type": "null"}`)
	if err := Validate(schemaJSON, json.RawMessage(`null`)); err != nil {
		t.Errorf("null should be valid: %v", err)
	}
	if err := Validate(schemaJSON, json.RawMessage(`"not null"`)); err == nil {
		t.Error("string should not be valid for null type")
	}
}

func TestValidate_NestedObject(t *testing.T) {
	schemaJSON := json.RawMessage(`{
		"type": "object",
		"properties": {
			"address": {
				"type": "object",
				"properties": {
					"city": {"type": "string"},
					"zip":  {"type": "string"}
				},
				"required": ["city"]
			}
		},
		"required": ["address"]
	}`)

	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{
			name:    "valid nested",
			value:   `{"address": {"city": "NYC", "zip": "10001"}}`,
			wantErr: false,
		},
		{
			name:    "missing nested required",
			value:   `{"address": {"zip": "10001"}}`,
			wantErr: true,
		},
		{
			name:    "missing top-level required",
			value:   `{}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(schemaJSON, json.RawMessage(tt.value))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidate_NoTypeConstraint(t *testing.T) {
	schemaJSON := json.RawMessage(`{}`)
	// Any value should be accepted when no type is specified.
	for _, v := range []string{`"string"`, `42`, `true`, `null`, `[1,2]`, `{"a":1}`} {
		if err := Validate(schemaJSON, json.RawMessage(v)); err != nil {
			t.Errorf("value %s should be valid with no type constraint: %v", v, err)
		}
	}
}

func TestValidate_InvalidSchema(t *testing.T) {
	err := Validate(json.RawMessage(`not json`), json.RawMessage(`"value"`))
	if err == nil {
		t.Error("expected error for invalid schema JSON")
	}
}

func TestValidate_InvalidValue(t *testing.T) {
	err := Validate(json.RawMessage(`{"type":"string"}`), json.RawMessage(`not json`))
	if err == nil {
		t.Error("expected error for invalid value JSON")
	}
}

func TestValidationError_Error(t *testing.T) {
	ve := &ValidationError{
		Errors: []FieldError{
			{Path: "/name", Message: "required field missing"},
			{Path: "/age", Message: "expected integer"},
		},
	}
	got := ve.Error()
	if got == "" {
		t.Error("Error() returned empty string")
	}
}

func TestValidate_ToolParameterSchema(t *testing.T) {
	// Realistic tool parameter schema from an LLM function-calling scenario.
	schemaJSON := json.RawMessage(`{
		"type": "object",
		"properties": {
			"expression": {
				"type": "string",
				"description": "Mathematical expression to evaluate"
			}
		},
		"required": ["expression"]
	}`)

	valid := json.RawMessage(`{"expression": "2 + 2"}`)
	if err := Validate(schemaJSON, valid); err != nil {
		t.Errorf("valid tool params rejected: %v", err)
	}

	invalid := json.RawMessage(`{}`)
	if err := Validate(schemaJSON, invalid); err == nil {
		t.Error("missing required field should fail validation")
	}
}
