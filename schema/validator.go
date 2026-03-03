// Package schema provides lightweight JSON Schema validation for tool parameters.
//
// It supports the subset of JSON Schema needed for LLM tool-calling: object
// types with typed properties, required fields, enums, and basic numeric and
// string constraints. This avoids pulling in a heavy external dependency while
// covering the practical needs of agent tool validation.
package schema

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Schema represents a JSON Schema document (subset).
type Schema struct {
	Type        string             `json:"type"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Enum        []any              `json:"enum,omitempty"`
	MinLength   *int               `json:"minLength,omitempty"`
	MaxLength   *int               `json:"maxLength,omitempty"`
	Minimum     *float64           `json:"minimum,omitempty"`
	Maximum     *float64           `json:"maximum,omitempty"`
	Description string             `json:"description,omitempty"`
}

// ValidationError collects one or more validation failures with JSON-pointer paths.
type ValidationError struct {
	Errors []FieldError
}

// FieldError is a single validation failure.
type FieldError struct {
	Path    string // JSON pointer, e.g. "/foo/bar"
	Message string
}

func (e *ValidationError) Error() string {
	msgs := make([]string, len(e.Errors))
	for i, fe := range e.Errors {
		msgs[i] = fe.Path + ": " + fe.Message
	}
	return "validation failed: " + strings.Join(msgs, "; ")
}

// Validate checks a raw JSON value against a raw JSON schema.
func Validate(rawSchema, rawValue json.RawMessage) error {
	var s Schema
	if err := json.Unmarshal(rawSchema, &s); err != nil {
		return fmt.Errorf("schema: invalid schema: %w", err)
	}

	var value any
	if err := json.Unmarshal(rawValue, &value); err != nil {
		return fmt.Errorf("schema: invalid JSON value: %w", err)
	}

	var errs []FieldError
	validate(&s, value, "", &errs)
	if len(errs) > 0 {
		return &ValidationError{Errors: errs}
	}
	return nil
}

func validate(s *Schema, value any, path string, errs *[]FieldError) {
	if s == nil {
		return
	}

	// Enum check (applies regardless of type).
	if len(s.Enum) > 0 {
		if !enumContains(s.Enum, value) {
			*errs = append(*errs, FieldError{
				Path:    pathOrRoot(path),
				Message: fmt.Sprintf("value not in enum %v", s.Enum),
			})
		}
	}

	switch s.Type {
	case "object":
		validateObject(s, value, path, errs)
	case "array":
		validateArray(s, value, path, errs)
	case "string":
		validateString(s, value, path, errs)
	case "number", "integer":
		validateNumber(s, value, path, errs)
	case "boolean":
		if _, ok := value.(bool); !ok {
			*errs = append(*errs, FieldError{
				Path:    pathOrRoot(path),
				Message: fmt.Sprintf("expected boolean, got %T", value),
			})
		}
	case "null":
		if value != nil {
			*errs = append(*errs, FieldError{
				Path:    pathOrRoot(path),
				Message: fmt.Sprintf("expected null, got %T", value),
			})
		}
	case "":
		// No type constraint - accept any value.
	default:
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("unsupported type %q", s.Type),
		})
	}
}

func validateObject(s *Schema, value any, path string, errs *[]FieldError) {
	obj, ok := value.(map[string]any)
	if !ok {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("expected object, got %T", value),
		})
		return
	}

	// Check required fields.
	for _, req := range s.Required {
		if _, exists := obj[req]; !exists {
			*errs = append(*errs, FieldError{
				Path:    path + "/" + req,
				Message: "required field missing",
			})
		}
	}

	// Validate properties.
	for name, propSchema := range s.Properties {
		if val, exists := obj[name]; exists {
			validate(propSchema, val, path+"/"+name, errs)
		}
	}
}

func validateArray(s *Schema, value any, path string, errs *[]FieldError) {
	arr, ok := value.([]any)
	if !ok {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("expected array, got %T", value),
		})
		return
	}

	if s.Items != nil {
		for i, item := range arr {
			validate(s.Items, item, fmt.Sprintf("%s/%d", path, i), errs)
		}
	}
}

func validateString(s *Schema, value any, path string, errs *[]FieldError) {
	str, ok := value.(string)
	if !ok {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("expected string, got %T", value),
		})
		return
	}

	if s.MinLength != nil && len(str) < *s.MinLength {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("string length %d < minLength %d", len(str), *s.MinLength),
		})
	}
	if s.MaxLength != nil && len(str) > *s.MaxLength {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("string length %d > maxLength %d", len(str), *s.MaxLength),
		})
	}
}

func validateNumber(s *Schema, value any, path string, errs *[]FieldError) {
	num, ok := toFloat64(value)
	if !ok {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("expected number, got %T", value),
		})
		return
	}

	if s.Type == "integer" && num != float64(int64(num)) {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("expected integer, got %v", num),
		})
	}

	if s.Minimum != nil && num < *s.Minimum {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("value %v < minimum %v", num, *s.Minimum),
		})
	}
	if s.Maximum != nil && num > *s.Maximum {
		*errs = append(*errs, FieldError{
			Path:    pathOrRoot(path),
			Message: fmt.Sprintf("value %v > maximum %v", num, *s.Maximum),
		})
	}
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func enumContains(enum []any, value any) bool {
	for _, e := range enum {
		if fmt.Sprintf("%v", e) == fmt.Sprintf("%v", value) {
			return true
		}
	}
	return false
}

func pathOrRoot(path string) string {
	if path == "" {
		return "/"
	}
	return path
}
