// Package conformance checks that agentflow's trace encoder, its committed
// fixtures, and the published JSON Schema (../trace-v1.schema.json) agree.
// It is a separate module so that the schema validator never becomes a
// dependency of agentflow itself.
package conformance
