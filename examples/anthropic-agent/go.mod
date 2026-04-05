module github.com/kumarlokesh/agentflow/examples/anthropic-agent

go 1.24

require (
	github.com/kumarlokesh/agentflow v0.0.0
	github.com/kumarlokesh/agentflow/providers/anthropic v0.0.0
	github.com/kumarlokesh/agentflow/providers/retry v0.0.0
)

require github.com/google/uuid v1.6.0 // indirect

replace (
	github.com/kumarlokesh/agentflow => ../../
	github.com/kumarlokesh/agentflow/providers/anthropic => ../../providers/anthropic
	github.com/kumarlokesh/agentflow/providers/retry => ../../providers/retry
)
