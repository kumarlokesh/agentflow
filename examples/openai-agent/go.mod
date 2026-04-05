module github.com/kumarlokesh/agentflow/examples/openai-agent

go 1.24

require (
	github.com/kumarlokesh/agentflow v0.0.0
	github.com/kumarlokesh/agentflow/providers/openai v0.0.0
	github.com/kumarlokesh/agentflow/providers/retry v0.0.0
)

require github.com/google/uuid v1.6.0 // indirect

replace (
	github.com/kumarlokesh/agentflow => ../../
	github.com/kumarlokesh/agentflow/providers/openai => ../../providers/openai
	github.com/kumarlokesh/agentflow/providers/retry => ../../providers/retry
)
