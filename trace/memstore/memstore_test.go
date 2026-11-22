package memstore_test

import (
	"testing"

	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/memstore"
	"github.com/kumarlokesh/agentflow/trace/storetest"
)

func TestContract(t *testing.T) {
	storetest.Run(t, func(t *testing.T) trace.Store { return memstore.New() })
}
