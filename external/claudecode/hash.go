package claudecode

import (
	"bytes"
	"crypto/sha256"
	"encoding"
	"encoding/hex"
	"hash"

	"github.com/kumarlokesh/agentflow/llm"
	"github.com/kumarlokesh/agentflow/trace"
	"github.com/kumarlokesh/agentflow/trace/canonical"
)

// requestHasher computes llm.started fingerprints (trace.RequestFingerprint
// of the reconstructed request) in time linear in the conversation: the
// canonical form of a request is {"messages":[…],"model":…,…}, with keys
// sorted, so the messages are a prefix that grows, and the hash state after
// them is reused. A long session has thousands of model calls; hashing
// each whole conversation again would be quadratic.
type requestHasher struct {
	h     hash.Hash
	n     int
	tail  []byte        // `],"model":…}` for the current model
	msgs  []llm.Message // kept only for the fallback
	exact bool
	model string
}

var messagesPrefix = []byte(`{"messages":[`)

func newRequestHasher(model string) *requestHasher {
	r := &requestHasher{}
	r.setModel(model)
	r.reset()
	return r
}

func (r *requestHasher) view(msgs []llm.Message) any {
	return trace.FingerprintView(&llm.Request{Model: r.model, System: llm.Blocks{}, Messages: msgs, Tools: []llm.ToolSpec{}})
}

func (r *requestHasher) setModel(model string) {
	r.model = model
	c, err := canonical.Marshal(r.view([]llm.Message{}))
	r.exact = err == nil && bytes.HasPrefix(c, append(append([]byte{}, messagesPrefix...), ']'))
	if r.exact {
		r.tail = c[len(messagesPrefix):]
	}
}

func (r *requestHasher) reset() {
	r.h = sha256.New()
	r.h.Write(messagesPrefix)
	r.n = 0
	r.msgs = nil
}

func (r *requestHasher) add(m llm.Message) {
	r.msgs = append(r.msgs, m)
	c, err := canonical.Marshal(m)
	if err != nil {
		r.exact = false
		return
	}
	if r.n > 0 {
		r.h.Write([]byte{','})
	}
	r.h.Write(c)
	r.n++
}

// sum returns the fingerprint of the request with the conversation so far.
func (r *requestHasher) sum() string {
	if !r.exact {
		fp, _ := trace.Fingerprint(r.view(r.msgs))
		return fp
	}
	state, err := r.h.(encoding.BinaryMarshaler).MarshalBinary()
	if err != nil {
		fp, _ := trace.Fingerprint(r.view(r.msgs))
		return fp
	}
	h := sha256.New()
	if err := h.(encoding.BinaryUnmarshaler).UnmarshalBinary(state); err != nil {
		fp, _ := trace.Fingerprint(r.view(r.msgs))
		return fp
	}
	h.Write(r.tail)
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
