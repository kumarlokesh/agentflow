package trace

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

// MaxEventSize is the largest encoded event (after blob externalization) a
// writer accepts. Anything larger indicates a bug.
const MaxEventSize = 4 << 20

// DefaultBlobThreshold is the string length (bytes) above which a string is
// moved out of the event into the blob store.
const DefaultBlobThreshold = 32 << 10

// BlobRef addresses a blob by content: "sha256:<hex>".
type BlobRef string

// BlobRefOf returns the reference for data.
func BlobRefOf(data []byte) BlobRef {
	sum := sha256.Sum256(data)
	return BlobRef("sha256:" + hex.EncodeToString(sum[:]))
}

// Valid reports whether r is a well-formed sha256 reference.
func (r BlobRef) Valid() bool {
	h, ok := strings.CutPrefix(string(r), "sha256:")
	if !ok || len(h) != 64 {
		return false
	}
	_, err := hex.DecodeString(h)
	return err == nil
}

// Reserved JSON object keys (spec §11).
const (
	keyBlob     = "$blob"
	keySize     = "size"
	keyRedacted = "$redacted"
	keyLit      = "$lit"
)

// EncodeOptions control event encoding.
type EncodeOptions struct {
	// BlobThreshold is the string length above which strings are
	// externalized. Zero or negative disables externalization.
	BlobThreshold int
	// PutBlob stores a blob. Required when BlobThreshold > 0.
	PutBlob func(data []byte) (BlobRef, error)
	// Redact, if set, is consulted for every JSON string value in a body;
	// redacted values are stored as {"$redacted":"<rule>"}. Fingerprints are
	// computed before encoding, so they are unaffected.
	Redact Redactor
}

// Redactor decides whether a string value in an event body must not be
// stored. path is the value's location in the body, e.g.
// "message.blocks[0].text". It returns the rule name to record.
type Redactor func(kind Kind, path string, value string) (rule string, redact bool)

// EncodeEvent encodes e as a single JSON line (without the trailing newline).
func EncodeEvent(e Event, opts EncodeOptions) ([]byte, error) {
	if e.Body == nil {
		return nil, errors.New("trace: event has no body")
	}
	body, err := json.Marshal(e.Body)
	if err != nil {
		return nil, fmt.Errorf("trace: encode %s body: %w", e.Kind(), err)
	}
	body, err = externalize(e.Kind(), body, opts)
	if err != nil {
		return nil, err
	}
	env := envelope{
		V:       Version,
		TraceID: e.TraceID,
		RunID:   e.RunID,
		Seq:     e.Seq,
		Time:    e.Time.UTC().Format(timeLayout),
		Kind:    e.Kind(),
		Cause:   e.Cause,
		Body:    body,
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(env); err != nil {
		return nil, fmt.Errorf("trace: encode event: %w", err)
	}
	line := bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
	if len(line) > MaxEventSize {
		return nil, fmt.Errorf("%w: %s seq %d is %d bytes (max %d)", ErrEventTooLarge, e.Kind(), e.Seq, len(line), MaxEventSize)
	}
	return line, nil
}

// DecodeEvent decodes one encoded event, resolving blob references with
// getBlob (which may be nil if the event has none).
func DecodeEvent(line []byte, getBlob func(BlobRef) ([]byte, error)) (Event, error) {
	var env envelope
	if err := json.Unmarshal(line, &env); err != nil {
		return Event{}, fmt.Errorf("trace: decode event: %w", err)
	}
	if env.V == 0 && env.SchemaVersion != nil {
		return Event{}, ErrPrototypeTrace
	}
	if env.V != Version {
		return Event{}, fmt.Errorf("%w: %d", ErrUnsupportedVersion, env.V)
	}
	t, err := parseTime(env.Time)
	if err != nil {
		return Event{}, err
	}
	raw := []byte(env.Body)
	if bytes.Contains(raw, []byte(`"$`)) {
		if raw, err = resolve(raw, getBlob); err != nil {
			return Event{}, fmt.Errorf("trace: %s seq %d: %w", env.Kind, env.Seq, err)
		}
	}
	body, err := decodeBody(env.Kind, raw)
	if err != nil {
		return Event{}, err
	}
	return Event{
		TraceID: env.TraceID,
		RunID:   env.RunID,
		Seq:     env.Seq,
		Time:    t,
		Cause:   env.Cause,
		Body:    body,
	}, nil
}

func parseTime(s string) (time.Time, error) {
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return time.Time{}, fmt.Errorf("trace: event time %q: %w", s, err)
	}
	return t.UTC(), nil
}

// --- order-preserving JSON tree ---

type nodeKind byte

const (
	nObject nodeKind = iota
	nArray
	nString
	nLiteral // number, true, false, null: kept as raw text
)

type node struct {
	kind    nodeKind
	members []member // nObject
	items   []*node  // nArray
	str     string   // nString
	lit     string   // nLiteral
}

type member struct {
	key string
	val *node
}

func parseTree(data []byte) (*node, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	n, err := parseNode(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("trailing data after JSON value")
	}
	return n, nil
}

func parseNode(dec *json.Decoder) (*node, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch v := tok.(type) {
	case json.Delim:
		switch v {
		case '{':
			n := &node{kind: nObject}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, _ := kt.(string)
				val, err := parseNode(dec)
				if err != nil {
					return nil, err
				}
				n.members = append(n.members, member{key: key, val: val})
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return n, nil
		case '[':
			n := &node{kind: nArray, items: []*node{}}
			for dec.More() {
				item, err := parseNode(dec)
				if err != nil {
					return nil, err
				}
				n.items = append(n.items, item)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return n, nil
		}
		return nil, fmt.Errorf("unexpected delimiter %q", v)
	case string:
		return &node{kind: nString, str: v}, nil
	case json.Number:
		return &node{kind: nLiteral, lit: string(v)}, nil
	case bool:
		if v {
			return &node{kind: nLiteral, lit: "true"}, nil
		}
		return &node{kind: nLiteral, lit: "false"}, nil
	case nil:
		return &node{kind: nLiteral, lit: "null"}, nil
	}
	return nil, fmt.Errorf("unexpected token %T", tok)
}

func (n *node) write(buf *bytes.Buffer) {
	switch n.kind {
	case nObject:
		buf.WriteByte('{')
		for i, m := range n.members {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeJSONString(buf, m.key)
			buf.WriteByte(':')
			m.val.write(buf)
		}
		buf.WriteByte('}')
	case nArray:
		buf.WriteByte('[')
		for i, it := range n.items {
			if i > 0 {
				buf.WriteByte(',')
			}
			it.write(buf)
		}
		buf.WriteByte(']')
	case nString:
		writeJSONString(buf, n.str)
	case nLiteral:
		buf.WriteString(n.lit)
	}
}

func writeJSONString(buf *bytes.Buffer, s string) {
	var tmp bytes.Buffer
	enc := json.NewEncoder(&tmp)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s) // encoding a string cannot fail
	buf.Write(bytes.TrimSuffix(tmp.Bytes(), []byte("\n")))
}

func (n *node) has(key string) bool {
	for _, m := range n.members {
		if m.key == key {
			return true
		}
	}
	return false
}

func (n *node) get(key string) *node {
	for _, m := range n.members {
		if m.key == key {
			return m.val
		}
	}
	return nil
}

// --- externalize (write side) ---

func externalize(kind Kind, body []byte, opts EncodeOptions) ([]byte, error) {
	// Fast path: nothing reserved, nothing to redact, nothing large enough to move.
	if opts.Redact == nil && !bytes.Contains(body, []byte(`"$`)) && (opts.BlobThreshold <= 0 || len(body) <= opts.BlobThreshold) {
		return reencode(body)
	}
	if opts.BlobThreshold > 0 && opts.PutBlob == nil {
		return nil, errors.New("trace: BlobThreshold set without PutBlob")
	}
	root, err := parseTree(body)
	if err != nil {
		return nil, fmt.Errorf("trace: externalize: %w", err)
	}
	out, err := externalizeNode(root, kind, "", opts)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	out.write(&buf)
	return buf.Bytes(), nil
}

// reencode rewrites body without HTML escaping so files stay readable.
func reencode(body []byte) ([]byte, error) {
	if !bytes.ContainsAny(body, `\`) {
		return body, nil
	}
	root, err := parseTree(body)
	if err != nil {
		return nil, fmt.Errorf("trace: encode: %w", err)
	}
	var buf bytes.Buffer
	root.write(&buf)
	return buf.Bytes(), nil
}

func externalizeNode(n *node, kind Kind, path string, opts EncodeOptions) (*node, error) {
	switch n.kind {
	case nString:
		if opts.Redact != nil {
			if rule, ok := opts.Redact(kind, path, n.str); ok {
				return &node{kind: nObject, members: []member{{key: keyRedacted, val: &node{kind: nString, str: rule}}}}, nil
			}
		}
		if opts.BlobThreshold > 0 && len(n.str) > opts.BlobThreshold {
			ref, err := opts.PutBlob([]byte(n.str))
			if err != nil {
				return nil, fmt.Errorf("trace: put blob: %w", err)
			}
			return &node{kind: nObject, members: []member{
				{key: keyBlob, val: &node{kind: nString, str: string(ref)}},
				{key: keySize, val: &node{kind: nLiteral, lit: fmt.Sprint(len(n.str))}},
			}}, nil
		}
		return n, nil
	case nArray:
		for i, it := range n.items {
			x, err := externalizeNode(it, kind, fmt.Sprintf("%s[%d]", path, i), opts)
			if err != nil {
				return nil, err
			}
			n.items[i] = x
		}
		return n, nil
	case nObject:
		reserved := n.has(keyBlob) || n.has(keyRedacted) || n.has(keyLit)
		for i, m := range n.members {
			child := m.key
			if path != "" {
				child = path + "." + m.key
			}
			x, err := externalizeNode(m.val, kind, child, opts)
			if err != nil {
				return nil, err
			}
			n.members[i].val = x
		}
		if reserved {
			// Application data that uses a reserved key: wrap it so readers
			// don't mistake it for a reference.
			return &node{kind: nObject, members: []member{{key: keyLit, val: n}}}, nil
		}
		return n, nil
	}
	return n, nil
}

// --- resolve (read side) ---

func resolve(body []byte, getBlob func(BlobRef) ([]byte, error)) ([]byte, error) {
	root, err := parseTree(body)
	if err != nil {
		return nil, fmt.Errorf("resolve: %w", err)
	}
	out, err := resolveNode(root, getBlob)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	out.write(&buf)
	return buf.Bytes(), nil
}

func resolveNode(n *node, getBlob func(BlobRef) ([]byte, error)) (*node, error) {
	switch n.kind {
	case nArray:
		for i, it := range n.items {
			x, err := resolveNode(it, getBlob)
			if err != nil {
				return nil, err
			}
			n.items[i] = x
		}
		return n, nil
	case nObject:
		switch {
		case len(n.members) == 1 && n.members[0].key == keyLit:
			inner := n.members[0].val
			if inner.kind != nObject {
				return nil, errors.New(`"$lit" must wrap an object`)
			}
			return resolveMembers(inner, getBlob)
		case n.has(keyBlob):
			refNode := n.get(keyBlob)
			if refNode.kind != nString {
				return nil, errors.New(`"$blob" must be a string`)
			}
			ref := BlobRef(refNode.str)
			if getBlob == nil {
				return nil, fmt.Errorf("%w: %s (no blob store)", ErrBlobNotFound, ref)
			}
			data, err := getBlob(ref)
			if err != nil {
				return nil, err
			}
			return &node{kind: nString, str: string(data)}, nil
		case len(n.members) == 1 && n.members[0].key == keyRedacted:
			rule := n.members[0].val.str
			return &node{kind: nString, str: "[redacted: " + rule + "]"}, nil
		}
		return resolveMembers(n, getBlob)
	}
	return n, nil
}

func resolveMembers(n *node, getBlob func(BlobRef) ([]byte, error)) (*node, error) {
	for i, m := range n.members {
		x, err := resolveNode(m.val, getBlob)
		if err != nil {
			return nil, err
		}
		n.members[i].val = x
	}
	return n, nil
}
