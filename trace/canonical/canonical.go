// Package canonical implements the JSON Canonicalization Scheme (JCS,
// RFC 8785): a deterministic serialization of JSON, used by agentflow to
// fingerprint requests.
//
// Properties of the output:
//   - no insignificant whitespace;
//   - object members sorted by their names' UTF-16 code units;
//   - numbers formatted as ECMAScript's Number.prototype.toString (IEEE 754
//     doubles, shortest round-trip form);
//   - strings escaped minimally (only '"', '\\', and control characters).
//
// Input must be valid UTF-8 JSON without duplicate object member names.
package canonical

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// Transform returns the canonical form of the JSON document in data.
func Transform(data []byte) ([]byte, error) {
	if !utf8.Valid(data) {
		return nil, errors.New("canonical: input is not valid UTF-8")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var out bytes.Buffer
	if err := writeValue(dec, &out); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("canonical: trailing data after JSON value")
	}
	return out.Bytes(), nil
}

// Marshal encodes v with encoding/json and returns its canonical form.
func Marshal(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return Transform(raw)
}

func writeValue(dec *json.Decoder, out *bytes.Buffer) error {
	tok, err := dec.Token()
	if err != nil {
		return fmt.Errorf("canonical: %w", err)
	}
	return writeToken(dec, tok, out)
}

func writeToken(dec *json.Decoder, tok json.Token, out *bytes.Buffer) error {
	switch v := tok.(type) {
	case json.Delim:
		switch v {
		case '{':
			return writeObject(dec, out)
		case '[':
			return writeArray(dec, out)
		default:
			return fmt.Errorf("canonical: unexpected delimiter %q", v)
		}
	case string:
		writeString(out, v)
	case json.Number:
		f, err := strconv.ParseFloat(string(v), 64)
		if err != nil {
			return fmt.Errorf("canonical: number %s: %w", v, err)
		}
		s, err := FormatNumber(f)
		if err != nil {
			return err
		}
		out.WriteString(s)
	case bool:
		if v {
			out.WriteString("true")
		} else {
			out.WriteString("false")
		}
	case nil:
		out.WriteString("null")
	default:
		return fmt.Errorf("canonical: unexpected token %T", tok)
	}
	return nil
}

type member struct {
	key   string
	units []uint16
	value []byte
}

func writeObject(dec *json.Decoder, out *bytes.Buffer) error {
	var members []member
	seen := map[string]bool{}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return fmt.Errorf("canonical: %w", err)
		}
		key, ok := tok.(string)
		if !ok {
			return fmt.Errorf("canonical: object key is %T", tok)
		}
		if seen[key] {
			return fmt.Errorf("canonical: duplicate object member %q", key)
		}
		seen[key] = true
		var val bytes.Buffer
		if err := writeValue(dec, &val); err != nil {
			return err
		}
		members = append(members, member{key: key, units: utf16.Encode([]rune(key)), value: val.Bytes()})
	}
	if _, err := dec.Token(); err != nil { // closing '}'
		return fmt.Errorf("canonical: %w", err)
	}
	sort.Slice(members, func(i, j int) bool { return lessUTF16(members[i].units, members[j].units) })
	out.WriteByte('{')
	for i, m := range members {
		if i > 0 {
			out.WriteByte(',')
		}
		writeString(out, m.key)
		out.WriteByte(':')
		out.Write(m.value)
	}
	out.WriteByte('}')
	return nil
}

func writeArray(dec *json.Decoder, out *bytes.Buffer) error {
	out.WriteByte('[')
	first := true
	for dec.More() {
		if !first {
			out.WriteByte(',')
		}
		first = false
		if err := writeValue(dec, out); err != nil {
			return err
		}
	}
	if _, err := dec.Token(); err != nil { // closing ']'
		return fmt.Errorf("canonical: %w", err)
	}
	out.WriteByte(']')
	return nil
}

func lessUTF16(a, b []uint16) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return len(a) < len(b)
}

const hexDigits = "0123456789abcdef"

func writeString(out *bytes.Buffer, s string) {
	out.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		default:
			if r < 0x20 {
				out.WriteString(`\u00`)
				out.WriteByte(hexDigits[r>>4])
				out.WriteByte(hexDigits[r&0xf])
			} else {
				out.WriteRune(r)
			}
		}
	}
	out.WriteByte('"')
}

// FormatNumber formats f the way ECMAScript's Number.prototype.toString does,
// as required by RFC 8785. NaN and infinities are rejected.
func FormatNumber(f float64) (string, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "", fmt.Errorf("canonical: %v is not a valid JSON number", f)
	}
	if f == 0 {
		return "0", nil // also covers -0
	}
	sign := ""
	if f < 0 {
		sign = "-"
		f = -f
	}
	// Shortest digits that round-trip, in the form d.ddde±x.
	sci := strconv.FormatFloat(f, 'e', -1, 64)
	mant, expPart, _ := strings.Cut(sci, "e")
	digits := strings.Replace(mant, ".", "", 1)
	exp, err := strconv.Atoi(expPart)
	if err != nil {
		return "", fmt.Errorf("canonical: format %v: %w", f, err)
	}
	k := len(digits) // number of significant digits
	n := exp + 1     // value = 0.digits × 10^n

	var s string
	switch {
	case k <= n && n <= 21:
		s = digits + strings.Repeat("0", n-k)
	case 0 < n && n <= 21:
		s = digits[:n] + "." + digits[n:]
	case -6 < n && n <= 0:
		s = "0." + strings.Repeat("0", -n) + digits
	default:
		e := n - 1
		esign := "+"
		if e < 0 {
			esign = "-"
			e = -e
		}
		if k == 1 {
			s = digits + "e" + esign + strconv.Itoa(e)
		} else {
			s = digits[:1] + "." + digits[1:] + "e" + esign + strconv.Itoa(e)
		}
	}
	return sign + s, nil
}
