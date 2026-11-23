package replay

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// DiffOp is how a field differs.
type DiffOp string

const (
	OpAdded   DiffOp = "added"
	OpRemoved DiffOp = "removed"
	OpChanged DiffOp = "changed"
)

// FieldDiff is one differing field between a recorded and a current value.
type FieldDiff struct {
	// Path names the field with semantic keys, e.g. "system[0].text",
	// "tools{calculator}.input_schema.properties.x",
	// "messages[3].blocks[0].content[0].text".
	Path     string
	Op       DiffOp
	Recorded json.RawMessage // absent for OpAdded
	Current  json.RawMessage // absent for OpRemoved
	// TextDiff is a line diff when both sides are long or multi-line strings.
	TextDiff string
}

// diffJSON compares two JSON documents structurally.
func diffJSON(recorded, current []byte) ([]FieldDiff, error) {
	a, err := decode(recorded)
	if err != nil {
		return nil, fmt.Errorf("replay: diff recorded value: %w", err)
	}
	b, err := decode(current)
	if err != nil {
		return nil, fmt.Errorf("replay: diff current value: %w", err)
	}
	var out []FieldDiff
	diffValue("", a, b, &out)
	return out, nil
}

func decode(raw []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	err := dec.Decode(&v)
	return v, err
}

func join(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

func diffValue(path string, a, b any, out *[]FieldDiff) {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok {
			*out = append(*out, changed(path, a, b))
			return
		}
		keys := map[string]bool{}
		for k := range av {
			keys[k] = true
		}
		for k := range bv {
			keys[k] = true
		}
		sorted := make([]string, 0, len(keys))
		for k := range keys {
			sorted = append(sorted, k)
		}
		sort.Strings(sorted)
		for _, k := range sorted {
			x, inA := av[k]
			y, inB := bv[k]
			p := join(path, k)
			switch {
			case !inA:
				*out = append(*out, FieldDiff{Path: p, Op: OpAdded, Current: raw(y)})
			case !inB:
				*out = append(*out, FieldDiff{Path: p, Op: OpRemoved, Recorded: raw(x)})
			default:
				diffValue(p, x, y, out)
			}
		}
	case []any:
		bv, ok := b.([]any)
		if !ok {
			*out = append(*out, changed(path, a, b))
			return
		}
		if byName, ok := namedElements(av, bv); ok && strings.HasSuffix(path, "tools") {
			diffNamed(path, byName, av, bv, out)
			return
		}
		diffArray(path, av, bv, out)
	default:
		if !scalarEqual(a, b) {
			*out = append(*out, changed(path, a, b))
		}
	}
}

// diffArray aligns elements with a longest common subsequence, so an
// inserted or removed element is reported as such instead of as a cascade
// of changes. Between aligned elements, leftovers are paired positionally
// (changed), and the rest are added or removed. Paths use the current index
// for added and changed elements, the recorded index for removed ones.
func diffArray(path string, a, b []any, out *[]FieldDiff) {
	if len(a) == len(b) {
		for i := range a {
			diffValue(fmt.Sprintf("%s[%d]", path, i), a[i], b[i], out)
		}
		return
	}
	ka, kb := make([]string, len(a)), make([]string, len(b))
	for i, x := range a {
		ka[i] = string(raw(x))
	}
	for j, y := range b {
		kb[j] = string(raw(y))
	}
	lcs := make([][]int, len(a)+1)
	for i := range lcs {
		lcs[i] = make([]int, len(b)+1)
	}
	for i := len(a) - 1; i >= 0; i-- {
		for j := len(b) - 1; j >= 0; j-- {
			if ka[i] == kb[j] {
				lcs[i][j] = lcs[i+1][j+1] + 1
			} else {
				lcs[i][j] = max(lcs[i+1][j], lcs[i][j+1])
			}
		}
	}
	var ga, gb []int // unmatched indexes since the last aligned pair
	flush := func() {
		n := min(len(ga), len(gb))
		for k := 0; k < n; k++ {
			diffValue(fmt.Sprintf("%s[%d]", path, gb[k]), a[ga[k]], b[gb[k]], out)
		}
		for _, i := range ga[n:] {
			*out = append(*out, FieldDiff{Path: fmt.Sprintf("%s[%d]", path, i), Op: OpRemoved, Recorded: raw(a[i])})
		}
		for _, j := range gb[n:] {
			*out = append(*out, FieldDiff{Path: fmt.Sprintf("%s[%d]", path, j), Op: OpAdded, Current: raw(b[j])})
		}
		ga, gb = nil, nil
	}
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		switch {
		case i < len(a) && j < len(b) && ka[i] == kb[j]:
			flush()
			i++
			j++
		case j < len(b) && (i == len(a) || lcs[i][j+1] >= lcs[i+1][j]):
			gb = append(gb, j)
			j++
		default:
			ga = append(ga, i)
			i++
		}
	}
	flush()
}

// namedElements reports whether every element of both arrays is an object
// with a string "name" (e.g. tool specs), so they can be matched by name.
func namedElements(a, b []any) (map[string]bool, bool) {
	names := map[string]bool{}
	for _, xs := range [][]any{a, b} {
		for _, x := range xs {
			m, ok := x.(map[string]any)
			if !ok {
				return nil, false
			}
			n, ok := m["name"].(string)
			if !ok {
				return nil, false
			}
			names[n] = true
		}
	}
	return names, true
}

func diffNamed(path string, names map[string]bool, a, b []any, out *[]FieldDiff) {
	index := func(xs []any) map[string]any {
		m := map[string]any{}
		for _, x := range xs {
			m[x.(map[string]any)["name"].(string)] = x
		}
		return m
	}
	ia, ib := index(a), index(b)
	sorted := make([]string, 0, len(names))
	for n := range names {
		sorted = append(sorted, n)
	}
	sort.Strings(sorted)
	for _, n := range sorted {
		p := path + "{" + n + "}"
		x, inA := ia[n]
		y, inB := ib[n]
		switch {
		case !inA:
			*out = append(*out, FieldDiff{Path: p, Op: OpAdded, Current: raw(y)})
		case !inB:
			*out = append(*out, FieldDiff{Path: p, Op: OpRemoved, Recorded: raw(x)})
		default:
			diffValue(p, x, y, out)
		}
	}
}

func scalarEqual(a, b any) bool {
	an, aok := a.(json.Number)
	bn, bok := b.(json.Number)
	if aok && bok {
		af, e1 := strconv.ParseFloat(string(an), 64)
		bf, e2 := strconv.ParseFloat(string(bn), 64)
		if e1 == nil && e2 == nil {
			return af == bf
		}
		return an == bn
	}
	return a == b
}

func changed(path string, a, b any) FieldDiff {
	d := FieldDiff{Path: path, Op: OpChanged, Recorded: raw(a), Current: raw(b)}
	as, aok := a.(string)
	bs, bok := b.(string)
	switch {
	case !aok || !bok:
	case strings.Contains(as, "\n") || strings.Contains(bs, "\n"):
		d.TextDiff = lineDiff(as, bs)
	case len(as) > 80 || len(bs) > 80:
		d.TextDiff = windowDiff(as, bs)
	}
	return d
}

// windowDiff shows long single-line strings (signatures, IDs, one-line
// prompts) around their first difference instead of in full.
func windowDiff(a, b string) string {
	p := 0
	for p < len(a) && p < len(b) && a[p] == b[p] {
		p++
	}
	const before, after = 40, 80
	start := max(p-before, 0)
	cut := func(s string) string {
		lo, end := min(start, len(s)), min(p+after, len(s))
		for lo > 0 && !utf8.RuneStart(s[lo]) {
			lo--
		}
		for end < len(s) && !utf8.RuneStart(s[end]) {
			end++
		}
		out := s[lo:end]
		if lo > 0 {
			out = "…" + out
		}
		if end < len(s) {
			out += "…"
		}
		return out
	}
	return "- " + cut(a) + "\n+ " + cut(b)
}

func raw(v any) json.RawMessage {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
}

// lineDiff returns a minimal line diff ("- " removed, "+ " added, "  " kept).
func lineDiff(a, b string) string {
	x, y := strings.Split(a, "\n"), strings.Split(b, "\n")
	// LCS table.
	lcs := make([][]int, len(x)+1)
	for i := range lcs {
		lcs[i] = make([]int, len(y)+1)
	}
	for i := len(x) - 1; i >= 0; i-- {
		for j := len(y) - 1; j >= 0; j-- {
			if x[i] == y[j] {
				lcs[i][j] = lcs[i+1][j+1] + 1
			} else {
				lcs[i][j] = max(lcs[i+1][j], lcs[i][j+1])
			}
		}
	}
	var sb strings.Builder
	i, j := 0, 0
	for i < len(x) || j < len(y) {
		switch {
		case i < len(x) && j < len(y) && x[i] == y[j]:
			sb.WriteString("  " + x[i] + "\n")
			i++
			j++
		case i < len(x) && (j == len(y) || lcs[i+1][j] >= lcs[i][j+1]):
			sb.WriteString("- " + x[i] + "\n") // removals before additions, as in diff(1)
			i++
		default:
			sb.WriteString("+ " + y[j] + "\n")
			j++
		}
	}
	return strings.TrimSuffix(sb.String(), "\n")
}

// ignoreMatcher matches diff paths against Ignore patterns. In a pattern,
// "*" matches any characters except "."; a pattern also matches every path
// below it ("params" ignores "params.temperature").
type ignoreMatcher []*regexp.Regexp

func compileIgnore(patterns []string) (ignoreMatcher, error) {
	var m ignoreMatcher
	for _, p := range patterns {
		quoted := regexp.QuoteMeta(p)
		quoted = strings.ReplaceAll(quoted, `\*`, `[^.]*`)
		re, err := regexp.Compile(`^` + quoted + `($|[.\[{])`)
		if err != nil {
			return nil, fmt.Errorf("replay: ignore pattern %q: %w", p, err)
		}
		m = append(m, re)
	}
	return m, nil
}

func (m ignoreMatcher) match(path string) bool {
	for _, re := range m {
		if re.MatchString(path) {
			return true
		}
	}
	return false
}

func (m ignoreMatcher) filter(diffs []FieldDiff) []FieldDiff {
	if len(m) == 0 {
		return diffs
	}
	var out []FieldDiff
	for _, d := range diffs {
		if !m.match(d.Path) {
			out = append(out, d)
		}
	}
	return out
}
