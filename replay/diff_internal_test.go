package replay

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// Long single-line strings (reasoning signatures, one-line prompts) are shown
// around their first difference, not in full.
func TestLongSingleLineDiffIsWindowed(t *testing.T) {
	a := strings.Repeat("x", 500) + "AAA" + strings.Repeat("y", 500)
	b := strings.Repeat("x", 500) + "BBB" + strings.Repeat("y", 500)
	d := changed("sig", a, b)
	lines := strings.Split(d.TextDiff, "\n")
	if len(lines) != 2 || !strings.HasPrefix(lines[0], "- …") || !strings.HasPrefix(lines[1], "+ …") {
		t.Fatalf("diff = %q", d.TextDiff)
	}
	if !strings.Contains(lines[0], "AAA") || !strings.Contains(lines[1], "BBB") {
		t.Errorf("window misses the difference: %q", d.TextDiff)
	}
	if len(d.TextDiff) > 300 {
		t.Errorf("diff is %d bytes, want a short window", len(d.TextDiff))
	}

	// Cuts never split a multi-byte character.
	u := strings.Repeat("é", 100)
	d = changed("u", u+"1", u+"2")
	if !utf8.ValidString(d.TextDiff) {
		t.Errorf("invalid UTF-8 in %q", d.TextDiff)
	}

	// Multi-line strings keep the line diff.
	if d := changed("p", "a\nb", "a\nc"); d.TextDiff != "  a\n- b\n+ c" {
		t.Errorf("line diff = %q", d.TextDiff)
	}

	// Whitespace-only changes are visible.
	if d := changed("p", "a\nb", "a\nb\n"); d.TextDiff != "  a\n  b\n+ (empty line)" {
		t.Errorf("trailing newline = %q", d.TextDiff)
	}
	if d := changed("p", "a\nb", "a\nb "); d.TextDiff != "  a\n- b\n+ \"b \"" {
		t.Errorf("trailing space = %q", d.TextDiff)
	}
}
