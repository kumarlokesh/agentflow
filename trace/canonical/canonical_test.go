package canonical_test

import (
	"math"
	"strings"
	"testing"

	"github.com/kumarlokesh/agentflow/trace/canonical"
)

// The example from RFC 8785 §3.2.2.
func TestRFC8785Example(t *testing.T) {
	in := `{
  "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
  "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
  "literals": [null, true, false]
}`
	want := `{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}`
	got, err := canonical.Transform([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("\ngot  %s\nwant %s", got, want)
	}
}

// Sorting is by UTF-16 code units, not bytes or code points (RFC 8785 §3.2.3).
func TestSortByUTF16CodeUnits(t *testing.T) {
	in := `{"€":"Euro Sign","\r":"Carriage Return","דּ":"Hebrew Letter Dalet With Dagesh","1":"One","😀":"Emoji: Grinning Face","\u0080":"Control","ö":"Latin Small Letter O With Diaeresis"}`
	got, err := canonical.Transform([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	order := []string{"Carriage Return", "One", "Control", "Latin Small", "Euro Sign", "Emoji", "Hebrew"}
	pos := -1
	for _, name := range order {
		i := strings.Index(string(got), name)
		if i < 0 || i < pos {
			t.Fatalf("member %q out of order in %s", name, got)
		}
		pos = i
	}
}

// Number serialization vectors from RFC 8785 Appendix B (IEEE 754 bit
// patterns → expected ECMAScript string).
func TestNumberVectors(t *testing.T) {
	vectors := []struct {
		bits uint64
		want string
	}{
		{0x0000000000000000, "0"},
		{0x8000000000000000, "0"},
		{0x0000000000000001, "5e-324"},
		{0x8000000000000001, "-5e-324"},
		{0x7fefffffffffffff, "1.7976931348623157e+308"},
		{0xffefffffffffffff, "-1.7976931348623157e+308"},
		{0x4340000000000000, "9007199254740992"},
		{0xc340000000000000, "-9007199254740992"},
		{0x4430000000000000, "295147905179352830000"},
		{0x44b52d02c7e14af5, "9.999999999999997e+22"},
		{0x44b52d02c7e14af6, "1e+23"},
		{0x44b52d02c7e14af7, "1.0000000000000001e+23"},
		{0x444b1ae4d6e2ef4e, "999999999999999700000"},
		{0x444b1ae4d6e2ef4f, "999999999999999900000"},
		{0x444b1ae4d6e2ef50, "1e+21"},
		{0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"},
		{0x3eb0c6f7a0b5ed8d, "0.000001"},
		{0x41b3de4355555553, "333333333.3333332"},
		{0x41b3de4355555554, "333333333.33333325"},
		{0x41b3de4355555555, "333333333.3333333"},
		{0x41b3de4355555556, "333333333.3333334"},
		{0x41b3de4355555557, "333333333.33333343"},
		{0xbecbf647612f3696, "-0.0000033333333333333333"},
		{0x43143ff3c1cb0959, "1424953923781206.2"},
	}
	for _, v := range vectors {
		f := math.Float64frombits(v.bits)
		got, err := canonical.FormatNumber(f)
		if err != nil {
			t.Errorf("%#016x: %v", v.bits, err)
			continue
		}
		if got != v.want {
			t.Errorf("%#016x (%v): got %s, want %s", v.bits, f, got, v.want)
		}
	}
}

func TestRejects(t *testing.T) {
	for _, in := range []string{
		`{"a":1,"a":2}`, // duplicate member
		`{"a":1} {}`,    // trailing data
		"\"\xff\"",      // invalid UTF-8
		`{"a":`,         // truncated
	} {
		if out, err := canonical.Transform([]byte(in)); err == nil {
			t.Errorf("Transform(%q) = %s, want error", in, out)
		}
	}
	for _, f := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := canonical.FormatNumber(f); err == nil {
			t.Errorf("FormatNumber(%v) should fail", f)
		}
	}
}

func TestIdempotentAndOrderIndependent(t *testing.T) {
	a := `{"b":[1, 2.0, {"z":null,"y":"é"}],"a":true}`
	b := `{"a":true,"b":[1,2,{"y":"é","z":null}]}`
	ca, err := canonical.Transform([]byte(a))
	if err != nil {
		t.Fatal(err)
	}
	cb, err := canonical.Transform([]byte(b))
	if err != nil {
		t.Fatal(err)
	}
	if string(ca) != string(cb) {
		t.Fatalf("equivalent documents differ:\n%s\n%s", ca, cb)
	}
	again, err := canonical.Transform(ca)
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != string(ca) {
		t.Fatalf("not idempotent:\n%s\n%s", ca, again)
	}
}

func FuzzCanonical(f *testing.F) {
	for _, seed := range []string{`{}`, `[]`, `{"a":[1,"x",null,true]}`, `1e300`, `"\u0001"`, `{"😀":0.1}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		out, err := canonical.Transform(data)
		if err != nil {
			return
		}
		again, err := canonical.Transform(out)
		if err != nil {
			t.Fatalf("canonical output is not valid input: %s: %v", out, err)
		}
		if string(again) != string(out) {
			t.Fatalf("not idempotent: %s -> %s", out, again)
		}
	})
}
