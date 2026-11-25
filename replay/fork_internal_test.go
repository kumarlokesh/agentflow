package replay

import (
	"math"
	"testing"
)

func TestWilson(t *testing.T) {
	for _, c := range []struct {
		k, n      int
		low, high float64
	}{{0, 10, 0, 0.2775}, {5, 10, 0.2366, 0.7634}, {10, 10, 0.7225, 1}, {4, 5, 0.3755, 0.9638}, {0, 0, 0, 1}} {
		_, low, high := wilson(c.k, c.n)
		if math.Abs(low-c.low) > 1e-4 || math.Abs(high-c.high) > 1e-4 {
			t.Errorf("wilson(%d, %d) = [%.4f, %.4f], want [%.4f, %.4f]", c.k, c.n, low, high, c.low, c.high)
		}
	}
}
