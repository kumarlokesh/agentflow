package replay

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/kumarlokesh/agentflow"
)

// neverVerified are custom effects that are nondeterministic by definition.
var neverVerified = map[string]bool{"clock.now": true, "id.new": true}

type verifyPattern struct {
	kind  agentflow.EffectKind
	name  *regexp.Regexp
	exact bool
}

// verifySet selects effects to verify (Options.Verify).
type verifySet []verifyPattern

func compileVerify(patterns []string) (verifySet, error) {
	var out verifySet
	for _, p := range patterns {
		var kind agentflow.EffectKind
		var name string
		switch {
		case strings.HasPrefix(p, "tool:"):
			kind, name = agentflow.EffectTool, strings.TrimPrefix(p, "tool:")
		case strings.HasPrefix(p, "effect:"):
			kind, name = agentflow.EffectCustom, strings.TrimPrefix(p, "effect:")
		default:
			return nil, fmt.Errorf("replay: Verify pattern %q: want \"tool:<name>\" or \"effect:<name>\" (model calls can't be verified)", p)
		}
		if name == "" {
			return nil, fmt.Errorf("replay: Verify pattern %q has no name", p)
		}
		exact := !strings.Contains(name, "*")
		if kind == agentflow.EffectCustom && exact && neverVerified[name] {
			return nil, fmt.Errorf("replay: Verify pattern %q: %s is nondeterministic by definition and can't be verified", p, name)
		}
		re, err := regexp.Compile("^" + strings.ReplaceAll(regexp.QuoteMeta(name), `\*`, ".*") + "$")
		if err != nil {
			return nil, fmt.Errorf("replay: Verify pattern %q: %w", p, err)
		}
		out = append(out, verifyPattern{kind: kind, name: re, exact: exact})
	}
	return out, nil
}

// selects reports whether call is to be verified. A tool that a wildcard
// matches but that isn't annotated ReadOnly is added to skipped instead.
func (vs verifySet) selects(call *agentflow.Call, skipped map[string]bool) bool {
	if call.Kind == agentflow.EffectLLM || (call.Kind == agentflow.EffectCustom && neverVerified[call.Name]) {
		return false
	}
	wildcardSkip := false
	for _, p := range vs {
		if p.kind != call.Kind || !p.name.MatchString(call.Name) {
			continue
		}
		if p.exact || call.Kind == agentflow.EffectCustom || readOnly(call) {
			return true
		}
		wildcardSkip = true
	}
	if wildcardSkip {
		skipped[call.Name] = true
	}
	return false
}

func readOnly(call *agentflow.Call) bool {
	tc, ok := call.Request.(*agentflow.ToolCall)
	return ok && tc.Annotations != nil && tc.Annotations.ReadOnly
}
