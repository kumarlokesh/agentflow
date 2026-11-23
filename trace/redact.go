package trace

import "regexp"

// secretPatterns match common credential formats. They are deliberately
// conservative: each requires a distinctive prefix and enough length that
// ordinary prose doesn't match.
var secretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`sk-ant-[A-Za-z0-9_\-]{20,}`),                // Anthropic
	regexp.MustCompile(`sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}`), // OpenAI
	regexp.MustCompile(`gh[pousr]_[A-Za-z0-9]{36,}`),                // GitHub tokens
	regexp.MustCompile(`github_pat_[A-Za-z0-9_]{22,}`),              // GitHub fine-grained PATs
	regexp.MustCompile(`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`),             // AWS access key IDs
	regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`),        // PEM private keys
}

// SecretPatterns is an opt-in Redactor that redacts any string value
// containing a common credential format (Anthropic, OpenAI and GitHub
// tokens, AWS access key IDs, PEM private keys). The whole value is
// replaced, under the rule name "secret".
func SecretPatterns(_ Kind, _ string, value string) (string, bool) {
	for _, re := range secretPatterns {
		if re.MatchString(value) {
			return "secret", true
		}
	}
	return "", false
}
