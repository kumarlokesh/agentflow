package inspect

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"unicode/utf8"
)

//go:embed html.tmpl
var htmlTemplate string

var page = template.Must(template.New("trace").Parse(htmlTemplate))

// maxHTMLString bounds strings in the HTML view; the bundle export is the
// lossless format.
const maxHTMLString = 64 << 10

type htmlBlame struct {
	Symptom      *Anomaly    `json:"symptom"`
	RootCauses   []htmlCause `json:"root_causes"`
	Consequences int         `json:"consequences"`
}

type htmlCause struct {
	Anomaly     *Anomaly `json:"anomaly"`
	Chain       []Hop    `json:"chain"`
	SharedState string   `json:"shared_state,omitempty"`
}

// WriteHTML writes a self-contained HTML page for browsing the trace: the
// run tree, each run's effects and conversation, its anomalies, and the
// blame result if b is not nil. The page loads nothing from the network.
func WriteHTML(w io.Writer, t *Trace, b *BlameResult) error {
	payload := map[string]any{"trace": t}
	if b != nil && b.Symptom != nil {
		hb := htmlBlame{Symptom: b.Symptom, Consequences: b.Consequences}
		for _, c := range b.RootCauses {
			hb.RootCauses = append(hb.RootCauses, htmlCause(c))
		}
		payload["blame"] = hb
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("inspect: html: %w", err)
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return err
	}
	// json.Marshal escapes <, > and &, so the data can't close its script
	// element.
	data, err := json.Marshal(truncateStrings(v))
	if err != nil {
		return err
	}
	return page.Execute(w, map[string]any{
		"Title": "agentflow trace " + t.ID,
		"Data":  template.JS(data), //nolint:gosec // JSON from json.Marshal, HTML-escaped
	})
}

func truncateStrings(v any) any {
	switch x := v.(type) {
	case string:
		if len(x) <= maxHTMLString {
			return x
		}
		cut := maxHTMLString
		for cut > 0 && !utf8.RuneStart(x[cut]) {
			cut--
		}
		return x[:cut] + fmt.Sprintf("… [%d more bytes not shown; the bundle export has everything]", len(x)-cut)
	case map[string]any:
		for k, e := range x {
			x[k] = truncateStrings(e)
		}
	case []any:
		for i, e := range x {
			x[i] = truncateStrings(e)
		}
	}
	return v
}
