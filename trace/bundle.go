package trace

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"time"
)

// bundleFormat is the header's "bundle" value.
const bundleFormat = "agentflow.trace"

type bundleHeader struct {
	Bundle  string `json:"bundle"`
	V       int    `json:"v"`
	TraceID string `json:"trace_id"`
	Created string `json:"created"`
}

type bundleBlob struct {
	Blob BlobRef `json:"blob"`
	Data string  `json:"data"` // base64
}

// BundleOptions configure Export.
type BundleOptions struct {
	// BlobThreshold for strings moved to blob lines. Zero means
	// DefaultBlobThreshold; negative keeps everything inline.
	BlobThreshold int
	// Now stamps the header. Defaults to time.Now.
	Now func() time.Time
}

// Export writes one trace as a portable bundle (spec §12.2): a header
// line, every event of every run (grouped by run in start order, seq order
// within a run), then every referenced blob.
func Export(ctx context.Context, w io.Writer, s Store, traceID string, opts BundleOptions) error {
	threshold := opts.BlobThreshold
	if threshold == 0 {
		threshold = DefaultBlobThreshold
	}
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	var runs []RunInfo
	for info, err := range s.Runs(ctx, RunFilter{TraceID: traceID}) {
		if err != nil {
			return err
		}
		runs = append(runs, info)
	}
	if len(runs) == 0 {
		return fmt.Errorf("trace: export %s: %w", traceID, ErrRunNotFound)
	}

	blobs := map[BlobRef][]byte{}
	put := func(data []byte) (BlobRef, error) {
		ref := BlobRefOf(data)
		blobs[ref] = data
		return ref, nil
	}
	bw := bufio.NewWriter(w)
	header, err := json.Marshal(bundleHeader{Bundle: bundleFormat, V: Version, TraceID: traceID, Created: now().UTC().Format(timeLayout)})
	if err != nil {
		return err
	}
	if err := writeLine(bw, header); err != nil {
		return err
	}
	for _, run := range runs {
		for e, err := range s.Events(ctx, run.RunID, 0) {
			if err != nil {
				return err
			}
			line, err := EncodeEvent(e, EncodeOptions{BlobThreshold: max(threshold, 0), PutBlob: put})
			if err != nil {
				return err
			}
			if err := writeLine(bw, line); err != nil {
				return err
			}
		}
	}
	refs := make([]BlobRef, 0, len(blobs))
	for ref := range blobs {
		refs = append(refs, ref)
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i] < refs[j] })
	for _, ref := range refs {
		line, err := json.Marshal(bundleBlob{Blob: ref, Data: base64.StdEncoding.EncodeToString(blobs[ref])})
		if err != nil {
			return err
		}
		if err := writeLine(bw, line); err != nil {
			return err
		}
	}
	return bw.Flush()
}

func writeLine(w *bufio.Writer, line []byte) error {
	if _, err := w.Write(line); err != nil {
		return fmt.Errorf("trace: export: %w", err)
	}
	if err := w.WriteByte('\n'); err != nil {
		return fmt.Errorf("trace: export: %w", err)
	}
	return nil
}

// Import reads a bundle and appends all of its runs to s. It returns the
// bundle's trace ID.
func Import(ctx context.Context, r io.Reader, s Store) (string, error) {
	br := bufio.NewReader(r)
	var (
		header bundleHeader
		lines  [][]byte
		blobs  = map[BlobRef][]byte{}
		first  = true
	)
	for {
		line, err := br.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if len(line) > 0 {
			if first {
				if err := json.Unmarshal(line, &header); err != nil || header.Bundle != bundleFormat {
					return "", errors.New("trace: import: not an agentflow trace bundle")
				}
				if header.V != Version {
					return "", fmt.Errorf("trace: import: %w: %d", ErrUnsupportedVersion, header.V)
				}
				first = false
			} else if bytes.HasPrefix(line, []byte(`{"blob":`)) {
				var b bundleBlob
				if err := json.Unmarshal(line, &b); err != nil {
					return "", fmt.Errorf("trace: import: blob line: %w", err)
				}
				data, err := base64.StdEncoding.DecodeString(b.Data)
				if err != nil {
					return "", fmt.Errorf("trace: import: blob %s: %w", b.Blob, err)
				}
				if BlobRefOf(data) != b.Blob {
					return "", fmt.Errorf("trace: import: blob %s: content does not match its hash", b.Blob)
				}
				blobs[b.Blob] = data
			} else {
				lines = append(lines, line)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("trace: import: %w", err)
		}
	}
	if first {
		return "", errors.New("trace: import: empty bundle")
	}
	get := func(ref BlobRef) ([]byte, error) {
		data, ok := blobs[ref]
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrBlobNotFound, ref)
		}
		return data, nil
	}
	// Group consecutive events of the same run into one Append.
	var batch []Event
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		err := s.Append(ctx, batch...)
		batch = nil
		return err
	}
	for _, line := range lines {
		e, err := DecodeEvent(line, get)
		if err != nil {
			return "", fmt.Errorf("trace: import: %w", err)
		}
		if e.TraceID != header.TraceID {
			return "", fmt.Errorf("trace: import: event of trace %s in bundle of trace %s", e.TraceID, header.TraceID)
		}
		if len(batch) > 0 && batch[0].RunID != e.RunID {
			if err := flush(); err != nil {
				return "", err
			}
		}
		batch = append(batch, e)
	}
	if err := flush(); err != nil {
		return "", err
	}
	return header.TraceID, nil
}
