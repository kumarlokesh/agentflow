// Package filestore is a trace.Store backed by JSONL files (RFC-0001 §10):
//
//	<root>/traces/<trace_id>/<run_id>.jsonl   one event per line, seq order
//	<root>/blobs/sha256/<hh>/<remaining-hex>  content-addressed blobs
//
// Each run has exactly one writer (one process). A crash can leave a
// truncated final line; readers drop it and report the run as incomplete.
package filestore

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/kumarlokesh/agentflow/trace"
)

const (
	dirPerm  = 0o700
	filePerm = 0o600
	ext      = ".jsonl"
)

// Options configure a Store.
type Options struct {
	// BlobThreshold is the string length above which strings are moved to
	// blob files. Zero means trace.DefaultBlobThreshold; negative disables.
	BlobThreshold int
	// Sync fsyncs the run file after every Append.
	Sync bool
}

// Store is a file-backed trace.Store. Call Close when done writing.
type Store struct {
	root string
	opts Options

	mu    sync.Mutex
	open  map[string]*runFile // runs being written, by run ID
	where map[string]string   // run ID → trace ID cache
}

type runFile struct {
	traceID string
	f       *os.File
	w       *bufio.Writer
	next    uint64
}

var _ trace.Store = (*Store)(nil)

// Open opens (creating if needed) a store rooted at dir.
func Open(dir string, opts Options) (*Store, error) {
	if opts.BlobThreshold == 0 {
		opts.BlobThreshold = trace.DefaultBlobThreshold
	}
	for _, d := range []string{dir, filepath.Join(dir, "traces"), filepath.Join(dir, "blobs", "sha256")} {
		if err := os.MkdirAll(d, dirPerm); err != nil {
			return nil, fmt.Errorf("filestore: %w", err)
		}
	}
	return &Store{root: dir, opts: opts, open: map[string]*runFile{}, where: map[string]string{}}, nil
}

// Root returns the store's directory.
func (s *Store) Root() string { return s.root }

// Close flushes and closes every run file still open for writing.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var errs []error
	for id, rf := range s.open {
		errs = append(errs, rf.close())
		delete(s.open, id)
	}
	return errors.Join(errs...)
}

func (rf *runFile) close() error {
	err := rf.w.Flush()
	return errors.Join(err, rf.f.Close())
}

func (s *Store) runPath(traceID, runID string) string {
	return filepath.Join(s.root, "traces", traceID, runID+ext)
}

func validID(id string) bool {
	return id != "" && id != "." && id != ".." && !strings.ContainsAny(id, `/\`)
}

// Append implements trace.Store.
func (s *Store) Append(_ context.Context, events ...trace.Event) error {
	if len(events) == 0 {
		return nil
	}
	runID, traceID := events[0].RunID, events[0].TraceID
	if !validID(runID) || !validID(traceID) {
		return fmt.Errorf("filestore: invalid trace/run id %q/%q", traceID, runID)
	}
	for _, e := range events {
		if e.RunID != runID || e.TraceID != traceID {
			return fmt.Errorf("filestore: Append mixes runs (%s/%s and %s/%s)", traceID, runID, e.TraceID, e.RunID)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	rf, err := s.writerFor(traceID, runID, events[0])
	if err != nil {
		return err
	}
	for i, e := range events {
		if want := rf.next + uint64(i); e.Seq != want {
			return fmt.Errorf("%w: run %s: got seq %d, want %d", trace.ErrSeqConflict, runID, e.Seq, want)
		}
	}
	var buf bytes.Buffer
	finished := false
	for _, e := range events {
		line, err := trace.EncodeEvent(e, trace.EncodeOptions{
			BlobThreshold: max(s.opts.BlobThreshold, 0),
			PutBlob:       s.putBlobLocked,
		})
		if err != nil {
			return err
		}
		buf.Write(line)
		buf.WriteByte('\n')
		if _, ok := e.Body.(*trace.RunFinished); ok {
			finished = true
		}
	}
	if _, err := rf.w.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("filestore: write run %s: %w", runID, err)
	}
	if err := rf.w.Flush(); err != nil {
		return fmt.Errorf("filestore: write run %s: %w", runID, err)
	}
	if s.opts.Sync {
		if err := rf.f.Sync(); err != nil {
			return fmt.Errorf("filestore: sync run %s: %w", runID, err)
		}
	}
	rf.next += uint64(len(events))
	if finished {
		delete(s.open, runID)
		if err := rf.close(); err != nil {
			return fmt.Errorf("filestore: close run %s: %w", runID, err)
		}
	}
	return nil
}

// writerFor returns the open writer for a run, opening or creating its file.
// Called with s.mu held.
func (s *Store) writerFor(traceID, runID string, first trace.Event) (*runFile, error) {
	if rf, ok := s.open[runID]; ok {
		if rf.traceID != traceID {
			return nil, fmt.Errorf("filestore: run %s belongs to trace %s, not %s", runID, rf.traceID, traceID)
		}
		return rf, nil
	}
	if existing, ok := s.locate(runID); ok && existing != traceID {
		return nil, fmt.Errorf("filestore: run %s belongs to trace %s, not %s", runID, existing, traceID)
	}
	path := s.runPath(traceID, runID)
	var next uint64
	if _, err := os.Stat(path); err == nil {
		// Reopening an existing run: count complete lines and drop a
		// truncated tail left by a crash, so appends stay well-formed.
		n, valid, err := countLines(path)
		if err != nil {
			return nil, err
		}
		if err := os.Truncate(path, valid); err != nil {
			return nil, fmt.Errorf("filestore: %w", err)
		}
		next = n
	} else {
		if !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("filestore: %w", err)
		}
		if first.Seq != 0 {
			return nil, fmt.Errorf("%w: run %s: got seq %d, want 0", trace.ErrSeqConflict, runID, first.Seq)
		}
		if _, ok := first.Body.(*trace.RunStarted); !ok {
			return nil, fmt.Errorf("filestore: run %s: first event must be run.started, got %s", runID, first.Kind())
		}
		if err := os.MkdirAll(filepath.Dir(path), dirPerm); err != nil {
			return nil, fmt.Errorf("filestore: %w", err)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, filePerm)
	if err != nil {
		return nil, fmt.Errorf("filestore: %w", err)
	}
	rf := &runFile{traceID: traceID, f: f, w: bufio.NewWriter(f), next: next}
	s.open[runID] = rf
	s.where[runID] = traceID
	return rf, nil
}

// countLines returns the number of complete (newline-terminated) lines and
// the byte length they occupy.
func countLines(path string) (uint64, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, fmt.Errorf("filestore: %w", err)
	}
	defer func() { _ = f.Close() }()
	r := bufio.NewReader(f)
	var n uint64
	var valid int64
	for {
		line, err := r.ReadBytes('\n')
		if err == io.EOF {
			return n, valid, nil
		}
		if err != nil {
			return 0, 0, fmt.Errorf("filestore: %w", err)
		}
		n++
		valid += int64(len(line))
	}
}

// locate finds which trace a run belongs to. Called with s.mu held.
func (s *Store) locate(runID string) (string, bool) {
	if t, ok := s.where[runID]; ok {
		return t, true
	}
	matches, _ := filepath.Glob(filepath.Join(s.root, "traces", "*", runID+ext))
	if len(matches) == 0 {
		return "", false
	}
	t := filepath.Base(filepath.Dir(matches[0]))
	s.where[runID] = t
	return t, true
}

// Events implements trace.Store.
func (s *Store) Events(_ context.Context, runID string, from uint64) iter.Seq2[trace.Event, error] {
	return func(yield func(trace.Event, error) bool) {
		if !validID(runID) {
			yield(trace.Event{}, fmt.Errorf("filestore: invalid run id %q", runID))
			return
		}
		s.mu.Lock()
		traceID, ok := s.locate(runID)
		s.mu.Unlock()
		if !ok {
			yield(trace.Event{}, fmt.Errorf("%w: %s", trace.ErrRunNotFound, runID))
			return
		}
		f, err := os.Open(s.runPath(traceID, runID))
		if err != nil {
			yield(trace.Event{}, fmt.Errorf("filestore: %w", err))
			return
		}
		defer func() { _ = f.Close() }()
		r := bufio.NewReader(f)
		for i := uint64(0); ; i++ {
			line, err := r.ReadBytes('\n')
			if err == io.EOF {
				return // a final line without '\n' is a crash artifact: dropped
			}
			if err != nil {
				yield(trace.Event{}, fmt.Errorf("filestore: read run %s: %w", runID, err))
				return
			}
			if i < from {
				continue
			}
			e, err := trace.DecodeEvent(bytes.TrimSuffix(line, []byte("\n")), s.getBlob)
			if err == nil && e.Seq != i {
				err = fmt.Errorf("filestore: run %s line %d has seq %d", runID, i, e.Seq)
			}
			if !yield(e, err) || err != nil {
				return
			}
		}
	}
}

// Runs implements trace.Store.
func (s *Store) Runs(_ context.Context, f trace.RunFilter) iter.Seq2[trace.RunInfo, error] {
	return func(yield func(trace.RunInfo, error) bool) {
		traceIDs, err := listDir(filepath.Join(s.root, "traces"), true)
		if err != nil {
			yield(trace.RunInfo{}, err)
			return
		}
		sort.Sort(sort.Reverse(sort.StringSlice(traceIDs))) // UUIDv7: newest first
		var infos []trace.RunInfo
		for _, tid := range traceIDs {
			if f.TraceID != "" && tid != f.TraceID {
				continue
			}
			files, err := listDir(filepath.Join(s.root, "traces", tid), false)
			if err != nil {
				yield(trace.RunInfo{}, err)
				return
			}
			sort.Strings(files) // UUIDv7: start order
			for _, name := range files {
				if !strings.HasSuffix(name, ext) {
					continue
				}
				info, err := s.runInfo(filepath.Join(s.root, "traces", tid, name))
				if err != nil {
					if !yield(trace.RunInfo{}, err) {
						return
					}
					continue
				}
				infos = append(infos, info)
			}
		}
		trace.FilterRuns(infos, f, yield)
	}
}

func listDir(dir string, dirs bool) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("filestore: %w", err)
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() == dirs {
			out = append(out, e.Name())
		}
	}
	return out, nil
}

// runInfo reads a run file's first line and, from its tail, its last
// complete line, to build a RunInfo without reading the whole file.
func (s *Store) runInfo(path string) (trace.RunInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return trace.RunInfo{}, fmt.Errorf("filestore: %w", err)
	}
	defer func() { _ = f.Close() }()

	firstLine, err := bufio.NewReader(f).ReadBytes('\n')
	if err != nil {
		return trace.RunInfo{}, fmt.Errorf("filestore: %s: no complete first line: %w", path, err)
	}
	first, err := trace.DecodeEvent(bytes.TrimSuffix(firstLine, []byte("\n")), s.getBlob)
	if err != nil {
		return trace.RunInfo{}, fmt.Errorf("filestore: %s: %w", path, err)
	}

	last, err := lastCompleteLine(f)
	if err != nil {
		return trace.RunInfo{}, fmt.Errorf("filestore: %s: %w", path, err)
	}
	var finished *trace.Event
	if last != nil && isKind(last, trace.KindRunFinished) {
		e, err := trace.DecodeEvent(last, s.getBlob)
		if err != nil {
			return trace.RunInfo{}, fmt.Errorf("filestore: %s: %w", path, err)
		}
		finished = &e
	}
	return trace.RunInfoFrom(first, finished)
}

// lastCompleteLine returns the last newline-terminated line of f (without
// the newline), reading backwards from the end.
func lastCompleteLine(f *os.File) ([]byte, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	const chunk = 64 << 10
	var tail []byte
	for off := size; off > 0; {
		n := int64(chunk)
		if off < n {
			n = off
		}
		off -= n
		buf := make([]byte, n)
		if _, err := f.ReadAt(buf, off); err != nil && err != io.EOF {
			return nil, err
		}
		tail = append(buf, tail...)
		// Drop an unterminated final fragment (crash artifact).
		end := bytes.LastIndexByte(tail, '\n')
		if end < 0 {
			continue
		}
		start := bytes.LastIndexByte(tail[:end], '\n')
		if start >= 0 || off == 0 {
			return tail[start+1 : end], nil
		}
		if int64(len(tail)) > trace.MaxEventSize+chunk {
			return nil, errors.New("last line exceeds the maximum event size")
		}
	}
	return nil, nil
}

func isKind(line []byte, k trace.Kind) bool {
	var head struct {
		Kind trace.Kind `json:"kind"`
	}
	return json.Unmarshal(line, &head) == nil && head.Kind == k
}

// --- blobs ---

func (s *Store) blobPath(ref trace.BlobRef) (string, error) {
	if !ref.Valid() {
		return "", fmt.Errorf("filestore: invalid blob ref %q", ref)
	}
	h := strings.TrimPrefix(string(ref), "sha256:")
	return filepath.Join(s.root, "blobs", "sha256", h[:2], h[2:]), nil
}

// PutBlob implements trace.BlobStore.
func (s *Store) PutBlob(_ context.Context, data []byte) (trace.BlobRef, error) {
	return s.putBlobLocked(data)
}

// putBlobLocked writes a blob atomically (temp file + rename). It doesn't
// need s.mu: blob files are immutable and named by content.
func (s *Store) putBlobLocked(data []byte) (trace.BlobRef, error) {
	ref := trace.BlobRefOf(data)
	path, err := s.blobPath(ref)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(path); err == nil {
		return ref, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), dirPerm); err != nil {
		return "", fmt.Errorf("filestore: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return "", fmt.Errorf("filestore: %w", err)
	}
	_, werr := tmp.Write(data)
	cerr := tmp.Close()
	if err := errors.Join(werr, cerr); err != nil {
		_ = os.Remove(tmp.Name())
		return "", fmt.Errorf("filestore: write blob: %w", err)
	}
	if err := os.Chmod(tmp.Name(), filePerm); err != nil {
		_ = os.Remove(tmp.Name())
		return "", fmt.Errorf("filestore: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		_ = os.Remove(tmp.Name())
		return "", fmt.Errorf("filestore: %w", err)
	}
	return ref, nil
}

// Blob implements trace.BlobStore.
func (s *Store) Blob(_ context.Context, ref trace.BlobRef) ([]byte, error) {
	return s.getBlob(ref)
}

func (s *Store) getBlob(ref trace.BlobRef) ([]byte, error) {
	path, err := s.blobPath(ref)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%w: %s", trace.ErrBlobNotFound, ref)
	}
	if err != nil {
		return nil, fmt.Errorf("filestore: %w", err)
	}
	if trace.BlobRefOf(data) != ref {
		return nil, fmt.Errorf("filestore: blob %s is corrupt (content does not match its hash)", ref)
	}
	return data, nil
}
