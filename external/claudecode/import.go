package claudecode

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/kumarlokesh/agentflow/trace"
)

// ImportOptions configure Import.
type ImportOptions struct {
	Options
	// LockDir, if set, holds a lock file per session, so that concurrent
	// imports of one session (hooks firing together) take turns.
	LockDir string
}

// ErrRewritten is returned when the store has events for a session that
// the transcript no longer produces: the transcript was rewritten, or the
// trace was written by something else.
var ErrRewritten = errors.New("claudecode: the stored trace doesn't match the transcript")

// Import converts the session at path and appends to s what it doesn't have
// yet. It returns the trace ID (the session ID), or "" if there was nothing
// to record yet.
func Import(ctx context.Context, s trace.Store, path string, opts ImportOptions) (string, error) {
	id := sessionIDFromPath(path)
	if opts.LockDir != "" {
		unlock, err := lock(ctx, filepath.Join(opts.LockDir, "claude-code-"+id+".lock"))
		if err != nil {
			return "", err
		}
		defer unlock()
	}
	sess, err := Convert(path, opts.Options)
	if err != nil {
		return "", err
	}
	if len(sess.Runs) == 0 {
		return "", nil
	}
	for _, r := range sess.Runs {
		existing, err := trace.Collect(s.Events(ctx, r.RunID, 0))
		if err != nil && !errors.Is(err, trace.ErrRunNotFound) {
			return "", err
		}
		if len(existing) > len(r.Events) {
			return "", fmt.Errorf("%w: run %s has %d events stored, the transcript gives %d", ErrRewritten, r.Path, len(existing), len(r.Events))
		}
		for i, e := range existing {
			want := r.Events[i]
			if e.Kind() != want.Kind() || !sameCause(e.Cause, want.Cause) {
				return "", fmt.Errorf("%w: run %s seq %d is %s, the transcript gives %s", ErrRewritten, r.Path, i, e.Kind(), want.Kind())
			}
		}
		if rest := r.Events[len(existing):]; len(rest) > 0 {
			if err := s.Append(ctx, rest...); err != nil {
				return "", err
			}
		}
	}
	return sess.ID, nil
}

func sameCause(a, b *uint64) bool {
	return (a == nil) == (b == nil) && (a == nil || *a == *b)
}

func sessionIDFromPath(path string) string {
	return trimExt(filepath.Base(path))
}

func trimExt(name string) string {
	return name[:len(name)-len(filepath.Ext(name))]
}

// Sessions returns the session transcripts in a Claude Code project
// directory (~/.claude/projects/<project>), oldest first.
func Sessions(projectDir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(projectDir, "*.jsonl"))
	if err != nil {
		return nil, err
	}
	sort.Slice(files, func(i, j int) bool {
		a, _ := os.Stat(files[i])
		b, _ := os.Stat(files[j])
		if a == nil || b == nil {
			return files[i] < files[j]
		}
		return a.ModTime().Before(b.ModTime())
	})
	return files, nil
}

// lock takes a lock file, waiting for another holder; a lock older than two
// minutes is stale (its holder died) and is taken over.
func lock(ctx context.Context, path string) (func(), error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	deadline := time.Now().Add(30 * time.Second)
	for {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_ = f.Close()
			return func() { _ = os.Remove(path) }, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		if st, err := os.Stat(path); err == nil && time.Since(st.ModTime()) > 2*time.Minute {
			_ = os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("claudecode: %s is locked by another import", path)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}
