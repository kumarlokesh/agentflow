// Package mcp provides an MCP (Model Context Protocol) client for agentflow.
// It implements the MCP JSON-RPC 2.0 protocol over stdio, allowing agents to
// discover and use tools from any MCP-compatible server without writing custom
// tool implementations.
//
// MCP tool calls flow through agentflow's normal event-sourced pipeline -
// they are recorded as standard tool_call/tool_result events and are
// fully replayable.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
)

// Transport is the low-level MCP communication interface.
// Implementations send a JSON-RPC request and return the result payload.
// Implementations must be safe for concurrent use.
type Transport interface {
	// Send issues a JSON-RPC request with the given method and params, and
	// returns the result field from the response. It blocks until the server
	// replies or the context is cancelled.
	Send(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error)
	// Close shuts down the transport and releases its resources.
	Close() error
}

// --- JSON-RPC 2.0 wire types ---

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *jsonRPCError) Error() string {
	return fmt.Sprintf("mcp: JSON-RPC error %d: %s", e.Code, e.Message)
}

// pending tracks an in-flight request.
type pending struct {
	result chan jsonRPCResponse
}

// StdioTransport implements Transport by spawning a subprocess and
// communicating over its stdin/stdout using newline-delimited JSON.
// It is safe for concurrent use.
type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	mu     sync.Mutex // serialises writes to stdin

	nextID   atomic.Int64
	inflight sync.Map // int64 → *pending

	readErr  chan error // closed when the reader goroutine exits
	closeOnce sync.Once
}

// NewPipeTransport creates a StdioTransport from an existing reader/writer pair.
// This is primarily for testing - use NewStdioTransport for production use.
func NewPipeTransport(r io.Reader, w io.WriteCloser) *StdioTransport {
	t := &StdioTransport{
		stdin:   w,
		stdout:  bufio.NewScanner(r),
		readErr: make(chan error, 1),
	}
	go t.readLoop()
	return t
}

// NewStdioTransport spawns the given command and establishes a JSON-RPC
// session over its stdio streams. The subprocess is killed when Close is
// called or its own process exits.
func NewStdioTransport(command string, args ...string) (*StdioTransport, error) {
	cmd := exec.Command(command, args...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdin pipe: %w", err)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdout pipe: %w", err)
	}

	// Discard stderr so it doesn't block the subprocess.
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp: start subprocess: %w", err)
	}

	t := &StdioTransport{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewScanner(stdoutPipe),
		readErr: make(chan error, 1),
	}

	go t.readLoop()
	return t, nil
}

// Send marshals a JSON-RPC request, writes it to the subprocess stdin, and
// blocks until the matching response arrives or ctx is cancelled.
func (t *StdioTransport) Send(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	id := t.nextID.Add(1)

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	line, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("mcp: marshal request: %w", err)
	}
	line = append(line, '\n')

	p := &pending{result: make(chan jsonRPCResponse, 1)}
	t.inflight.Store(id, p)
	defer t.inflight.Delete(id)

	t.mu.Lock()
	_, writeErr := t.stdin.Write(line)
	t.mu.Unlock()
	if writeErr != nil {
		return nil, fmt.Errorf("mcp: write request: %w", writeErr)
	}

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("mcp: request cancelled: %w", ctx.Err())
	case resp := <-p.result:
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	case err := <-t.readErr:
		if err != nil {
			return nil, fmt.Errorf("mcp: reader error: %w", err)
		}
		return nil, fmt.Errorf("mcp: transport closed")
	}
}

// Close shuts down the subprocess (if any) and releases all resources.
func (t *StdioTransport) Close() error {
	var firstErr error
	t.closeOnce.Do(func() {
		t.stdin.Close()
		if t.cmd != nil && t.cmd.Process != nil {
			if err := t.cmd.Process.Kill(); err != nil {
				firstErr = fmt.Errorf("mcp: kill subprocess: %w", err)
			}
			t.cmd.Wait() //nolint:errcheck - best-effort reap
		}
	})
	return firstErr
}

// readLoop reads newline-delimited JSON responses and dispatches them to the
// corresponding in-flight pending channel.
func (t *StdioTransport) readLoop() {
	defer close(t.readErr)
	for t.stdout.Scan() {
		line := t.stdout.Bytes()
		if len(line) == 0 {
			continue
		}
		var resp jsonRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			// Non-JSON lines (e.g. server startup messages) are ignored.
			continue
		}
		if v, ok := t.inflight.Load(resp.ID); ok {
			p := v.(*pending)
			p.result <- resp
		}
	}
	if err := t.stdout.Err(); err != nil {
		t.readErr <- err
	}
}
