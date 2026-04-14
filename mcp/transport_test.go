package mcp_test

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/kumarlokesh/agentflow/mcp"
)

// mockServer is a minimal JSON-RPC 2.0 server that responds to a predefined
// set of methods. It runs in a goroutine and communicates over an io.Pipe.
type mockServer struct {
	handlers map[string]func(params json.RawMessage) (any, error)
	reader   *bufio.Scanner
	writer   io.Writer
}

func newMockServer(r io.Reader, w io.Writer) *mockServer {
	return &mockServer{
		handlers: make(map[string]func(json.RawMessage) (any, error)),
		reader:   bufio.NewScanner(r),
		writer:   w,
	}
}

func (s *mockServer) handle(method string, fn func(json.RawMessage) (any, error)) {
	s.handlers[method] = fn
}

func (s *mockServer) serve() {
	for s.reader.Scan() {
		line := s.reader.Bytes()
		var req struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      *int64          `json:"id"`
			Method  string          `json:"method"`
			Params  json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(line, &req); err != nil {
			continue
		}
		// Notifications (no ID) — skip response.
		if req.ID == nil {
			continue
		}
		fn, ok := s.handlers[req.Method]
		var resp map[string]any
		if !ok {
			resp = map[string]any{
				"jsonrpc": "2.0",
				"id":      *req.ID,
				"error":   map[string]any{"code": -32601, "message": "method not found"},
			}
		} else {
			result, err := fn(req.Params)
			if err != nil {
				resp = map[string]any{
					"jsonrpc": "2.0",
					"id":      *req.ID,
					"error":   map[string]any{"code": -32000, "message": err.Error()},
				}
			} else {
				resp = map[string]any{
					"jsonrpc": "2.0",
					"id":      *req.ID,
					"result":  result,
				}
			}
		}
		b, _ := json.Marshal(resp)
		b = append(b, '\n')
		s.writer.Write(b)
	}
}

// newPipedTransport creates a Transport backed by an in-process mock server.
func newPipedTransport(t *testing.T, setup func(*mockServer)) mcp.Transport {
	t.Helper()

	// client reads from serverOut, writes to serverIn.
	serverIn_r, serverIn_w := io.Pipe()
	serverOut_r, serverOut_w := io.Pipe()

	srv := newMockServer(serverIn_r, serverOut_w)
	if setup != nil {
		setup(srv)
	}
	go func() {
		srv.serve()
		serverOut_w.Close()
	}()

	transport := mcp.NewPipeTransport(serverOut_r, serverIn_w)
	t.Cleanup(func() {
		transport.Close()
		serverIn_w.Close()
	})
	return transport
}

func TestStdioTransport_SendReceive(t *testing.T) {
	transport := newPipedTransport(t, func(s *mockServer) {
		s.handle("ping", func(_ json.RawMessage) (any, error) {
			return map[string]string{"pong": "ok"}, nil
		})
	})

	result, err := transport.Send(context.Background(), "ping", nil)
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	var got map[string]string
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if got["pong"] != "ok" {
		t.Errorf("pong = %q, want ok", got["pong"])
	}
}

func TestStdioTransport_ConcurrentRequests(t *testing.T) {
	transport := newPipedTransport(t, func(s *mockServer) {
		s.handle("echo", func(params json.RawMessage) (any, error) {
			return json.RawMessage(params), nil
		})
	})

	const n = 10
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			params, _ := json.Marshal(map[string]int{"i": i})
			result, err := transport.Send(context.Background(), "echo", params)
			if err != nil {
				errs <- fmt.Errorf("request %d: %w", i, err)
				return
			}
			var got map[string]int
			if err := json.Unmarshal(result, &got); err != nil {
				errs <- fmt.Errorf("request %d unmarshal: %w", i, err)
				return
			}
			errs <- nil
		}(i)
	}
	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Error(err)
		}
	}
}

func TestStdioTransport_ContextCancellation(t *testing.T) {
	transport := newPipedTransport(t, func(s *mockServer) {
		// No handlers — requests block forever.
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := transport.Send(ctx, "slow", nil)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestStdioTransport_ServerError(t *testing.T) {
	transport := newPipedTransport(t, func(s *mockServer) {
		s.handle("fail", func(_ json.RawMessage) (any, error) {
			return nil, fmt.Errorf("something went wrong")
		})
	})

	_, err := transport.Send(context.Background(), "fail", nil)
	if err == nil {
		t.Fatal("expected error from server")
	}
	if !strings.Contains(err.Error(), "something went wrong") {
		t.Errorf("error = %q, want to contain 'something went wrong'", err)
	}
}

// TestRealStdioTransport_EchoServer uses a real subprocess (the Go echo helper
// embedded below) to verify the full stdio path. Skipped if 'go' is not in PATH.
func TestRealStdioTransport_EchoServer(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not in PATH")
	}

	// Write a tiny echo server to a temp file and run it.
	echoSrc := `package main
import ("bufio";"encoding/json";"os")
func main() {
	s := bufio.NewScanner(os.Stdin)
	for s.Scan() {
		var req map[string]any
		if json.Unmarshal(s.Bytes(), &req) != nil { continue }
		if req["id"] == nil { continue }
		resp := map[string]any{"jsonrpc":"2.0","id":req["id"],"result":req["params"]}
		b, _ := json.Marshal(resp)
		os.Stdout.Write(append(b, '\n'))
	}
}`
	dir := t.TempDir()
	srcFile := dir + "/main.go"
	if err := os.WriteFile(srcFile, []byte(echoSrc), 0644); err != nil {
		t.Fatal(err)
	}

	transport, err := mcp.NewStdioTransport("go", "run", srcFile)
	if err != nil {
		t.Fatalf("NewStdioTransport: %v", err)
	}
	defer transport.Close()

	params, _ := json.Marshal(map[string]string{"hello": "world"})
	result, err := transport.Send(context.Background(), "test", params)
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	var got map[string]string
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["hello"] != "world" {
		t.Errorf("echo got %q, want world", got["hello"])
	}
}
