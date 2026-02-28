package multi

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// --- Mock Runner ---

type mockRunner struct {
	name   string
	output string
	err    error
	delay  time.Duration
	mu     sync.Mutex
	calls  []string
}

func newMockRunner(name, output string) *mockRunner {
	return &mockRunner{name: name, output: output}
}

func (r *mockRunner) Name() string { return r.name }

func (r *mockRunner) Run(ctx context.Context, task string) (string, error) {
	r.mu.Lock()
	r.calls = append(r.calls, task)
	delay := r.delay
	err := r.err
	output := r.output
	r.mu.Unlock()

	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	if err != nil {
		return "", err
	}
	return output, nil
}

func (r *mockRunner) callCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.calls)
}

// --- Mailbox Tests ---

func TestMailbox_SendReceive(t *testing.T) {
	mb := NewMailbox("test", 10)

	msg := Message{ID: "m1", Content: "hello"}
	if err := mb.Send(msg); err != nil {
		t.Fatalf("Send() error = %v", err)
	}

	if mb.Len() != 1 {
		t.Errorf("Len = %d, want 1", mb.Len())
	}

	got, err := mb.Receive(context.Background())
	if err != nil {
		t.Fatalf("Receive() error = %v", err)
	}
	if got.Content != "hello" {
		t.Errorf("Content = %q, want hello", got.Content)
	}
}

func TestMailbox_Full(t *testing.T) {
	mb := NewMailbox("test", 1)

	mb.Send(Message{ID: "m1"})
	err := mb.Send(Message{ID: "m2"})
	if !errors.Is(err, ErrMailboxFull) {
		t.Errorf("expected ErrMailboxFull, got %v", err)
	}
}

func TestMailbox_TryReceive(t *testing.T) {
	mb := NewMailbox("test", 10)

	// Empty mailbox.
	_, ok := mb.TryReceive()
	if ok {
		t.Error("expected false from empty mailbox")
	}

	mb.Send(Message{ID: "m1", Content: "hi"})
	msg, ok := mb.TryReceive()
	if !ok {
		t.Error("expected true after send")
	}
	if msg.Content != "hi" {
		t.Errorf("Content = %q, want hi", msg.Content)
	}
}

func TestMailbox_ReceiveContextCancel(t *testing.T) {
	mb := NewMailbox("test", 10)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := mb.Receive(ctx)
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestMailbox_DefaultCapacity(t *testing.T) {
	mb := NewMailbox("test", 0) // 0 → default 100
	if mb.capacity != 100 {
		t.Errorf("capacity = %d, want 100", mb.capacity)
	}
}

func TestMailbox_Concurrent(t *testing.T) {
	mb := NewMailbox("test", 200)
	var wg sync.WaitGroup

	// 100 senders.
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			mb.Send(Message{ID: fmt.Sprintf("m%d", i)})
		}(i)
	}
	wg.Wait()

	if mb.Len() != 100 {
		t.Errorf("Len = %d, want 100", mb.Len())
	}
}

// --- Registry Tests ---

func TestRegistry_RegisterAndGet(t *testing.T) {
	reg := NewRegistry()
	agent := newMockRunner("agent-1", "output")

	if err := reg.Register(agent); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	got, ok := reg.Get("agent-1")
	if !ok {
		t.Fatal("Get() returned false")
	}
	if got.Name() != "agent-1" {
		t.Errorf("Name = %q, want agent-1", got.Name())
	}
}

func TestRegistry_DuplicateRegistration(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("agent-1", ""))

	err := reg.Register(newMockRunner("agent-1", ""))
	if err == nil {
		t.Error("expected error for duplicate registration")
	}
}

func TestRegistry_EmptyName(t *testing.T) {
	reg := NewRegistry()
	err := reg.Register(newMockRunner("", ""))
	if err == nil {
		t.Error("expected error for empty name")
	}
}

func TestRegistry_GetNotFound(t *testing.T) {
	reg := NewRegistry()
	_, ok := reg.Get("nonexistent")
	if ok {
		t.Error("expected false for nonexistent agent")
	}
}

func TestRegistry_Mailbox(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("a1", ""))

	mb, ok := reg.Mailbox("a1")
	if !ok || mb == nil {
		t.Error("expected mailbox for registered agent")
	}

	_, ok = reg.Mailbox("nonexistent")
	if ok {
		t.Error("expected false for nonexistent agent mailbox")
	}
}

func TestRegistry_Names(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("b", ""))
	reg.Register(newMockRunner("a", ""))

	names := reg.Names()
	if len(names) != 2 {
		t.Fatalf("Names() len = %d, want 2", len(names))
	}
}

func TestRegistry_Len(t *testing.T) {
	reg := NewRegistry()
	if reg.Len() != 0 {
		t.Error("Len should be 0")
	}
	reg.Register(newMockRunner("a", ""))
	if reg.Len() != 1 {
		t.Error("Len should be 1")
	}
}

func TestRegistry_Concurrent(t *testing.T) {
	reg := NewRegistry()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			reg.Register(newMockRunner(fmt.Sprintf("agent-%d", i), ""))
			reg.Get(fmt.Sprintf("agent-%d", i))
			reg.Names()
			reg.Len()
		}(i)
	}
	wg.Wait()

	if reg.Len() != 50 {
		t.Errorf("Len = %d, want 50", reg.Len())
	}
}

// --- Coordinator Tests ---

func TestCoordinator_Delegate(t *testing.T) {
	reg := NewRegistry()
	agent := newMockRunner("solver", "42")
	reg.Register(agent)

	coord := NewCoordinator(reg, CoordinatorConfig{})

	task, err := coord.Delegate(context.Background(), "solver", "what is 6*7?")
	if err != nil {
		t.Fatalf("Delegate() error = %v", err)
	}
	if task.Status != TaskCompleted {
		t.Errorf("Status = %q, want completed", task.Status)
	}
	if task.Result != "42" {
		t.Errorf("Result = %q, want 42", task.Result)
	}
	if task.AssignedTo != "solver" {
		t.Errorf("AssignedTo = %q, want solver", task.AssignedTo)
	}
	if agent.callCount() != 1 {
		t.Errorf("callCount = %d, want 1", agent.callCount())
	}
}

func TestCoordinator_DelegateNotFound(t *testing.T) {
	reg := NewRegistry()
	coord := NewCoordinator(reg, CoordinatorConfig{})

	task, err := coord.Delegate(context.Background(), "nonexistent", "task")
	if !errors.Is(err, ErrAgentNotFound) {
		t.Errorf("expected ErrAgentNotFound, got %v", err)
	}
	if task.Status != TaskFailed {
		t.Errorf("Status = %q, want failed", task.Status)
	}
}

func TestCoordinator_DelegateError(t *testing.T) {
	reg := NewRegistry()
	agent := newMockRunner("failer", "")
	agent.err = errors.New("boom")
	reg.Register(agent)

	coord := NewCoordinator(reg, CoordinatorConfig{})
	task, err := coord.Delegate(context.Background(), "failer", "fail please")
	if err == nil {
		t.Fatal("expected error")
	}
	if task.Status != TaskFailed {
		t.Errorf("Status = %q, want failed", task.Status)
	}
	if task.Error != "boom" {
		t.Errorf("Error = %q, want boom", task.Error)
	}
}

func TestCoordinator_MaxDelegationDepth(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("agent", "ok"))

	coord := NewCoordinator(reg, CoordinatorConfig{MaxDelegationDepth: 2})

	_, err := coord.DelegateFrom(context.Background(), "agent", "task", "parent", 2)
	if !errors.Is(err, ErrDelegationDepth) {
		t.Errorf("expected ErrDelegationDepth, got %v", err)
	}
}

func TestCoordinator_DelegateFrom_SendsResult(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("worker", "result-data"))
	reg.Register(newMockRunner("manager", ""))

	coord := NewCoordinator(reg, CoordinatorConfig{})

	task, err := coord.DelegateFrom(context.Background(), "worker", "do work", "manager", 1)
	if err != nil {
		t.Fatalf("DelegateFrom() error = %v", err)
	}
	if task.Result != "result-data" {
		t.Errorf("Result = %q, want result-data", task.Result)
	}

	// Manager should have a result message in mailbox.
	mb, _ := reg.Mailbox("manager")
	msg, ok := mb.TryReceive()
	if !ok {
		t.Fatal("expected message in manager mailbox")
	}
	if msg.Type != MsgResult {
		t.Errorf("Type = %q, want result", msg.Type)
	}
	if msg.Content != "result-data" {
		t.Errorf("Content = %q, want result-data", msg.Content)
	}
}

func TestCoordinator_DelegateFrom_SendsError(t *testing.T) {
	reg := NewRegistry()
	worker := newMockRunner("worker", "")
	worker.err = errors.New("failed")
	reg.Register(worker)
	reg.Register(newMockRunner("manager", ""))

	coord := NewCoordinator(reg, CoordinatorConfig{})
	coord.DelegateFrom(context.Background(), "worker", "do work", "manager", 0)

	mb, _ := reg.Mailbox("manager")
	msg, ok := mb.TryReceive()
	if !ok {
		t.Fatal("expected error message in manager mailbox")
	}
	if msg.Type != MsgError {
		t.Errorf("Type = %q, want error", msg.Type)
	}
}

func TestCoordinator_Stop(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("agent", "ok"))
	coord := NewCoordinator(reg, CoordinatorConfig{})

	coord.Stop()

	_, err := coord.Delegate(context.Background(), "agent", "task")
	if !errors.Is(err, ErrCoordinatorStopped) {
		t.Errorf("expected ErrCoordinatorStopped, got %v", err)
	}
}

func TestCoordinator_Tasks(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("a", "ok"))
	coord := NewCoordinator(reg, CoordinatorConfig{})

	coord.Delegate(context.Background(), "a", "task1")
	coord.Delegate(context.Background(), "a", "task2")

	tasks := coord.Tasks()
	if len(tasks) != 2 {
		t.Errorf("Tasks() len = %d, want 2", len(tasks))
	}
}

func TestCoordinator_GetTask(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("a", "ok"))
	coord := NewCoordinator(reg, CoordinatorConfig{})

	coord.Delegate(context.Background(), "a", "task")

	task, ok := coord.GetTask("task-1")
	if !ok {
		t.Fatal("GetTask returned false")
	}
	if task.Status != TaskCompleted {
		t.Errorf("Status = %q", task.Status)
	}
}

func TestCoordinator_SendMessage(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("a", ""))
	reg.Register(newMockRunner("b", ""))
	coord := NewCoordinator(reg, CoordinatorConfig{})

	err := coord.SendMessage("a", "b", MsgInfo, "hello from a")
	if err != nil {
		t.Fatalf("SendMessage() error = %v", err)
	}

	mb, _ := reg.Mailbox("b")
	msg, ok := mb.TryReceive()
	if !ok {
		t.Fatal("expected message")
	}
	if msg.From != "a" || msg.Content != "hello from a" {
		t.Errorf("got From=%q Content=%q", msg.From, msg.Content)
	}
}

func TestCoordinator_SendMessage_NotFound(t *testing.T) {
	reg := NewRegistry()
	coord := NewCoordinator(reg, CoordinatorConfig{})

	err := coord.SendMessage("a", "nonexistent", MsgInfo, "hi")
	if !errors.Is(err, ErrAgentNotFound) {
		t.Errorf("expected ErrAgentNotFound, got %v", err)
	}
}

func TestCoordinator_MaxConcurrent(t *testing.T) {
	reg := NewRegistry()
	// Create 5 agents with small delays.
	for i := 0; i < 5; i++ {
		agent := newMockRunner(fmt.Sprintf("agent-%d", i), "ok")
		agent.delay = 50 * time.Millisecond
		reg.Register(agent)
	}

	coord := NewCoordinator(reg, CoordinatorConfig{MaxConcurrent: 2})

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			coord.Delegate(context.Background(), fmt.Sprintf("agent-%d", i), "work")
		}(i)
	}
	wg.Wait()

	tasks := coord.Tasks()
	completed := 0
	for _, t := range tasks {
		if t.Status == TaskCompleted {
			completed++
		}
	}
	if completed != 5 {
		t.Errorf("completed = %d, want 5", completed)
	}
}

func TestCoordinator_ContextCancellation(t *testing.T) {
	reg := NewRegistry()
	agent := newMockRunner("slow", "ok")
	agent.delay = 5 * time.Second
	reg.Register(agent)

	coord := NewCoordinator(reg, CoordinatorConfig{})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := coord.Delegate(ctx, "slow", "slow task")
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

// --- FanOut Tests ---

func TestCoordinator_FanOut(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("a1", "result-a"))
	reg.Register(newMockRunner("a2", "result-b"))
	reg.Register(newMockRunner("a3", "result-c"))

	coord := NewCoordinator(reg, CoordinatorConfig{})

	results, err := coord.FanOut(context.Background(), []string{"a1", "a2", "a3"}, "shared task")
	if err != nil {
		t.Fatalf("FanOut() error = %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("results len = %d, want 3", len(results))
	}

	for _, task := range results {
		if task.Status != TaskCompleted {
			t.Errorf("task %s Status = %q", task.ID, task.Status)
		}
	}
}

func TestCoordinator_FanOut_PartialFailure(t *testing.T) {
	reg := NewRegistry()
	reg.Register(newMockRunner("good", "ok"))
	bad := newMockRunner("bad", "")
	bad.err = errors.New("fail")
	reg.Register(bad)

	coord := NewCoordinator(reg, CoordinatorConfig{})

	results, err := coord.FanOut(context.Background(), []string{"good", "bad"}, "task")
	if err == nil {
		t.Error("expected error from partial failure")
	}
	if len(results) != 2 {
		t.Fatalf("results len = %d, want 2", len(results))
	}

	var completed, failed int
	for _, task := range results {
		switch task.Status {
		case TaskCompleted:
			completed++
		case TaskFailed:
			failed++
		}
	}
	if completed != 1 || failed != 1 {
		t.Errorf("completed=%d, failed=%d, want 1,1", completed, failed)
	}
}
