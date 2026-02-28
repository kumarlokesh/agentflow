// Package multi provides multi-agent orchestration for agentflow.
//
// It implements an agent registry, inter-agent messaging via mailboxes,
// task delegation, and concurrency control. The Coordinator orchestrates
// multiple agents working together on complex tasks.
//
// Design rationale: Multi-agent coordination is built on channels and
// context cancellation rather than shared mutable state. Each agent has
// its own mailbox (buffered channel), and the coordinator manages
// delegation and result aggregation.
package multi

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// --- Sentinel errors ---

var (
	// ErrAgentNotFound is returned when a referenced agent is not registered.
	ErrAgentNotFound = errors.New("multi: agent not found")
	// ErrMailboxFull is returned when an agent's mailbox is at capacity.
	ErrMailboxFull = errors.New("multi: mailbox full")
	// ErrDelegationDepth is returned when delegation exceeds the max depth.
	ErrDelegationDepth = errors.New("multi: max delegation depth exceeded")
	// ErrCoordinatorStopped is returned when the coordinator is no longer running.
	ErrCoordinatorStopped = errors.New("multi: coordinator stopped")
)

// --- Message Types ---

// Message is the unit of inter-agent communication.
type Message struct {
	// ID uniquely identifies this message.
	ID string
	// From is the sender agent name.
	From string
	// To is the recipient agent name.
	To string
	// Type classifies the message.
	Type MessageType
	// Content is the message payload.
	Content string
	// TaskID links the message to a delegated task.
	TaskID string
	// Timestamp is when the message was created.
	Timestamp time.Time
	// Metadata holds arbitrary key-value pairs.
	Metadata map[string]string
}

// MessageType classifies a message.
type MessageType string

const (
	// MsgTask is a task delegation request.
	MsgTask MessageType = "task"
	// MsgResult is a task result response.
	MsgResult MessageType = "result"
	// MsgError is an error notification.
	MsgError MessageType = "error"
	// MsgInfo is an informational message.
	MsgInfo MessageType = "info"
)

// --- Agent Runner Interface ---

// Runner is the interface that agents must implement to participate
// in multi-agent orchestration. It wraps the core agent's Run method.
type Runner interface {
	// Name returns the agent's unique name.
	Name() string
	// Run executes the agent on the given task and returns the output.
	Run(ctx context.Context, task string) (string, error)
}

// --- Mailbox ---

// Mailbox is a buffered channel for inter-agent messaging.
type Mailbox struct {
	name     string
	messages chan Message
	capacity int
}

// NewMailbox creates a mailbox with the given capacity.
func NewMailbox(name string, capacity int) *Mailbox {
	if capacity <= 0 {
		capacity = 100
	}
	return &Mailbox{
		name:     name,
		messages: make(chan Message, capacity),
		capacity: capacity,
	}
}

// Send delivers a message to the mailbox. Returns ErrMailboxFull if at capacity.
func (mb *Mailbox) Send(msg Message) error {
	select {
	case mb.messages <- msg:
		return nil
	default:
		return fmt.Errorf("%w: agent %q", ErrMailboxFull, mb.name)
	}
}

// Receive returns the next message, blocking until one arrives or ctx is cancelled.
func (mb *Mailbox) Receive(ctx context.Context) (Message, error) {
	select {
	case msg := <-mb.messages:
		return msg, nil
	case <-ctx.Done():
		return Message{}, ctx.Err()
	}
}

// TryReceive returns a message if one is available, without blocking.
func (mb *Mailbox) TryReceive() (Message, bool) {
	select {
	case msg := <-mb.messages:
		return msg, true
	default:
		return Message{}, false
	}
}

// Len returns the number of pending messages.
func (mb *Mailbox) Len() int {
	return len(mb.messages)
}

// --- Registry ---

// Registry manages named agents and their mailboxes.
type Registry struct {
	mu        sync.RWMutex
	agents    map[string]Runner
	mailboxes map[string]*Mailbox
}

// NewRegistry creates an empty agent registry.
func NewRegistry() *Registry {
	return &Registry{
		agents:    make(map[string]Runner),
		mailboxes: make(map[string]*Mailbox),
	}
}

// Register adds an agent to the registry with a default mailbox.
func (r *Registry) Register(agent Runner) error {
	return r.RegisterWithCapacity(agent, 100)
}

// RegisterWithCapacity adds an agent with a specified mailbox capacity.
func (r *Registry) RegisterWithCapacity(agent Runner, mailboxCapacity int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := agent.Name()
	if name == "" {
		return fmt.Errorf("multi: agent name must not be empty")
	}
	if _, exists := r.agents[name]; exists {
		return fmt.Errorf("multi: agent %q already registered", name)
	}
	r.agents[name] = agent
	r.mailboxes[name] = NewMailbox(name, mailboxCapacity)
	return nil
}

// Get retrieves an agent by name.
func (r *Registry) Get(name string) (Runner, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agent, ok := r.agents[name]
	return agent, ok
}

// Mailbox returns the mailbox for an agent.
func (r *Registry) Mailbox(name string) (*Mailbox, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	mb, ok := r.mailboxes[name]
	return mb, ok
}

// Names returns all registered agent names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.agents))
	for name := range r.agents {
		names = append(names, name)
	}
	return names
}

// Len returns the number of registered agents.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.agents)
}

// --- Task ---

// Task represents a unit of work delegated to an agent.
type Task struct {
	// ID uniquely identifies this task.
	ID string
	// AssignedTo is the agent responsible for execution.
	AssignedTo string
	// DelegatedBy is the agent that created this task (empty for top-level).
	DelegatedBy string
	// Description is the task prompt.
	Description string
	// Status tracks the task lifecycle.
	Status TaskStatus
	// Result holds the output once completed.
	Result string
	// Error holds the error message on failure.
	Error string
	// Depth tracks how many delegation levels deep this task is.
	Depth int
	// CreatedAt is when the task was created.
	CreatedAt time.Time
	// CompletedAt is when the task finished.
	CompletedAt time.Time
}

// TaskStatus tracks the lifecycle of a delegated task.
type TaskStatus string

const (
	TaskPending    TaskStatus = "pending"
	TaskRunning    TaskStatus = "running"
	TaskCompleted  TaskStatus = "completed"
	TaskFailed     TaskStatus = "failed"
)

// --- Coordinator ---

// CoordinatorConfig configures the multi-agent coordinator.
type CoordinatorConfig struct {
	// MaxDelegationDepth limits how deep task delegation can go.
	// Zero means DefaultMaxDelegationDepth.
	MaxDelegationDepth int
	// MaxConcurrent limits simultaneous agent executions.
	// Zero means no limit.
	MaxConcurrent int
	// Logger is the structured logger. If nil, slog.Default() is used.
	Logger *slog.Logger
}

// DefaultMaxDelegationDepth is the default maximum delegation depth.
const DefaultMaxDelegationDepth = 5

// Coordinator orchestrates multiple agents, managing task delegation,
// message routing, and concurrency control.
type Coordinator struct {
	registry   *Registry
	maxDepth   int
	logger     *slog.Logger
	semaphore  chan struct{}

	mu      sync.Mutex
	tasks   map[string]*Task
	stopped bool
	counter int
}

// NewCoordinator creates a coordinator with the given registry and config.
func NewCoordinator(registry *Registry, cfg CoordinatorConfig) *Coordinator {
	maxDepth := cfg.MaxDelegationDepth
	if maxDepth <= 0 {
		maxDepth = DefaultMaxDelegationDepth
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	var sem chan struct{}
	if cfg.MaxConcurrent > 0 {
		sem = make(chan struct{}, cfg.MaxConcurrent)
	}

	return &Coordinator{
		registry:  registry,
		maxDepth:  maxDepth,
		logger:    logger,
		semaphore: sem,
		tasks:     make(map[string]*Task),
	}
}

// Delegate creates a task and assigns it to the named agent.
// It returns the task result synchronously.
func (c *Coordinator) Delegate(ctx context.Context, agentName, taskDescription string) (*Task, error) {
	return c.delegateInternal(ctx, agentName, taskDescription, "", 0)
}

// DelegateFrom creates a task delegated by another agent, tracking depth.
func (c *Coordinator) DelegateFrom(ctx context.Context, agentName, taskDescription, delegatedBy string, depth int) (*Task, error) {
	return c.delegateInternal(ctx, agentName, taskDescription, delegatedBy, depth)
}

func (c *Coordinator) delegateInternal(ctx context.Context, agentName, taskDescription, delegatedBy string, depth int) (*Task, error) {
	c.mu.Lock()
	if c.stopped {
		c.mu.Unlock()
		return nil, ErrCoordinatorStopped
	}

	if depth >= c.maxDepth {
		c.mu.Unlock()
		return nil, fmt.Errorf("%w: depth %d >= max %d", ErrDelegationDepth, depth, c.maxDepth)
	}

	c.counter++
	taskID := fmt.Sprintf("task-%d", c.counter)
	task := &Task{
		ID:          taskID,
		AssignedTo:  agentName,
		DelegatedBy: delegatedBy,
		Description: taskDescription,
		Status:      TaskPending,
		Depth:       depth,
		CreatedAt:   time.Now().UTC(),
	}
	c.tasks[taskID] = task
	c.mu.Unlock()

	agent, ok := c.registry.Get(agentName)
	if !ok {
		task.Status = TaskFailed
		task.Error = ErrAgentNotFound.Error()
		return task, fmt.Errorf("%w: %q", ErrAgentNotFound, agentName)
	}

	c.logger.Info("delegating task",
		"task_id", taskID,
		"agent", agentName,
		"depth", depth,
		"delegated_by", delegatedBy,
	)

	// Acquire semaphore if concurrency is limited.
	if c.semaphore != nil {
		select {
		case c.semaphore <- struct{}{}:
			defer func() { <-c.semaphore }()
		case <-ctx.Done():
			task.Status = TaskFailed
			task.Error = ctx.Err().Error()
			return task, ctx.Err()
		}
	}

	task.Status = TaskRunning
	output, err := agent.Run(ctx, taskDescription)
	task.CompletedAt = time.Now().UTC()

	if err != nil {
		task.Status = TaskFailed
		task.Error = err.Error()
		c.logger.Error("task failed", "task_id", taskID, "agent", agentName, "error", err)

		// Send error message to the delegating agent if applicable.
		if delegatedBy != "" {
			c.sendMessage(agentName, delegatedBy, MsgError, err.Error(), taskID)
		}
		return task, err
	}

	task.Status = TaskCompleted
	task.Result = output
	c.logger.Info("task completed", "task_id", taskID, "agent", agentName,
		"duration", task.CompletedAt.Sub(task.CreatedAt))

	// Send result message to the delegating agent if applicable.
	if delegatedBy != "" {
		c.sendMessage(agentName, delegatedBy, MsgResult, output, taskID)
	}

	return task, nil
}

// SendMessage sends a message between agents via their mailboxes.
func (c *Coordinator) SendMessage(from, to string, msgType MessageType, content string) error {
	return c.sendMessage(from, to, msgType, content, "")
}

func (c *Coordinator) sendMessage(from, to string, msgType MessageType, content, taskID string) error {
	mb, ok := c.registry.Mailbox(to)
	if !ok {
		return fmt.Errorf("%w: %q", ErrAgentNotFound, to)
	}

	c.mu.Lock()
	c.counter++
	msgID := fmt.Sprintf("msg-%d", c.counter)
	c.mu.Unlock()

	return mb.Send(Message{
		ID:        msgID,
		From:      from,
		To:        to,
		Type:      msgType,
		Content:   content,
		TaskID:    taskID,
		Timestamp: time.Now().UTC(),
	})
}

// Tasks returns all tracked tasks.
func (c *Coordinator) Tasks() []*Task {
	c.mu.Lock()
	defer c.mu.Unlock()

	tasks := make([]*Task, 0, len(c.tasks))
	for _, t := range c.tasks {
		tasks = append(tasks, t)
	}
	return tasks
}

// GetTask returns a task by ID.
func (c *Coordinator) GetTask(id string) (*Task, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t, ok := c.tasks[id]
	return t, ok
}

// Stop prevents new delegations.
func (c *Coordinator) Stop() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stopped = true
}

// --- Fan-out helper ---

// FanOut delegates the same task to multiple agents concurrently and
// collects their results. Useful for parallel execution patterns.
func (c *Coordinator) FanOut(ctx context.Context, agentNames []string, task string) ([]*Task, error) {
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		results []*Task
		firstErr error
	)

	for _, name := range agentNames {
		wg.Add(1)
		go func(agentName string) {
			defer wg.Done()
			t, err := c.Delegate(ctx, agentName, task)

			mu.Lock()
			defer mu.Unlock()
			if t != nil {
				results = append(results, t)
			}
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}(name)
	}

	wg.Wait()
	return results, firstErr
}
