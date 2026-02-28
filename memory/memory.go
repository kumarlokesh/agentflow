// Package memory provides working and episodic memory for agentflow agents.
//
// Memory enables agents to retrieve relevant context from past interactions,
// maintain state across steps, and manage information within budget constraints.
//
// The design is pluggable: the Memory interface abstracts storage, while
// retrieval strategies and eviction policies are composable.
package memory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// --- Core Interface ---

// Entry represents a single memory item.
type Entry struct {
	// ID uniquely identifies this entry.
	ID string
	// Content is the text content of the memory.
	Content string
	// Embedding is the vector representation (may be nil if not computed).
	Embedding []float64
	// Metadata holds arbitrary key-value pairs.
	Metadata map[string]string
	// CreatedAt is when the entry was stored.
	CreatedAt time.Time
	// AccessedAt is when the entry was last retrieved.
	AccessedAt time.Time
	// AccessCount tracks how often this entry has been retrieved.
	AccessCount int
	// Score is the relevance score from the last retrieval (transient).
	Score float64
}

// Store is the interface for memory persistence. Implementations must be
// safe for concurrent use.
type Store interface {
	// Add stores a new memory entry.
	Add(ctx context.Context, entry Entry) error
	// Get retrieves an entry by ID.
	Get(ctx context.Context, id string) (*Entry, error)
	// Search finds entries matching the query. The query can be text or
	// a vector depending on the implementation.
	Search(ctx context.Context, query Query) ([]Entry, error)
	// Delete removes an entry by ID.
	Delete(ctx context.Context, id string) error
	// List returns all entries.
	List(ctx context.Context) ([]Entry, error)
	// Count returns the total number of entries.
	Count(ctx context.Context) (int, error)
}

// Query describes a memory retrieval request.
type Query struct {
	// Text is the natural language query (used for keyword/semantic search).
	Text string
	// Embedding is a vector query (used for vector similarity search).
	Embedding []float64
	// TopK limits the number of results.
	TopK int
	// MinScore filters out results below this threshold (0-1).
	MinScore float64
	// Metadata filters entries by metadata key-value pairs.
	Metadata map[string]string
}

// SummarizeFunc is a callback for summarizing memory content.
// It receives a list of entries and returns a condensed summary.
type SummarizeFunc func(ctx context.Context, entries []Entry) (string, error)

// --- In-Memory Vector Store ---

// InMemory is a thread-safe, in-memory Store with vector similarity search.
// Suitable for development, testing, and short-lived agents.
type InMemory struct {
	mu      sync.RWMutex
	entries map[string]Entry
}

// NewInMemory creates an empty in-memory store.
func NewInMemory() *InMemory {
	return &InMemory{
		entries: make(map[string]Entry),
	}
}

// Add stores a new entry.
func (m *InMemory) Add(_ context.Context, entry Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if entry.ID == "" {
		return fmt.Errorf("memory: entry ID must not be empty")
	}
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now().UTC()
	}
	entry.AccessedAt = entry.CreatedAt
	m.entries[entry.ID] = entry
	return nil
}

// Get retrieves an entry by ID, updating its access metadata.
func (m *InMemory) Get(_ context.Context, id string) (*Entry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry, ok := m.entries[id]
	if !ok {
		return nil, nil
	}
	entry.AccessedAt = time.Now().UTC()
	entry.AccessCount++
	m.entries[id] = entry
	return &entry, nil
}

// Search finds entries matching the query using vector similarity or keyword matching.
func (m *InMemory) Search(_ context.Context, q Query) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	topK := q.TopK
	if topK <= 0 {
		topK = 10
	}

	var results []Entry

	for _, entry := range m.entries {
		// Apply metadata filter.
		if !matchesMetadata(entry, q.Metadata) {
			continue
		}

		var score float64
		if len(q.Embedding) > 0 && len(entry.Embedding) > 0 {
			// Vector similarity search.
			score = cosineSimilarity(q.Embedding, entry.Embedding)
		} else if q.Text != "" {
			// Keyword search fallback.
			score = keywordScore(q.Text, entry.Content)
		} else {
			score = 1.0 // No query filter — return all.
		}

		if score >= q.MinScore {
			e := entry
			e.Score = score
			results = append(results, e)
		}
	}

	// Sort by score descending.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

// Delete removes an entry by ID.
func (m *InMemory) Delete(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.entries, id)
	return nil
}

// List returns all entries sorted by creation time.
func (m *InMemory) List(_ context.Context) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entries := make([]Entry, 0, len(m.entries))
	for _, e := range m.entries {
		entries = append(entries, e)
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].CreatedAt.Before(entries[j].CreatedAt)
	})
	return entries, nil
}

// Count returns the number of entries.
func (m *InMemory) Count(_ context.Context) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries), nil
}

// --- Budget & Eviction ---

// BudgetConfig controls memory size limits and eviction behavior.
type BudgetConfig struct {
	// MaxEntries is the maximum number of entries. Zero means unlimited.
	MaxEntries int
	// EvictionStrategy determines which entries to remove when over budget.
	EvictionStrategy EvictionStrategy
}

// EvictionStrategy determines which entries to remove.
type EvictionStrategy int

const (
	// EvictOldest removes the oldest entries first (FIFO).
	EvictOldest EvictionStrategy = iota
	// EvictLRU removes the least recently accessed entries.
	EvictLRU
	// EvictLFU removes the least frequently accessed entries.
	EvictLFU
)

// BudgetEnforcer wraps a Store and enforces entry count limits.
type BudgetEnforcer struct {
	store    Store
	config   BudgetConfig
}

// NewBudgetEnforcer creates a budget-enforcing wrapper around a store.
func NewBudgetEnforcer(store Store, config BudgetConfig) *BudgetEnforcer {
	return &BudgetEnforcer{store: store, config: config}
}

// Add stores an entry, evicting old entries if over budget.
func (b *BudgetEnforcer) Add(ctx context.Context, entry Entry) error {
	if err := b.store.Add(ctx, entry); err != nil {
		return err
	}
	return b.enforce(ctx)
}

// Get delegates to the underlying store.
func (b *BudgetEnforcer) Get(ctx context.Context, id string) (*Entry, error) {
	return b.store.Get(ctx, id)
}

// Search delegates to the underlying store.
func (b *BudgetEnforcer) Search(ctx context.Context, q Query) ([]Entry, error) {
	return b.store.Search(ctx, q)
}

// Delete delegates to the underlying store.
func (b *BudgetEnforcer) Delete(ctx context.Context, id string) error {
	return b.store.Delete(ctx, id)
}

// List delegates to the underlying store.
func (b *BudgetEnforcer) List(ctx context.Context) ([]Entry, error) {
	return b.store.List(ctx)
}

// Count delegates to the underlying store.
func (b *BudgetEnforcer) Count(ctx context.Context) (int, error) {
	return b.store.Count(ctx)
}

func (b *BudgetEnforcer) enforce(ctx context.Context) error {
	if b.config.MaxEntries <= 0 {
		return nil
	}

	count, err := b.store.Count(ctx)
	if err != nil {
		return err
	}

	if count <= b.config.MaxEntries {
		return nil
	}

	entries, err := b.store.List(ctx)
	if err != nil {
		return err
	}

	// Sort by eviction priority.
	switch b.config.EvictionStrategy {
	case EvictLRU:
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].AccessedAt.Before(entries[j].AccessedAt)
		})
	case EvictLFU:
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].AccessCount < entries[j].AccessCount
		})
	default: // EvictOldest
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].CreatedAt.Before(entries[j].CreatedAt)
		})
	}

	// Remove excess entries.
	toRemove := count - b.config.MaxEntries
	for i := 0; i < toRemove && i < len(entries); i++ {
		if err := b.store.Delete(ctx, entries[i].ID); err != nil {
			return err
		}
	}
	return nil
}

// --- Helper Functions ---

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// keywordScore computes a simple keyword match score (0-1).
func keywordScore(query, content string) float64 {
	queryLower := strings.ToLower(query)
	contentLower := strings.ToLower(content)

	words := strings.Fields(queryLower)
	if len(words) == 0 {
		return 0
	}

	matches := 0
	for _, w := range words {
		if strings.Contains(contentLower, w) {
			matches++
		}
	}
	return float64(matches) / float64(len(words))
}

func matchesMetadata(entry Entry, filter map[string]string) bool {
	for k, v := range filter {
		if entry.Metadata == nil {
			return false
		}
		if entry.Metadata[k] != v {
			return false
		}
	}
	return true
}
