package memory

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

// --- InMemory Tests ---

func TestInMemory_AddAndGet(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	entry := Entry{
		ID:      "e1",
		Content: "The capital of France is Paris.",
		Metadata: map[string]string{
			"source": "wiki",
		},
	}

	if err := m.Add(ctx, entry); err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	got, err := m.Get(ctx, "e1")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got == nil {
		t.Fatal("Get() returned nil")
	}
	if got.Content != entry.Content {
		t.Errorf("Content = %q, want %q", got.Content, entry.Content)
	}
	if got.AccessCount != 1 {
		t.Errorf("AccessCount = %d, want 1", got.AccessCount)
	}
}

func TestInMemory_Add_EmptyID(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	err := m.Add(ctx, Entry{ID: "", Content: "test"})
	if err == nil {
		t.Fatal("expected error for empty ID")
	}
}

func TestInMemory_Get_NotFound(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	got, err := m.Get(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got != nil {
		t.Error("expected nil for nonexistent entry")
	}
}

func TestInMemory_Delete(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	m.Add(ctx, Entry{ID: "e1", Content: "test"})
	m.Delete(ctx, "e1")

	got, _ := m.Get(ctx, "e1")
	if got != nil {
		t.Error("entry should be deleted")
	}

	count, _ := m.Count(ctx)
	if count != 0 {
		t.Errorf("Count = %d, want 0", count)
	}
}

func TestInMemory_List(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()
	now := time.Now().UTC()

	m.Add(ctx, Entry{ID: "e2", Content: "second", CreatedAt: now.Add(time.Second)})
	m.Add(ctx, Entry{ID: "e1", Content: "first", CreatedAt: now})

	entries, err := m.List(ctx)
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("List() len = %d, want 2", len(entries))
	}
	// Should be sorted by creation time.
	if entries[0].ID != "e1" {
		t.Errorf("first entry = %q, want e1", entries[0].ID)
	}
}

func TestInMemory_Count(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	m.Add(ctx, Entry{ID: "e1", Content: "a"})
	m.Add(ctx, Entry{ID: "e2", Content: "b"})

	count, _ := m.Count(ctx)
	if count != 2 {
		t.Errorf("Count = %d, want 2", count)
	}
}

// --- Search Tests ---

func TestInMemory_Search_KeywordMatch(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	m.Add(ctx, Entry{ID: "e1", Content: "Go is a compiled programming language"})
	m.Add(ctx, Entry{ID: "e2", Content: "Python is an interpreted language"})
	m.Add(ctx, Entry{ID: "e3", Content: "The weather is nice today"})

	results, err := m.Search(ctx, Query{Text: "programming language", TopK: 5})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}

	// "Go is a compiled programming language" should rank highest (2/2 words match).
	if results[0].ID != "e1" {
		t.Errorf("top result = %q, want e1", results[0].ID)
	}
}

func TestInMemory_Search_VectorSimilarity(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	m.Add(ctx, Entry{ID: "e1", Content: "cats", Embedding: []float64{1, 0, 0}})
	m.Add(ctx, Entry{ID: "e2", Content: "dogs", Embedding: []float64{0.9, 0.1, 0}})
	m.Add(ctx, Entry{ID: "e3", Content: "cars", Embedding: []float64{0, 0, 1}})

	results, err := m.Search(ctx, Query{
		Embedding: []float64{1, 0, 0}, // Most similar to "cats"
		TopK:      2,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len = %d, want 2", len(results))
	}
	if results[0].ID != "e1" {
		t.Errorf("top result = %q, want e1 (cats)", results[0].ID)
	}
	if results[0].Score < 0.99 {
		t.Errorf("top score = %f, want ~1.0", results[0].Score)
	}
}

func TestInMemory_Search_MinScore(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	m.Add(ctx, Entry{ID: "e1", Content: "cats", Embedding: []float64{1, 0, 0}})
	m.Add(ctx, Entry{ID: "e2", Content: "cars", Embedding: []float64{0, 0, 1}})

	results, _ := m.Search(ctx, Query{
		Embedding: []float64{1, 0, 0},
		TopK:      10,
		MinScore:  0.5,
	})

	// Only "cats" should pass the 0.5 threshold.
	if len(results) != 1 {
		t.Fatalf("len = %d, want 1", len(results))
	}
	if results[0].ID != "e1" {
		t.Errorf("result = %q, want e1", results[0].ID)
	}
}

func TestInMemory_Search_MetadataFilter(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()

	m.Add(ctx, Entry{ID: "e1", Content: "a", Metadata: map[string]string{"type": "fact"}})
	m.Add(ctx, Entry{ID: "e2", Content: "b", Metadata: map[string]string{"type": "opinion"}})

	results, _ := m.Search(ctx, Query{
		TopK:     10,
		Metadata: map[string]string{"type": "fact"},
	})
	if len(results) != 1 {
		t.Fatalf("len = %d, want 1", len(results))
	}
	if results[0].ID != "e1" {
		t.Errorf("result = %q, want e1", results[0].ID)
	}
}

func TestInMemory_Search_NoQuery(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()
	m.Add(ctx, Entry{ID: "e1", Content: "a"})
	m.Add(ctx, Entry{ID: "e2", Content: "b"})

	results, _ := m.Search(ctx, Query{TopK: 10})
	if len(results) != 2 {
		t.Errorf("len = %d, want 2 (return all when no query)", len(results))
	}
}

func TestInMemory_Search_DefaultTopK(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()
	for i := 0; i < 20; i++ {
		m.Add(ctx, Entry{ID: fmt.Sprintf("e%d", i), Content: "content"})
	}

	results, _ := m.Search(ctx, Query{})
	if len(results) != 10 { // default topK
		t.Errorf("len = %d, want 10 (default topK)", len(results))
	}
}

func TestInMemory_Concurrent(t *testing.T) {
	ctx := context.Background()
	m := NewInMemory()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("e%d", i)
			m.Add(ctx, Entry{ID: id, Content: "content"})
			m.Get(ctx, id)
			m.Search(ctx, Query{Text: "content", TopK: 5})
			m.List(ctx)
			m.Count(ctx)
		}(i)
	}
	wg.Wait()

	count, _ := m.Count(ctx)
	if count != 50 {
		t.Errorf("Count = %d, want 50", count)
	}
}

// --- Cosine Similarity Tests ---

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"identical", []float64{1, 0, 0}, []float64{1, 0, 0}, 1.0},
		{"orthogonal", []float64{1, 0, 0}, []float64{0, 1, 0}, 0.0},
		{"opposite", []float64{1, 0}, []float64{-1, 0}, -1.0},
		{"similar", []float64{1, 1}, []float64{1, 0}, 1.0 / math.Sqrt(2)},
		{"empty", nil, nil, 0.0},
		{"different lengths", []float64{1}, []float64{1, 2}, 0.0},
		{"zero vectors", []float64{0, 0}, []float64{0, 0}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("cosineSimilarity() = %f, want %f", got, tt.want)
			}
		})
	}
}

// --- Keyword Score Tests ---

func TestKeywordScore(t *testing.T) {
	tests := []struct {
		query, content string
		want           float64
	}{
		{"hello world", "Hello World!", 1.0},
		{"hello world", "Hello there", 0.5},
		{"hello world", "nothing here", 0.0},
		{"", "anything", 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			got := keywordScore(tt.query, tt.content)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("keywordScore(%q, %q) = %f, want %f", tt.query, tt.content, got, tt.want)
			}
		})
	}
}

// --- BudgetEnforcer Tests ---

func TestBudgetEnforcer_EvictOldest(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	b := NewBudgetEnforcer(inner, BudgetConfig{
		MaxEntries:       3,
		EvictionStrategy: EvictOldest,
	})

	now := time.Now().UTC()
	b.Add(ctx, Entry{ID: "e1", Content: "oldest", CreatedAt: now})
	b.Add(ctx, Entry{ID: "e2", Content: "middle", CreatedAt: now.Add(time.Second)})
	b.Add(ctx, Entry{ID: "e3", Content: "newest", CreatedAt: now.Add(2 * time.Second)})

	// Adding a 4th should evict the oldest.
	b.Add(ctx, Entry{ID: "e4", Content: "latest", CreatedAt: now.Add(3 * time.Second)})

	count, _ := b.Count(ctx)
	if count != 3 {
		t.Errorf("Count = %d, want 3", count)
	}

	// e1 should be evicted.
	got, _ := b.Get(ctx, "e1")
	if got != nil {
		t.Error("e1 should have been evicted")
	}

	// e4 should exist.
	got, _ = b.Get(ctx, "e4")
	if got == nil {
		t.Error("e4 should exist")
	}
}

func TestBudgetEnforcer_EvictLRU(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	b := NewBudgetEnforcer(inner, BudgetConfig{
		MaxEntries:       2,
		EvictionStrategy: EvictLRU,
	})

	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	// e1: accessed long ago, e2: accessed slightly later.
	b.Add(ctx, Entry{ID: "e1", Content: "a", CreatedAt: now})
	b.Add(ctx, Entry{ID: "e2", Content: "b", CreatedAt: now.Add(time.Second)})

	// Access e1 via the inner store directly to control timing.
	// This sets e1.AccessedAt to time.Now(), much later than e2.AccessedAt (now+1s).
	time.Sleep(2 * time.Millisecond)
	b.Get(ctx, "e1")

	// Add e3 — should evict e2 (oldest AccessedAt = now+1s).
	b.Add(ctx, Entry{ID: "e3", Content: "c", CreatedAt: now.Add(2 * time.Second)})

	// e2 should be evicted (AccessedAt = now+1s, oldest).
	count, _ := b.Count(ctx)
	if count != 2 {
		t.Fatalf("Count = %d, want 2", count)
	}

	entries, _ := b.List(ctx)
	ids := make(map[string]bool)
	for _, e := range entries {
		ids[e.ID] = true
	}

	if ids["e2"] {
		t.Error("e2 should have been evicted (LRU)")
	}
	if !ids["e1"] {
		t.Error("e1 should still exist (was accessed recently)")
	}
	if !ids["e3"] {
		t.Error("e3 should exist (just added)")
	}
}

func TestBudgetEnforcer_EvictLFU(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	b := NewBudgetEnforcer(inner, BudgetConfig{
		MaxEntries:       2,
		EvictionStrategy: EvictLFU,
	})

	now := time.Now().UTC()
	b.Add(ctx, Entry{ID: "e1", Content: "popular", CreatedAt: now})
	b.Add(ctx, Entry{ID: "e2", Content: "unpopular", CreatedAt: now.Add(time.Second)})

	// Access e1 multiple times to increase frequency.
	b.Get(ctx, "e1")
	b.Get(ctx, "e1")
	b.Get(ctx, "e1")

	// Add e3 — should evict e2 (least frequently accessed).
	b.Add(ctx, Entry{ID: "e3", Content: "c", CreatedAt: now.Add(2 * time.Second)})

	got, _ := b.Get(ctx, "e2")
	if got != nil {
		t.Error("e2 should have been evicted (LFU)")
	}
}

func TestBudgetEnforcer_UnlimitedBudget(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	b := NewBudgetEnforcer(inner, BudgetConfig{MaxEntries: 0}) // unlimited

	for i := 0; i < 100; i++ {
		b.Add(ctx, Entry{ID: fmt.Sprintf("e%d", i), Content: "c"})
	}

	count, _ := b.Count(ctx)
	if count != 100 {
		t.Errorf("Count = %d, want 100", count)
	}
}

func TestBudgetEnforcer_DelegatesMethods(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	b := NewBudgetEnforcer(inner, BudgetConfig{MaxEntries: 10})

	b.Add(ctx, Entry{ID: "e1", Content: "test"})

	// Search
	results, _ := b.Search(ctx, Query{Text: "test", TopK: 5})
	if len(results) != 1 {
		t.Errorf("Search len = %d, want 1", len(results))
	}

	// List
	list, _ := b.List(ctx)
	if len(list) != 1 {
		t.Errorf("List len = %d, want 1", len(list))
	}

	// Delete
	b.Delete(ctx, "e1")
	count, _ := b.Count(ctx)
	if count != 0 {
		t.Errorf("Count after delete = %d, want 0", count)
	}
}

// --- StoreProvider Tests ---

func TestStoreProvider_Recall(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	inner.Add(ctx, Entry{ID: "e1", Content: "Go is a great language"})
	inner.Add(ctx, Entry{ID: "e2", Content: "Python for data science"})
	inner.Add(ctx, Entry{ID: "e3", Content: "Rust for systems programming"})

	p := AsProvider(inner)
	results, err := p.Recall(ctx, "programming language", 2)
	if err != nil {
		t.Fatalf("Recall() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Recall() returned no results")
	}
	if len(results) > 2 {
		t.Errorf("Recall() returned %d results, want ≤2", len(results))
	}
	for _, r := range results {
		if r == "" {
			t.Error("Recall() returned empty string entry")
		}
	}
}

func TestStoreProvider_Recall_Empty(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	p := AsProvider(inner)

	results, err := p.Recall(ctx, "anything", 5)
	if err != nil {
		t.Fatalf("Recall() error = %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Recall() on empty store = %d results, want 0", len(results))
	}
}

func TestStoreProvider_Recall_SkipsEmptyContent(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemory()
	inner.Add(ctx, Entry{ID: "e1", Content: ""})
	inner.Add(ctx, Entry{ID: "e2", Content: "non-empty"})

	p := AsProvider(inner)
	results, err := p.Recall(ctx, "", 10)
	if err != nil {
		t.Fatalf("Recall() error = %v", err)
	}
	for _, r := range results {
		if r == "" {
			t.Error("Recall() returned an empty-content entry")
		}
	}
}
