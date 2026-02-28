// Command agentflow is the CLI for the agentflow runtime. It provides
// subcommands for running agents, replaying recorded runs, listing runs,
// and diffing two runs.
//
// Usage:
//
//	agentflow <command> [flags]
//
// Commands:
//
//	runs     List all recorded runs
//	replay   Replay a recorded run deterministically
//	diff     Compare two recorded runs
//	version  Print version information
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"text/tabwriter"

	"github.com/kumarlokesh/agentflow/replay"
	"github.com/kumarlokesh/agentflow/store"
)

// Build-time variables set via ldflags.
var (
	version = "dev"
	commit  = "none"
)

const defaultStoreDir = ".agentflow/runs"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	// Set up structured logging.
	logLevel := slog.LevelInfo
	if os.Getenv("AGENTFLOW_LOG_LEVEL") == "debug" {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	}))

	storeDir := os.Getenv("AGENTFLOW_STORE_DIR")
	if storeDir == "" {
		storeDir = defaultStoreDir
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	var err error
	switch os.Args[1] {
	case "runs":
		err = cmdRuns(ctx, storeDir)
	case "replay":
		err = cmdReplay(ctx, storeDir, logger, os.Args[2:])
	case "diff":
		err = cmdDiff(ctx, storeDir, os.Args[2:])
	case "version":
		fmt.Printf("agentflow %s (%s)\n", version, commit)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `agentflow — deterministic agent runtime

Usage:
  agentflow <command> [flags]

Commands:
  runs      List all recorded runs
  replay    Replay a recorded run
  diff      Compare two recorded runs
  version   Print version information
  help      Show this help

Environment:
  AGENTFLOW_STORE_DIR    Event store directory (default: %s)
  AGENTFLOW_LOG_LEVEL    Log level: debug, info, warn, error (default: info)
`, defaultStoreDir)
}

// cmdRuns lists all recorded run IDs from the file store.
func cmdRuns(ctx context.Context, storeDir string) error {
	fs, err := store.NewFile(storeDir)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}

	runs, err := fs.ListRuns(ctx)
	if err != nil {
		return fmt.Errorf("list runs: %w", err)
	}

	if len(runs) == 0 {
		fmt.Println("No recorded runs found.")
		return nil
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "RUN ID")
	fmt.Fprintln(tw, "------")
	for _, r := range runs {
		fmt.Fprintln(tw, r)
	}
	return tw.Flush()
}

// cmdReplay replays a recorded run.
func cmdReplay(ctx context.Context, storeDir string, logger *slog.Logger, args []string) error {
	fs := flag.NewFlagSet("replay", flag.ExitOnError)
	runID := fs.String("run", "", "Run ID to replay (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID == "" {
		fs.Usage()
		return fmt.Errorf("--run is required")
	}

	fileStore, err := store.NewFile(storeDir)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}

	engine := replay.NewEngine(fileStore, logger)
	result, err := engine.Replay(ctx, *runID)
	if err != nil {
		return err
	}

	fmt.Printf("Replay of run %s\n", result.RunID)
	fmt.Printf("  Replay Run ID: %s\n", result.ReplayRunID)
	fmt.Printf("  Match:         %v\n", result.Match)
	fmt.Printf("  Steps:         %d\n", result.Steps)
	fmt.Printf("  Duration:      %s\n", result.Duration)
	fmt.Printf("  Output:        %s\n", result.Output)

	if !result.Match {
		fmt.Println("\n⚠ WARNING: Replay output differs from original run!")
	}
	return nil
}

// cmdDiff compares two runs.
func cmdDiff(ctx context.Context, storeDir string, args []string) error {
	fs := flag.NewFlagSet("diff", flag.ExitOnError)
	runA := fs.String("a", "", "First run ID (required)")
	runB := fs.String("b", "", "Second run ID (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runA == "" || *runB == "" {
		fs.Usage()
		return fmt.Errorf("--a and --b are both required")
	}

	fileStore, err := store.NewFile(storeDir)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}

	result, err := replay.Diff(ctx, fileStore, *runA, *runB)
	if err != nil {
		return err
	}

	fmt.Print(result.Summary)
	return nil
}
