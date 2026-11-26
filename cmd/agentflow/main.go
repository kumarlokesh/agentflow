// Command agentflow inspects recorded traces.
//
// Usage:
//
//	agentflow [-store dir] <command> [arguments]
//
// Commands:
//
//	runs     List recorded traces
//	show     Show a trace's run tree, or its events
//	diff     Compare two traces run by run
//	blame    Find the root causes of a failure
//	export   Write a trace as a portable bundle, or as one HTML page
//	version  Print version information
//
// Replay and fork need your agents' code, so they aren't commands here: call
// cli.Main with your agents in your own program to get them (package cli),
// or use replay.Run and replay.Fork directly.
package main

import "github.com/kumarlokesh/agentflow/cli"

// Build-time variables set via ldflags.
var (
	version = "dev"
	commit  = "none"
)

func main() {
	cli.Main(cli.Config{Name: "agentflow", Version: version + " (" + commit + ")"})
}
