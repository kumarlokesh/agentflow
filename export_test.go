package agentflow

import "time"

// Export functions for testing. These are only available in test binaries
// (the _test.go suffix ensures they're excluded from production builds).

// ExportNewUUID returns the current newUUID function.
func ExportNewUUID() func() string { return newUUID }

// SetNewUUID replaces the UUID generator (for deterministic tests).
func SetNewUUID(fn func() string) { newUUID = fn }

// ExportNowUTC returns the current nowUTC function.
func ExportNowUTC() func() time.Time { return nowUTC }

// SetNowUTC replaces the time source (for deterministic tests).
func SetNowUTC(fn func() time.Time) { nowUTC = fn }
