package agentflow

import (
	"time"

	"github.com/google/uuid"
)

// newUUID generates a new UUID v4 string. Extracted to a package-level var so
// tests can replace it with a deterministic generator.
var newUUID = func() string {
	return uuid.New().String()
}

// nowUTC returns the current time in UTC. Extracted to a package-level var so
// tests can replace it with a fixed clock.
var nowUTC = func() time.Time {
	return time.Now().UTC()
}
