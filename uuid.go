package agentflow

import (
	"time"

	"github.com/google/uuid"
)

// newUUID generates a new UUID v4 string.
var newUUID = func() string {
	return uuid.New().String()
}

// nowUTC returns the current time in UTC.
var nowUTC = func() time.Time {
	return time.Now().UTC()
}
