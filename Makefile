.PHONY: build test cover lint fmt vet clean docker-build docker-run all check

BINARY_NAME := agentflow
BUILD_DIR   := bin
COVER_FILE  := coverage.out
GO          := go
GOFLAGS     := -race -count=1
LDFLAGS     := -s -w

# Build the CLI binary
build:
	@mkdir -p $(BUILD_DIR)
	$(GO) build -ldflags="$(LDFLAGS)" -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/agentflow

# Run all tests with race detector
test:
	$(GO) test $(GOFLAGS) ./...

# Run tests with coverage
cover:
	$(GO) test -race -coverprofile=$(COVER_FILE) -covermode=atomic ./...
	$(GO) tool cover -func=$(COVER_FILE)
	@echo "---"
	@echo "HTML report: coverage.html"
	$(GO) tool cover -html=$(COVER_FILE) -o coverage.html

# Run linter
lint:
	golangci-lint run ./...

# Format code
fmt:
	gofmt -s -w .
	$(GO) mod tidy

# Run go vet
vet:
	$(GO) vet ./...

# Remove build artifacts
clean:
	rm -rf $(BUILD_DIR) $(COVER_FILE) coverage.html

# Build Docker image
docker-build:
	docker build -t $(BINARY_NAME):latest .

# Run in Docker
docker-run:
	docker run --rm -it $(BINARY_NAME):latest

# Full check: format, vet, lint, test
check: fmt vet lint test

# Full pipeline: check + build
all: check build
