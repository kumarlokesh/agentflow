# syntax=docker/dockerfile:1

# ---- Build Stage ----
FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /bin/agentflow \
    ./cmd/agentflow

# ---- Runtime Stage ----
FROM alpine:3.21

RUN apk --no-cache add ca-certificates

COPY --from=builder /bin/agentflow /usr/local/bin/agentflow

ENTRYPOINT ["agentflow"]
