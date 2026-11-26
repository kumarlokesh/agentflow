# agentflow trace format, version 1

This document specifies how agentflow records agent runs. It is written for anyone who reads or writes traces without the Go library: viewers, converters, analysis scripts, or a runtime in another language.

- **Status:** stable as of v0.1.0. Changes follow [§13 Versioning](#13-versioning).
- **Schema:** [`trace-v1.schema.json`](trace-v1.schema.json) (JSON Schema 2020-12) validates the shape of single lines. Rules that involve more than one line (ordering, causes, the conversation fold, fingerprints) are stated here only.
- **Reference implementation:** the Go package `github.com/kumarlokesh/agentflow/trace`.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## Contents

1. [Concepts](#1-concepts)
2. [Encoding](#2-encoding)
3. [Identifiers and paths](#3-identifiers-and-paths)
4. [The event envelope](#4-the-event-envelope)
5. [Ordering and causes](#5-ordering-and-causes)
6. [Event kinds](#6-event-kinds)
7. [Shared objects](#7-shared-objects)
8. [Messages and content blocks](#8-messages-and-content-blocks)
9. [Conversation and request reconstruction](#9-conversation-and-request-reconstruction)
10. [Fingerprints](#10-fingerprints)
11. [Blobs, redaction and reserved keys](#11-blobs-redaction-and-reserved-keys)
12. [Storage: run files and bundles](#12-storage-run-files-and-bundles)
13. [Versioning](#13-versioning)
14. [Security considerations](#14-security-considerations)
15. [Example](#15-example)

## 1. Concepts

| Term | Meaning |
|---|---|
| **Trace** | Everything recorded for one top-level invocation. A replay or fork of a trace is a new trace whose `origin` names the source. |
| **Run** | One execution of one task, by an agent (a model-and-tools loop) or a workflow (code that orchestrates agents). The runs of a trace form a tree. |
| **Event** | One immutable record in a run, addressed by `(run_id, seq)`. |
| **Effect** | An interaction with the world: a model call, a tool call, a custom effect (any other nondeterministic operation, such as reading the clock or a memory lookup), or starting a child run. Recorded as a `*.started` event followed by a `*.completed` or `*.failed` event (for a child run, `run.spawned` followed by `run.joined`). |
| **Conversation** | The messages a run has accumulated. Defined exactly by its conversation events ([§9](#9-conversation-and-request-reconstruction)). |
| **Blob** | A large string stored once, outside the event, addressed by its SHA-256 hash. |

## 2. Encoding

- An event is one JSON object (RFC 8259), encoded as UTF-8 on a single line terminated by `\n`.
- Writers SHOULD NOT escape `<`, `>` and `&` (Go's default HTML escaping), so that files stay readable. Readers MUST accept either form.
- An encoded event, after blob externalization ([§11](#11-blobs-redaction-and-reserved-keys)), MUST NOT exceed 4 MiB.
- Object members may appear in any order. The order agentflow writes is fixed, but readers MUST NOT depend on it.
- Durations are integers in nanoseconds, in fields ending in `_ns`.

## 3. Identifiers and paths

- `trace_id` and `run_id` are strings. agentflow writes UUIDv7 values, which begin with a timestamp, so they sort in creation order. Readers MUST treat IDs as opaque.
- `seq` is an unsigned integer. It starts at 0 in each run and increases by 1 with each event, with no gaps.
- A run's **path** is its logical address within its trace. The root run's path is `/`. A child's path appends one segment to its parent's path, of the form `<agent>#<n>` (the parent's n-th child running that agent, counting from 0) or `<agent>@<key>` (a child with a key). An example is `/planner#0/sql#1`. Paths are unique within a trace.
- agentflow assigns segments as follows. A child started inside a tool call gets the generated key `<step>.<call>.<n>`, e.g. `/researcher@2.1.0`. Here `<step>` is the model step whose response requested the call, `<call>` is the call's position in that response (from 0), and `<n>` counts that call's earlier children running the same agent. Calls of one response may run concurrently, so this keeps paths independent of scheduling. A delegation (a tool whose call starts exactly one child) is the exception: it gets `<agent>#<n>`, with ordinals assigned in response order. Agent names never contain `/`, `#` or `@`, and keys never contain `/`.
- A path is the same in a recording, its replays and its forks. Tools that match runs across traces MUST match them by path, not by ID.
- IDs and `time` are never part of a fingerprint, and replay never compares them.

## 4. The event envelope

```json
{"v":1,"trace_id":"…","run_id":"…","seq":5,"time":"2026-11-23T10:00:00.123456Z","kind":"tool.started","cause":3,"body":{…}}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `v` | integer | yes | Format version. Always `1` in this version. |
| `trace_id` | string | yes | The trace this event belongs to. |
| `run_id` | string | yes | The run this event belongs to. |
| `seq` | integer ≥ 0 | yes | Position in the run ([§5](#5-ordering-and-causes)). |
| `time` | string | yes | Wall-clock time, RFC 3339. Writers SHOULD use UTC with microseconds (`2026-11-23T10:00:00.123456Z`). Informational only: order comes from `seq`. |
| `kind` | string | yes | The event kind ([§6](#6-event-kinds)). Dotted lowercase, e.g. `llm.started`. |
| `cause` | integer ≥ 0 | no | The `seq` of an earlier event in the same run that this event answers ([§5](#5-ordering-and-causes)). |
| `body` | object | yes | The kind-specific payload. |

## 5. Ordering and causes

Within a run:

1. `seq` 0 is always `run.started`. A run has exactly one `run.started`.
2. If a run has `run.finished`, it is the run's last event. A run without one is **incomplete** (it crashed or is still running). Readers report such runs with status `incomplete`, a status that is never written.
3. Every `*.started` event is answered by exactly one `*.completed` or `*.failed` event of the same family (`llm`, `tool` or `effect`), and every `run.spawned` by exactly one `run.joined`. The answer's `cause` is the start's `seq`. In an incomplete run, the last starts may have no answer. A run that ends while an effect is unanswered (it diverged, failed or was cancelled) answers it before `run.finished`, with `*.failed` code `cancelled` or a `run.joined` with status `cancelled`.
4. `cause` always refers to an earlier event in the same run.

Where agentflow sets `cause`:

| Event | `cause` |
|---|---|
| `llm.completed`, `llm.failed`, `tool.completed`, `tool.failed`, `effect.completed`, `effect.failed` | The matching `*.started`. |
| `tool.started` | The `llm.completed` whose response requested the call. |
| `effect.started` or `run.spawned` made while a tool call runs (a *nested* effect) | The enclosing `tool.started`. |
| `run.mode_changed` written for a tool call | That call's `tool.started`. |
| `run.joined` | The matching `run.spawned`. |

**Tool batches.** The tool calls from one model response form a batch, and are written in this layout:

1. a `tool.started` for every call, in the order of the response;
2. for each call, in the same order: that call's nested events, then a `run.mode_changed` if the call switched the run to live, then its `tool.completed` or `tool.failed`;
3. a `message.appended` with the tool results.

The tools themselves may run concurrently, but the layout never depends on which finished first.

**Spawn batches.** Child runs started together by code (`SpawnAll`) follow the same rule: a `run.spawned` for every child, in order, then a `run.joined` for every child, in the same order. A single child is a batch of one.

**Across runs.** Each run has its own `seq`. No order is defined between events of different runs; the links between runs (below) and `cause` give a causal partial order. A child's events are written as it runs. A `run.spawned` nested in a tool call is written only when its batch completes, which is after the child has run. Readers MUST link a child to its parent through the child's `run.started.parent.run_id`. They MUST NOT wait for the parent's `run.spawned`.

## 6. Event kinds

Tables list body fields. Fields marked *opt* may be absent. Content blocks (`blocks`) are defined in [§8](#8-messages-and-content-blocks), and other shared objects in [§7](#7-shared-objects).

A reader MUST accept event kinds it doesn't know, and SHOULD preserve them unchanged when copying a trace.

### 6.1 Run lifecycle

**`run.started`**: always `seq` 0.

| Field | Type | Meaning |
|---|---|---|
| `kind` | `"agent"` \| `"workflow"` | What executes the run. |
| `path` | string | The run's path ([§3](#3-identifiers-and-paths)). |
| `parent` | `{"run_id": string}` or `null` | The parent run. `null` for the root run. |
| `agent` | `{"name": string, "version"?: string}` | The agent or workflow definition. |
| `task` | blocks | The task the run was given. |
| `config` | config or `null` | The model configuration ([§7](#7-shared-objects)). `null` for workflows. |
| `config_fp` | fingerprint, *opt* | Fingerprint of `config` ([§10](#10-fingerprints)). Absent for workflows. |
| `origin` | origin, *opt* | How the trace came to exist ([§7](#7-shared-objects)). Only on the root run. |

**`run.finished`**: always the last event.

| Field | Type | Meaning |
|---|---|---|
| `status` | `"completed"` \| `"failed"` \| `"cancelled"` \| `"max_steps"` \| `"diverged"` | How the run ended. `diverged`: a replay stopped at a divergence. |
| `output` | blocks | The run's final output. |
| `error` | error, *opt* | Why the run failed. |
| `usage` | usage | Tokens used by this run only (not its children). |
| `steps` | integer | Number of model calls made. |

**`run.spawned`**: written in the parent when a child run starts.

| Field | Type | Meaning |
|---|---|---|
| `child_run_id` | string | The child's `run_id`. |
| `path` | string | The child's path. |
| `agent` | string | The child's agent name. |
| `task` | blocks | The task given to the child. |
| `fingerprint` | fingerprint | See [§10](#10-fingerprints). |

**`run.joined`**: written in the parent when that child finishes, or when the spawn ends without one. `cause` is the `run.spawned`.

| Field | Type | Meaning |
|---|---|---|
| `child_run_id` | string | The child's `run_id`. |
| `status` | status | The child's final status. |
| `output` | blocks | The child's output. |
| `usage` | usage | Tokens used by the child and all its descendants. |
| `duration_ns` | integer | How long the child ran. |
| `error` | error, *opt* | Present when the spawn itself failed or was abandoned: refused, couldn't start, or the parent ended before the child joined. `status` is then `failed` or `cancelled`. A child that ran and failed reports its error in its own `run.finished`. |
| `override` | boolean, *opt* | `true` if a fork supplied this result instead of running the child. |
| `intercepted` | boolean, *opt* | `true` if an interceptor or a limit on the run tree produced this outcome. In that case no child ran. |

A child's `run.started.parent.run_id` and the parent's `run.spawned.child_run_id` link the two runs in both directions. A `child_run_id` with no run in the trace means that no child ran there. Either the spawn was refused, or a replay answered it from the recording.

**`run.mode_changed`**: written during a replay or fork when a run stops being answered from the recording. A run never switches back.

| Field | Type | Meaning |
|---|---|---|
| `mode` | `"live"` | The new mode. |
| `reason` | `"divergence"` \| `"fork_point"` \| `"shared_state"` \| `"new_run"` | Why it switched. |
| `detail` | any JSON, *opt* | For example, the divergence that caused the switch. |

### 6.2 Conversation

**`message.appended`**: appends one message to the conversation. Exactly one of the two fields is present.

| Field | Type | Meaning |
|---|---|---|
| `message` | message | The message, inline. |
| `source_seq` | integer | The `seq` of an earlier `llm.completed` in this run. Its `message` is appended unchanged. This avoids storing model responses twice. |

**`conversation.replaced`**: replaces the whole conversation, for example after compaction.

| Field | Type | Meaning |
|---|---|---|
| `messages` | array of message | The new conversation. |

**`config.changed`**: changes part of the configuration mid-run. At least one field is present, and absent fields are unchanged. `tools: []` removes all tools.

| Field | Type |
|---|---|
| `model` | string, *opt* |
| `system` | blocks, *opt* |
| `tools` | array of tool spec, *opt* |
| `params` | params, *opt* |

### 6.3 Model calls

**`llm.started`**: a model call. The request is recorded as a reference, not a copy ([§9](#9-conversation-and-request-reconstruction)).

| Field | Type | Meaning |
|---|---|---|
| `step` | integer | The step number, from 0. Each model call starts a step. |
| `context` | `{"messages": integer, "config_seq": integer}` | The request uses the first `messages` messages of the conversation and the effective config at event `config_seq`. |
| `ephemeral` | array of `{"index": integer, "message": message}`, *opt* | Messages sent with this request only, not added to the conversation. |
| `fingerprint` | fingerprint | Fingerprint of the exact request ([§10](#10-fingerprints)). |

**`llm.completed`**

| Field | Type | Meaning |
|---|---|---|
| `step` | integer | Same as the start. |
| `message` | message | The assistant message. |
| `stop_reason` | `"end_turn"` \| `"tool_use"` \| `"max_tokens"` \| `"stop_sequence"` \| `"refusal"` \| `"other"` | Why generation stopped. |
| `usage` | usage | Tokens used by this call. |
| `model` | string, *opt* | The model as reported by the provider. |
| `attempts` | integer, *opt* | Number of attempts, if the call was retried. |
| `duration_ns` | integer | Duration of the call. |
| `ext` | ext, *opt* | Provider data about the response, such as a request ID. |
| `override` | boolean, *opt* | `true` if a fork supplied this response. |
| `intercepted` | boolean, *opt* | `true` if an interceptor (a policy, for example) produced this outcome and the call never reached the model. |

**`llm.failed`**

| Field | Type | Meaning |
|---|---|---|
| `step` | integer | Same as the start. |
| `error` | error | The failure. |
| `attempts` | integer, *opt* | Number of attempts. |
| `duration_ns` | integer | Duration of the call. |
| `intercepted` | boolean, *opt* | As for `llm.completed`. |

### 6.4 Tool calls

**`tool.started`**

| Field | Type | Meaning |
|---|---|---|
| `step` | integer | The step whose response requested the call. |
| `call_id` | string | The call ID from the model's `tool_use` block. |
| `tool` | string | Tool name. |
| `input` | any JSON | The tool input. |
| `fingerprint` | fingerprint | See [§10](#10-fingerprints). |
| `accesses` | array of `{"resource": string, "mode": "read" \| "write"}`, *opt* | Shared resources the tool declared it touches. Absent means unknown. `[]` means the tool touches nothing shared. Writers MUST keep the two distinct. |

**`tool.completed`**: the tool returned a result. A result can still be an error *the model sees*, which is marked by `is_error`.

| Field | Type | Meaning |
|---|---|---|
| `call_id` | string | Same as the start. |
| `content` | blocks | The result. |
| `is_error` | boolean | The result reports an error to the model. |
| `duration_ns` | integer | Duration of the call. |
| `override` | boolean, *opt* | `true` if a fork supplied this result. |
| `intercepted` | boolean, *opt* | `true` if an interceptor produced this result. |

**`tool.failed`**: the call did not produce a result, for example because the tool was not found, the call was denied, the tool timed out or panicked.

| Field | Type |
|---|---|
| `call_id` | string |
| `error` | error |
| `duration_ns` | integer |
| `intercepted` | boolean, *opt* |

### 6.5 Custom effects

Any other nondeterministic operation, such as reading the clock, generating an ID, or a memory lookup. Context lookups made before a model call are custom effects named `context.<name>`.

| Kind | Fields |
|---|---|
| `effect.started` | `name` (string), `input` (any JSON), `fingerprint` |
| `effect.completed` | `name`, `output` (any JSON), `duration_ns`, `override` *opt*, `intercepted` *opt* |
| `effect.failed` | `name`, `error`, `duration_ns`, `intercepted` *opt* |

### 6.6 Annotations

**`annotation`**: a marker written by an application or by tests, such as a label or a verdict.

| Field | Type | Meaning |
|---|---|---|
| `key` | string | Keys starting with `agentflow.` are reserved for agentflow. |
| `value` | any JSON | |

Reserved keys:

| Key | Value | Meaning |
|---|---|---|
| `agentflow.expectation` | `{"name": string, "pass": boolean, "detail"?: string}` | A check recorded by the application: whether something the run produced is what it should be. A failed expectation is a symptom that tools such as `blame` start from. |

An annotation written while a tool call runs is buffered with the call's nested events, like a nested effect ([§5](#5-ordering-and-causes)).

## 7. Shared objects

**error**

| Field | Type | Meaning |
|---|---|---|
| `code` | string | A machine-readable code (below). |
| `message` | string | A human-readable description. |
| `retryable` | boolean, *opt* | The operation may succeed if retried. |
| `detail` | string, *opt* | Extra detail, such as a stack trace. |

Codes in use: model calls: `rate_limited`, `overloaded`, `invalid_request`, `auth`, `context_length`, `server`, `network`, `cancelled`. Tools and effects: `not_found`, `invalid_input`, `denied`, `timeout`, `panic`, `cancelled`, `internal`. Child runs: `limit_exceeded`, `denied`, `cancelled`. Readers MUST accept other codes.

**usage**: `input_tokens` and `output_tokens` (required), and `cache_read_tokens`, `cache_write_tokens` and `reasoning_tokens` (optional). Reasoning tokens are a subset of output tokens.

**config**: an agent's model configuration.

| Field | Type | Meaning |
|---|---|---|
| `model` | string | Model name. |
| `system` | blocks | System prompt. |
| `tools` | array of tool spec | Sorted by `name`. |
| `tool_choice` | `{"mode": "auto" \| "any" \| "none" \| "tool", "name"?: string}` or `null` | `name` is set when `mode` is `tool`. |
| `params` | params | Generation parameters. |
| `max_steps` | integer | The run's step limit. |

**tool spec**: `name`, `description`, `input_schema` (a JSON Schema, any JSON), and optional `annotations` (`{"read_only"?, "destructive"?, "idempotent"?}`, all booleans). Annotations describe the tool to agentflow. They are not sent to the model.

**params**: all fields optional, and an unset field is absent: `max_tokens` (integer), `temperature` (number), `top_p` (number), `stop` (array of strings), `reasoning` (`{"effort"?: string, "budget_tokens"?: integer}`), `ext`.

**origin**: on the root run's `run.started`.

| Field | Type | Meaning |
|---|---|---|
| `kind` | `"live"` \| `"replay"` \| `"fork"` | How the trace was made. |
| `source_trace_id` | string, *opt* | The trace replayed or forked. |
| `mode` | string, *opt* | The replay mode. |
| `at`, `actual_at` | fork point, *opt* | The fork point requested, and the one used if it moved. A fork point is `{"path": string, "effect"?: string, "seq"?: integer}`. `effect` is an effect label ([§10.1](#101-effect-labels)). |
| `overrides` | any JSON, *opt* | The fork's overrides, as given. |
| `group`, `sample`, `arm` | string, integer, string, *opt* | Link the samples of one fork experiment. |
| `producer` | string, *opt* | The system that recorded the trace when it isn't the agentflow runtime, e.g. `"claude-code"` for an imported Claude Code session. See [§9](#9-conversation-and-request-reconstruction). |

**ext**: an object of provider-specific data, keyed by provider name, e.g. `{"anthropic": {"signature": "…"}}`. Values are any JSON. Readers MUST preserve `ext` byte for byte (after parsing): some providers reject a conversation whose opaque data was changed.

## 8. Messages and content blocks

A **message** is `{"role": "user" | "assistant", "blocks": [block, …]}`. There is no system role (the system prompt is in the config) and no tool role (tool results are `tool_result` blocks in a user message).

A **block** is an object with a `type` field:

| `type` | Fields | Meaning |
|---|---|---|
| `text` | `text`, `ext`*opt* | Plain text. |
| `media` | `media_type`, `data`*opt* (base64), `url`*opt*, `ext`*opt* | An image, document or audio clip. Normally exactly one of `data` and `url` is set. |
| `tool_use` | `call_id`, `name`, `input` (any JSON), `ext`*opt* | A tool call requested by the model. |
| `tool_result` | `call_id`, `content` (blocks), `is_error`*opt*, `ext`*opt* | The result of a tool call. |
| `reasoning` | `text`*opt*, `redacted`*opt* (boolean), `ext`*opt* | Model reasoning. Signatures or encrypted content are kept in `ext`. A reasoning block MUST NOT be modified: providers verify it. |
| `data` | `value` (any JSON), `ext`*opt* | Structured content, such as structured output. |

A reader MUST accept block types it doesn't know and SHOULD preserve them unchanged. Writers use them to carry provider content that has no mapping yet.

## 9. Conversation and request reconstruction

The conversation of a run is a fold over its events in `seq` order:

```
conv := []
for each event e in seq order:
    message.appended with message     → append e.body.message to conv
    message.appended with source_seq  → append the message of the llm.completed at e.body.source_seq
    conversation.replaced             → conv := e.body.messages
```

The **effective config** at an event is the `config` of `run.started`, with every `config.changed` up to that event applied field by field.

The exact request of the `llm.started` event at `seq = s` is:

| Request field | Value |
|---|---|
| `model`, `system`, `tool_choice`, `params` | From the effective config at event `context.config_seq`. That event MUST be the `run.started` or a `config.changed`. |
| `tools` | The same config's tools, sorted by `name`. |
| `messages` | The first `context.messages` messages of `conv` as folded up to (not including) `s`. Then each `ephemeral[i].message` is inserted at `ephemeral[i].index`, in order; each index refers to the list as it stands after the previous insertions. |

Each message is stored once, and each request is a small reference, so a trace grows linearly with the length of the run.

**Traces from other producers.** When the root run's `origin.producer` is set, the trace was recorded by another system (for example, converted from a Claude Code transcript). Its `config` holds only what that system recorded (typically the model), so the request rebuilt by this section is a **reconstruction**: the system prompt, tools or hidden context may be missing, and fingerprints are computed over the reconstruction. Such traces are for reading. Tools MUST NOT replay or fork them as if their requests were exact.

## 10. Fingerprints

A fingerprint is `"sha256:"` followed by 64 lowercase hex digits: the SHA-256 of the **RFC 8785 (JCS) canonical form** of a JSON value. JCS sorts object members, removes insignificant whitespace, and formats numbers and strings deterministically, so two values have the same fingerprint exactly when they are equal as JSON.

Fingerprints are computed over **logical values**: the content before blob externalization and before redaction ([§11](#11-blobs-redaction-and-reserved-keys)).

| Where | Value hashed |
|---|---|
| `llm.started.fingerprint` | `{"model", "system", "messages", "tools", "tool_choice", "params"}` of the request rebuilt in [§9](#9-conversation-and-request-reconstruction), encoded as in this format. Each tool is reduced to `{"name", "description", "input_schema"}`, because annotations are not sent to the model. `messages`, `system` and `tools` are `[]` when empty. A missing `input_schema` is `null`. |
| `tool.started.fingerprint` | `{"tool": <tool>, "input": <input>}`. An empty input is `{}`. |
| `effect.started.fingerprint` | `{"name": <name>, "input": <input>}`. An empty input is `null`. |
| `run.spawned.fingerprint` | `{"agent": <agent>, "task": <task>, "key"?: <key>}`, where `key` is the text after `@` in the last segment of `path`, absent for an `<agent>#<n>` segment. The child's config is excluded. |
| `run.started.config_fp` | The `config` object, as encoded in the event. |

Fingerprints are exact: tool call IDs inside messages, for example, are part of the request. Replay compares a new request's fingerprint with the recorded one, and the request is answered from the recording only when they are equal.

### 10.1 Effect labels

Replay results and fork points name effects with **labels** that are derived from a run's events:

| Effect | Label |
|---|---|
| The n-th `llm.started` in the run (from 0) | `llm#<n>` |
| A `tool.started` | `tool:<call_id>`. When a `call_id` repeats within a run (some providers reuse IDs across responses), the k-th repeat is `tool:<call_id>#<k>`, counting from 1. |
| The n-th `effect.started` with a given name, outside tool calls | `effect:<name>#<n>` |
| A `run.spawned` | `spawn:<path>`, with the child's path, e.g. `spawn:/researcher#0`. |
| An effect nested in a tool call | The tool's label, then `/`, then the nested label, e.g. `tool:toolu_01/effect:clock.now#0` or `tool:toolu_01/spawn:/researcher#0`. Counters restart inside each call. |

## 11. Blobs, redaction and reserved keys

Inside a `body`, any JSON **string** may be replaced by one of two reference objects. Readers MUST resolve references before interpreting a body.

| Reference | Meaning | Resolves to |
|---|---|---|
| `{"$blob": "sha256:<hex>", "size": <n>}` | The string was moved to the blob store. `<hex>` is the SHA-256 of its UTF-8 bytes, and `size` its length in bytes. | The blob's content, as a string. A missing blob is an error. |
| `{"$redacted": "<rule>"}` | The string was removed before storage by the redaction rule named `<rule>`. | A placeholder. agentflow uses `"[redacted: <rule>]"`. |

- agentflow externalizes strings longer than 32 KiB by default (configurable per store). Identical content is stored once. Base64 media data is a string, so large media is externalized too.
- Redaction is opt-in and configured by the application.
- `$blob`, `$redacted` and `$lit` are **reserved keys**. If application data (a tool input, an effect output, an annotation value, …) contains an object with any of these keys, the writer wraps that object as `{"$lit": <object>}`. A reader unwraps it and takes the wrapped object's own keys literally. The wrapped object's member values are still resolved as usual, since they may contain references themselves.
- The envelope is never externalized or redacted.

## 12. Storage: run files and bundles

### 12.1 File store

agentflow's file store keeps one file per run:

```
<root>/traces/<trace_id>/<run_id>.jsonl     the run's events, one per line, in seq order
<root>/blobs/sha256/<hh>/<rest-of-hex>      blob content (raw bytes), named by hash
```

- Directories are created with mode `0700` and files with `0600`.
- A crash can leave a final line without its `\n`, or cut off mid-event. Readers MUST ignore such a line and report the run as incomplete. Before appending to a run again, a writer truncates the file back to its last complete line.

### 12.2 Bundles

A **bundle** is a single file holding one trace, used for sharing and as a test fixture:

```
{"bundle":"agentflow.trace","v":1,"trace_id":"…","created":"…"}   header
{…event…}                                                           events
{"blob":"sha256:…","data":"<base64>"}                               blobs
```

1. The first line is the header. `bundle` is always `"agentflow.trace"`, and `v` is the format version.
2. Then every event of the trace, grouped by run (in run start order), in `seq` order within each run.
3. Then one line per blob referenced by the events, sorted by hash. `data` is the blob's content in standard base64. A reader MUST check that the content matches the hash.

Every event's `trace_id` MUST equal the header's `trace_id`.

## 13. Versioning

- Every event carries `"v": 1`. A reader MUST reject an event with a `v` it doesn't support, and SHOULD say so explicitly rather than misreading the event.
- **Additive changes keep `v`.** These are new event kinds, new optional fields, new block types and new error codes. Readers therefore MUST ignore unknown fields and accept unknown kinds and block types. The published schema allows unknown fields for this reason.
- **Breaking changes increase `v`.** agentflow readers support the current and the previous version.
- Traces written by agentflow's unreleased prototype carry `schema_version` instead of `v`. Readers SHOULD reject them as unsupported.

## 14. Security considerations

- **Traces contain everything the agent saw**: prompts, tool inputs and outputs, and model responses. Treat trace files like logs that may contain secrets. The file store creates them readable by the owner only.
- **Redaction covers stored content, not fingerprints.** A fingerprint is computed before redaction, so that replay can still match a request whose stored copy is redacted. A fingerprint of a *low-entropy* secret (a PIN or a short password) can be recovered by trying every candidate. High-entropy secrets such as API keys and tokens cannot practically be recovered.
- **Blobs are verified by hash.** A reader that loads a blob from an untrusted bundle MUST check the content against its hash.
- **Bundles are data.** Importing a bundle never runs code. Replaying one runs *your* code, with outcomes taken from the bundle.

## 15. Example

A run in which the model calls a calculator tool once and then answers. Fingerprints are shortened here; the real values are 64 hex digits.

```json
{"v":1,"trace_id":"T","run_id":"R","seq":0,"time":"2026-11-23T10:00:00.000000Z","kind":"run.started","body":{"kind":"agent","path":"/","parent":null,"agent":{"name":"calculator"},"task":[{"type":"text","text":"What is 6*7?"}],"config":{"model":"claude-opus-5-5","system":[{"type":"text","text":"Use the calculator tool."}],"tools":[{"name":"calculator","description":"Evaluates an expression.","input_schema":{"type":"object"}}],"tool_choice":null,"params":{"max_tokens":1024},"max_steps":10},"config_fp":"sha256:3f…","origin":{"kind":"live"}}}
{"v":1,"trace_id":"T","run_id":"R","seq":1,"time":"2026-11-23T10:00:00.000100Z","kind":"message.appended","body":{"message":{"role":"user","blocks":[{"type":"text","text":"What is 6*7?"}]}}}
{"v":1,"trace_id":"T","run_id":"R","seq":2,"time":"2026-11-23T10:00:00.000200Z","kind":"llm.started","body":{"step":0,"context":{"messages":1,"config_seq":0},"fingerprint":"sha256:91…"}}
{"v":1,"trace_id":"T","run_id":"R","seq":3,"time":"2026-11-23T10:00:00.800000Z","kind":"llm.completed","cause":2,"body":{"step":0,"message":{"role":"assistant","blocks":[{"type":"tool_use","call_id":"toolu_01","name":"calculator","input":{"expression":"6*7"}}]},"stop_reason":"tool_use","usage":{"input_tokens":112,"output_tokens":21},"duration_ns":799800000}}
{"v":1,"trace_id":"T","run_id":"R","seq":4,"time":"2026-11-23T10:00:00.800100Z","kind":"message.appended","body":{"source_seq":3}}
{"v":1,"trace_id":"T","run_id":"R","seq":5,"time":"2026-11-23T10:00:00.800200Z","kind":"tool.started","cause":3,"body":{"step":0,"call_id":"toolu_01","tool":"calculator","input":{"expression":"6*7"},"fingerprint":"sha256:0c…"}}
{"v":1,"trace_id":"T","run_id":"R","seq":6,"time":"2026-11-23T10:00:00.800300Z","kind":"tool.completed","cause":5,"body":{"call_id":"toolu_01","content":[{"type":"text","text":"42"}],"is_error":false,"duration_ns":41000}}
{"v":1,"trace_id":"T","run_id":"R","seq":7,"time":"2026-11-23T10:00:00.800400Z","kind":"message.appended","body":{"message":{"role":"user","blocks":[{"type":"tool_result","call_id":"toolu_01","content":[{"type":"text","text":"42"}]}]}}}
{"v":1,"trace_id":"T","run_id":"R","seq":8,"time":"2026-11-23T10:00:00.800500Z","kind":"llm.started","body":{"step":1,"context":{"messages":3,"config_seq":0},"fingerprint":"sha256:5d…"}}
{"v":1,"trace_id":"T","run_id":"R","seq":9,"time":"2026-11-23T10:00:01.400000Z","kind":"llm.completed","cause":8,"body":{"step":1,"message":{"role":"assistant","blocks":[{"type":"text","text":"6 × 7 = 42"}]},"stop_reason":"end_turn","usage":{"input_tokens":140,"output_tokens":9},"duration_ns":599500000}}
{"v":1,"trace_id":"T","run_id":"R","seq":10,"time":"2026-11-23T10:00:01.400100Z","kind":"message.appended","body":{"source_seq":9}}
{"v":1,"trace_id":"T","run_id":"R","seq":11,"time":"2026-11-23T10:00:01.400200Z","kind":"run.finished","body":{"status":"completed","output":[{"type":"text","text":"6 × 7 = 42"}],"usage":{"input_tokens":252,"output_tokens":30},"steps":2}}
```

The request of `llm#1` (`seq` 8) is rebuilt from the config at `seq` 0 and the first 3 messages of the conversation: the task (`seq` 1), the assistant's tool call (`seq` 3, through `source_seq` at `seq` 4), and the tool result (`seq` 7).
