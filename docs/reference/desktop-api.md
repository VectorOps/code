# Desktop API reference

The desktop worker exposes Runner's control API over WebSocket and adds desktop-specific methods. This page covers the transport, desktop settings, and prompt history. For workflow and tool methods, see [VectorOps Runner](https://github.com/vectorops/runner).

## Transport

Packets are version-2 Runner control packets sent as JSON text messages. Authenticate before sending requests. The [connection guide](../connections.md) describes tokens and allowed origins.

The client subscribes to events before fetching saved state. After reconnecting, it authenticates, subscribes again, and reloads snapshots and history. Event sequence numbers belong to one connection, not a durable replay log. Commands are not automatically replayed after disconnect.

Start and input-response retries reuse their idempotency key when retried with unchanged content in the same mounted interface. If a request loses its response, its outcome can be unknown; do not assume it failed or resend it with a new key without checking state.

## Desktop settings

`desktop.settings.get` takes an empty parameter object. `desktop.settings.update` accepts a patch of writable settings and returns the resulting settings snapshot.

| Field | Values or default | Writable |
| --- | --- | --- |
| `appearance` | `system` (default), `light`, `dark` | Yes |
| `default_workflow` | Workflow reference; default `agent` | Yes |
| `history_limits` | Counts listed below | Yes |
| `debug_mode` | `auto` (default), `on`, `off` | Yes |
| `development` | Whether this worker was launched in development mode | No |
| `debug_enabled` | Effective debug visibility, resolved by the backend | No |

The `history_limits` object has these fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `initial` | 1000 | Steps loaded when opening history |
| `maximum` | 2000 | Retention ceiling in the client |
| `target` | 1000 | Number of steps to retain after trimming |
| `older` | 200 | Steps per request for earlier history; maximum 500 |

Counts must be positive integers. `initial` and `target` must not exceed `maximum`. Updates merge top-level fields; `history_limits` is replaced as a group, with defaults for omitted members. Unknown fields are rejected.

For example, enable diagnostics:

```json
{
  "kind": "request",
  "id": "debug-setting",
  "method": "desktop.settings.update",
  "params": {"debug_mode": "on"}
}
```

Updates use a filesystem lock and atomic replacement. Invalid values return `invalid_argument`; storage failures return `settings_storage_error`. See [data storage](../usage.md#data-storage) for the settings location.

## Prompt history

These methods require an authenticated desktop worker. The worker supplies the project identity; clients do not pass a `project_id` parameter.

### List prompts

```json
{
  "kind": "request",
  "id": "recall",
  "method": "prompt_history.list",
  "params": {"scope": "global", "query": "fix tests", "limit": 20}
}
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `scope` | `global` | `global`, `project`, or `session` |
| `execution_id` | Omitted | Required for `session` scope |
| `query` | Empty string | Case-insensitive literal search text |
| `before` | Omitted | Positive entry ID used as a pagination cursor |
| `limit` | 20 | Page size, from 1 to 20 |

The result contains `entries` and `next_before`. Each entry has `id`, `project_id`, `execution_id`, `timestamp`, and `text`. Pass `next_before` as `before` to request an older page; `null` means there are no more results. Repeated text appears once, at its latest position in the selected scope.

### Append a prompt

`prompt_history.append` requires:

- `text`: nonblank text, at most 65,536 characters.
- `execution_id`: a nonempty execution identifier, at most 128 characters.
- `idempotency_key`: a nonempty key, at most 256 characters. Reuse it only for an unchanged retry of the same entry.

The result is the stored entry. A reused key with different text, execution, or project is rejected. The store retains the latest 1,000 entries across projects for that host account.

See [prompt recall](../usage.md#write-and-recall-messages) for interface behavior and [data storage](../usage.md#data-storage) for privacy details.
