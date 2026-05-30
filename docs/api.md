# SSNS HTTP API Reference

The `ssns-server` binary exposes a small REST + static-file API used by the bundled web UI. All payloads are JSON unless noted. All endpoints respond with `application/json` on success and a `{"error": "..."}` body on failure.

Server defaults: host `127.0.0.1`, port `8765`, UI directory `ui/`.

```sh
./build/ssns-server --host 127.0.0.1 --port 8765 --ui-dir ui
```

### `ssns-server` CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--host <ip>` | `127.0.0.1` | Bind address |
| `--port <int>` | `8765` | Listen port |
| `--ui-dir <path>` | `ui` | Frontend SPA root (served under `/` and `/ui/*`) |
| `--data <path>` | `ui_data/training_data.json` | Training log file the API reads/writes |
| `--status <path>` | `ui_data/training_status.json` | Live status file (epoch counter, running flag) |
| `--benchmark <path>` | `./ssns-benchmark` | Path to the `ssns-benchmark` binary spawned by `/api/run_training` |
| `--help`, `-h` | | Show usage |

---

## Endpoints

### `GET /`
Serves `ui/index.html`. The single-page web UI for driving training and keygen interactively.

Response: `text/html`.

### `GET /ui/<path>`
Serves any static asset under `ui/` (CSS, JS, images). Path traversal (`..`) is rejected.

Response: file content with content-type by extension. `404` if the file does not exist.

---

### `GET /api/training_data`
Returns the full training log produced by the last (or in-progress) `ssns-benchmark` run.

Response shape (truncated):
```json
{
  "metadata": {
    "T_input": 32, "T_hidden": 128, "S_input": 32, "S_hidden": 768,
    "output_dim": 1280, "cluster_size": 5, "dz": 0.09,
    "epochs": 30000, "batch_size": 64, "lr_max": 0.01,
    "use_fhe": false, "bimodality_alpha": 0.0
  },
  "snapshots": [
    {"epoch": 0,    "loss": 1.234, "W1_S": [...], "W2_S": [...], "teacher_W1": [...], "teacher_W2": [...]},
    {"epoch": 3000, "loss": 0.045, ...},
    ...
  ]
}
```

Status codes:
- `200` on success
- `404` if no training log exists (run `/api/run_training` first)
- `503` if the file is mid-write (retry)

### `GET /api/training_status`
Returns live status while a training subprocess is running.

Response shape:
```json
{
  "running": true,
  "epoch": 1234,
  "total_epochs": 30000,
  "loss": 0.0123,
  "started_at": "2026-05-30T12:00:00Z",
  "stopped": false,
  "completed_at": null
}
```

When the run finishes, `running` flips to `false` and `completed_at` is populated. If the subprocess was killed via `/api/stop_training`, `stopped` is `true`.

Status codes:
- `200` on success
- `404` if no status file exists yet
- `503` if the file is mid-write

---

### `POST /api/run_training`
Spawns `ssns-benchmark` as a detached subprocess with the supplied hyperparameters. Returns immediately; the actual training happens in the background. Poll `/api/training_status` and `/api/training_data` for progress.

**Required body fields** (all integers unless noted, doubles where marked):

| Field | Type | Range / notes |
|---|---|---|
| `T_input` | int | Teacher input dimension |
| `T_hidden` | int | Teacher hidden width |
| `S_input` | int | Student input dimension (must equal `T_input` for direct comparison) |
| `S_hidden` | int | Student hidden width (e.g. 768) |
| `output_dim` | int | `key_bits * cluster_size` (e.g. 1280 for 256 bits) |
| `cluster_size` | int | Neurons per output bit (e.g. 5) |
| `batch_size` | int | e.g. 64 |
| `epochs` | int | e.g. 30 000 |
| `dz` | double | Dead-zone half-width, e.g. 0.09 |
| `lr_max` | double | Peak learning rate, e.g. 0.01 |
| `warmup_frac` | double | Fraction of epochs for linear warmup, e.g. 0.05 |

**Optional body fields**:

| Field | Type | Default | Notes |
|---|---|---|---|
| `samples_to_log` | int | 20 | Snapshots written per run, range `[1, 1024]` |
| `snapshot_count` | int | 10 | Layout snapshots saved, range `[2, 200]` |
| `use_fhe` | bool | false | Enable real CKKS path (much slower) |
| `bimodality_alpha` | double | 0.0 | Loss term pushing outputs to `{0,1}`, range `[0, 5]` |
| `simulate_fhe_noise` | double | 0.0 | Plain + gaussian noise (fast FHE proxy), range `[0, 10]` |
| `key_confirmation` | bool | false | Append `--key-confirmation N` to benchmark CLI |
| `key_confirmation_trials` | int | 10 000 | If `key_confirmation` is true, range `[1, 1 000 000]` |
| `teacher_seed` | uint64 | (CLI default) | Set both `teacher_seed` and `bfa_seed` together for reproducible runs |
| `bfa_seed` | uint64 | (CLI default) | |

Response on success (`200`):
```json
{
  "status": "started",
  "message": "Training subprocess spawned...",
  "pid": 12345,
  "params": { ...echo of request body... },
  "cmd": ["./build/ssns-benchmark", "--t-input", "32", ...]
}
```

Status codes:
- `200` started successfully
- `422` invalid JSON body or missing/out-of-range field
- `500` `ssns-benchmark` binary not found or not executable

### `POST /api/stop_training`
Sends `SIGTERM` to a running benchmark subprocess. The server verifies (via `/proc/<pid>/comm` and `/proc/<pid>/cmdline`) that the pid actually belongs to a benchmark process, then rewrites `training_status.json` with `running=false stopped=true` so polling UIs settle even if the child died before flushing its own status.

Request body:
```json
{ "pid": 12345 }
```

Response (`200`):
```json
{ "status": "stopped", "pid": 12345 }
```

Status codes:
- `200` SIGTERM delivered
- `404` no such process, or pid is not an `ssns-benchmark`
- `422` missing or invalid `pid` field

---

### `POST /api/manual_test`
Single-shot keygen test. Loads the most recent trained Teacher + Student from `training_data.json`, runs them on the user-supplied input `X`, returns both sigmoid output vectors plus the extracted bits and their agreement.

Request body:
```json
{ "X": [0.12, -0.34, ...] }
```

`X` must have exactly `S_input` elements (matched against the loaded training log's metadata).

Response (`200`):
```json
{
  "Y_T": [...], "Y_S": [...],
  "bits_T": [0, 1, 1, 0, ...], "idx_T": [0, 2, 3, 7, ...],
  "bits_S": [0, 1, 1, 0, ...], "idx_S": [0, 2, 3, 5, 7, ...],
  "shared_indices": [0, 2, 3, 7, ...],
  "mismatches": 0,
  "n_confident_T": 38, "n_confident_S": 42,
  "match": true,
  "epoch_used": 30000
}
```

Status codes:
- `200` success
- `422` malformed body or wrong `X` length
- `404` no trained pair available (train first)

### `POST /api/stress_test`
Runs `n_trials` independent keygen rounds against the loaded Teacher/Student pair using random gaussian inputs. Aggregates per-trial confident-bit counts, shared-bit counts, mismatches, and SHA-256 hash-exchange acceptance.

Request body:
```json
{ "n_trials": 200, "seed": 42 }
```

| Field | Type | Default | Range |
|---|---|---|---|
| `n_trials` | int | 200 | `[1, 200 000]` |
| `seed` | int or null | random | any non-negative int |

Response (`200`):
```json
{
  "n_trials": 200,
  "epoch_used": 30000,
  "total_clusters": 256,
  "cluster_size": 5,
  "dead_zone": 0.09,
  "teacher_confident": {"mean": 113.5, "std": 4.2, "min": 102, "max": 124},
  "student_confident": {"mean": 113.2, "std": 4.3, "min": 101, "max": 125},
  "shared":            {"mean": 113.6, "std": 4.0, "min": 100, "max": 124},
  "mismatches":        {"mean": 0.0,   "std": 0.0, "min": 0,   "max": 0},
  "match_rate_pct": 100.0,
  "perfect_trials": 200,
  "total_mismatches": 0,
  "total_shared_bits": 22720,
  "hash_dropped": 0,
  "hash_accepted": 200,
  "hash_silent_fail": 0,
  "histogram_shared": [{"lo": 100, "hi": 105, "count": 12}, ...],
  "seed": 42
}
```

`hash_silent_fail` should always be `0`. A non-zero value means a trial passed bit-comparison but failed SHA-256 confirmation, indicating a real bug.

Status codes:
- `200` success
- `422` `n_trials` out of `[1, 200 000]` or seed not an int
- `404` / `409` / `503` no usable training log

---

## Error format

Every non-2xx response uses:
```json
{ "error": "human-readable message" }
```
