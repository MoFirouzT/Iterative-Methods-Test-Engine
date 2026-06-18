# Persistence & Experiment Naming

Two formats are always written together for every experiment:

| Format | Purpose |
| -------- | --------- |
| `result.jld2` | Full binary — preserves all Julia types, fast reload |
| `run{N}_{MethodName}.csv` | Per-method per-run CSV — human-readable, grep-able |
| `manifest.json` | Experiment metadata and human name; no binary load needed |

## File Layout on Disk

```text
logs/
└── 20260417/
    ├── 001/
    │   ├── manifest.json            ← {name, timestamp, host, methods, n_runs, tags}
    │   ├── result.jld2
    │   ├── run1_GradientDescent[step_size=Armijo].csv
    │   ├── run1_GradientDescent[step_size=BB1].csv
    │   └── run2_GradientDescent[step_size=Armijo].csv
    └── 002/
        ├── manifest.json
        └── ...
```

CSV sidecar contract:
The CSV writer records the fixed IterationLog fields plus all extras whose values are CSV-scalar:
numbers (incl. Bool), strings, symbols, missing, nothing.
Vector-valued and composite extras (e.g. :x_iter, :sub_logs, custom dicts) are intentionally omitted —
they have no stable text representation and inflating them inline would defeat the CSV's purpose as a grep-able artifact.
The full payload remains in result.jld2 and is recovered transparently by load_experiment.
The set of skipped keys is recorded in manifest.json under csv_skipped_extras, with a human-readable note pointing to JLD2 for the full payload.
The classifier rule is all-or-nothing per key: a single non-scalar occurrence demotes the whole column to JLD2-only, so the CSV never contains half a column.
Adding a new CSV-scalar type is one method:_is_csv_scalar(::DateTime) = true in persistence.jl.
Method names are unique per run, but if two sanitize to the same filename the writer raises rather than silently overwriting one run with another.

## Durability

Each experiment directory is reserved atomically: `next_experiment_path` creates
the numbered directory with `mkdir` (throws on `EEXIST`), so concurrent writers
can't collide on the same counter. The two load-bearing artifacts — `result.jld2`
and `manifest.json` — are written to a temp sibling and atomically renamed into
place, so a crash mid-write never leaves a truncated binary for `load_experiment`
or a half-written manifest for `list_experiments`. CSV sidecars are written
directly; a truncated one is self-evident and not consulted by reload or indexing.

## API

```julia
# Save — called automatically at the end of run_experiment.
# `compress = false` (default) writes the JLD2 binary uncompressed; see the
# "JLD2 compression" section below for why off is the empirical default and
# when to override. `true` selects JLD2's built-in codec; a specific
# `TranscodingStreams` codec value is also accepted and passed through verbatim.
# `persist` prunes heavy per-iteration extras from the binary (see "Selective
# saving" below); the default keeps everything.
# `extra_manifest` merges its keys into manifest.json after the framework's own
# fields, so an experiment can record stage-specific results (milestone iters,
# tolerances, ...) for later `jq` access without loading the JLD2.
save_experiment(result::ExperimentResult;
                compress = false,
                extra_manifest::Dict{String,Any} = Dict{String,Any}(),
                persist::PersistPolicy = PersistPolicy())

# Reload for analysis — one call restores everything
result = load_experiment("logs/20260417/001/")

# Quick metadata access without loading the binary
manifest = load_manifest("logs/20260417/001/manifest.json")

# List all experiments across all days
list_experiments(log_root="logs") :: Vector{NamedTuple}
# Returns [{path, date, number, name, timestamp, n_methods, n_runs}, ...]
```

## Manifest schema — base fields

Every `manifest.json` carries:

- `name`, `timestamp`, `host`, `tags` — from `ExperimentConfig`.
- `n_runs`, `n_methods`, `methods` — derived from `result.run_results`.
- `method_results` — per-method, per-run termination summary:
  `n_iters`, `stop_reason`, `f_final`, `grad_final`, `dist_final`.
  This is derived from `MethodResult.n_iters` / `.stop_reason` and the
  final `IterationLog` entry. It makes `jq '.method_results.BB1[0]'`
  enough to know how a method terminated without loading the JLD2.
  Non-finite metrics serialize as JSON `null` (e.g. `dist_final` is `null`
  when the problem has no known `x_opt`, so `dist_to_opt = Inf`) — JSON has
  no `Inf`/`NaN`, so the manifest sanitizes them on write.
- `csv_skipped_extras` (only present when non-empty) — keys whose values
  were non-scalar and therefore omitted from the CSV sidecar.
- `persist_dropped_extras` / `persist_decimated` (only when a non-default
  `PersistPolicy` was used) — which extras keys were pruned from `result.jld2`
  and the per-key decimation factors. Makes the pruning auditable from the
  manifest alone.

Stage-specific summary data (e.g. Stage 4's iters-to-milestone and the
distance/gradient tolerances used) is delivered via `extra_manifest` and
merged after the base fields. Stage 4 uses this to record the milestone
threshold and per-method iters-to-milestone so cold-restart `jq` queries
can answer "did BB1 cross 1e-6 here?" without recomputing from the CSVs.

## On-disk layout — column-major

In memory a `MethodResult` holds `Vector{IterationLog}`, one `Dict`-bearing
struct per iteration. Serializing that array-of-structs directly would make JLD2
store the dict's type machinery *per row*, which both bloats the file and gives
the compressor nothing to chew on.

Instead, `result.jld2` stores each method's iter logs **column-major** (a
`JLD2.writeas` shim on `MethodResult`): parallel typed vectors for the fixed
metrics (`iter`, `objective`, `gradient_norm`, …) plus one vector per `extras`
key (`Dict{Symbol,Vector}`, with a private `_ABSENT` sentinel for iters that
lack a key — distinct from a genuine `missing` *value*, which round-trips
unchanged). The conversion is invisible to callers: `load_experiment`
reconstructs the identical `Vector{IterationLog}`, so analysis code and
`to_dataframe` are unaffected.

Measured impact (single-method synthetic payloads), against the naive
array-of-structs serialization:

| payload | array-of-structs | column-major |
| --- | --- | --- |
| low-dim (2-D `x_iter`, 20 000 iters) | 15.5 MB | **3.7 MB** (≈4.2×) |
| high-dim (1000-D `x_iter`, 5 000 iters) | 43.8 MB | 40.9 MB |

The columnar layout wins big when the per-row dict overhead dominates (low-dim,
many iters). It barely helps high-dim payloads, where the bytes *are* the
`:x_iter` trajectory — that case is for **selective saving** below.

## Selective saving (`PersistPolicy`)

The columnar layout shrinks the per-row overhead but not the trajectory data
itself. On high-dimensional problems `:x_iter` (a full-iterate snapshot every
iteration) dominates the file, and it's only needed for trajectory plots.
`save_experiment(...; persist = PersistPolicy(...))` (and the `persist` field on
`ExperimentConfig`, threaded through `run_experiment`) prunes it from the binary:

```julia
# drop a key entirely from result.jld2
ExperimentConfig(...; persist = PersistPolicy(drop = [:x_iter]))

# or thin it: keep :x_iter on iter 0 and every 10th iteration
ExperimentConfig(...; persist = PersistPolicy(decimate = Dict(:x_iter => 10)))
```

Scalar metric columns are **never** touched — they're always full-resolution.
Only named extras keys are pruned, and only from `result.jld2`; CSV sidecars are
written from the full result and carry scalar extras regardless. What was pruned
is recorded in `manifest.json` (`persist_dropped_extras` / `persist_decimated`).

Measured impact (same payloads, `drop = [:x_iter]`):

| payload | keep all | drop `:x_iter` | + `compress = true` |
| --- | --- | --- | --- |
| low-dim, 20 000 iters | 3.7 MB | 1.15 MB | **0.58 MB** |
| high-dim, 5 000 iters | 40.9 MB | 0.31 MB | **0.17 MB** |

## JLD2 compression

`save_experiment` accepts a `compress` kwarg passed verbatim to `JLD2.save`
(`false` default; `true` for the built-in codec; or a specific
`TranscodingStreams` codec like `ZstdCompressor()`).

The default is off because compression only pays *after* the heavy extras are
gone. On the columnar layout with `:x_iter` still present, the vector-of-vectors
columns don't compress and the codec is a net loss (3.7 → 4.0 MB). Drop or
decimate `:x_iter` first, and then `compress = true` is worth it (1.15 → 0.58 MB
low-dim; 0.31 → 0.17 MB high-dim, per the table above).

CSV sidecars and `manifest.json` are never compressed — they exist to be
`grep`-able / `jq`-able. `load_experiment` reads any codec transparently (JLD2
detects it from the file header), so no matching kwarg is needed on load.

---
