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

## API

```julia
# Save — called automatically at the end of run_experiment.
# `compress = true` (default) writes the JLD2 binary with on-the-fly compression.
# `extra_manifest` merges its keys into manifest.json after the framework's own
# fields, so an experiment can record stage-specific results (milestone iters,
# tolerances, ...) for later `jq` access without loading the JLD2.
save_experiment(result::ExperimentResult;
                compress::Bool = true,
                extra_manifest::Dict{String,Any} = Dict{String,Any}())

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

Stage-specific summary data (e.g. Stage 4's iters-to-milestone and the
distance/gradient tolerances used) is delivered via `extra_manifest` and
merged after the base fields. Stage 4 uses this to record the milestone
threshold and per-method iters-to-milestone so cold-restart `jq` queries
can answer "did BB1 cross 1e-6 here?" without recomputing from the CSVs.

## JLD2 compression

`save_experiment` accepts a `compress` kwarg that is passed verbatim to
`JLD2.save`. Default is **`compress = false`** — the choice is empirical,
not stylistic.

**Why default off.** Measured on a single-method 20 000-iter Rosenbrock
iter-log payload, JLD2's built-in codec is a net loss:

| variant | size |
| --- | --- |
| `compress = false` | 19.3 MB |
| `compress = true` | 20.2 MB (**104.8%** — overhead exceeds savings) |

The MethodResult payload is dominated by `extras::Dict{Symbol,Any}` per
`IterationLog`, whose typing/dispatch overhead doesn't compress and adds
codec block headers. The scalar numeric columns (objective, gradient norm,
step norm, dist-to-opt, core time) are too thinly slotted across the
Dict-per-row structure for the codec to find repeated patterns.

**When to override.**

- Pass `compress = true` if you're storing problem families where the
  iter-log payload is denser numeric arrays (e.g. high-dim x_iter
  snapshots stored directly, not via Dict extras), where compression
  may actually pay off.
- Pass a specific `TranscodingStreams` codec (e.g. `ZstdCompressor()`
  from CodecZstd.jl) if you've benchmarked it against your payload and
  it wins.
- The kwarg exists precisely to keep that escape hatch available; the
  default just reflects what wins on the routine path *today*.

**What's NOT compressed.**

CSV sidecars and `manifest.json` are plain text — they exist precisely
to be `grep`-able / `jq`-able and compressing them would defeat that.

**Loading.** `load_experiment` reads either form transparently — JLD2
detects the codec from the file header, no matching kwarg needed. Old
files written under either default load without change.

**Future work — schema migration.** The realistic path to shrinking
`result.jld2` is to change the on-disk layout, not the codec.

- *Current (array-of-structs):* one `IterationLog` per iter, each carrying
  its own `Dict{Symbol,Any}` extras — JLD2 stores the dict's type machinery
  per row, which is what defeats the codec.
- *Proposed (struct-of-arrays per method):* one column-major struct per
  method holding `iter::Vector{Int}`, `objective::Vector{Float64}`,
  `gradient_norm::Vector{Float64}`, … plus a single
  `extras::Dict{Symbol,Vector{Any}}` keyed by extras name (one cell per
  iter, `missing` where absent).

Estimated payoff: **5–10×** on Rosenbrock-style payloads where the columns
are uniformly typed and densely populated. Cost: a versioned-manifest
persistence migration plus `to_dataframe` / `iter_logs` rewrites — deferred,
not blocking. (Listed under planned work in
[experiments/README.md](https://github.com/MoFirouzT/Iterative-Methods-Test-Engine/blob/main/experiments/README.md).)

---
