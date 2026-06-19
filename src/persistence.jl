"""
	Persistence & Experiment Naming

Writes and restores experiment outputs in three formats:
- Full binary: result.jld2  (everything; all Julia types preserved)
- Per-method per-run CSV sidecars  (scalar extras only — vectors stay in JLD2)
- Lightweight manifest.json metadata, including the list of extras keys that
  were dropped from the CSV so the omission is auditable.
"""

using Dates
using CSV
using DataFrames
using JLD2
using JSON3


# ── JLD2 serialization shim for VariantGrid ────────────────────────────────
# `VariantGrid.builder` is `Function` and is most often supplied as a closure
# (see experiments/stages/stage5.jl: `(; step_size, kwargs...) -> GradientDescent(...)`).
# JLD2 cannot reliably round-trip closures (`#14#15` etc.) and emits a
# warning during `save_experiment`. Round-tripped closures would also be
# unsafe to call — the captured environment is gone.
#
# The fix: serialize a stripped-down record that drops `builder` and
# `filters` (both `Function`-typed). The expanded `VariantSpec`s inside the
# `ExperimentResult` already carry concrete `IterativeMethod` instances, so
# nothing downstream of `expand(grid)` needs the closure. On load, the grid
# is reconstructed with a sentinel builder that errors loudly if anyone
# tries to re-expand it post-mortem.
struct _SerializedVariantGrid
	base_name::String
	axes::Vector{VariantAxis}
	shared_params::NamedTuple
	role::Symbol
end

JLD2.writeas(::Type{VariantGrid}) = _SerializedVariantGrid

Base.convert(::Type{_SerializedVariantGrid}, g::VariantGrid) =
	_SerializedVariantGrid(g.base_name, g.axes, g.shared_params, g.role)

_deserialized_grid_builder(args...; kwargs...) = error(
	"VariantGrid.builder was not serialized — re-construct the grid " *
	"in-process if you need to re-expand it. The expanded VariantSpecs " *
	"inside ExperimentResult already carry concrete methods; consume those " *
	"instead.")

Base.convert(::Type{VariantGrid}, s::_SerializedVariantGrid) = VariantGrid(
	base_name     = s.base_name,
	axes          = s.axes,
	builder       = _deserialized_grid_builder,
	shared_params = s.shared_params,
	role          = s.role,
)


# ── Columnar (struct-of-arrays) serialization shim for MethodResult ─────────
# In memory a MethodResult holds `Vector{IterationLog}`, one Dict-bearing struct
# per iteration. Serializing that directly makes JLD2 store the Dict's type
# machinery per row, which both bloats the file and defeats the compressor
# (measured: a 2-D 20k-iter payload is 15.5 MB and *grows* under zlib). Instead
# `result.jld2` stores the data column-major — parallel typed vectors for the
# fixed metrics plus one vector per extras key — which is ~4× smaller and lets
# compression pay. The conversion is invisible to callers: `load_experiment`
# reconstructs the identical `Vector{IterationLog}`.
#
# Absent extras: a key present on only some iters gets an `_Absent` sentinel in
# the rows that lack it, so reconstruction restores exactly which entries had the
# key — distinct from a genuine `missing` *value*, which round-trips unchanged.
struct _Absent end
const _ABSENT = _Absent()

struct _SerializedMethodResult
	method_name::String
	iter::Vector{Int}
	core_time_ns::Vector{Int64}
	objective::Vector{Float64}
	gradient_norm::Vector{Float64}
	step_norm::Vector{Float64}
	dist_to_opt::Vector{Float64}
	extras::Dict{Symbol,Vector}        # key => per-iter column (_ABSENT where absent)
	final_state::Any
	stop_reason::Symbol
	n_iters::Int
	events::Vector{NamedTuple}
end

# Build one column per extras key. Keys present on every iter become a narrowly
# typed vector (e.g. Vector{Float64}, Vector{Vector{Float64}}) that compresses
# well; sparse keys fall back to a sentinel-padded Vector{Any}.
function _columnize_extras(iter_logs::Vector{IterationLog})
	all_keys = Set{Symbol}()
	for e in iter_logs
		union!(all_keys, keys(e.extras))
	end
	cols = Dict{Symbol,Vector}()
	n = length(iter_logs)
	for k in all_keys
		if all(e -> haskey(e.extras, k), iter_logs)
			cols[k] = identity.([e.extras[k] for e in iter_logs])   # narrows eltype
		else
			col = Vector{Any}(undef, n)
			for (i, e) in enumerate(iter_logs)
				col[i] = haskey(e.extras, k) ? e.extras[k] : _ABSENT
			end
			cols[k] = col
		end
	end
	return cols
end

function _decolumnize_extras(cols::Dict{Symbol,Vector}, n::Int)
	out = [Dict{Symbol,Any}() for _ in 1:n]
	for (k, col) in cols
		for i in 1:n
			v = col[i]
			v === _ABSENT && continue
			out[i][k] = v
		end
	end
	return out
end

JLD2.writeas(::Type{MethodResult}) = _SerializedMethodResult

function Base.convert(::Type{_SerializedMethodResult}, m::MethodResult)
	logs = m.iter_logs
	_SerializedMethodResult(
		m.method_name,
		[e.iter         for e in logs],
		[e.core_time_ns for e in logs],
		[e.objective    for e in logs],
		[e.gradient_norm for e in logs],
		[e.step_norm    for e in logs],
		[e.dist_to_opt  for e in logs],
		_columnize_extras(logs),
		m.final_state,
		m.stop_reason,
		m.n_iters,
		m.events,
	)
end

function Base.convert(::Type{MethodResult}, s::_SerializedMethodResult)
	n = length(s.iter)
	extras = _decolumnize_extras(s.extras, n)
	logs = [IterationLog(
				iter          = s.iter[i],
				core_time_ns  = s.core_time_ns[i],
				objective     = s.objective[i],
				gradient_norm = s.gradient_norm[i],
				step_norm     = s.step_norm[i],
				dist_to_opt   = s.dist_to_opt[i],
				extras        = extras[i],
			) for i in 1:n]
	return MethodResult(s.method_name, logs, s.final_state, s.stop_reason,
	                    s.n_iters, s.events)
end


# ── CSV scalar predicate and classifier ────────────────────────────────────
# A value is "CSV-friendly" if CSV.write can render it as a single cell
# without information loss. Composite values (Vector, Tuple, Dict, sub-logs,
# custom structs) have no stable text representation and are dropped from
# the CSV; they remain in result.jld2.
#
# Extension point: define `_is_csv_scalar(::YourType) = true` if you add a
# new extras type that should be CSV-preserved (e.g. `Dates.DateTime`).
_is_csv_scalar(x::Number)         = true   # covers Int, Float, Bool, Rational, ...
_is_csv_scalar(x::AbstractString) = true
_is_csv_scalar(x::Symbol)         = true
_is_csv_scalar(::Missing)         = true
_is_csv_scalar(::Nothing)         = true
_is_csv_scalar(x)                 = false


"""
	_classify_extras(iter_logs) -> (scalar_keys, vector_keys)

Splits the union of `extras` keys across `iter_logs` into those whose values
are CSV-scalar on every entry that has them, and those that are not. The rule
is all-or-nothing per key: a single non-scalar occurrence demotes the whole
column to JLD2-only, so we never write half a CSV column.
"""
function _classify_extras(iter_logs::Vector{IterationLog})
	all_keys = Set{Symbol}()
	for entry in iter_logs
		union!(all_keys, keys(entry.extras))
	end
	scalar_keys = Symbol[]
	vector_keys = Symbol[]
	for k in sort!(collect(all_keys))
		ok = all(iter_logs) do entry
			!haskey(entry.extras, k) || _is_csv_scalar(entry.extras[k])
		end
		push!(ok ? scalar_keys : vector_keys, k)
	end
	return scalar_keys, vector_keys
end


function _csv_row(run_id::Int, method_name::String, entry::IterationLog,
                       scalar_keys::Vector{Symbol})
	row = Dict{Symbol,Any}(
		:run_id => run_id,
		:method_name => method_name,
		:iter => entry.iter,
		:core_time_ns => entry.core_time_ns,
		:objective => entry.objective,
		:gradient_norm => entry.gradient_norm,
		:step_norm => entry.step_norm,
		:dist_to_opt => entry.dist_to_opt,
	)

	for k in scalar_keys
		row[k] = get(entry.extras, k, missing)
	end

	return row
end


"""
	_methodresult_dataframe(run_id, method_result) -> (df, vector_keys)

Builds a DataFrame for one method's run, dropping vector-valued extras.
Returns the dropped keys so save_experiment can aggregate them for the
manifest. Column order is stable: fixed fields first, scalar extras after.
"""
function _methodresult_dataframe(run_id::Int, method_result::MethodResult)
	iter_logs = method_result.iter_logs
	scalar_keys, vector_keys = _classify_extras(iter_logs)
	rows = [_csv_row(run_id, method_result.method_name, entry, scalar_keys)
	        for entry in iter_logs]

	if isempty(rows)
		return DataFrame(), vector_keys
	end

	df = DataFrame(rows)
	fixed = [:run_id, :method_name, :iter, :core_time_ns, :objective,
	         :gradient_norm, :step_norm, :dist_to_opt]
	df = df[!, vcat(fixed, scalar_keys)]

	return df, vector_keys
end


function _sanitize_filename(name::String)
	replace(name, r"[^A-Za-z0-9._\-\[\]\(\)=]+" => "_")
end


function _write_csv_sidecars(result::ExperimentResult)
	skipped = Set{Symbol}()
	# Method names are unique per run (Dict keys), but `_sanitize_filename` can
	# collapse two distinct names to the same on-disk file. Detect that and fail
	# loudly rather than silently overwrite one method's run with another's.
	written = Set{String}()
	for run_result in result.run_results
		for (_, method_result) in run_result.method_results
			df, vector_keys = _methodresult_dataframe(run_result.run_id, method_result)
			union!(skipped, vector_keys)
			csv_name = string("run", run_result.run_id, "_",
			                   _sanitize_filename(method_result.method_name), ".csv")
			csv_path = joinpath(result.experiment_path, csv_name)
			if csv_path in written
				error("CSV sidecar name collision on $(repr(csv_name)): two " *
				      "method names sanitize to the same file. Rename a method " *
				      "so its on-disk artifact stays distinct.")
			end
			push!(written, csv_path)
			CSV.write(csv_path, df)
		end
	end
	return sort!(collect(skipped))
end


# Per-method, per-run termination summary derived from MethodResult and the
# final IterationLog entry. Embedded in every manifest so `jq '.method_results.BB1[0]'`
# is enough to know how a method terminated without loading result.jld2.
function _method_results_summary(result::ExperimentResult)
	out = Dict{String, Vector{Dict{String, Any}}}()
	for run_result in result.run_results
		for (name, mres) in run_result.method_results
			isempty(mres.iter_logs) && continue
			last_entry = mres.iter_logs[end]
			entry = Dict{String, Any}(
				"run_id"       => run_result.run_id,
				"n_iters"      => mres.n_iters,
				"stop_reason"  => string(mres.stop_reason),
				"f_final"      => last_entry.objective,
				"grad_final"   => last_entry.gradient_norm,
				"dist_final"   => last_entry.dist_to_opt,
			)
			push!(get!(out, name, Dict{String, Any}[]), entry)
		end
	end
	# Stable ordering inside each method's vector.
	for v in values(out)
		sort!(v, by = e -> e["run_id"])
	end
	return out
end


function _manifest_payload(result::ExperimentResult, skipped_extras::Vector{Symbol})
	methods = String[]
	if !isempty(result.run_results)
		methods = sort(collect(keys(result.run_results[1].method_results)))
	end

	manifest = Dict{String,Any}(
		"name" => result.config.name,
		"timestamp" => string(result.timestamp),
		"host" => result.host,
		"n_runs" => length(result.run_results),
		"n_methods" => length(methods),
		"methods" => methods,
		"tags" => result.config.tags,
		"method_results" => _method_results_summary(result),
	)

	if !isempty(skipped_extras)
		manifest["csv_skipped_extras"] = sort!(string.(skipped_extras))
		manifest["csv_skipped_extras_note"] = (
			"These extras keys are non-scalar (vectors, sub-logs, ...) " *
			"and are stored in result.jld2 only. Reload via load_experiment " *
			"for the full payload."
		)
	end

	return manifest
end


# Write via a temp sibling then atomically rename, so a crash mid-write never
# leaves a truncated `result.jld2` (which `load_experiment` would choke on) or a
# half-written `manifest.json` (which `list_experiments` keys off). `tempname`
# places the temp inside the destination directory, so the rename stays on one
# filesystem and is therefore atomic. CSV sidecars are written directly — they
# are per-method grep artifacts, individually self-evident if truncated, and not
# load-bearing for reload or indexing.
function _atomic_write(f::Function, path::String)
	# Keep the original extension: `JLD2.save` dispatches on it via FileIO, and
	# a temp without `.jld2` raises "No applicable_savers found".
	tmp = string(tempname(dirname(path); cleanup = false), splitext(path)[2])
	try
		f(tmp)
		mv(tmp, path; force = true)
	catch
		isfile(tmp) && rm(tmp; force = true)
		rethrow()
	end
	return nothing
end


# ── Save-time extras pruning (PersistPolicy) ────────────────────────────────
# Produce a copy of `result` whose iter-log `extras` have been pruned per the
# policy, leaving every scalar metric column untouched. Returns `result`
# unchanged (no copy) when the policy is a no-op, so the common path is free.
_persist_is_noop(p::PersistPolicy) = isempty(p.drop) && isempty(p.decimate)

# iter 0 (the init/warm-up row) is always kept for any decimated key so the
# trajectory still has a starting point; otherwise keep iters where iter % k == 0.
function _prune_extras(extras::Dict{Symbol,Any}, iter::Int, policy::PersistPolicy)
	out = Dict{Symbol,Any}()
	for (k, v) in extras
		k in policy.drop && continue
		if haskey(policy.decimate, k)
			k_keep = policy.decimate[k]
			(iter == 0 || iter % k_keep == 0) || continue
		end
		out[k] = v
	end
	return out
end

function _apply_persist_policy(result::ExperimentResult, policy::PersistPolicy)
	_persist_is_noop(policy) && return result

	new_runs = RunResult[]
	for rr in result.run_results
		new_methods = Dict{String,MethodResult}()
		for (name, m) in rr.method_results
			new_logs = [IterationLog(
							iter          = e.iter,
							core_time_ns  = e.core_time_ns,
							objective     = e.objective,
							gradient_norm = e.gradient_norm,
							step_norm     = e.step_norm,
							dist_to_opt   = e.dist_to_opt,
							extras        = _prune_extras(e.extras, e.iter, policy),
						) for e in m.iter_logs]
			new_methods[name] = MethodResult(m.method_name, new_logs, m.final_state,
			                                 m.stop_reason, m.n_iters, m.events)
		end
		push!(new_runs, RunResult(rr.run_id, new_methods))
	end
	return ExperimentResult(result.config, result.experiment_path,
	                        result.timestamp, result.host, new_runs)
end


# JSON has no Inf/NaN. Map non-finite reals to `nothing` (→ null) so manifests
# stay valid for problems without a known optimum (where dist_to_opt = Inf).
_json_safe(x::AbstractFloat)  = isfinite(x) ? x : nothing
_json_safe(x::AbstractDict)   = Dict{String,Any}(string(k) => _json_safe(v) for (k, v) in x)
_json_safe(x::AbstractVector) = Any[_json_safe(v) for v in x]
_json_safe(x) = x


"""
	save_experiment(result::ExperimentResult;
	                compress = false,
	                extra_manifest = Dict{String,Any}())

Writes:
- `result.jld2`  (full binary; uncompressed by default — see `compress`)
- `run{N}_{method}.csv`  (scalar extras only)
- `manifest.json`  (metadata, base `method_results`, CSV-skipped extras keys,
   and any keys from `extra_manifest`)

CSVs are written before the manifest so the manifest can record which
extras were dropped.

# Keyword arguments
- `compress` — controls JLD2 compression for `result.jld2`. `false`
  (default) writes uncompressed; `true` uses JLD2's built-in codec; a
  specific `TranscodingStreams` codec value is also accepted and passed
  through verbatim. **Default is `false`**: with the iter logs stored
  column-major, the remaining bulk is `:x_iter`-style
  vector-of-vectors columns, which don't compress and where the codec is a
  net loss. Compression only pays once those heavy extras are removed via
  `persist` — then `compress = true` is worth it. See
  [docs/src/modules/persistence.md — JLD2 compression] for measurements.
  `load_experiment` reads either form transparently — no matching kwarg required.
- `extra_manifest::Dict{String,Any}` — keys merged into `manifest.json`
  after the framework's own fields. Use this to record stage-specific
  results (e.g. iters-to-milestone, tolerances used) so cold-restart
  queries can answer questions without recomputing from CSVs. Keys in
  `extra_manifest` override the base fields if they collide — caller's
  problem if that's not intended.
- `persist::PersistPolicy` — save-time pruning of heavy per-iteration extras
  (e.g. `drop = [:x_iter]`) from `result.jld2`. The default keeps everything.
  CSV sidecars are unaffected (they carry only scalar extras and are written
  from the full result); the JLD2 binary honours the policy and the manifest
  records what was pruned under `persist_dropped_extras` / `persist_decimated`.
"""
function save_experiment(result::ExperimentResult;
                         compress = false,
                         extra_manifest::Dict{String,Any} = Dict{String,Any}(),
                         persist::PersistPolicy = PersistPolicy())
	mkpath(result.experiment_path)

	# Prune heavy extras for the binary only; CSVs below use the full result.
	to_store = _apply_persist_policy(result, persist)

	jld_path = joinpath(result.experiment_path, "result.jld2")
	_atomic_write(jld_path) do tmp
		JLD2.save(tmp, Dict("result" => to_store); compress = compress)
	end

	skipped = _write_csv_sidecars(result)

	manifest = _manifest_payload(result, skipped)
	if !isempty(persist.drop)
		manifest["persist_dropped_extras"] = sort!(string.(persist.drop))
	end
	if !isempty(persist.decimate)
		manifest["persist_decimated"] =
			Dict(string(k) => v for (k, v) in persist.decimate)
	end
	merge!(manifest, extra_manifest)

	manifest_path = joinpath(result.experiment_path, "manifest.json")
	_atomic_write(manifest_path) do tmp
		open(tmp, "w") do io
			JSON3.pretty(io, _json_safe(manifest))
		end
	end

	return nothing
end


"""
	load_experiment(path::String) -> ExperimentResult

Loads an experiment from `path/result.jld2`.
"""
function load_experiment(path::String)
	jld_path = joinpath(path, "result.jld2")
	data = JLD2.load(jld_path)
	return data["result"]
end


"""
	load_manifest(path::String)

Loads metadata from a manifest path.
"""
function load_manifest(path::String)
	return JSON3.read(read(path, String))
end


function _parse_experiment_id(path::String)
	number = try
		parse(Int, basename(path))
	catch
		0
	end

	date = basename(dirname(path))
	return date, number
end


function _manifest_get(manifest, key::Symbol, default)
	if hasproperty(manifest, key)
		return getproperty(manifest, key)
	end
	return default
end


"""
	list_experiments(log_root::String = "logs")

Returns a vector of metadata entries:
`(path, date, number, name, timestamp, n_methods, n_runs)`
"""
function list_experiments(log_root::String = "logs")
	isdir(log_root) || return NamedTuple[]

	out = NamedTuple[]
	for date_dir in readdir(log_root; join = true)
		isdir(date_dir) || continue
		date_name = basename(date_dir)
		occursin(r"^\d{8}$", date_name) || continue

		for exp_dir in readdir(date_dir; join = true)
			isdir(exp_dir) || continue
			exp_name = basename(exp_dir)
			occursin(r"^\d{3,}$", exp_name) || continue

			manifest_path = joinpath(exp_dir, "manifest.json")
			if isfile(manifest_path)
				manifest = load_manifest(manifest_path)
				push!(out, (
					path = exp_dir,
					date = String(date_name),
					number = parse(Int, exp_name),
					name = String(_manifest_get(manifest, :name, "")),
					timestamp = String(_manifest_get(manifest, :timestamp, "")),
					n_methods = Int(_manifest_get(manifest, :n_methods, 0)),
					n_runs = Int(_manifest_get(manifest, :n_runs, 0)),
				))
			else
				date, number = _parse_experiment_id(exp_dir)
				push!(out, (
					path = exp_dir,
					date = date,
					number = number,
					name = "",
					timestamp = "",
					n_methods = 0,
					n_runs = 0,
				))
			end
		end
	end

	sort!(out, by = x -> (x.date, x.number))
	return out
end
