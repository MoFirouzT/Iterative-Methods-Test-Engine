#!/usr/bin/env julia
# scripts/reproduce.jl — one command from `git clone` to every portfolio figure.
#
#     julia --project scripts/reproduce.jl
#
# Runs each portfolio experiment in its own process (so the scripts can't clash
# on globals) and writes the figures into figures/. The lasso flagship figure is
# produced first. The first run pays Julia's compile/load cost for each script;
# subsequent figures are quicker.

const PROJECT = dirname(@__DIR__)

# Ordered so the flagship figure (lasso) lands first.
const SCRIPTS = [
    ("experiments/exp_lasso_ista_fista.jl",  "figures/lasso_ista_fista.png   (★ flagship figure: ISTA vs FISTA + support recovery)"),
    ("experiments/exp_ls1_dimension.jl",     "figures/ls1_dimension.png      (dimension scaling + timing pillar)"),
    ("experiments/exp_ls2_conditioning.jl",  "figures/ls2_conditioning.png   (GD rate vs conditioning: slope 1 vs √κ)"),
    ("experiments/exp_precond_grid.jl",      "figures/precond_grid.png       (Jacobi ≈ Newton: variant-grid sweep)"),
    ("experiments/exp_tr_steihaug_cg.jl",    "figures/tr_steihaug_cg.png     (trust-region + Steihaug-CG: nested optimization)"),
]

function main()
    @info "Reproducing portfolio figures into figures/ — $(length(SCRIPTS)) experiments"
    t0 = time()
    for (i, (script, desc)) in enumerate(SCRIPTS)
        @info "[$i/$(length(SCRIPTS))] ▶ $script"
        println("        → $desc")
        path = joinpath(PROJECT, script)
        run(`$(Base.julia_cmd()) --project=$PROJECT $path`)
    end
    @info "✓ done in $(round(time() - t0, digits = 1))s — see figures/"
end

main()
