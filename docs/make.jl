using Documenter

# The architecture reference is hand-written Markdown under docs/src/ (sliced from the
# former monolithic architecture.md, one page per module). No @docs/@autodocs blocks,
# so we do not load the TestEngine project here — see docs/Project.toml.

const REPO = "github.com/MoFirouzT/Iterative-Methods-Test-Engine"

makedocs(
    sitename = "Iterative-Methods Test Engine",
    authors  = "Mohammad Firouztabar",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://MoFirouzT.github.io/Iterative-Methods-Test-Engine",
        edit_link  = "main",
        repolink   = "https://$REPO",
        assets     = String[],
        # This is a script-style project (no version in Project.toml); set an explicit
        # empty inventory version so HTMLWriter doesn't warn about the missing version.
        inventory_version = "",
    ),
    repo  = "https://$REPO/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Design Philosophy" => "design.md",
        "Modules" => [
            "Problem Interface"            => "modules/problem-interface.md",
            "Algorithm & Core Timing"      => "modules/algorithm-core.md",
            "Stopping Criteria"            => "modules/stopping-criteria.md",
            "Variant Grid Engine"          => "modules/variant-grid.md",
            "Nested Algorithms"            => "modules/nested-algorithms.md",
            "Logging & Verbosity"          => "modules/logging.md",
            "Experiment Orchestration"     => "modules/orchestration.md",
            "Persistence"                  => "modules/persistence.md",
            "Debug Mode"                   => "modules/debug-mode.md",
            "Analysis & Plotting"          => "modules/analysis-plotting.md",
        ],
        "Repository Internals" => "internals.md",
        "Extension Guide"      => "extending.md",
    ],
)

deploydocs(
    repo      = REPO,
    devbranch = "main",
)
