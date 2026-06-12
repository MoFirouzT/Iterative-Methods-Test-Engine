module TestEngine

include("logging.jl")
include("problems.jl")
include("core.jl")
include("stopping.jl")
include("variants.jl")

# NOTE: concrete methods and problems are deliberately NOT included here.
# The engine module ships only abstractions + machinery + dependency-free
# utilities. Content (algorithms/, problems/) is loaded by experiments and
# tests via experiments/_bootstrap.jl and extends the engine through
# `import .TestEngine`. This keeps the engine standalone and dependency-lean.

include("debug.jl")
include("experiment.jl")
include("persistence.jl")
include("analysis.jl")

export Objective, Regularizer, Hessian, MatrixHessian, OperatorHessian, DiagonalHessian
export value, grad, grad!, hessian, hessian_vec, apply, materialize, diagonal
export Problem, total_objective, objective, ProblemSpec, AnalyticProblem, FileProblem, RandomProblem
export register_analytic_problem!, register_random_problem!, register_file_loader!, make_problem

export IterativeMethod, ConventionalMethod, ExperimentalMethod
export IterateGroup, MetricsGroup, TimingGroup
export init_state, step!, extract_log_entry, run_method, run_sub_method
export SubRunConfig, SubResult, @core_timed

export StoppingCriterion, MaxIterations, TimeLimit, GradientTolerance
export ObjectiveStagnation, StepTolerance, DistanceToOptimal, CompositeCriterion, stop_when_any, stop_when_all, should_stop
export NegativeCurvature, TrustRegionBoundary, _tr_status

# Concrete methods, their components (StepSize/DescentDirection/MinorUpdate/
# HessianApprox + concretes), GradientDescent, and concrete problems
# (Rosenbrock, LeastSquares, regularizers) are CONTENT — defined in algorithms/
# and problems/, loaded via experiments/_bootstrap.jl. Not exported here.

export VariantAxis, VariantGrid, VariantSpec, expand
export ABBREVIATIONS, abbreviate, register_abbreviation!

export VerbosityLevel, VerbosityConfig, IterationLog, Logger
export log_init!, log_iter!, log_event!, attach_sub_logs!, finalize!
export SILENT, MILESTONE, SUMMARY, DETAILED, DEBUG
export make_logger
export elapsed_core_s, elapsed_wall_s

export ExperimentConfig, MethodResult, RunResult, ExperimentResult
export resolve_methods, next_experiment_path, run_experiment
export WarmupStrategy, NoWarmup, IterativeWarmup, FunctionWarmup
export register_warmup!, run_warmup, WARMUP_FUNCTIONS

export DebugConfig, DebugCheck
export CheckObjectiveMonotonicity, CheckGradientNormBound, CheckStepDecay, CheckNumericalGradient
export run_debug_checks!, trigger_debug!, numerical_gradient

export save_experiment, load_experiment, load_manifest, list_experiments

export MethodStyle, METHOD_PALETTE, METHOD_COLOR_REGISTRY
export method_color, register_method_color!, get_method_color
export PlotSpec, FigureLayout, render_plot!, render_figure, save_figure
export to_dataframe, filter_methods, aggregate_runs

end
