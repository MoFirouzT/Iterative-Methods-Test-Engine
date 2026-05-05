module TestEngine

include("logging.jl")
include("problems.jl")
include("core.jl")
include("stopping.jl")
include("variants.jl")

include(joinpath(@__DIR__, "..", "algorithms", "conventional", "components", "descent_directions.jl"))
include(joinpath(@__DIR__, "..", "algorithms", "conventional", "components", "step_sizes.jl"))
include(joinpath(@__DIR__, "..", "algorithms", "conventional", "gradient_descent.jl"))

include(joinpath(@__DIR__, "..", "problems", "rosenbrock", "rosenbrock.jl"))

include("experiment.jl")
include("persistence.jl")
include("analysis.jl")

export Objective, Regularizer, Hessian, MatrixHessian, OperatorHessian, DiagonalHessian
export value, grad, grad!, hessian, hessian_vec, apply, materialize, diagonal
export Problem, total_objective, objective, ProblemSpec, AnalyticProblem, FileProblem, RandomProblem
export register_analytic_problem!, register_random_problem!, make_problem

export IterativeMethod, ConventionalMethod, ExperimentalMethod
export IterateGroup, MetricsGroup, TimingGroup
export init_state, step!, extract_log_entry, run_method, run_sub_method
export SubRunConfig, SubResult, @core_timed

export StoppingCriterion, MaxIterations, TimeLimit, GradientTolerance
export ObjectiveStagnation, StepTolerance, CompositeCriterion, stop_when_any, stop_when_all, should_stop

export DescentDirection, SteepestDescent, StepSize, LineSearch
export FixedStep, ArmijoLS, WolfeLS, CauchyStep, BarzilaiBorwein
export GradientDescent, GradientDescentNumerics, GradientDescentState
export compute_direction, compute_step_size

export RosenbrockKernel, RosenbrockObjective

export HessianApprox, FullHessian, BFGS, SR1, LBFGS, DiagBFGS
export MinorUpdate, NoMinorUpdate, MomentumStep, NesterovStep, CorrectionStep
export VariantAxis, VariantGrid, VariantSpec, expand

export VerbosityLevel, VerbosityConfig, IterationLog, Logger
export log_init!, log_iter!, log_event!, attach_sub_logs!, finalize!
export elapsed_core_s, elapsed_wall_s

export ExperimentConfig, MethodResult, RunResult, ExperimentResult
export resolve_methods, next_experiment_path, run_experiment

export save_experiment, load_experiment, load_manifest, list_experiments

export MethodStyle, METHOD_PALETTE, METHOD_COLOR_REGISTRY
export method_color, register_method_color!, get_method_color
export PlotSpec, FigureLayout, render_figure, save_figure
export to_dataframe, filter_methods, aggregate_runs

end
