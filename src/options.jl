module Options

export IpoptOptions
export PiccoloOptions
export set!

using Ipopt
using Libdl
using ExponentialAction
using Base: @kwdef

abstract type AbstractOptions end

"""
    Solver options for Ipopt

    https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_print_options_documentation
"""
@kwdef mutable struct IpoptOptions{T} <: AbstractOptions
    tol::T = 1e-8
    s_max::T = 100.0
    max_iter::Int = 1_000
    max_cpu_time = 1_000_000.0
    dual_inf_tol::T = 1.0
    constr_viol_tol::T = 1.0e-6
    compl_inf_tol::T = 1.0e-3
    acceptable_tol::T = 1.0e-6
    acceptable_iter::Int = 15
    acceptable_dual_inf_tol::T = 1.0e10
    acceptable_constr_viol_tol::T = 1.0e-2
    acceptable_compl_inf_tol::T = 1.0e-2
    acceptable_obj_change_tol::T = 1.0e-5
    diverging_iterates_tol::T = 1.0e8
    mu_target::T = 1.0e-4
    print_level::Int = 5
    output_file = nothing
    print_user_options = "no"
    print_options_documentation = "no"
    print_timing_statistics = "no"
    print_options_mode = "text"
    print_advanced_options = "no"
    print_info_string = "no"
    inf_pr_output = "original"
    print_frequency_iter = 1
    print_frequency_time = 0.0
    skip_finalize_solution_call = "no"
    hsllib = nothing
    hessian_approximation = "exact"
    recalc_y = "no"
    recalc_y_feas_tol = 1.0e-6
    linear_solver = "mumps"
    watchdog_shortened_iter_trigger = 10
    watchdog_trial_iter_max = 3
end


"""
    PiccoloOptions

Options for the Piccolo quantum optimal control library.

# Fields
- `verbose::Bool = true`: Print verbose output
- `verbose_evaluator::Bool = false`: Print verbose output from the evaluator
- `free_time::Bool = true`: Allow free time optimization
- `timesteps_all_equal::Bool = true`: Use equal timesteps
- `integrator::Symbol = :pade`: Integrator to use
- `pade_order::Int = 4`: Order of the Pade approximation
- `rollout_integrator::Function = expv`: Integrator to use for rollout
- `jacobian_structure::Bool = integrator == :pade`: Use the integrator's Jacobian structure
- `eval_hessian::Bool = false`: Evaluate the Hessian
- `geodesic = true`: Use the geodesic to initialize the optimization.
- `blas_multithreading::Bool = true`: Use BLAS multithreading.
- `build_trajectory_constraints::Bool = true`: Build trajectory constraints.
- `complex_control_norm_constraint_name::Union{Nothing, Symbol} = nothing`: Name of the complex control norm constraint.
- `complex_control_norm_constraint_radius::Float64 = 1.0`: Radius of the complex control norm constraint.
- `bound_state::Bool = false`: Bound the state.
- `leakage_suppression::Bool = false`: Suppress leakage.
- `R_leakage::Float64 = 1.0`: Leakage suppression parameter.
- `free_phase_infidelity::Bool = false`: Free phase infidelity.
- `phase_operators::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Complex}}} = nothing`: Phase operators.
- `phase_name::Symbol = :ϕ`: Name of the phase.
"""
@kwdef mutable struct PiccoloOptions <: AbstractOptions
    verbose::Bool = true
    verbose_evaluator::Bool = false
    free_time::Bool = true
    timesteps_all_equal::Bool = true
    integrator::Symbol = :pade
    pade_order::Int = 4
    jacobian_structure::Bool = integrator == :pade
    rollout_integrator::Function = expv
    eval_hessian::Bool = false
    geodesic = true
    blas_multithreading::Bool = true
    build_trajectory_constraints::Bool = true
    complex_control_norm_constraint_name::Union{Nothing, Symbol} = nothing
    complex_control_norm_constraint_radius::Float64 = 1.0
    bound_state::Bool = false
    leakage_suppression::Bool = false
    R_leakage::Float64 = 1.0
    free_phase_infidelity::Bool = false
    phase_operators::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Complex}}} = nothing
    phase_name::Symbol = :ϕ
end

function set!(optimizer::Ipopt.Optimizer, options::AbstractOptions)
    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name == :linear_solver
            if value == "pardiso"
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
            end
        end
        if !isnothing(value)
           optimizer.options[String(name)] = value
        end
    end
end

end
