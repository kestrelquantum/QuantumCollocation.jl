module Problems

export AbstractProblem
export FixedTimeProblem
export QuantumControlProblem

export set_trajectory!
export update_trajectory!
export get_objective
export get_constraints

using ..QuantumSystems
using ..Integrators
using ..Evaluators
using ..Options
using ..Constraints
using ..Dynamics
using ..Objectives

using TrajectoryIndexingUtils
using NamedTrajectories

using LinearAlgebra
using JLD2
using Ipopt
using TestItemRunner
using MathOptInterface
using LinearAlgebra
const MOI = MathOptInterface

abstract type AbstractProblem end

"""
    mutable struct QuantumControlProblem <: AbstractProblem

Stores all the information needed to set up and solve a QuantumControlProblem as well as the solution
after the solver terminates.

# Fields
- `optimizer::Ipopt.Optimizer`: Ipopt optimizer object
"""
mutable struct QuantumControlProblem <: AbstractProblem
    optimizer::Ipopt.Optimizer
    variables::Vector{MOI.VariableIndex}
    system::AbstractQuantumSystem
    trajectory::NamedTrajectory
    integrators::Union{Nothing,Vector{<:AbstractIntegrator}}
    ipopt_options::IpoptOptions
    piccolo_options::PiccoloOptions
    params::Dict{Symbol, Any}
end

function QuantumControlProblem(
    system::AbstractQuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    dynamics::QuantumDynamics;
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    scale_factor_objective::Float64=1.0,
    additional_objective::Union{Nothing, Objective}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    params::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    return_evaluator=false,
    kwargs...
)
    # Save internal copy of the options to allow modification
    ipopt_options = deepcopy(ipopt_options)
    piccolo_options = deepcopy(piccolo_options)

    if !piccolo_options.blas_multithreading
        BLAS.set_num_threads(1)
    end

    if !piccolo_options.eval_hessian
        ipopt_options.hessian_approximation = "limited-memory"
    end

    nonlinear_constraints = NonlinearConstraint[con for con ∈ constraints if con isa NonlinearConstraint]

    if scale_factor_objective != 1
        obj = scale_factor_objective * obj
    end

    if !isnothing(additional_objective)
        obj += additional_objective
    end

    if piccolo_options.verbose
        println("    building evaluator...")
    end

    evaluator = PicoEvaluator(
        traj,
        obj,
        dynamics,
        nonlinear_constraints;
        eval_hessian=piccolo_options.eval_hessian,
    )

    if return_evaluator
        return evaluator
    end

    n_dynamics_constraints = dynamics.dim * (traj.T - 1)
    n_variables = traj.dim * traj.T

    # add globabl variables to n_variables
    for global_var ∈ keys(traj.global_data)
        global_var_dim = length(traj.global_data[global_var])
        n_variables += global_var_dim
    end

    linear_constraints = LinearConstraint[con for con ∈ constraints if con isa LinearConstraint]

    if piccolo_options.build_trajectory_constraints
        linear_constraints = LinearConstraint[trajectory_constraints(traj); linear_constraints]
    end

    optimizer = Ipopt.Optimizer()

    if piccolo_options.verbose
        println("    initializing optimizer...")
    end

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        traj,
        linear_constraints,
        n_dynamics_constraints,
        nonlinear_constraints,
        n_variables,
        verbose=piccolo_options.verbose
    )

    # Container for saving constraints and objectives
    params = merge(kwargs, params)
    params[:linear_constraints] = linear_constraints
    params[:nonlinear_constraints] = [
        nl_constraint.params for nl_constraint ∈ nonlinear_constraints
    ]
    params[:objective_terms] = obj.terms

    return QuantumControlProblem(
        optimizer,
        variables,
        system,
        traj,
        dynamics.integrators,
        ipopt_options,
        piccolo_options,
        params
    )
end

function QuantumControlProblem(
    system::AbstractQuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator};
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
    # Save internal copy of the options to allow modification
    ipopt_options = deepcopy(ipopt_options)
    piccolo_options = deepcopy(piccolo_options)

    if piccolo_options.verbose
        println("    building dynamics from integrators...")
    end
    dynamics = QuantumDynamics(integrators, traj;
        jacobian_structure=piccolo_options.jacobian_structure,
        eval_hessian=piccolo_options.eval_hessian,
        verbose=piccolo_options.verbose
    )
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        piccolo_options=piccolo_options,
        kwargs...
    )
end

# constructor that accepts just an AbstractIntegrator
function QuantumControlProblem(
    system::AbstractQuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrator::AbstractIntegrator;
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
    # Save internal copy of the options to allow modification
    ipopt_options = deepcopy(ipopt_options)
    piccolo_options = deepcopy(piccolo_options)

    if piccolo_options.verbose
        println("    building dynamics from integrator...")
    end
    dynamics = QuantumDynamics(integrator, traj;
        jacobian_structure=piccolo_options.jacobian_structure,
        eval_hessian=piccolo_options.eval_hessian,
        verbose=piccolo_options.verbose
    )
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        piccolo_options=piccolo_options,
        kwargs...
    )
end

function QuantumControlProblem(
    system::AbstractQuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    f::Function;
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
    # Save internal copy of the options to allow modification
    ipopt_options = deepcopy(ipopt_options)
    piccolo_options = deepcopy(piccolo_options)

    if piccolo_options.verbose
        println("    building dynamics from function...")
    end
    dynamics = QuantumDynamics(f, traj;
        jacobian_structure=piccolo_options.jacobian_structure,
        eval_hessian=piccolo_options.eval_hessian,
        verbose=piccolo_options.verbose
    )
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        piccolo_options=piccolo_options,
        kwargs...
    )
end

function initialize_optimizer!(
    optimizer::Ipopt.Optimizer,
    evaluator::PicoEvaluator,
    trajectory::NamedTrajectory,
    linear_constraints::Vector{LinearConstraint},
    n_dynamics_constraints::Int,
    nonlinear_constraints::Vector{NonlinearConstraint},
    n_variables::Int;
    verbose=true
)
    nl_cons = fill(
        MOI.NLPBoundsPair(0.0, 0.0),
        n_dynamics_constraints
    )

    for nl_con ∈ nonlinear_constraints
        if nl_con isa NonlinearEqualityConstraint
            append!(nl_cons, fill(MOI.NLPBoundsPair(0.0, 0.0), nl_con.dim))
        elseif nl_con isa NonlinearInequalityConstraint
            append!(nl_cons, fill(MOI.NLPBoundsPair(0.0, Inf), nl_con.dim))
        else
            error("Unknown nonlinear constraint type")
        end
    end

    # build NLP block data
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)

    # set NLP block data
    MOI.set(optimizer, MOI.NLPBlock(), block_data)

    # set objective sense: minimize
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add variables
    variables = MOI.add_variables(optimizer, n_variables)

    # add linear constraints
    constrain!(optimizer, variables, linear_constraints, trajectory, verbose=verbose)

    return variables
end

function set_trajectory!(
    prob::QuantumControlProblem,
    traj::NamedTrajectory
)
    # initialize n variables with trajectory data
    n_vars = traj.dim * traj.T

    # set trajectory data
    MOI.set(
        prob.optimizer,
        MOI.VariablePrimalStart(),
        prob.variables[1:n_vars],
        collect(traj.datavec)
    )

    # set global variables
    for global_vars_i ∈ values(traj.global_data)
        n_global_vars = length(global_vars_i)
        MOI.set(
            prob.optimizer,
            MOI.VariablePrimalStart(),
            prob.variables[n_vars .+ (1:n_global_vars)],
            global_vars_i
        )
        n_vars += n_global_vars
    end
end

set_trajectory!(prob::QuantumControlProblem) =
    set_trajectory!(prob, prob.trajectory)

function get_datavec(prob::QuantumControlProblem)
    n_vars = prob.trajectory.dim * prob.trajectory.T

    # get trajectory data
    return MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        prob.variables[1:n_vars]
    )
end

function get_global_data(prob::QuantumControlProblem)

    # get global variables after trajectory data
    global_keys = keys(prob.trajectory.global_data)
    global_values = []
    n_vars = prob.trajectory.dim * prob.trajectory.T
    for global_var ∈ global_keys
        n_global_vars = length(prob.trajectory.global_data[global_var])
        push!(global_values, MOI.get(
            prob.optimizer,
            MOI.VariablePrimal(),
            prob.variables[n_vars .+ (1:n_global_vars)]
        ))
        n_vars += n_global_vars
    end
    return (; (global_keys .=> global_values)...)
end

@views function update_trajectory!(prob::QuantumControlProblem)
    datavec = get_datavec(prob)
    global_data = get_global_data(prob)
    prob.trajectory = NamedTrajectory(datavec, global_data, prob.trajectory)
    return nothing
end

"""
    get_objective(prob::QuantumControlProblem)

Return the objective function of the `prob::QuantumControlProblem`.
"""
function get_objective(
    prob::QuantumControlProblem;
    match::Union{Nothing, AbstractVector{<:Symbol}}=nothing,
    invert::Bool=false
)
    objs = deepcopy(prob.params[:objective_terms])
    if isnothing(match)
        return Objective(objs)
    else
        if invert
            return Objective([term for term ∈ objs if term[:type] ∉ match])
        else
            return Objective([term for term ∈ objs if term[:type] ∈ match])
        end
    end
end

"""
    get_constraints(prob::QuantumControlProblem)

Return the constraints of the `prob::QuantumControlProblem`.
"""
function get_constraints(prob::QuantumControlProblem)
    linear_constraints = deepcopy(prob.params[:linear_constraints])
    nonlinear_constraints = deepcopy(prob.params[:nonlinear_constraints])
    return AbstractConstraint[
        linear_constraints...,
        NonlinearConstraint.(nonlinear_constraints)...
    ]
end

# ============================================================================= #

@testitem "Additional Objective" begin
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 50
    Δt = 0.2

    prob_vanilla = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false),
    )

    J_extra = QuadraticSmoothnessRegularizer(:dda, prob_vanilla.trajectory, 10.0)

    prob_additional = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false),
        additional_objective=J_extra,
    )

    J_prob_vanilla = Problems.get_objective(prob_vanilla)

    J_additional = Problems.get_objective(prob_additional)

    Z = prob_vanilla.trajectory
    Z⃗ = vec(prob_vanilla.trajectory)

    @test J_prob_vanilla.L(Z⃗, Z) + J_extra.L(Z⃗, Z) ≈ J_additional.L(Z⃗, Z)

end

end
