module Problems

export AbstractProblem
export FixedTimeProblem
export QuantumControlProblem

export initialize_trajectory!
export update_trajectory!
export get_traj_data
export get_datavec

using ..QuantumSystems
using ..Integrators
using ..Evaluators
using ..IpoptOptions
using ..Constraints
using ..Dynamics
using ..Objectives

using TrajectoryIndexingUtils
using NamedTrajectories
using JLD2
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

abstract type AbstractProblem end

mutable struct QuantumControlProblem <: AbstractProblem
    optimizer::Ipopt.Optimizer
    variables::Matrix{MOI.VariableIndex}
    system::QuantumSystem
    trajectory::NamedTrajectory
    options::Options
    params::Dict{Symbol, Any}
end

function QuantumControlProblem(
    system::QuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    dynamics::QuantumDynamics;
    eval_hessian::Bool=true,
    options::Options=Options(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    params::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    max_iter::Int=options.max_iter,
    linear_solver::String=options.linear_solver,
    verbose=false,
    kwargs...
)
    options.max_iter = max_iter
    options.linear_solver = linear_solver

    nonlinear_constraints = NonlinearConstraint[con for con ∈ constraints if con isa NonlinearConstraint]

    if verbose
        println("    building evaluator...")
    end
    evaluator = PicoEvaluator(traj, obj, dynamics, nonlinear_constraints, eval_hessian)

    n_dynamics_constraints = dynamics.dim * (traj.T - 1)
    n_variables = traj.dim * traj.T

    linear_constraints = [con for con ∈ constraints if con isa LinearConstraint]
    linear_constraints = LinearConstraint[trajectory_constraints(traj); linear_constraints]

    optimizer = Ipopt.Optimizer()

    if verbose
        println("    initializing optimizer...")
    end
    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        linear_constraints,
        n_dynamics_constraints,
        nonlinear_constraints,
        n_variables
    )

    variables = reshape(variables, traj.dim, traj.T)

    params = merge(kwargs, params)

    params[:eval_hessian] = eval_hessian
    params[:options] = options
    params[:linear_constraints] = constraints
    params[:nonlinear_constraints] = [nl_constraint.params for nl_constraint ∈ nonlinear_constraints]
    params[:objective_terms] = obj.terms

    return QuantumControlProblem(
        optimizer,
        variables,
        system,
        traj,
        options,
        params
    )
end

function QuantumControlProblem(
    system::QuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator};
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    verbose=false,
    kwargs...
)
    if verbose
        println("    building dynamics from integrators...")
    end
    dynamics = QuantumDynamics(integrators, traj; verbose=verbose)
    params[:dynamics] = integrators
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        params=params,
        verbose=verbose,
        kwargs...
    )
end

# constructor that accepts just an AbstractIntegrator
function QuantumControlProblem(
    system::QuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrator::AbstractIntegrator;
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    verbose=false,
    kwargs...
)
    if verbose
        println("    building dynamics from integrator...")
    end
    integrators = [integrator]
    dynamics = QuantumDynamics(integrators, traj; verbose=verbose)
    params[:dynamics] = integrators
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        params=params,
        verbose=verbose,
        kwargs...
    )
end

function QuantumControlProblem(
    system::QuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    f::Function;
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    verbose=false,
    kwargs...
)
    if verbose
        println("    building dynamics from function...")
    end
    dynamics = QuantumDynamics(f, traj; verbose=verbose)
    params[:dynamics] = :function
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        params=params,
        verbose=verbose,
        kwargs...
    )
end

function initialize_optimizer!(
    optimizer::Ipopt.Optimizer,
    evaluator::PicoEvaluator,
    linear_constraints::Vector{LinearConstraint},
    n_dynamics_constraints::Int,
    nonlinear_constraints::Vector{NonlinearConstraint},
    n_variables::Int
)
    nl_cons = fill(
        MOI.NLPBoundsPair(0.0, 0.0),
        n_dynamics_constraints
    )

    for nl_con ∈ nonlinear_constraints
        if nl_con isa NonlinearEqualityConstraint
            push!(nl_cons, MOI.NLPBoundsPair(0.0, 0.0))
        elseif nl_con isa NonlinearInequalityConstraint
            push!(nl_cons, MOI.NLPBoundsPair(0.0, Inf))
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
    constrain!(optimizer, variables, linear_constraints, verbose=true)

    return variables
end

function initialize_trajectory!(
    prob::QuantumControlProblem,
    traj::NamedTrajectory
)
    MOI.set(
        prob.optimizer,
        MOI.VariablePrimalStart(),
        vec(prob.variables),
        collect(traj.datavec)
    )
end

initialize_trajectory!(prob::QuantumControlProblem) =
    initialize_trajectory!(prob, prob.trajectory)

function get_datavec(prob::QuantumControlProblem)
    Z⃗ = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        vec(prob.variables)
    )
    return Z⃗
end

@views function update_trajectory!(prob::QuantumControlProblem)
    Z⃗ = get_datavec(prob)
    prob.trajectory = NamedTrajectory(Z⃗, prob.trajectory)
end


end
