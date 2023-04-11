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
    trajectory::NamedTrajectory
    system::QuantumSystem
    params::Dict{Symbol, Any}
end

function QuantumControlProblem(
    traj::NamedTrajectory,
    system::QuantumSystem,
    obj::Objective,
    dynamics::AbstractDynamics;
    eval_hessian::Bool=true,
    options::Options=Options(),
    constraints::Vector{LinearConstraint}=LinearConstraint[],
    nl_constraints::Vector{NonlinearConstraint}=NonlinearConstraint[],
    params::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    max_iter::Int=options.max_iter,
    linear_solver::String=options.linear_solver,
    kwargs...
)
    options.max_iter = max_iter
    options.linear_solver = linear_solver

    evaluator = PicoEvaluator(traj, obj, dynamics, nl_constraints, eval_hessian)

    n_dynamics_constraints = dynamics.dim * (traj.T - 1)
    n_variables = traj.dim * traj.T

    linear_trajectory_constraints = trajectory_constraints(traj)

    linear_constraints = vcat(linear_trajectory_constraints, constraints)

    optimizer = Ipopt.Optimizer()

    variables = initialize_optimizer!(
        optimizer,
        evaluator,
        linear_constraints,
        n_dynamics_constraints,
        nl_constraints,
        n_variables
    )

    variables = reshape(variables, traj.dim, traj.T)

    params = merge(kwargs, params)

    params[:eval_hessian] = eval_hessian
    params[:options] = options
    params[:constraints] = constraints
    params[:nl_constraints] = [nl_constraint.params for nl_constraint ∈ nl_constraints]
    params[:objective_terms] = obj.terms

    return QuantumControlProblem(
        optimizer,
        variables,
        traj,
        system,
        params
    )
end

function QuantumControlProblem(
    traj::NamedTrajectory,
    system::QuantumSystem,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator};
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    kwargs...
)
    dynamics = QuantumDynamics(integrators, traj)
    params[:dynamics] = integrators
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        params=params,
        kwargs...
    )
end

# constructor that accepts just an AbstractIntegrator
function QuantumControlProblem(
    traj::NamedTrajectory,
    system::QuantumSystem,
    obj::Objective,
    integrator::AbstractIntegrator;
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    kwargs...
)
    integrators = [integrator]
    dynamics = QuantumDynamics(integratorsd, traj)
    params[:dynamics] = integrators
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        params=params,
        kwargs...
    )
end

function QuantumControlProblem(
    traj::NamedTrajectory,
    system::QuantumSystem,
    obj::Objective,
    f::Function;
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    kwargs...
)
    dynamics = QuantumDynamics(f, traj)
    params[:dynamics] = :function
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        params=params,
        kwargs...
    )
end

function initialize_optimizer!(
    optimizer::Ipopt.Optimizer,
    evaluator::PicoEvaluator,
    constraints::Vector{AbstractConstraint},
    n_dynamics_constraints::Int,
    nl_constraints::Vector{NonlinearConstraint},
    n_variables::Int
)
    dynamics_cons = fill(
        MOI.NLPBoundsPair(0.0, 0.0),
        n_dynamics_constraints
    )

    general_nl_cons = []

    for nl_con ∈ nl_constraints
        if nl_con isa NonlinearEqualityConstraint
            push!(general_nl_cons, MOI.NLPBoundsPair(0.0, 0.0))
        elseif nl_con isa NonlinearInequalityConstraint
            push!(general_nl_cons, MOI.NLPBoundsPair(0.0, Inf))
        else
            error("Unknown nonlinear constraint type")
        end
    end

    nl_cons = vcat(dynamics_cons, general_nl_cons)

    # build NLP block data
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)

    # set NLP block data
    MOI.set(optimizer, MOI.NLPBlock(), block_data)

    # set objective sense: minimize
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add variables
    variables = MOI.add_variables(optimizer, n_variables)

    # add constraints
    constrain!(optimizer, variables, constraints, verbose=true)

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
