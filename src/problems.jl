module Problems

export AbstractProblem
export FixedTimeProblem
export QuantumControlProblem

export set_trajectory!
export update_trajectory!
export get_traj_data
export get_datavec
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
using JLD2
using Ipopt
using MathOptInterface
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
    variables::Matrix{MOI.VariableIndex}
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
    kwargs...
)
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
        traj, obj, dynamics, nonlinear_constraints, eval_hessian=piccolo_options.eval_hessian
    )

    n_dynamics_constraints = dynamics.dim * (traj.T - 1)
    n_variables = traj.dim * traj.T

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

    variables = reshape(variables, traj.dim, traj.T)

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
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
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
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        params=params,
        kwargs...
    )
end

# constructor that accepts just an AbstractIntegrator
function QuantumControlProblem(
    system::AbstractQuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrator::AbstractIntegrator;
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
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
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        params=params,
        kwargs...
    )
end

function QuantumControlProblem(
    system::AbstractQuantumSystem,
    traj::NamedTrajectory,
    obj::Objective,
    f::Function;
    params::Dict{Symbol,Any}=Dict{Symbol, Any}(),
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
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
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        params=params,
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
    MOI.set(
        prob.optimizer,
        MOI.VariablePrimalStart(),
        vec(prob.variables),
        collect(traj.datavec)
    )
end

set_trajectory!(prob::QuantumControlProblem) =
    set_trajectory!(prob, prob.trajectory)

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

"""
    get_objective(prob::QuantumControlProblem)

Return the objective function of the `prob::QuantumControlProblem`.
"""
function get_objective(prob::QuantumControlProblem)
    return Objective(prob.params[:objective_terms])
end

"""
    get_constraints(prob::QuantumControlProblem)

Return the constraints of the `prob::QuantumControlProblem`.
"""
function get_constraints(prob::QuantumControlProblem)
    return AbstractConstraint[
        prob.params[:linear_constraints]...,
        NonlinearConstraint.(prob.params[:nonlinear_constraints])...
    ]
end

end
