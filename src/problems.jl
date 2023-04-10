module Problems

export AbstractProblem
export FixedTimeProblem
export QuantumControlProblem

export initialize_trajectory!
export update_traj_data!
export get_traj_data
export get_variables
export solve!
export generate_file_path
export save_problem
export load_problem

using TrajectoryIndexingUtils
using ..QuantumSystems
using ..Integrators
using ..Evaluators
using ..IpoptOptions
using ..Constraints
using ..Dynamics
using ..Objectives


using JLD2
using NamedTrajectories
using Libdl
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

abstract type AbstractProblem end

mutable struct QuantumControlProblem <: AbstractProblem
    system::AbstractSystem
    variables::Matrix{MOI.VariableIndex}
    optimizer::Ipopt.Optimizer
    trajectory::NamedTrajectory
    params::Dict{Symbol, Any}
end

function QuantumControlProblem(
    system::AbstractSystem,
    traj::NamedTrajectory,
    obj::Objective,
    dynamics::AbstractDynamics;
    eval_hessian::Bool=true,
    options::Options=Options(),
    constraints::Vector{LinearConstraint}=LinearConstraint[],
    nl_constraints::Vector{NonlinearConstraint}=NonlinearConstraint[],
    params::Dict{Symbol, Any}=Dict{Symbol, Any}(),
    kwargs...
)
    optimizer = Ipopt.Optimizer()
    set!(optimizer, options)

    evaluator = PicoEvaluator(traj, obj, dynamics, nl_constraints, eval_hessian)

    n_dynamics_constraints = dynamics.dim * (traj.T - 1)
    n_variables = traj.dim * traj.T

    linear_trajectory_constraints = trajectory_constraints(traj)

    linear_constraints = vcat(linear_trajectory_constraints, constraints)

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

    return QuantumControlProblem(
        system,
        variables,
        optimizer,
        traj,
        params
    )

end

function QuantumControlProblem(
    system::AbstractSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator};
    kwargs...
)
    dynamics = QuantumDynamics(integrators, traj)

    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
        kwargs...
    )
end

# constructor that accepts just an AbstractIntegrator
function QuantumControlProblem(
    system::AbstractSystem,
    traj::NamedTrajectory,
    obj::Objective,
    integrator::AbstractIntegrator;
    kwargs...
)
    return QuantumControlProblem(
        system,
        traj,
        obj,
        [integrator];
        kwargs...
    )
end

function QuantumControlProblem(
    system::AbstractSystem,
    traj::NamedTrajectory,
    obj::Objective,
    f::Function;
    kwargs...
)
    dynamics = QuantumDynamics(f, traj)
    return QuantumControlProblem(
        system,
        traj,
        obj,
        dynamics;
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

function get_variables(prob::QuantumControlProblem)
    Z⃗ = MOI.get(
        prob.optimizer,
        MOI.VariablePrimal(),
        vec(prob.variables)
    )
    return Z⃗
end

@views function update_traj_data!(prob::QuantumControlProblem)
    Z⃗ = get_variables(prob)
    prob.trajectory = NamedTrajectory(Z⃗, prob.trajectory)
end




function save_problem(prob::AbstractProblem, path::String)
    mkpath(dirname(path))
    @save path prob
end

function load_problem(path::String)
    @load path prob
    return prob
end

function solve!(
    prob::QuantumControlProblem;
    init_traj=prob.trajectory,
    save_path=nothing,
    controls_save_path=nothing,
)
    initialize_trajectory!(prob, init_traj)

    MOI.optimize!(prob.optimizer)

    update_traj_data!(prob)

    if !isnothing(save_path)
        save_problem(prob, save_path)
    end

    if !isnothing(controls_save_path)
        save_controls(prob, controls_save_path)
    end
end


function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_suffix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("$(file_name)", file_name_) && occursin(".$(extension)", file_name_)

                numeric_suffix = parse(
                    Int,
                    split(split(file_name_, "_")[end], ".")[1]
                )

                max_numeric_suffix = max(
                    numeric_suffix,
                    max_numeric_suffix
                )
            end
        end
    end

    file_path = joinpath(
        path,
        file_name *
        "_$(lpad(max_numeric_suffix + 1, 5, '0')).$(extension)"
    )

    return file_path
end



end
