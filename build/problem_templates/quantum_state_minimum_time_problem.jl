"""
    QuantumStateMinimumTimeProblem

TODO: Add documentation
"""
function QuantumStateMinimumTimeProblem end

function QuantumStateMinimumTimeProblem(
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    state_symbol::Symbol=:ψ̃,
    D=1.0,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert state_symbol ∈ trajectory.names

    objective += MinimumTimeObjective(trajectory; D=D)

    final_fidelity = fidelity(trajectory[end][state_symbol], trajectory.goal[state_symbol])

    fidelity_constraint = FinalQuantumStateFidelityConstraint(
        state_symbol,
        final_fidelity,
        trajectory
    )

    push!(constraints, fidelity_constraint)

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        verbose=verbose,
        ipopt_options=ipopt_options,
        kwargs...
    )
end

function QuantumStateMinimumTimeProblem(
    data_path::String;
    kwargs...
)
    data = load(data_path)
    system = data["system"]
    trajectory = data["trajectory"]
    objective = Objective(data["params"][:objective_terms])
    integrators = data["params"][:dynamics]
    constraints = AbstractConstraint[
        data["params"][:linear_constraints]...,
        NonlinearConstraint.(data["params"][:nonlinear_constraints])...
    ]
    return QuantumStateMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end
