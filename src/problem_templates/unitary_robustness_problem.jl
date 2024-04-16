@doc raw"""
    UnitaryRobustnessProblem(Hₑ, trajectory, system, objective, integrators, constraints;
        unitary_symbol=:Ũ⃗,
        final_fidelity=unitary_fidelity(trajectory[end][unitary_symbol], trajectory.goal[unitary_symbol]),
        subspace=nothing,
        eval_hessian=false,
        verbose=false,
        ipopt_options=Options(),
        kwargs...
    )

    UnitaryRobustnessProblem(Hₑ, prob::QuantumControlProblem; kwargs...)

Create a quantum control problem for robustness optimization of a unitary trajectory.
"""
function UnitaryRobustnessProblem end


function UnitaryRobustnessProblem(
    Hₑ::AbstractMatrix{<:Number},
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    final_fidelity::Float64=unitary_fidelity(trajectory[end][unitary_symbol], trajectory.goal[unitary_symbol]),
    subspace::AbstractVector{<:Integer}=1:size(Hₑ, 1),
    eval_hessian::Bool=false,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    if !eval_hessian
        ipopt_options.hessian_approximation = "limited-memory"
    end

    objective += InfidelityRobustnessObjective(
        Hₑ,
        trajectory,
        eval_hessian=eval_hessian,
        subspace=subspace
    )

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory;
        subspace=subspace
    )

    constraints = AbstractConstraint[constraints..., fidelity_constraint]

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        verbose=verbose,
        ipopt_options=ipopt_options,
        eval_hessian=eval_hessian,
        kwargs...
    )
end

function UnitaryRobustnessProblem(
    Hₑ::AbstractMatrix{<:Number},
    prob::QuantumControlProblem;
    kwargs...
)
    params = deepcopy(prob.params)
    trajectory = copy(prob.trajectory)
    system = prob.system
    objective = Objective(params[:objective_terms])
    integrators = prob.integrators
    constraints = [
        params[:linear_constraints]...,
        NonlinearConstraint.(params[:nonlinear_constraints])...
    ]

    return UnitaryRobustnessProblem(
        Hₑ,
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end
