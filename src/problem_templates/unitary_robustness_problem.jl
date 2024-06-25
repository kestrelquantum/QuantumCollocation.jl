@doc raw"""
    UnitaryRobustnessProblem(
        H_error, trajectory, system, objective, integrators, constraints;
        unitary_symbol=:Ũ⃗,
        final_fidelity=nothing,
        subspace=nothing,
        ipopt_options=IpoptOptions(),
        piccolo_options=PiccoloOptions(),
        kwargs...
    )

    UnitaryRobustnessProblem(Hₑ, prob::QuantumControlProblem; kwargs...)

Create a quantum control problem for robustness optimization of a unitary trajectory.
"""
function UnitaryRobustnessProblem end


function UnitaryRobustnessProblem(
    H_error::AbstractMatrix{<:Number},
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    final_fidelity::Union{Real, Nothing}=nothing,
    subspace::AbstractVector{<:Integer}=1:size(H_error, 1),
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    if isnothing(final_fidelity)
        final_fidelity = unitary_fidelity(
            trajectory[unitary_symbol][:, end], trajectory.goal[unitary_symbol]
        )
    end

    objective += InfidelityRobustnessObjective(
        H_error,
        trajectory,
        eval_hessian=piccolo_options.eval_hessian,
        subspace=subspace
    )

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory;
        subspace=subspace
    )
    push!(constraints, fidelity_constraint)

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

function UnitaryRobustnessProblem(
    H_error::AbstractMatrix{<:Number},
    prob::QuantumControlProblem;
    objective::Objective=get_objective(prob),
    constraints::AbstractVector{<:AbstractConstraint}=get_constraints(prob),
    ipopt_options::IpoptOptions=deepcopy(prob.ipopt_options),
    piccolo_options::PiccoloOptions=deepcopy(prob.piccolo_options),
    build_trajectory_constraints=false,
    kwargs...
)
    piccolo_options.build_trajectory_constraints = build_trajectory_constraints

    return UnitaryRobustnessProblem(
        H_error,
        copy(prob.trajectory),
        prob.system,
        objective,
        prob.integrators,
        constraints;
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Robust and Subspace Templates" begin
    # TODO: Improve these tests.
    # --------------------------------------------
    # Initialize with UnitarySmoothPulseProblem
    # --------------------------------------------
    H_error = GATES[:Z]
    H_drift = zeros(3, 3)
    H_drives = [create(3) + annihilate(3), im * (create(3) - annihilate(3))]
    sys = QuantumSystem(H_drift, H_drives)
    U_goal = EmbeddedOperator(:X, sys)
    subspace = U_goal.subspace_indices
    T = 51
    Δt = 0.2

    #  test initial problem
    # ---------------------
    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        geodesic=true,
        verbose=false,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )
    before = unitary_fidelity(prob, subspace=subspace)
    solve!(prob, max_iter=20)
    after = unitary_fidelity(prob, subspace=subspace)

    # Subspace gate success
    @test after > before


    #  test robustness from previous problem
    # --------------------------------------
    final_fidelity = 0.99
    rob_prob = UnitaryRobustnessProblem(
        H_error, prob,
        final_fidelity=final_fidelity,
        subspace=subspace,
        ipopt_options=IpoptOptions(recalc_y="yes", recalc_y_feas_tol=10.0, print_level=1),
    )
    solve!(rob_prob, max_iter=50)

    loss(Z⃗) = InfidelityRobustnessObjective(H_error, prob.trajectory).L(Z⃗, prob.trajectory)

    # Robustness improvement over default
    @test loss(rob_prob.trajectory.datavec) < loss(prob.trajectory.datavec)

    # Fidelity constraint approximately satisfied
    @test isapprox(unitary_fidelity(rob_prob; subspace=subspace), 0.99, atol=0.025)
end
