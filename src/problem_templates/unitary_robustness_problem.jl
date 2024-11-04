export UnitaryRobustnessProblem


@doc raw"""
    UnitaryRobustnessProblem(
        H_error,
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        kwargs...
    )

    UnitaryRobustnessProblem(Hₑ, prob::QuantumControlProblem; kwargs...)

Create a quantum control problem for robustness optimization of a unitary trajectory.

# Keyword Arguments
- `unitary_symbol::Symbol=:Ũ⃗`: The symbol for the unitary trajectory in `trajectory`.
- `final_fidelity::Union{Real, Nothing}=nothing`: The target fidelity for the final unitary.
- `ipopt_options::IpoptOptions=IpoptOptions()`: Options for the Ipopt solver.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: Options for the Piccolo solver.
- `kwargs...`: Additional keyword arguments passed to `QuantumControlProblem`.
"""
function UnitaryRobustnessProblem end


function UnitaryRobustnessProblem(
    H_error::OperatorType,
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    final_fidelity::Union{Real, Nothing}=nothing,
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

    objective += UnitaryRobustnessObjective(
        H_error=H_error,
        eval_hessian=piccolo_options.eval_hessian,
    )

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory;
        subspace=H_error isa EmbeddedOperator ? H_error.subspace_indices : nothing
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
    H_error::OperatorType,
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
    H_drift = zeros(3, 3)
    H_drives = [create(3) + annihilate(3), im * (create(3) - annihilate(3))]
    sys = QuantumSystem(H_drift, H_drives)

    U_goal = EmbeddedOperator(:X, sys)
    H_embed = EmbeddedOperator(:Z, sys)
    T = 51
    Δt = 0.2

    #  test initial problem
    # ---------------------
    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        R=1e-12,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )
    before = unitary_rollout_fidelity(prob, subspace=U_goal.subspace_indices)
    solve!(prob, max_iter=15)
    after = unitary_rollout_fidelity(prob, subspace=U_goal.subspace_indices)

    # Subspace gate success
    @test after > before


    #  test robustness from previous problem
    # --------------------------------------
    final_fidelity = 0.99
    rob_prob = UnitaryRobustnessProblem(
        H_embed, prob,
        final_fidelity=final_fidelity,
        ipopt_options=IpoptOptions(recalc_y="yes", recalc_y_feas_tol=100.0, print_level=1),
    )
    solve!(rob_prob, max_iter=50)

    loss(Z⃗) = UnitaryRobustnessObjective(H_error=H_embed).L(Z⃗, prob.trajectory)

    # Robustness improvement over default (or small initial)
    # TODO: Can this test be improved? (might fail if unlucky)
    after = loss(rob_prob.trajectory.datavec)
    before = loss(prob.trajectory.datavec)
    @test (after < before) || (before < 0.25)

    # TODO: Fidelity constraint approximately satisfied
    @test_skip isapprox(unitary_rollout_fidelity(rob_prob; subspace=U_goal.subspace_indices), 0.99, atol=0.05)
end
