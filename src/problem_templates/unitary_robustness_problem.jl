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
- `unitary_name::Symbol=:Ũ⃗`: The symbol for the unitary trajectory in `trajectory`.
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
    unitary_name::Symbol=:Ũ⃗,
    final_fidelity::Union{Real, Nothing}=nothing,
    phase_name::Symbol=:ϕ,
    phase_operators::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    subspace=nothing,
    kwargs...
)
    @assert unitary_name ∈ trajectory.names

    objective += UnitaryRobustnessObjective(
        H_error=H_error,
        eval_hessian=piccolo_options.eval_hessian
    )

    U_T = trajectory[unitary_name][:, end]
    U_G = trajectory.goal[unitary_name]
    subspace = isnothing(subspace) ? axes(iso_vec_to_operator(U_T), 1) : subspace

    if isnothing(phase_operators)
        if isnothing(final_fidelity)
            final_fidelity = iso_vec_unitary_fidelity(U_T, U_G, subspace=subspace)
        end
        
        fidelity_constraint = FinalUnitaryFidelityConstraint(
            unitary_name,
            final_fidelity,
            trajectory;
            subspace=subspace,
            eval_hessian=piccolo_options.eval_hessian
        )
    else
        if isnothing(final_fidelity)
            phases = trajectory.global_data[phase_name]
            final_fidelity = iso_vec_unitary_free_phase_fidelity(
                U_T, U_G, phases, phase_operators; subspace=subspace
            )
        end

        fidelity_constraint = FinalUnitaryFreePhaseFidelityConstraint(
            unitary_name,
            phase_name,
            phase_operators,
            final_fidelity,
            trajectory;
            subspace=subspace,
            eval_hessian=piccolo_options.eval_hessian
        )
    end

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


    # set up without a final fidelity
    # -------------------------------
    @test UnitaryRobustnessProblem(H_embed, prob) isa QuantumControlProblem


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

@testitem "Set up a free phase problem" begin
    using LinearAlgebra
    δ1 = δ2 = -0.1
    T = 75
    Δt = 1.0
    n_levels = 3
    a = annihilate(n_levels)
    id = I(n_levels)
    a1 = kron(a, id)
    a2 = kron(id, a)
    H_drift = δ1 / 2 * a1' * a1' * a1 * a1 + δ2 / 2 * a2' * a2' * a2 * a2
    H_drives = [a1'a1, a2'a2, a1'a2 + a1*a2', im * (a1'a2 - a1 * a2')]
    system = QuantumSystem(H_drift, H_drives)
    U_goal = EmbeddedOperator(
        GATES[:CZ], 
        get_subspace_indices([1:2, 1:2], [n_levels, n_levels]),
        [n_levels, n_levels]
    )

    phase_operators = [PAULIS[:Z], PAULIS[:Z]]
    prob = UnitarySmoothPulseProblem(
        system, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, eval_hessian=false),
        phase_operators=phase_operators,
    )

    ZZ = EmbeddedOperator(
        reduce(kron, phase_operators), 
        get_subspace_indices([1:2, 1:2], [n_levels, n_levels]), 
        [n_levels, n_levels]
    )

    @test UnitaryRobustnessProblem(
        ZZ,
        prob,
        phase_operators=phase_operators,
        subspace=U_goal.subspace_indices,
    ) isa QuantumControlProblem
end

@testitem "Set up a free phase problem" begin
    using LinearAlgebra
    δ1 = δ2 = -0.1
    T = 75
    Δt = 1.0
    n_levels = 3
    a = annihilate(n_levels)
    id = I(n_levels)
    a1 = kron(a, id)
    a2 = kron(id, a)
    H_drift = δ1 / 2 * a1' * a1' * a1 * a1 + δ2 / 2 * a2' * a2' * a2 * a2
    H_drives = [a1'a1, a2'a2, a1'a2 + a1*a2', im * (a1'a2 - a1 * a2')]
    system = QuantumSystem(H_drift, H_drives)
    U_goal = EmbeddedOperator(
        GATES[:CZ], 
        get_subspace_indices([1:2, 1:2], [n_levels, n_levels]),
        [n_levels, n_levels]
    )

    phase_operators = [PAULIS[:Z], PAULIS[:Z]]
    prob = UnitarySmoothPulseProblem(
        system, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, eval_hessian=false),
        phase_operators=phase_operators,
    )

    ZZ = EmbeddedOperator(
        reduce(kron, phase_operators), 
        get_subspace_indices([1:2, 1:2], [n_levels, n_levels]), 
        [n_levels, n_levels]
    )

    @test UnitaryRobustnessProblem(
        ZZ,
        prob,
        phase_operators=phase_operators,
        subspace=U_goal.subspace_indices,
    ) isa QuantumControlProblem
end
