@doc raw"""
    UnitaryRobustnessProblem(Hₑ, trajectory, system, objective, integrators, constraints;
        unitary_symbol=:Ũ⃗,
        final_fidelity=unitary_fidelity(trajectory[end][unitary_symbol], trajectory.goal[unitary_symbol]),
        subspace=nothing,
        eval_hessian=false,
        verbose=false,
        ipopt_options=IpoptOptions(),
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
    ipopt_options::IpoptOptions=IpoptOptions(),
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
    probs = Dict()

    # --------------------------------------------
    #   1. test UnitarySmoothPulseProblem with subspace
    #   - rely on linear interpolation of unitary
    # --------------------------------------------
    probs["transmon"] = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        geodesic=false,
        verbose=false,
        ipopt_options=IpoptOptions(print_level=1)
    )
    solve!(probs["transmon"], max_iter=200)

    # Subspace gate success
    @test unitary_fidelity(probs["transmon"], subspace=subspace) > 0.99


    # --------------------------------------------
    #   2. test UnitaryRobustnessProblem from previous problem
    # --------------------------------------------
    probs["robust"] = UnitaryRobustnessProblem(
        H_error, probs["transmon"],
        final_fidelity=0.99,
        subspace=subspace,
        verbose=false,
        ipopt_options=IpoptOptions(recalc_y="yes", recalc_y_feas_tol=1.0, print_level=1)
    )
    solve!(probs["robust"], max_iter=200)

    eval_loss(problem, Loss) = Loss(vec(problem.trajectory.data), problem.trajectory)
    loss = InfidelityRobustnessObjective(H_error, probs["transmon"].trajectory).L

    # Robustness improvement over default
    @test eval_loss(probs["robust"], loss) < eval_loss(probs["transmon"], loss)

    # Fidelity constraint approximately satisfied
    @test isapprox(unitary_fidelity(probs["robust"]; subspace=subspace), 0.99, atol=0.025)

    # --------------------------------------------
    #   3. test UnitaryRobustnessProblem from default struct
    # --------------------------------------------
    params = deepcopy(probs["transmon"].params)
    trajectory = copy(probs["transmon"].trajectory)
    system = probs["transmon"].system
    objective = QuadraticRegularizer(:dda, trajectory, 1e-4)
    integrators = probs["transmon"].integrators
    constraints = AbstractConstraint[]

    probs["unconstrained"] = UnitaryRobustnessProblem(
        H_error, trajectory, system, objective, integrators, constraints,
        final_fidelity=0.99,
        subspace=subspace,
        ipopt_options=IpoptOptions(recalc_y="yes", recalc_y_feas_tol=1e-1, print_level=4)
    )
    solve!(probs["unconstrained"]; max_iter=100)

    # Additonal robustness improvement after relaxed objective
    @test eval_loss(probs["unconstrained"], loss) < eval_loss(probs["transmon"], loss)

    # Fidelity constraint approximately satisfied
    @test isapprox(unitary_fidelity(probs["unconstrained"]; subspace=subspace), 0.99, atol=0.025)
end
