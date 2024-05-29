"""
    Testing problems

    - test problem creation
    - test problem solving
    - test problem saving
    - test problem loading
"""

@testitem "Problems" begin
    # initializing test system
    T = 5
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    # test problem creation


end

@testitem "Additional Objective" begin
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 50
    Δt = 0.2

    prob_vanilla = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=Options(print_level=4)
    )

    J_extra = QuadraticSmoothnessRegularizer(:dda, prob_vanilla.trajectory, 10.0)

    prob_additional = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=Options(print_level=4),
        additional_objective=J_extra,
    )

    J_prob_vanilla = Problems.get_objective(prob_vanilla)

    J_additional = Problems.get_objective(prob_additional)

    Z = prob_vanilla.trajectory
    Z⃗ = prob_vanilla.trajectory.datavec

    @test J_prob_vanilla.L(Z⃗, Z) + J_extra.L(Z⃗, Z) ≈ J_additional.L(Z⃗, Z)


end
