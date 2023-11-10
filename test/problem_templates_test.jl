# ------------------------------------------------
# Test: ProblemTemplates
#
# 1. test UnitarySmoothPulseProblem
# 2. test UnitaryMinimumTimeProblem
# ------------------------------------------------


@testset "Problem Templates" begin

    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 50
    Δt = 0.2

    # --------------------------------------------
    # 1. test UnitarySmoothPulseProblem
    # --------------------------------------------

    prob = UnitarySmoothPulseProblem(H_drift, H_drives, U_goal, T, Δt)

    solve!(prob; max_iter=100)

    @test unitary_fidelity(prob) > 0.99

    # --------------------------------------------
    # 2. test UnitaryMinimumTimeProblem
    # --------------------------------------------

    final_fidelity = 0.99

    mintime_prob = UnitaryMinimumTimeProblem(prob; final_fidelity=final_fidelity)

    solve!(mintime_prob; max_iter=100)

    @test unitary_fidelity(mintime_prob) > final_fidelity

    @test times(mintime_prob.trajectory)[end] < times(prob.trajectory)[end]


end
