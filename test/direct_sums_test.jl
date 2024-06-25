"""
Tests: DirectSums submodule
"""

@testitem "Apply suffix to trajectories" begin
    using NamedTrajectories
    include("../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    suffix = "_new"
    new_traj = add_suffix(traj, suffix)
    
    @test new_traj.state_names == add_suffix(traj.state_names, suffix)
    @test new_traj.control_names == add_suffix(traj.control_names, suffix)

    same_traj = add_suffix(traj, "")
    @test traj == same_traj
end

@testitem "Merge trajectories" begin
    using NamedTrajectories
    include("../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    
    # apply suffix
    pf_traj1 = add_suffix(traj, "_1")
    pf_traj2 = add_suffix(traj, "_2")

    # merge
    new_traj = direct_sum(pf_traj1, pf_traj2)

    @test issetequal(new_traj.state_names, vcat(pf_traj1.state_names..., pf_traj2.state_names...))
    @test issetequal(new_traj.control_names, vcat(pf_traj1.control_names..., pf_traj2.control_names...))

    # merge2
    new_traj2 = direct_sum([pf_traj1, pf_traj2])

    @test new_traj == new_traj2
end

@testitem "Merge free time trajectories" begin
    using NamedTrajectories
    include("../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    
    # apply suffix
    pf_traj1 = add_suffix(traj, "_1")
    pf_traj2 = add_suffix(traj, "_2")
    pf_traj3 = add_suffix(traj, "_3")
    state_names = vcat(pf_traj1.state_names..., pf_traj2.state_names..., pf_traj3.state_names...)
    control_names = vcat(pf_traj1.control_names..., pf_traj2.control_names..., pf_traj3.control_names...)

    # merge (without reduce)
    new_traj_1 = direct_sum(direct_sum(pf_traj1, pf_traj2), pf_traj3, free_time=true)
    @test new_traj_1.timestep isa Symbol
    @test issetequal(new_traj_1.state_names, state_names)
    @test issetequal(setdiff(new_traj_1.control_names, control_names), [new_traj_1.timestep])

    # merge (with reduce)
    new_traj_2 = direct_sum([pf_traj1, pf_traj2, pf_traj3], free_time=true)
    @test new_traj_2.timestep isa Symbol
    @test issetequal(new_traj_2.state_names, state_names)
    @test issetequal(setdiff(new_traj_2.control_names, control_names), [new_traj_2.timestep])

    # check equality
    for c in new_traj_1.control_names
        @test new_traj_1[c] == new_traj_2[c]
    end
    for s in new_traj_1.state_names
        @test new_traj_1[s] == new_traj_2[s]
    end
end

@testitem "Merge systems" begin
    using NamedTrajectories
    include("../test/test_utils.jl")

    H_drift = 0.01 * GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    T = 50
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T))
        
    # apply suffix and sum
    sys2 = direct_sum(
        add_suffix(sys, "_1"),
        add_suffix(sys, "_2")
    )

    @test length(sys2.H_drives) == 4
    @test sys2.params[:T_1] == T
    @test sys2.params[:T_2] == T

    # add another system
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T, :S=>2T))
    sys3 = direct_sum(sys2, add_suffix(sys, "_3"))
    @test length(sys3.H_drives) == 6
    @test sys3.params[:T_3] == T
    @test sys3.params[:S_3] == 2T
end

@testitem "Get suffix" begin
    using NamedTrajectories

    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    T = 50
    Δt = 0.2
    ip_ops = IpoptOptions(print_level=1)
    pi_ops = PiccoloOptions(verbose=false, free_time=false)
    prob1 = UnitarySmoothPulseProblem(sys, GATES[:X], T, Δt, piccolo_options=pi_ops, ipopt_options=ip_ops)
    prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_ops, ipopt_options=ip_ops)
    
    # Direct sum problem with suffix extraction
    # Note: Turn off control reset
    direct_sum_prob = UnitaryDirectSumProblem([prob1, prob2], 0.99, drive_reset_ratio=0.0, ipopt_options=ip_ops)
    prob1_got = get_suffix(direct_sum_prob, "1")
    @test prob1_got.trajectory == add_suffix(prob1.trajectory, "1")

    # Mutate the direct sum problem
    update!(prob1_got.trajectory, :a1, ones(size(prob1_got.trajectory[:a1])))
    @test prob1_got.trajectory != add_suffix(prob1.trajectory, "1")

    # Remove suffix during extraction
    prob1_got_without = get_suffix(direct_sum_prob, "1", remove=true)
    @test prob1_got_without.trajectory == prob1.trajectory
end

@testitem "Append to default integrators" begin
    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:Y]])
    T = 50
    Δt = 0.2
    ip_ops = IpoptOptions(print_level=1)
    pi_false_ops = PiccoloOptions(verbose=false, free_time=false)
    pi_true_ops = PiccoloOptions(verbose=false, free_time=true)
    prob1 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_false_ops, ipopt_options=ip_ops)
    prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_true_ops, ipopt_options=ip_ops)

    suffix = "_new"
    # UnitaryPadeIntegrator
    prob1_new = add_suffix(prob1.integrators, suffix)
    @test prob1_new[1].unitary_symb == add_suffix(prob1.integrators[1].unitary_symb, suffix)
    @test prob1_new[1].drive_symb == add_suffix(prob1.integrators[1].drive_symb, suffix)

    # DerivativeIntegrator
    @test prob1_new[2].variable == add_suffix(prob1.integrators[2].variable, suffix)

    # UnitaryPadeIntegrator with free time
    prob2_new = add_suffix(prob2.integrators, suffix)
    @test prob2_new[1].unitary_symb == add_suffix(prob2.integrators[1].unitary_symb, suffix)
    @test prob2_new[1].drive_symb == add_suffix(prob2.integrators[1].drive_symb, suffix)

    # DerivativeIntegrator
    @test prob2_new[2].variable == add_suffix(prob2.integrators[2].variable, suffix)
end

@testitem "Free time get suffix" begin
    using NamedTrajectories

    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:Y]])
    T = 50
    Δt = 0.2
    ops = IpoptOptions(print_level=1)
    pi_false_ops = PiccoloOptions(verbose=false, free_time=false)
    pi_true_ops = PiccoloOptions(verbose=false, free_time=true)
    suffix = "_new"
    timestep_symbol = :Δt

    prob1 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_false_ops, ipopt_options=ops)
    traj1 = direct_sum(prob1.trajectory, add_suffix(prob1.trajectory, suffix), free_time=true)

    # Direct sum (shared timestep name)
    @test get_suffix(traj1, suffix).timestep == timestep_symbol
    @test get_suffix(traj1, suffix, remove=true).timestep == timestep_symbol

    prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, ipopt_options=ops, piccolo_options=pi_true_ops)
    traj2 = add_suffix(prob2.trajectory, suffix)
   
    # Trajectory (unique timestep name)
    @test get_suffix(traj2, suffix).timestep == add_suffix(timestep_symbol, suffix)
    @test get_suffix(traj2, suffix, remove=true).timestep == timestep_symbol
end