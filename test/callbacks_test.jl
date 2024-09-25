"""
    Testing  callback features

    - callback gives early stopping
    - callback gives trajectory data
    - callback example test with full argument list
"""

@testitem "Callback returns false early stops" begin
    using MathOptInterface
    const MOI = MathOptInterface

    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = [1.0, 0.0]
    ψ_target = [0.0, 1.0]

    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1), 
        piccolo_options=PiccoloOptions(verbose=false)
    )

    my_callback = (kwargs...) -> false

    initial = fidelity(prob)
    solve!(prob, max_iter=20, callback=my_callback)
    final = fidelity(prob)

    # callback forces problem to exit early as per Ipopt documentation
    @test MOI.get(prob.optimizer, MOI.TerminationStatus()) == MOI.INTERRUPTED
    @test initial ≈ final atol=1e-2
end


@testitem "Callback can get internal history" begin
    using MathOptInterface
    using NamedTrajectories
    const MOI = MathOptInterface

    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = [1.0, 0.0]
    ψ_target = [0.0, 1.0]

    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1), 
        piccolo_options=PiccoloOptions(verbose=false)
    )

    trajectory_history = NamedTrajectory[]
    function get_history_callback(
        kwargs...
    )
        x_vals = map(x -> MOI.get(prob.optimizer, MOI.CallbackVariablePrimal(prob.optimizer), x), prob.variables)
        push!(trajectory_history, NamedTrajectory(x_vals, prob.trajectory))
        return true
    end

    initial = fidelity(prob)
    solve!(prob, max_iter=20, callback=get_history_callback)

    # for (iter, traj) in enumerate(trajectory_history)
    #     plot("./iteration-$iter-trajectory.png", traj, [:ψ̃1, :a])
    # end
    @test length(trajectory_history) == 21
end

@testitem "Callback with full parameter test" begin
    using MathOptInterface
    using NamedTrajectories
    const MOI = MathOptInterface

    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = [1.0, 0.0]
    ψ_target = [0.0, 1.0]

    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1), 
        piccolo_options=PiccoloOptions(verbose=false)
    )

    obj_vals = []
    function get_history_callback(
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint,
    )
        push!(obj_vals, obj_value)
        return iter_count < 3
    end

    solve!(prob, max_iter=20, callback=get_history_callback)

    @test MOI.get(prob.optimizer, MOI.TerminationStatus()) == MOI.INTERRUPTED
    @test length(obj_vals) == 4   # problem init, iter 1, iter 2, iter 3 (terminate)
end