module Callbacks

export best_trajectory_callback
export trajectory_history_callback

using NamedTrajectories
using TestItemRunner

using ..Problems

function best_trajectory_callback(prob::QuantumControlProblem)
    best_traj = deepcopy(prob.trajectory)
    best_value = Inf64

    function callback(args...)
        obj = get_objective(prob)
        Z⃗ = Problems.get_datavec(prob)
        value = obj.L(Z⃗, prob.trajectory)
        if value < best_value
            best_value = value
            update!(best_traj, Z⃗)
        end
        return true
    end

    return callback, best_traj
end

function trajectory_history_callback(prob::QuantumControlProblem)
    trajectory_history = []

    function callback(args...)
        push!(trajectory_history, get_datavec(prob))
        return true
    end

    return callback, trajectory_history
end

# ========================================================================== # 

@testitem "Callback returns false early stops" begin
    using MathOptInterface
    const MOI = MathOptInterface
    include("../test/test_utils.jl")

    prob = smooth_quantum_state_problem()

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
    include("../test/test_utils.jl")

    prob = smooth_quantum_state_problem()

    callback, trajectory_history = trajectory_history_callback(prob)

    solve!(prob, max_iter=20, callback=callback)
    @test length(trajectory_history) == 21
end

@testitem "Callback can get best trajectory" begin
    using MathOptInterface
    using NamedTrajectories
    const MOI = MathOptInterface
    include("../test/test_utils.jl")

    prob, system = smooth_quantum_state_problem(return_system=true)

    callback, best_traj = best_trajectory_callback(prob)
    @test best_traj isa NamedTrajectory

    before = fidelity(best_traj, system)
    solve!(prob, max_iter=20, callback=callback)
    after = fidelity(prob.trajectory, system)
    best = fidelity(best_traj, system)
    
    @test best_traj isa NamedTrajectory
    @test before < after
    @test before < best
    @test after ≤ best
end

@testitem "Callback with full parameter test" begin
    using MathOptInterface
    using NamedTrajectories
    const MOI = MathOptInterface
    include("../test/test_utils.jl")

    prob = smooth_quantum_state_problem()

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

end