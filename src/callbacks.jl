module Callbacks

export best_rollout_fidelity_callback
export best_unitary_rollout_fidelity_callback
export trajectory_history_callback

using NamedTrajectories
using TestItemRunner

using ..Losses
using ..Problems: QuantumControlProblem, get_datavec
using ..QuantumSystems
using ..Rollouts


function best_rollout_callback(
    prob::QuantumControlProblem, rollout_fidelity::Function;
    system::Union{AbstractQuantumSystem, AbstractVector{<:AbstractQuantumSystem}}=prob.system
)
    best_value = 0.0
    best_trajectories = []

    function callback(args...)
        traj = NamedTrajectory(get_datavec(prob), prob.trajectory)
        value = rollout_fidelity(traj, system)
        if value > best_value
            best_value = value
            push!(best_trajectories, traj)
        end
        return true
    end

    return callback, best_trajectories
end

function best_rollout_fidelity_callback(prob::QuantumControlProblem)
    return best_rollout_callback(prob, fidelity)
end

function best_unitary_rollout_fidelity_callback(prob::QuantumControlProblem)
    return best_rollout_callback(prob, unitary_fidelity)
end

function trajectory_history_callback(prob::QuantumControlProblem)
    trajectory_history = []
    function callback(args...)
        push!(trajectory_history, NamedTrajectory(get_datavec(prob), prob.trajectory))
        return true
    end

    return callback, trajectory_history
end

# *************************************************************************** #

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

@testitem "Callback can get best state trajectory" begin
    using MathOptInterface
    using NamedTrajectories
    const MOI = MathOptInterface
    include("../test/test_utils.jl")

    prob, system = smooth_quantum_state_problem(return_system=true)

    callback, best_trajs = best_rollout_fidelity_callback(prob)
    @test length(best_trajs) == 0

    # measure fidelity
    before = fidelity(prob)
    solve!(prob, max_iter=20, callback=callback)

    # length must increase if iterations are made
    @test length(best_trajs) > 0
    @test best_trajs[end] isa NamedTrajectory
    
    # fidelity ranking
    after = fidelity(prob)
    best = fidelity(best_trajs[end], system)
    
    @test before < after
    @test before < best
    @test after ≤ best
end

@testitem "Callback can get best unitary trajectory" begin
    using MathOptInterface
    using NamedTrajectories
    const MOI = MathOptInterface
    include("../test/test_utils.jl")

    prob, system = smooth_unitary_problem(return_system=true)

    callback, best_trajs = best_unitary_rollout_fidelity_callback(prob)
    @test length(best_trajs) == 0

    # measure fidelity
    before = unitary_fidelity(prob)
    solve!(prob, max_iter=20, callback=callback)

    # length must increase if iterations are made
    @test length(best_trajs) > 0
    @test best_trajs[end] isa NamedTrajectory
    
    # fidelity ranking
    after = unitary_fidelity(prob)
    best = unitary_fidelity(best_trajs[end], system)
    
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