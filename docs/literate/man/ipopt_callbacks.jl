# # IpOpt Solver Callbacks

# This page describes the callback functions that can be used with the IpOpt solver (in the future, may describe more general callback behavior).

# ## Callbacks

# By default, IpOpt callbacks are called at each optimization step with the following signature:
using QuantumCollocation
using NamedTrajectories

import ..QuantumStateSmoothPulseProblem
import ..Callbacks

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
    return true
end

# This gives the user access to some of the optimization state internals at each iteration.
# A callback function with any subset of these arguments can be passed into the `solve!` function via the `callback` keyword argument see below.
# ```@docs
# ProblemSolvers.solve!(prob::QuantumControlProblem; callback=nothing)
# ```

# The callback function can be used to monitor the optimization progress, save intermediate results, or modify the optimization process.
# For example, the following callback function saves the optimization trajectory at each iteration:

trajectory_history = []
function get_history_callback(
    kwargs...
)
    push!(trajectory_history, QuantumCollocation.Problems.get_datavec(prob))
    return true
end

# The callback function can also be used to stop the optimization early by returning `false`. The following callback when passed to `solve!` will stop the optimization after the first iteration:
my_callback = (kwargs...) -> false

T = 50
Δt = 0.2
sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
ψ_init =  Vector{ComplexF64}([1.0, 0.0])
ψ_target =  Vector{ComplexF64}([0.0, 1.0])

# Single initial and target states
# --------------------------------
prob = QuantumStateSmoothPulseProblem(
    sys, ψ_init, ψ_target, T, Δt;
    ipopt_options=IpoptOptions(print_level=1), 
    piccolo_options=PiccoloOptions(verbose=false)
)

trajectory_history = []
# using the get_history_callback function to save the optimization trajectory at each iteration into the trajectory_history 
solve!(prob, max_iter=20, callback=get_history_callback)

for (iter, traj) in enumerate(trajectory_history)
    # get the length of the trajectory history depending on length and left pad the index with leading zeros
    str_index = lpad(iter, length(string(length(trajectory_history))), "0")
    # plot the trajectory but on fixed xaxis and yaxis
    plot("./iteration-$str_index-trajectory.png", NamedTrajectory(traj, prob.trajectory),  [:ψ̃1, :a], xlims=(-Δt, (T+5)*Δt), plot_ylims=(ψ̃1 = (-2, 2), a = (-1.1, 1.1)))
end

# Using a callback to get the best trajectory from all the optimization iterations
T = 50
Δt = 0.2
sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
ψ_init =  Vector{ComplexF64}([0.0, 1.0])
ψ_target =  Vector{ComplexF64}([1.0, 0.0])

# Single initial and target states
# --------------------------------
prob = QuantumStateSmoothPulseProblem(
    sys, ψ_init, ψ_target, T, Δt;
    ipopt_options=IpoptOptions(print_level=1), 
    piccolo_options=PiccoloOptions(verbose=false)
)

(best_traj_callback, best_traj) = make_save_best_trajectory_callback(prob, prob.trajectory)
solve!(prob, max_iter=20, callback=best_traj_callback)
# fidelity of the last iterate
@show fidelity(prob)
# fidelity of the best iterate
@show fidelity(best_traj)
