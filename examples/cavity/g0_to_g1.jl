using NamedTrajectories
using QuantumCollocation

system = MultiModeSystem(3, 14)

ψ_init = multimode_state("g0", 3, 14)
ψ_goal = multimode_state("g1", 3, 14)

T = 100
Δt = 10.0
Δt_max = Δt
Δt_min = 0.2Δt
dda_bound = 1e-5
a_bounds = [0.153, 0.153, 0.193, 0.193]
max_iter = 1000

plot_dir = joinpath(@__DIR__, "plots")
experiment = "g0_to_g1_test"

plot_path = joinpath(plot_dir, experiment*".png")

prob = QuantumStateSmoothPulseProblem(system, ψ_init, ψ_goal, T, Δt;
    Δt_max=Δt_max,
    Δt_min=Δt_min,
    dda_bound=dda_bound,
    a_bounds=a_bounds,
)

plot(plot_path, prob.trajectory, [:ψ̃, :a]; ignored_labels=[:ψ̃])

solve!(prob; max_iter=max_iter)

plot(plot_path, prob.trajectory, [:ψ̃, :a]; ignored_labels=[:ψ̃])
