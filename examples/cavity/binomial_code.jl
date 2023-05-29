using QuantumCollocation
using NamedTrajectories

transmon_levels = 3
cavity_levels = 14

system = MultiModeSystem(transmon_levels, cavity_levels)

g0 = multimode_state("g0", transmon_levels, cavity_levels)
e0 = multimode_state("e0", transmon_levels, cavity_levels)

g1 = multimode_state("g1", transmon_levels, cavity_levels)
g2 = multimode_state("g1", transmon_levels, cavity_levels)
g4 = multimode_state("g4", transmon_levels, cavity_levels)

ψ_init = [g0, e0]
ψ_goal = [(g0 + g4) / √2, g2]

qubit_a_bound = 0.153
cavity_a_bound = 0.193

dda_bound = 1e-4

a_bounds = [qubit_a_bound, qubit_a_bound, cavity_a_bound, cavity_a_bound]

T = 200
Δt = 10.0
Δt_max = Δt
Δt_min = 0.2Δt
Q = 200.0
R_L1 = 1.0
max_iter = 1
cavity_forbidden_states = cavity_levels .* [1, 2, 3, 4]
transmon_forbidden_states = [collect((2 * transmon_levels - 1) * cavity_levels + 1 : 2 * transmon_levels * cavity_levels); 
                             collect((transmon_levels - 1) * cavity_levels + 1: cavity_levels * transmon_levels)

forbidden_states = [transmon_forbidden_states; cavity_forbidden_states]

prob = QuantumStateSmoothPulseProblem(system, ψ_init, ψ_goal, T, Δt;
    Δt_max=Δt_max,
    Δt_min=Δt_min,
    dda_bound=dda_bound,
    a_bounds=a_bounds,
    L1_regularized_names=[:ψ̃1, :ψ̃2],
    L1_regularized_indices=(ψ̃1 = forbidden_states, ψ̃2 = forbidden_states),
    Q=Q,
    R_L1=R_L1,
)

experiment = "T_$(T)_dt_$(Δt)_Q_$(Q)_R_L1_$(R_L1)_max_iter_$(max_iter)_dda_bound_$(dda_bound)"

plot_dir = joinpath(@__DIR__, "plots/binomial_code")
save_dir = joinpath(@__DIR__, "data/binomial_code")

plot_path = generate_file_path("png", experiment, plot_dir)
save_path = generate_file_path("jld2", experiment, save_dir)

plot(plot_path, prob.trajectory, [:ψ̃1, :ψ̃2, :a]; ignored_labels=[:ψ̃1, :ψ̃2])

solve!(prob; max_iter=max_iter, save_path=save_path)

plot(plot_path, prob.trajectory, [:ψ̃1, :ψ̃2, :a]; ignored_labels=[:ψ̃1, :ψ̃2])
