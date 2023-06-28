using NamedTrajectories
using QuantumCollocation

ω = 2π * 4.96 #GHz
α = -2π * 0.143 #GHz
levels = 2

U_goal = -im * [0 1; 1 0]

H_drift = α / 2 * quad(levels)
H_drives = [
    create(levels) + annihilate(levels),
    1im * (create(levels) - annihilate(levels))
]

a_bounds = [2π * 19e-3,  2π * 19e-3]

system = QuantumSystem(
    H_drift,
    H_drives
)

T = 100
Δt = 0.4
Q = 200.
R = 0.01
R_L1 = 1.0
cost = :infidelity_cost
dda_bound = 0.05

options = Options(
    watchdog_shortened_iter_trigger=0,
)

max_iter = 500

time = T * Δt

free_time = false

prob = UnitarySmoothPulseProblem(system, U_goal, T, Δt;
    free_time=free_time,
    a_bounds=a_bounds,
    dda_bound=dda_bound,
    max_iter=max_iter,
    Q=Q,
    R=R,
    ipopt_options=options,
)

solve!(prob)

Ũ⃗_final = unitary_rollout(prob.trajectory.initial.Ũ⃗, prob.trajectory.a, prob.trajectory.timestep, system)[:, end]

println("fidelity = ", unitary_fidelity(Ũ⃗_final, prob.trajectory.goal.Ũ⃗))

plot_dir = joinpath(@__DIR__, "plots/pi_gate/")

experiment = "T_$(T)_dt_$(Δt)_Q_$(Q)_R_$(R)_dda_bound_$(dda_bound)"

plot_path = generate_file_path("png", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:Ũ⃗, :a])

save_dir = joinpath(@__DIR__, "data/pi_gate/")

save_path = generate_file_path("jld2", experiment, save_dir)

save_problem(save_path, prob)
