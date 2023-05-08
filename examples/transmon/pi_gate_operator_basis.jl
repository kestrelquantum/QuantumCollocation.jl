using NamedTrajectories
using QuantumCollocation

ω = 2π * 4.96 #GHz
α = -2π * 0.143 #GHz
levels = 3

ψg = [1. + 0*im, 0 , 0]
ψe = [0, 1. + 0*im, 0]

ψ1 = [ψg, ψe]
ψf = [-im * ψe, -im * ψg]

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

T = 400
Δt = 0.1
Q = 200.
R = 0.01
R_L1 = 1.0
dda_bound = 0.05

max_iter = 1000

time = T * Δt

prob = QuantumStateSmoothPulseProblem(system, ψ1, ψf, T, Δt;
    a_bounds=a_bounds,
    dda_bound=dda_bound,
    L1_regularized_names=[:ψ̃1, :ψ̃2],
    L1_regularized_indices=(ψ̃1 = [3,6], ψ̃2 = [3,6]),
    max_iter=max_iter,
    Q=Q,
    R=R,
    R_L1=R_L1,
)

solve!(prob)

println("fidelity = ", fidelity(prob.trajectory[end].ψ̃1, prob.trajectory.goal.ψ̃1))

plot_dir = joinpath(@__DIR__, "plots/pi_gate/")

experiment = "T_$(T)_dt_$(Δt)_Q_$(Q)_R_$(R)_R_L1_$(R_L1)_dda_bound_$(dda_bound)"

plot_path = generate_file_path("png", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:ψ̃1, :a])
