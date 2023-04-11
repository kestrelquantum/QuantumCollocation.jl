using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using Distributions
using Manifolds

max_iter = 5000
linear_solver = "mumps"

e0 = [1, 0]
e1 = [0, 1]

Id = GATES[:I]

e00 = e0 * e0'
e11 = e1 * e1'
e01 = e0 * e1'
e10 = e1 * e0'

U_goal_analytic =
    e00 ⊗ Id ⊗ e00 +
    e10 ⊗ Id ⊗ e01 +
    e01 ⊗ Id ⊗ e10 +
    e11 ⊗ Id ⊗ e11

U_goal = Matrix{ComplexF64}(U_goal_analytic)

â_dag = create(2)
â = annihilate(2)

ωs = 2π .* [5.18, 5.12, 5.06]
ω_d = 2π * 5.12

ξs = 2π * [0.34, 0.34, 0.34]

J_12 = 2π * 5.0e-3
J_23 = 2π * 5.0e-3


H_drift = sum(
    (ωs[q] - ω_d) * lift(â_dag, q, 3) * lift(â, q, 3) -
    ξs[q] / 2 * lift(â_dag, q, 3) * lift(â_dag, q, 3) * lift(â, q, 3) * lift(â, q, 3)
        for q = 1:3
)

# dispersive coupling
H_drift +=
    J_12 * (
        lift(â_dag, 1, 3) * lift(â, 2, 3) +
        lift(â, 1, 3) * lift(â_dag, 2, 3)
    ) +
    J_23 * (
        lift(â_dag, 2, 3) * lift(â, 3, 3) +
        lift(â, 2, 3) * lift(â_dag, 3, 3)
    )

H_drives_Re = [lift(â, j, 3) + lift(â_dag, j, 3) for j = 1:3]
H_drives_Im = [1im * (lift(â, j, 3) - lift(â_dag, j, 3)) for j = 1:3]

H_drives = vcat(H_drives_Re, H_drives_Im)

a_bound = 2π * 0.04 # GHz, a guess!
dda_bound = 0.001

T = 100
Δt = 2.0
Δt_min = 0.5 * Δt
Δt_max = 1.5 * Δt

prob = UnitarySmoothPulseProblem(
    H_drift,
    H_drives,
    U_goal,
    T,
    Δt;
    a_bound = a_bound,
    dda_bound = dda_bound,
    Δt_min = Δt_min,
    Δt_max = Δt_max,
    max_iter = max_iter,
    linear_solver = linear_solver,
)

experiment = "T_$(T)_Δt_$(Δt)_a_bound_$(a_bound)_dda_bound_$(dda_bound)_dt_min_$(Δt_min)_dt_max_$(Δt_max)_max_iter_$(max_iter)"

save_dir = joinpath(@__DIR__, "results")
plot_dir = joinpath(@__DIR__, "plots")

save_path = generate_file_path(".jld2", experiment, save_dir)
plot_path = generate_file_path(".pdf", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:Ũ⃗, :a])

solve!(prob; save_path=save_path)

plot(plot_path, prob.trajectory, [:Ũ⃗, :a])

println("Final fidelity: ", fid)

# |0⟩ rollout test
ψ = qubit_system_state("100")
ψ̃ = ket_to_iso(ψ)
ψ̃_goal = ket_to_iso(U_goal * ψ)
Ψ̃ = rollout(ψ̃, prob.trajectory.u, prob.trajectory.Δt, system)
Ψ̃_exp = rollout(ψ̃, prob.trajectory.u, prob.trajectory.Δt, system; integrator=exp)
println("|100⟩ → U|100⟩ = |001⟩ pade rollout fidelity:  ", fidelity(Ψ̃[:, end], ψ̃_goal))
println("|100⟩ → U|100⟩ = |001⟩ exp rollout fidelity:   ", fidelity(Ψ̃_exp[:, end], ψ̃_goal))
