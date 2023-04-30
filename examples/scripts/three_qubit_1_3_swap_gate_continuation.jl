using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using Distributions
using Manifolds

max_iter = 200
linear_solver = "pardiso"

U_init = 1.0 * I(8)

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



# U_goal = [
#     1 0 0 0 0 0 0 0;
#     0 0 0 0 1 0 0 0;
#     0 0 1 0 0 0 0 0;
#     0 0 0 0 0 0 1 0;
#     0 1 0 0 0 0 0 0;
#     0 0 0 0 0 1 0 0;
#     0 0 0 1 0 0 0 0;
#     0 0 0 0 0 0 0 1
# ] |> Matrix{ComplexF64}

a_dag = create(2)
a = annihilate(2)

ωs = 2π * [5.18, 5.12, 5.06]
ω_d = 2π * 5.12

ξs = 2π * fill(0.340, 3) # ξ = 0.340 GHz

J_12 = 5.0e-3
J_23 = 5.0e-3


function lift(U, q, n; l=2)
    Is = Matrix{Number}[I(l) for i in 1:n]
    Is[q] = U
    return foldr(kron, Is)
end

lift(number(2), 1, 3)
lift(a_dag, 1, 3)
lift(a, 1, 3)

H_drift = sum(
    (ωs[q] - ω_d) * lift(a_dag, q, 3) * lift(a, q, 3) -
    ξs[q] / 2 * lift(a_dag, q, 3) * lift(a_dag, q, 3) * lift(a, q, 3) * lift(a, q, 3)
        for q = 1:3
)

# dispersive coupling
H_drift +=
    J_12 * (
        lift(a_dag, 1, 3) * lift(a, 2, 3) +
        lift(a, 1, 3) * lift(a_dag, 2, 3)
    ) +
    J_23 * (
        lift(a_dag, 2, 3) * lift(a, 3, 3) +
        lift(a, 2, 3) * lift(a_dag, 3, 3)
    )

H_drives_Re = [lift(a, j, 3) + lift(a_dag, j, 3) for j = 1:3]
H_drives_Im = [1im * (lift(a, j, 3) - lift(a_dag, j, 3)) for j = 1:3]

H_drives = vcat(H_drives_Re, H_drives_Im)
n_drives = length(H_drives)

system = QuantumSystem(H_drift, H_drives)

U_dim = *(size(U_init)...)

N = size(U_init, 1)

Ũ⃗_init = operator_to_iso_vec(U_init)
Ũ⃗_goal = operator_to_iso_vec(U_goal)

Ũ⃗_dim = length(Ũ⃗_init)

traj_path = "examples/scripts/trajectories/three_qubits/swap_gate/continuation/T_50_Q_200.0_iter_200_fidelity_0.999994882002093_00000.jld2"
init_traj = load_traj(traj_path)


T = init_traj.T
dt = 4.0
Δt_min = 0.5 * dt
Δt_max = 2.0 * dt
u_bound = 2π * 0.04 # GHz
u_dist = Uniform(-u_bound, u_bound)
ddu_bound = 0.001



u = foldr(hcat, [zeros(n_drives), rand(u_dist, n_drives, T - 2), zeros(n_drives)])
du = randn(n_drives, T)
ddu = randn(n_drives, T)



comps = (
    Ũ⃗ = init_traj.Ũ⃗,
    G⃗ = init_traj.G⃗,
    u = init_traj.u,
    du = init_traj.du,
    ddu = init_traj.ddu,
    Δt = init_traj.Δt
)

bounds = (
    u = fill(u_bound, n_drives),
    ddu = fill(ddu_bound, n_drives),
    Δt = (Δt_min, Δt_max),
)

initial = (
    Ũ⃗ = Ũ⃗_init,
    u = zeros(n_drives),
)

final = (
    u = zeros(n_drives),
)

goal = (
    Ũ⃗ = Ũ⃗_goal,
)

traj = NamedTrajectory(
    comps;
    controls=(:ddu, :G⃗, :Δt),
    timestep=dt,
    dynamical_dts=true,
    bounds=bounds,
    initial=initial,
    final=final,
    goal=goal
)


P = FourthOrderPade(system)

@views function f(zₜ, zₜ₊₁)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components.Ũ⃗]
    Ũ⃗ₜ = zₜ[traj.components.Ũ⃗]
    uₜ₊₁ = zₜ₊₁[traj.components.u]
    uₜ = zₜ[traj.components.u]
    duₜ₊₁ = zₜ₊₁[traj.components.du]
    duₜ = zₜ[traj.components.du]

    G⃗ₜ = zₜ[traj.components.G⃗]
    G = skew_symmetric(G⃗ₜ, 2N)

    dduₜ = zₜ[traj.components.ddu]
    Δtₜ = zₜ[traj.components.Δt][1]

    δŨ⃗ = P(Ũ⃗ₜ₊₁, Ũ⃗ₜ, uₜ, Δtₜ; operator=true, G_additional=G)
    δu = uₜ₊₁ - uₜ - duₜ * Δtₜ
    δdu = duₜ₊₁ - duₜ - dduₜ * Δtₜ

    return vcat(δŨ⃗, δu, δdu)
end

Q = 200.0

loss = :UnitaryInfidelityLoss

J = QuantumObjective(:Ũ⃗, traj, loss, Q)

R_u = 1e-2

J += QuadraticRegularizer(:u, traj, R_u * ones(n_drives))

R_du = 1e-2

J += QuadraticRegularizer(:du, traj, R_du * ones(n_drives))

R_ddu = 1e-2

J += QuadraticRegularizer(:ddu, traj, R_ddu * ones(n_drives))

R_G = 2e2

J += QuadraticRegularizer(:G⃗, traj, R_G * ones(traj.dims.G⃗))

options = Options(
    max_iter=max_iter,
    linear_solver=linear_solver,
)

prob = QuantumControlProblem(system, traj, J, f;
    options=options,
)

plot_dir = "examples/scripts/plots/three_qubit_swap/"

experiment = "T_$(T)_Q_$(Q)_iter_$(max_iter)"

plot_path = generate_file_path("png", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:Ũ⃗, :u]; ignored_labels=[:Ũ⃗], timestep_name=:Δt)

solve!(prob)

fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)

experiment *= "_fidelity_$(fid)"


println("Final fidelity: ", fid)

# |0⟩ rollout test
ψ = qubit_system_state("000")
ψ̃ = ket_to_iso(ψ)
ψ̃_goal = ket_to_iso(U_goal * ψ)
Ψ̃ = rollout(ψ̃, prob.trajectory.u, prob.trajectory.Δt, system)
println("|0⟩ Rollout fidelity:   ", fidelity(Ψ̃[:, end], ψ̃_goal))

plot_path = generate_file_path("png", experiment, plot_dir)


add_component!(prob.trajectory, :ψ̃, Ψ̃)

plot(plot_path, prob.trajectory, [:Ũ⃗, :ψ̃, :u]; ignored_labels=[:Ũ⃗, :ψ̃], timestep_name=:Δt)

save_dir = "examples/scripts/trajectories/three_qubits/swap_gate/continuation"
save_path = generate_file_path("jld2", experiment, save_dir)
save(save_path, prob.trajectory)
