using Pico
using NamedTrajectories
using Revise
using LinearAlgebra
using Distributions
using HSL

max_iter = 100
linear_solver = "mumps"
hsllib = nothing

U_init = 1.0 * I(8)

U_goal = [
    1 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0;
    0 0 1 0 0 0 0 0;
    0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0;
    0 0 0 0 0 1 0 0;
    0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 0 1
] |> Matrix{ComplexF64}

a_dag = create(2)
a = annihilate(2)

ωs = [5.18, 5.12, 5.06]
ω_d = 5.12

ξs = [0.01, 0.01, 0.01]

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
    J_12 * (lift(a_dag, 1, 3) * lift(a, 2, 3) + lift(a, 1, 3) * lift(a_dag, 2, 3)) +
    J_23 * (lift(a_dag, 2, 3) * lift(a, 3, 3) + lift(a, 2, 3) * lift(a_dag, 3, 3))

H_drives_Re = [lift(a, j, 3) + lift(a_dag, j, 3) for j = 1:3]
H_drives_Im = [1im * (lift(a, j, 3) - lift(a_dag, j, 3)) for j = 1:3]

H_drives = vcat(H_drives_Re, H_drives_Im)
n_drives = length(H_drives)

system = QuantumSystem(H_drift, H_drives)

U_dim = *(size(U_init)...)

Ũ⃗_init = unitary_to_iso_vec(U_init)
Ũ⃗_goal = unitary_to_iso_vec(U_goal)

Ũ⃗_dim = length(Ũ⃗_init)

T = 40
dt = 5.0
Δt_min = 0.5 * dt
Δt_max = 1.0 * dt
u_bound = 0.04 # GHz
u_dist = Uniform(-u_bound, u_bound)
ddu_bound = 0.1

load_saved_traj = false

if !load_saved_traj
    u = foldr(hcat, [zeros(n_drives), rand(u_dist, n_drives, T - 2), zeros(n_drives)])
    du = randn(n_drives, T)
    ddu = randn(n_drives, T)
    Δt = dt * ones(1, T)
end

Ũ⃗ = unitary_rollout(Ũ⃗_init, u, Δt, system)

comps = (
    Ũ⃗ = Ũ⃗,
    u = u,
    du = du,
    ddu = ddu,
    Δt = Δt
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
    controls=(:ddu, :Δt),
    dt=dt,
    dynamical_dts=true,
    bounds=bounds,
    initial=initial,
    final=final,
    goal=goal
)


P = FourthOrderPade(system)

function f(zₜ, zₜ₊₁)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components.Ũ⃗]
    Ũ⃗ₜ = zₜ[traj.components.Ũ⃗]
    uₜ₊₁ = zₜ₊₁[traj.components.u]
    uₜ = zₜ[traj.components.u]
    duₜ₊₁ = zₜ₊₁[traj.components.du]
    duₜ = zₜ[traj.components.du]

    dduₜ = zₜ[traj.components.ddu]
    Δtₜ = zₜ[traj.components.Δt][1]

    δŨ⃗ = P(Ũ⃗ₜ₊₁, Ũ⃗ₜ, uₜ, Δtₜ; operator=true)
    δu = uₜ₊₁ - uₜ - duₜ * Δtₜ
    δdu = duₜ₊₁ - duₜ - dduₜ * Δtₜ

    return vcat(δŨ⃗, δu, δdu)
end

Q = 100.0

loss = :UnitaryInfidelityLoss

J = QuantumObjective(:Ũ⃗, traj, loss, Q)

R_ddu = 1e-3

J += QuadraticRegularizer(:ddu, traj, R_ddu * ones(n_drives))


options = Options(
    max_iter=max_iter,
    linear_solver=linear_solver,
    hsllib=hsllib,
)

prob = QuantumControlProblem(system, traj, J, f;
    options=options,
)

plot_dir = "examples/scripts/plots/three_qubit_swap/"

experiment = "T_$(T)_Q_$(Q)_iter_$(max_iter)"

plot_path = generate_file_path("png", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:Ũ⃗, :u]; ignored_labels=[:Ũ⃗], dt_name=:Δt)

solve!(prob)

fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)

experiment *= "_fidelity_$(fid)"

plot_path = generate_file_path("png", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:Ũ⃗, :u]; ignored_labels=[:Ũ⃗], dt_name=:Δt)

println("Final fidelity: ", fid)
