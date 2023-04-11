using Revise
using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using Distributions
using Manifolds

max_iter = 5000
linear_solver = "mumps"

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

a_dag = create(2)
a = annihilate(2)

ωs = 2π * [5.18, 5.12, 5.06]
ω_d = 2π * 5.12

ξs = 2π * [0.34, 0.34, 0.34]

J_12 = 2π * 5.0e-3
J_23 = 2π * 5.0e-3


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

load = :no_load

if load == :continuation
    traj_path = "examples/scripts/trajectories/three_qubits/swap_gate/continuation/T_50_Q_200.0_iter_200_fidelity_0.9999813247231153_00000.jld2"
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
        controls=(:ddu, :Δt),
        timestep=dt,
        dynamical_dts=true,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
elseif load == :post_continuation
    traj_path = "examples/scripts/trajectories/three_qubits/swap_gate/post_continuation/T_50_Q_100.0_iter_200_fidelity_0.7857872051064816_00000.jld2"
    init_traj = load_traj(traj_path)

    T = init_traj.T
    dt = 4.0
    Δt_min = 0.5 * dt
    Δt_max = 2.0 * dt
    u_bound = 2π * 0.04 # GHz
    u_dist = Uniform(-u_bound, u_bound)
    ddu_bound = 0.0005

    comps = (
        Ũ⃗ = init_traj.Ũ⃗,
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
        controls=(:ddu, :Δt),
        timestep=dt,
        dynamical_dts=true,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
elseif load == :no_load
    T = 100
    dt = 2.0
    Δt_min = 0.5 * dt
    Δt_max = 2.0 * dt
    u_bound = 2π * 0.04 # GHz
    u_dist = Uniform(-u_bound, u_bound)
    ddu_bound = 0.01

    Ũ⃗ = unitary_geodesic(U_goal, T)

    u = foldr(hcat, [zeros(n_drives), rand(u_dist, n_drives, T - 2), zeros(n_drives)])
    du = randn(n_drives, T)
    ddu = randn(n_drives, T)


    comps = (
        Ũ⃗ = Ũ⃗,
        u = u,
        du = du,
        ddu = ddu,
        Δt = fill(dt, 1, T)
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
        controls=(:u, :Δt),
        timestep=dt,
        dynamical_timesteps=true,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end


f = [
    UnitaryPadeIntegrator(system, :Ũ⃗, :u, :Δt),
    DerivativeIntegrator(:u, :du, :Δt, traj.dims[:u]),
    DerivativeIntegrator(:du, :ddu, :Δt, traj.dims[:du]),
]

Q = 100.0

loss = :UnitaryInfidelityLoss

J = QuantumObjective(:Ũ⃗, traj, loss, Q)

R_u = 1e-2

J += QuadraticRegularizer(:u, traj, R_u * ones(n_drives))

# R_u_smoothness = 1e-2

# J += QuadraticSmoothnessRegularizer(:u, traj, R_u_smoothness * ones(n_drives))

R_du = 1e-2

J += QuadraticRegularizer(:du, traj, R_du * ones(n_drives))

R_ddu = 1e-2

J += QuadraticRegularizer(:ddu, traj, R_ddu * ones(n_drives))


options = Options(
    max_iter=max_iter,
    linear_solver=linear_solver,
)

prob = QuantumControlProblem(system, traj, J, f;
    options=options,
)

plot_dir = "examples/scripts/plots/three_qubit_swap/" * string(load)

experiment = "T_$(traj.T)_Q_$(Q)_iter_$(max_iter)"

plot_path = generate_file_path("png", experiment, plot_dir)

plot(plot_path, prob.trajectory, [:Ũ⃗, :u]; ignored_labels=[:Ũ⃗], dt_name=:Δt)

solve!(prob)

fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)

experiment *= "_fidelity_$(fid)"


println("Final fidelity: ", fid)

# |0⟩ rollout test
ψ = qubit_system_state("100")
ψ̃ = ket_to_iso(ψ)
ψ̃_goal = ket_to_iso(U_goal * ψ)
Ψ̃ = rollout(ψ̃, prob.trajectory.u, prob.trajectory.Δt, system)
Ψ̃_exp = rollout(ψ̃, prob.trajectory.u, prob.trajectory.Δt, system; integrator=exp)
println("|100⟩ → U|100⟩ = |001⟩ Pade rollout fidelity:  ", fidelity(Ψ̃[:, end], ψ̃_goal))
println("|100⟩ → U|100⟩ = |001⟩ exp rollout fidelity:   ", fidelity(Ψ̃_exp[:, end], ψ̃_goal))

plot_path = generate_file_path("png", experiment, plot_dir)

add_component!(prob.trajectory, :ψ̃, Ψ̃)

plot(plot_path, prob.trajectory, [:Ũ⃗, :ψ̃, :u]; ignored_labels=[:Ũ⃗, :ψ̃], dt_name=:Δt)

save_dir = "examples/scripts/trajectories/three_qubits/swap_gate/" * string(load)
save_path = generate_file_path("jld2", experiment, save_dir)
save(save_path, prob.trajectory)
