using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using Distributions

# setting maximum number of iterations
max_iter = 1000
linear_solver = "mumps"

# defining levels for single qubit system
n_levels = 2

# definining pauli matrices
σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

# definining initial value of unitary
U_init = 1.0 * I(n_levels)

# definining goal value of unitary
gate = :Y
U_goal = GATES[gate]

# defining pauli ladder operators
σ₋ = 0.5 * (σx + 1im * σy)
σ₊ = 0.5 * (σx - 1im * σy)

# defining drive Hamiltonians for system
H_drives = 1 / 2 * [σ₋ + σ₊, 1im * (σ₋ - σ₊), σz]

# building quantum system (no drift term)
system = QuantumSystem(H_drives)

# converting unitaries to isomorphic vector representation
Ũ⃗_init = operator_to_iso_vec(U_init)
Ũ⃗_goal = operator_to_iso_vec(U_goal)

# getting dimension of the isomorphic vector representation
Ũ⃗_dim = length(Ũ⃗_init)

# defining time parameters
max_duration =  1.678
T = 100
dt = max_duration / T
dt_max = 1.0 * dt
dt_min = 0.1 * dt

# boudns on controls
# TODO: implement nonlinear constraints for abs val of γ ∈ C
γ_bound = 0.3 * 2π # rad / μs = 2π * 3e5 rad / s
α_bound = 0.1 * 2π # rad / μs = 2π * 1e5 rad / s

# dimensions of controls
γ_dim = 2
α_dim = 1

# initialization distributions for controls
γ_dist = Uniform(-γ_bound, γ_bound)
α_dist = Uniform(-α_bound, α_bound)

# load saved trajectory
load_saved_traj = true

if load_saved_traj
    data_path = "examples/single_qubit/results/mintime/Y_gate_T_100_Q_100.0_R_0.0001_R_smoothness_0.001_iter_10000_fidelity_0.9999999962678012_00000.jld2"
    traj = load_problem(data_path; return_data=true)["trajectory"]
    new_bounds = (Δt = (fill(0.1traj.Δt[end], T), fill(traj.Δt[end], T)),)
    traj.bounds = merge(traj.bounds, new_bounds)
else
    γ = foldr(hcat, [zeros(γ_dim), rand(γ_dist, γ_dim, T - 2), zeros(γ_dim)])
    α = foldr(hcat, [zeros(α_dim), rand(α_dist, α_dim, T - 2), zeros(α_dim)])
    Δt = dt * ones(1, T)

    Ũ⃗ = unitary_geodesic(U_goal, T; return_generator=false)

    # defining components for trajectory
    comps = (
        Ũ⃗ = Ũ⃗,
        γ = γ,
        α = α,
        Δt = Δt
    )

    # defining bounds
    bounds = (;
        Δt = (dt_min, dt_max)
    )

    # defining initial values
    initial = (
        Ũ⃗ = Ũ⃗_init,
        γ = zeros(γ_dim),
        α = zeros(α_dim)
    )

    # defining final values
    final = (
        γ = zeros(γ_dim),
        α = zeros(α_dim),
    )

    # defining goal states
    goal = (
        Ũ⃗ = Ũ⃗_goal,
    )

    # creating named trajectory
    traj = NamedTrajectory(
        comps;
        # controls=(:ddγ, :ddα, :Δt),
        controls=(:γ, :α, :Δt),
        timestep=:Δt,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end

# creating fourth order pade integrator
P = UnitaryPadeIntegrator(system, :Ũ⃗, (:γ, :α), :Δt)

# quantum objective weight parameter
Q = 1.0e2

# creating quantum objective
J = QuantumUnitaryObjective(:Ũ⃗, traj, Q)

# regularization parameters
R = 1e-4
R_smoothness = 1e-3
drive_bound_ratio = γ_bound / α_bound

R_γ = R
R_α = R * drive_bound_ratio

R_γ_smoothness = R_smoothness
R_α_smoothness = R_smoothness * drive_bound_ratio

# adding regularization terms on γ to the objective
J += QuadraticRegularizer(:γ, traj, R_γ)
# J += QuadraticSmoothnessRegularizer(:γ, traj, R_γ_smoothness)

# adding regularization terms on α to the objective
J += QuadraticRegularizer(:α, traj, R_α)
# J += QuadraticSmoothnessRegularizer(:α, traj, R_α_smoothness)

# Ipopt options
options = Options(
    max_iter=max_iter,
    linear_solver=linear_solver,
)

# defining constraints
constraints = [
    TimeStepsAllEqualConstraint(:Δt, traj),
    ComplexModulusContraint(:γ, γ_bound, traj),
]

# defining quantum control problem
prob = QuantumControlProblem(system, traj, J, P;
    options=options,
    constraints=constraints,
    verbose=true
)

# plotting directory
plot_dir = joinpath(@__DIR__, "plots/freetime/")

# experiment name
experiment =
    "$(gate)_gate_T_$(T)_Q_$(Q)_" *
    "R_$(R)_R_smoothness_$(R_smoothness)_" *
    "iter_$(max_iter)"

# creating unique plotting path
plot_path = generate_file_path("png", experiment, plot_dir)

# plotting initial trajectory
plot(plot_path, prob.trajectory, [:Ũ⃗, :γ, :α]; ignored_labels=[:Ũ⃗])

# solving the problem
solve!(prob)

# calculating unitary fidelity
fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)
println("Final unitary fidelity: ", fid)

drives = vcat(prob.trajectory.γ, prob.trajectory.α)
Δts = vec(prob.trajectory.Δt)

# |0⟩ rollout test
ψ₁ = [1, 0]
ψ̃₁ = ket_to_iso(ψ₁)
ψ̃₁_goal = ket_to_iso(U_goal * ψ₁)
Ψ̃₁_fourth_order_pade = rollout(ψ̃₁, drives, Δts, system)
Ψ̃₁_exp = rollout(ψ̃₁, drives, Δts, system; integrator=exp)
println("|0⟩ Fourth order Pade rollout fidelity:   ", fidelity(Ψ̃₁_fourth_order_pade[:, end], ψ̃₁_goal))
println("|0⟩ Exponential rollout fidelity:         ", fidelity(Ψ̃₁_exp[:, end], ψ̃₁_goal))
println()


# |1⟩ rollout test
ψ₂ = [0, 1]
ψ̃₂ = ket_to_iso(ψ₂)
ψ̃₂_goal = ket_to_iso(U_goal * ψ₂)
Ψ̃₂_fourth_order_pade = rollout(ψ̃₂, drives, Δts, system)
Ψ̃₂_exp = rollout(ψ̃₂, drives, Δts, system; integrator=exp)
println("|1⟩ Fourth order Pade rollout fidelity:   ", fidelity(Ψ̃₂_fourth_order_pade[:, end], ψ̃₂_goal))
println("|1⟩ Exponential rollout fidelity:         ", fidelity(Ψ̃₂_exp[:, end], ψ̃₂_goal))


println()
println("duration: ", times(prob.trajectory)[end])

# new plot name with fidelity included
experiment *= "_fidelity_$(fid)"
plot_path = join(split(plot_path, ".")[1:end-1], ".") * "_fidelity_$(fid).png"

plot(plot_path, prob.trajectory;
    ignored_labels=[:Ũ⃗],
)

save_dir = "examples/single_qubit/results"
save_path = generate_file_path("jld2", experiment, save_dir)

save_problem(save_path, prob)