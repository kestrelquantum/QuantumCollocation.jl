using QuantumCollocation
import QuantumCollocation: lift
using NamedTrajectories
using LinearAlgebra

levels = 2

qubits = 4

α = [225.78, 100.33, 189.32, 172.15] * 1e-3 # GHz

χ = Symmetric([
    0 -5.10982939 -0.18457118 -0.50235316;
    0       0     -0.94914758 -1.07618574;
    0       0           0     -0.44607489;
    0       0           0           0
]) * 1e-3 # GHz

â_dag = create(levels)
â = annihilate(levels)

lift(op, i) = lift(op, i, qubits; l=levels)

# drift hamiltonian for ith qubit
H_q(i) = -α[i] / 2 * lift(â_dag, i)^2 * lift(â, i)^2

# drift interaction hamiltonian for ith and jth qubit
H_c_ij(i, j) = χ[i, j] * lift(â_dag, i) * lift(â, i) * lift(â_dag, j) * lift(â, j)

# drive hamiltonian for ith qubit, real part
H_d_real(i) = 1 / 2 * (lift(â_dag, i) + lift(â, i))

# drive hamiltonian for ith qubit, imaginary part
H_d_imag(i) = 1im / 2 * (lift(â_dag, i) - lift(â, i))

# total drift hamiltonian
H_drift =
    sum(H_q(i) for i = 1:qubits) +
    sum(H_c_ij(i, j) for i = 1:qubits, j = 1:qubits if j > i)

H_drift *= 2π

# make vector of drive hamiltonians: [H_d_real(1), H_d_imag(1), H_d_real(2), ...]
# there's probably a cleaner way to do this lol
# H_drives = collect.(vec(vcat(
#     transpose(Matrix{ComplexF64}.([H_d_real(i) for i = 1:qubits])),
#     transpose(Matrix{ComplexF64}.([H_d_imag(i) for i = 1:qubits]))
# )))

# H_drives = Matrix{ComplexF64}.([H_d_real(1), H_d_imag(1), H_d_real(2), H_d_imag(2)])
H_drives = Matrix{ComplexF64}.([H_d_real(2), H_d_imag(2)])
H_drives .*= 2π

# make quantum system
system = QuantumSystem(H_drift, H_drives)

# create goal unitary
Id = 1.0I(levels)
g = cavity_state(0, levels)
e = cavity_state(1, levels)
eg = e * g'
ge = g * e'
U_goal = Id ⊗ (eg + ge) ⊗ Id ⊗ Id

# time parameters
duration = 100.0 # ns
T = 100
Δt = duration / T
Δt_max = 1.2 * Δt
Δt_min = 0.1 * Δt

# drive constraint: 20 MHz (linear units)
a_bound = 20 * 1e-3 # GHz

# pulse acceleration (used to control smoothness)
dda_bound = 2e-3

# maximum number of iterations
max_iter = 500

# warm start
warm_start = false

if warm_start
    data_path = joinpath(@__DIR__, "data/limited_drives_T_100_dt_1.0_dda_0.001_a_0.02_max_iter_500_00000.jld2")
    data = load_problem(data_path; return_data=true)
    init_traj = data["trajectory"]
    init_drives = init_traj.a
    init_Δt = init_traj.Δt[end]
end

prob = UnitarySmoothPulseProblem(
    system,
    U_goal,
    T,
    warm_start ? init_Δt : Δt;
    Δt_max=Δt_max,
    Δt_min=Δt_min,
    a_bound=a_bound,
    dda_bound=dda_bound,
    max_iter=max_iter,
    a_guess=warm_start ? init_drives : nothing,
)

save_dir = joinpath(@__DIR__, "data")
plot_dir = joinpath(@__DIR__, "plots")

experiment_name = "limited_drives_T_$(T)_dt_$(Δt)_dda_$(dda_bound)_a_$(a_bound)_max_iter_$(max_iter)"

save_path = generate_file_path("jld2", experiment_name, save_dir)
plot_path = generate_file_path("png", experiment_name, plot_dir)

# plot the initial guess for the wavefunction and controls
plot(plot_path, prob.trajectory, [:a])

solve!(prob)

# plot the final solution for the wavefunction and controls
plot(plot_path, prob.trajectory, [:a])

A = prob.trajectory.a
Δt = prob.trajectory.Δt

Ũ⃗_final = unitary_rollout(A, Δt, system; integrator=exp)[:, end]
final_fidelity = unitary_fidelity(Ũ⃗_final, prob.trajectory.goal.Ũ⃗)
println("Final fidelity: $final_fidelity")

duration = times(prob.trajectory)[end]

info = Dict(
    "final_fidelity" => final_fidelity,
    "duration" => duration,
)

# save the solution
save_problem(save_path, prob, info)
