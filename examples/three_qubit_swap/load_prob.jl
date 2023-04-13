using NamedTrajectories
using QuantumCollocation

experiment = "upsampled_T_500"
experiment_path = joinpath(@__DIR__, "upsampled_problems/$(experiment).jld2")

save_dir = joinpath(@__DIR__, "results")
plot_dir = joinpath(@__DIR__, "plots")

save_path = generate_file_path("jld2", experiment, save_dir)
plot_path = generate_file_path("png", experiment, plot_dir)

prob = load_problem(experiment_path)

println("plotting initial guess...")

plot(plot_path, prob.trajectory, [:Ũ⃗, :a]; ignored_labels=[:Ũ⃗])

println("solving problem...")
solve!(prob)
println()

fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)

println("Final fidelity: ", fid)

U_goal = iso_vec_to_operator(prob.trajectory.goal.Ũ⃗)

# |0⟩ rollout test
ψ = qubit_system_state("100")
ψ̃ = ket_to_iso(ψ)
ψ̃_goal = ket_to_iso(U_goal * ψ)
Ψ̃ = rollout(ψ̃, prob.trajectory.a, prob.trajectory.Δt, prob.system)
Ψ̃_exp = rollout(ψ̃, prob.trajectory.a, prob.trajectory.Δt, prob.system; integrator=exp)
pade_rollout_fidelity = fidelity(Ψ̃[:, end], ψ̃_goal)
exp_rollout_fidelity = fidelity(Ψ̃_exp[:, end], ψ̃_goal)
println("|100⟩ → U|100⟩ = |001⟩ pade rollout fidelity:  ", pade_rollout_fidelity)
println("|100⟩ → U|100⟩ = |001⟩ exp rollout fidelity:   ", exp_rollout_fidelity)

println("plotting solution...")
plot(plot_path, prob.trajectory, [:Ũ⃗, :a]; ignored_labels=[:Ũ⃗])

info = Dict(
    "solver fidelity" => fid,
    "pade rollout fidelity" => pade_rollout_fidelity,
    "exp rollout fidelity" => exp_rollout_fidelity,
    "pulse duration" => times(prob.trajectory)[end],
)

println("saving results...")
save_problem(save_path, prob, info)
