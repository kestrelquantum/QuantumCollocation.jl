using QuantumCollocation
using NamedTrajectories

transmon_levels = 3
cavity_levels = 14

data_path = "/home/aditya/oc_projects/QuantumCollocation.jl/examples/cavity/data/binomial_code/transmon_3_T_200_dt_15.0_Q_200.0_R_L1_10.0_max_iter_5000_dda_bound_1.0e-5_00000.jld2"

# data = load_problem(data_path; return_data=true)
# init_traj = data["trajectory"]

# system = MultiModeSystem(transmon_levels, cavity_levels)

# g0 = multimode_state("g0", transmon_levels, cavity_levels)
# e0 = multimode_state("e0", transmon_levels, cavity_levels)

# g1 = multimode_state("g1", transmon_levels, cavity_levels)
# g2 = multimode_state("g1", transmon_levels, cavity_levels)
# g4 = multimode_state("g4", transmon_levels, cavity_levels)

# ψ_init = [g0, e0]
# ψ_goal = [(g0 + g4) / √2, g2]

# qubit_a_bound = 0.153
# cavity_a_bound = 0.193

# dda_bound = 1e-5

# a_bounds = [qubit_a_bound, qubit_a_bound, cavity_a_bound, cavity_a_bound]

T = 200
Δt = 15.0
Δt_max = Δt
Δt_min = 0.2Δt
Q = 200.0
R_L1 = 10.0
max_iter = 2000

ketdim = transmon_levels * cavity_levels

cavity_forbidden_states = [
    cavity_levels .* (1:(transmon_levels - 1));
    ketdim .+ (
        cavity_levels .* (1:(transmon_levels - 1))
    )
]

transmon_forbidden_states = [
    (cavity_levels * (transmon_levels - 1) + 1) : ketdim;
    ketdim .+ (
        (cavity_levels * (transmon_levels - 1) + 1) : ketdim
    )
]

# forbidden_states = cavity_forbidden_states #[transmon_forbidden_states; cavity_forbidden_states]

# prob = QuantumStateSmoothPulseProblem(system, ψ_init, ψ_goal, T, Δt;
#     init_trajectory=init_traj,
#     Δt_max=Δt_max,
#     Δt_min=Δt_min,
#     dda_bound=dda_bound,
#     a_bounds=a_bounds,
#     L1_regularized_names=[:ψ̃1, :ψ̃2],
#     L1_regularized_indices=(ψ̃1 = forbidden_states, ψ̃2 = forbidden_states),
#     Q=Q,
#     R_L1=R_L1,
# )

# experiment = "transmon_$(transmon_levels)_T_$(T)_dt_$(Δt)_Q_$(Q)_R_L1_$(R_L1)_max_iter_$(max_iter)_dda_bound_$(dda_bound)"

# plot_dir = joinpath(@__DIR__, "plots/binomial_code")
# save_dir = joinpath(@__DIR__, "data/binomial_code")

# plot_path = generate_file_path("png", experiment, plot_dir)
# save_path = generate_file_path("jld2", experiment, save_dir)

# plot(plot_path, prob.trajectory, [:ψ̃1, :ψ̃2, :a]; ignored_labels=[:ψ̃1, :ψ̃2])

# solve!(prob; max_iter=max_iter)

# plot(plot_path, prob.trajectory, [:ψ̃1, :ψ̃2, :a]; ignored_labels=[:ψ̃1, :ψ̃2])

# ψ̃¹_final = rollout(prob.trajectory.initial.ψ̃1, prob.trajectory.a, prob.trajectory.Δt, system; integrator=exp)[:, end]

# ψ̃²_final = rollout(prob.trajectory.initial.ψ̃2, prob.trajectory.a, prob.trajectory.Δt, system; integrator=exp)[:, end]

# final_fidelity_1 = fidelity(ψ̃¹_final, prob.trajectory.goal.ψ̃1)
# final_fidelity_2 = fidelity(ψ̃²_final, prob.trajectory.goal.ψ̃2)

# println("Final fidelity 1: $final_fidelity_1")
# println("Final fidelity 2: $final_fidelity_2")

# info = Dict(
#     "final fidelity 1" => final_fidelity_1,
#     "final fidelity 2" => final_fidelity_2,
# )

# save_problem(save_path, prob, info)
