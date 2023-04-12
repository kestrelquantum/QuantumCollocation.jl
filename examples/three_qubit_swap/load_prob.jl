using NamedTrajectories
using QuantumCollocation

experiment_path = "examples/three_qubit_swap/results/T_200_Δt_1.0_a_bound_0.25132741228718347_dda_bound_0.01_dt_min_0.5_dt_max_1.5_max_iter_100000_00000..jld2"

load_problem(experiment_path; return_data=true)
