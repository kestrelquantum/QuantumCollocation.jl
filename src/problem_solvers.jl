module ProblemSolvers

export solve!

using ..SaveLoadUtils

using NamedTrajectories
using QuantumCollocationCore
using MathOptInterface
using TestItemRunner
const MOI = MathOptInterface

function solve!(
    prob::QuantumControlProblem;
    init_traj=nothing,
    save_path=nothing,
    max_iter::Int=prob.ipopt_options.max_iter,
    linear_solver::String=prob.ipopt_options.linear_solver,
    print_level::Int=prob.ipopt_options.print_level,
    remove_slack_variables::Bool=false,
    # state_type::Symbol=:unitary,
    # print_fidelity::Bool=false,
)
    # @assert state_type in (:ket, :unitary, :density_matrix) "Invalid state type: $state_type must be one of :ket, :unitary, or :density_matrix"

    prob.ipopt_options.max_iter = max_iter
    prob.ipopt_options.linear_solver = linear_solver
    prob.ipopt_options.print_level = print_level

    set!(prob.optimizer, prob.ipopt_options)

    if !isnothing(init_traj)
        set_trajectory!(prob, init_traj)
    else
        set_trajectory!(prob)
    end

    # if print_fidelity
    #     if state_type == :ket
    #         fids = fidelity(prob)
    #         println("\nInitial Fidelities: $fids")
    #     elseif state_type == :unitary
    #         fids = unitary_rollout_fidelity(prob)
    #         println("\nInitial Fidelity: $fids")


    MOI.optimize!(prob.optimizer)

    update_trajectory!(prob)

    if remove_slack_variables
        slack_var_names = Symbol[]
        for con in prob.params[:linear_constraints]
            if con isa L1SlackConstraint
                append!(slack_var_names, con.slack_names)
            end
        end

        prob.trajectory = remove_components(prob.trajectory, slack_var_names)
    end

    if !isnothing(save_path)
        save_problem(save_path, prob)
    end
end

end
