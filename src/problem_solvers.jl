module ProblemSolvers

export solve!

using ..Constraints
using ..Problems
using ..SaveLoadUtils
using ..Options

using NamedTrajectories
using MathOptInterface
using Ipopt
const MOI = MathOptInterface


"""
   solve!(prob::QuantumControlProblem;
        init_traj=nothing,
        save_path=nothing,
        max_iter=prob.ipopt_options.max_iter,
        linear_solver=prob.ipopt_options.linear_solver,
        print_level=prob.ipopt_options.print_level,
        remove_slack_variables=false,
        callback=nothing
        # state_type=:unitary,
        # print_fidelity=false,
    )

    Call optimization solver to solve the quantum control problem with parameters and callbacks.

# Arguments
- `prob::QuantumControlProblem`: The quantum control problem to solve.
- `init_traj::NamedTrajectory`: Initial guess for the control trajectory. If not provided, a random guess will be generated.
- `save_path::String`: Path to save the problem after optimization.
- `max_iter::Int`: Maximum number of iterations for the optimization solver.
- `linear_solver::String`: Linear solver to use for the optimization solver (e.g., "mumps", "paradiso", etc).
- `print_level::Int`: Verbosity level for the solver.
- `remove_slack_variables::Bool`: Remove slack variables from the trajectory after optimization.
- `callback::Function`: Callback function to call during optimization steps.
"""
function solve!(
    prob::QuantumControlProblem;
    init_traj=nothing,
    save_path=nothing,
    max_iter::Int=prob.ipopt_options.max_iter,
    linear_solver::String=prob.ipopt_options.linear_solver,
    print_level::Int=prob.ipopt_options.print_level,
    remove_slack_variables::Bool=false,
    callback=nothing
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
    
    if !isnothing(callback)
        MOI.set(prob.optimizer, Ipopt.CallbackFunction(), callback)
    end

    # if print_fidelity
    #     if state_type == :ket
    #         fids = fidelity(prob)
    #         println("\nInitial Fidelities: $fids")
    #     elseif state_type == :unitary
    #         fids = unitary_fidelity(prob)
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
