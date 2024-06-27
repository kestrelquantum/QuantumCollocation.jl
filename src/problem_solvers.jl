module ProblemSolvers

export solve!

using ..Constraints
using ..Problems
using ..SaveLoadUtils
using ..Options

using NamedTrajectories
using MathOptInterface
const MOI = MathOptInterface

function solve!(
    prob::QuantumControlProblem;
    init_traj=nothing,
    save_path=nothing,
    max_iter::Int=prob.ipopt_options.max_iter,
    linear_solver::String=prob.ipopt_options.linear_solver,
    print_level::Int=prob.ipopt_options.print_level,
)
    prob.ipopt_options.max_iter = max_iter
    prob.ipopt_options.linear_solver = linear_solver
    prob.ipopt_options.print_level = print_level

    set!(prob.optimizer, prob.ipopt_options)

    if !isnothing(init_traj)
        set_trajectory!(prob, init_traj)
    else
        set_trajectory!(prob)
    end

    MOI.optimize!(prob.optimizer)

    update_trajectory!(prob)

    slack_var_names = Symbol[]
    for con in prob.params[:linear_constraints]
        if con isa L1SlackConstraint
            append!(slack_var_names, con.slack_names)
        end
    end

    prob.trajectory = remove_components(prob.trajectory, slack_var_names)

    if !isnothing(save_path)
        save_problem(save_path, prob)
    end
end

end
