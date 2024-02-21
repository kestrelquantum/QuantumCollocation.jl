module ProblemSolvers

export solve!

using ..Constraints
using ..Problems
using ..SaveLoadUtils
using ..IpoptOptions

using NamedTrajectories
using MathOptInterface
const MOI = MathOptInterface

function solve!(
    prob::QuantumControlProblem;
    init_traj=nothing,
    save_path=nothing,
    controls_save_path=nothing,
    max_iter::Int=prob.options.max_iter,
    linear_solver::String=prob.options.linear_solver,
)
    prob.options.max_iter = max_iter
    prob.options.linear_solver = linear_solver

    set!(prob.optimizer, prob.options)

    if !isnothing(init_traj)
        initialize_trajectory!(prob, init_traj)
    else
        initialize_trajectory!(prob)
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

    # TODO: sort this out
    # if !isnothing(controls_save_path)
    #     save_controls(prob, controls_save_path)
    # end
end

end
