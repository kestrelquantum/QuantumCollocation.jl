module Callbacks

export callback_get_best_iterate
export callback_get_trajectory_history

using NamedTrajectories

import ..Losses
import ..Problems
import ..QuantumControlProblem

function callback_get_best_iterate(prob::QuantumControlProblem)
    best_trajectory_list = []
    best_fidelity = 0.0

    function callback(args...)
        trajectory = NamedTrajectory(Problems.get_datavec(prob), prob.trajectory)
        fidelity = Losses.fidelity(trajectory, prob.system)
        if fidelity > best_fidelity
            push!(best_trajectory_list, trajectory)
            best_fidelity = fidelity
        end
        return true
    end

    return callback, best_trajectory_list
end

function callback_get_trajectory_history(prob::QuantumControlProblem)
    trajectory_history = []

    function callback(args...)
        push!(trajectory_history, NamedTrajectory(Problems.get_datavec(prob), prob.trajectory))
        return true
    end
    
    return callback, trajectory_history
end

end # module