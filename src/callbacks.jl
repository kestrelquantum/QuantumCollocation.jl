module Callbacks

export make_save_best_trajectory_callback

using NamedTrajectories

import ..QuantumControlProblem
import ..Problems: get_datavec

function make_save_best_trajectory_callback(prob::QuantumControlProblem, traj::NamedTrajectory)
    best_traj = NamedTrajectory(get_datavec(prob), prob.trajectory)
    best_fidelity = 0.0

    function save_best_trajectory_callback(prob::QuantumControlProblem)
        fidelity = prob.objective.fidelity(prob)
        if fidelity > best_fidelity
            best_fidelity = fidelity
            copy!(best_traj, prob.trajectory)
        end
        return true
    end

    return save_best_trajectory_callback, best_traj
end

end