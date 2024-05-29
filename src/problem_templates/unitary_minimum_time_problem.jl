@doc raw"""
    UnitaryMinimumTimeProblem(
        trajectory::NamedTrajectory,
        system::AbstractQuantumSystem,
        objective::Objective,
        integrators::Vector{<:AbstractIntegrator},
        constraints::Vector{<:AbstractConstraint};
        kwargs...
    )

    UnitaryMinimumTimeProblem(
        prob::QuantumControlProblem;
        kwargs...
    )

    UnitaryMinimumTimeProblem(
        data_path::String;
        kwargs...
    )

Create a minimum-time problem for unitary control.

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
J(\vec{\tilde{U}}, a, \dot{a}, \ddot{a}) + D \sum_t \Delta t_t \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& c(\vec{\tilde{U}}, a, \dot{a}, \ddot{a}) = 0 \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

# Arguments
- `trajectory::NamedTrajectory`: The initial trajectory.
- `system::AbstractQuantumSystem`: The quantum system.
- `objective::Objective`: The objective function (additional to the minimum-time objective).
- `integrators::Vector{<:AbstractIntegrator}`: The integrators.
- `constraints::Vector{<:AbstractConstraint}`: The constraints.

# Keyword Arguments
- `unitary_symbol::Symbol=:Ũ⃗`: The symbol for the unitary control.
- `final_fidelity::Float64=0.99`: The final fidelity.
- `D=1.0`: The weight for the minimum-time objective.
- `verbose::Bool=false`: Whether to print additional information.
- `ipopt_options::Options=Options()`: The options for the Ipopt solver.
- `kwargs...`: Additional keyword arguments to pass to `QuantumControlProblem`.
"""
function UnitaryMinimumTimeProblem end

function UnitaryMinimumTimeProblem(
    trajectory::NamedTrajectory,
    system::AbstractQuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    final_fidelity::Float64=0.99,
    D=1.0,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    subspace=nothing,
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    objective += MinimumTimeObjective(trajectory; D=D)

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory;
        subspace=subspace
    )

    constraints = AbstractConstraint[constraints..., fidelity_constraint]

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        verbose=verbose,
        ipopt_options=ipopt_options,
        kwargs...
    )
end

function UnitaryMinimumTimeProblem(
    prob::QuantumControlProblem;
    kwargs...
)
    params = deepcopy(prob.params)
    trajectory = copy(prob.trajectory)
    system = prob.system
    objective = Objective(params[:objective_terms])
    integrators = prob.integrators
    constraints = [
        params[:linear_constraints]...,
        NonlinearConstraint.(params[:nonlinear_constraints])...
    ]
    return UnitaryMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end

function UnitaryMinimumTimeProblem(
    data_path::String;
    kwargs...
)
    data = load(data_path)
    system = data["system"]
    trajectory = data["trajectory"]
    objective = Objective(data["params"][:objective_terms])
    integrators = data["integrators"]
    constraints = AbstractConstraint[
        data["params"][:linear_constraints]...,
        NonlinearConstraint.(data["params"][:nonlinear_constraints])...
    ]
    return UnitaryMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Minimum time Hadamard gate" begin
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=Options(print_level=1)
    )

    solve!(prob, max_iter=100)

    @test unitary_fidelity(prob) > 0.99

    final_fidelity = 0.99

    mintime_prob = UnitaryMinimumTimeProblem(
        prob,
        final_fidelity=final_fidelity,
        ipopt_options=Options(print_level=1)
    )

    solve!(mintime_prob; max_iter=100)

    @test unitary_fidelity(mintime_prob) > final_fidelity

    @test sum(mintime_prob.trajectory[:Δt]) < sum(prob.trajectory[:Δt])
end
