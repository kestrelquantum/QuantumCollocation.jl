export UnitaryMinimumTimeProblem


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
- `unitary_name::Symbol=:Ũ⃗`: The symbol for the unitary control.
- `final_fidelity::Float64=0.99`: The final fidelity.
- `D=1.0`: The weight for the minimum-time objective.
- `ipopt_options::IpoptOptions=IpoptOptions()`: The options for the Ipopt solver.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The options for the Piccolo solver.
- `kwargs...`: Additional keyword arguments to pass to `QuantumControlProblem`.
"""
function UnitaryMinimumTimeProblem end

function UnitaryMinimumTimeProblem(
    trajectory::NamedTrajectory,
    system::AbstractQuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_name::Symbol=:Ũ⃗,
    global_name::Union{Nothing, Symbol}=nothing,
    final_fidelity::Union{Real, Nothing}=nothing,
    D=1.0,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    subspace=nothing,
    kwargs...
)
    @assert unitary_name ∈ trajectory.names

    if isnothing(final_fidelity)
        final_fidelity = iso_vec_unitary_fidelity(
            trajectory[unitary_name][:, end], trajectory.goal[unitary_name]
        )
    end

    objective += MinimumTimeObjective(trajectory; D=D, eval_hessian=piccolo_options.eval_hessian)

    if isnothing(global_name)
        fidelity_constraint = FinalUnitaryFidelityConstraint(
            unitary_name,
            final_fidelity,
            trajectory;
            subspace=subspace,
            eval_hessian=piccolo_options.eval_hessian
        )
    else
        phase_operators= [
            GATES[:Z] for _ in eachindex(trajectory.global_components[global_name])
        ]
        fidelity_constraint = FinalUnitaryFreePhaseFidelityConstraint(
            unitary_name,
            global_name,
            phase_operators,
            final_fidelity,
            trajectory;
            subspace=subspace,
            eval_hessian=piccolo_options.eval_hessian
        )
    end

    constraints = push!(constraints, fidelity_constraint)

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

function UnitaryMinimumTimeProblem(
    prob::QuantumControlProblem;
    objective::Objective=get_objective(prob),
    constraints::AbstractVector{<:AbstractConstraint}=get_constraints(prob),
    ipopt_options::IpoptOptions=deepcopy(prob.ipopt_options),
    piccolo_options::PiccoloOptions=deepcopy(prob.piccolo_options),
    build_trajectory_constraints=false,
    kwargs...
)
    piccolo_options.build_trajectory_constraints = build_trajectory_constraints

    return UnitaryMinimumTimeProblem(
        copy(prob.trajectory),
        prob.system,
        objective,
        prob.integrators,
        constraints;
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Minimum time Hadamard gate" begin
    using NamedTrajectories

    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        H_drift, H_drives, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )

    before = unitary_fidelity(prob)
    solve!(prob, max_iter=50)
    after = unitary_fidelity(prob)
    @test after > before

    # Soft fidelity constraint
    final_fidelity = minimum([0.99, after])
    mintime_prob = UnitaryMinimumTimeProblem(prob, final_fidelity=final_fidelity)
    solve!(mintime_prob; max_iter=100)

    # Test fidelity is approximatley staying above the constraint
    @test unitary_fidelity(mintime_prob) ≥ (final_fidelity - 0.1 * final_fidelity)
    duration_after = sum(get_timesteps(mintime_prob.trajectory))
    duration_before = sum(get_timesteps(prob.trajectory))
    @test duration_after < duration_before

    # Set up without a final fidelity to check interface
    UnitaryMinimumTimeProblem(prob)

end
