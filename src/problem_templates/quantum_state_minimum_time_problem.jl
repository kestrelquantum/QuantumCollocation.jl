export QuantumStateMinimumTimeProblem


"""
    QuantumStateMinimumTimeProblem(traj, sys, obj, integrators, constraints; kwargs...)
    QuantumStateMinimumTimeProblem(prob; kwargs...)

Construct a `QuantumControlProblem` for the minimum time problem of reaching a target state.

# Arguments
- `traj::NamedTrajectory`: The initial trajectory.
- `sys::QuantumSystem`: The quantum system.
- `obj::Objective`: The objective function.
- `integrators::Vector{<:AbstractIntegrator}`: The integrators.
- `constraints::Vector{<:AbstractConstraint}`: The constraints.
or
- `prob::QuantumControlProblem`: The quantum control problem.

# Keyword Arguments
- `state_name::Symbol=:ψ̃`: The symbol for the state variables.
- `final_fidelity::Union{Real, Nothing}=nothing`: The final fidelity.
- `D=1.0`: The cost weight on the time.
- `ipopt_options::IpoptOptions=IpoptOptions()`: The Ipopt options.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The Piccolo options.
- `kwargs...`: Additional keyword arguments, passed to `QuantumControlProblem`.

"""
function QuantumStateMinimumTimeProblem end

function QuantumStateMinimumTimeProblem(
    traj::NamedTrajectory,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:a,
    final_fidelity::Union{Real, Nothing}=nothing,
    D=1.0,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
    state_names = [name for name in traj.names if startswith(name, state_name)]
    @assert length(state_names) ≥ 1 "No matching states found in trajectory"

    obj += MinimumTimeObjective(traj; D=D, eval_hessian=piccolo_options.eval_hessian)

    # Default to average state fidelity
    if isnothing(final_fidelity)
        vals = [iso_fidelity(traj[n][:, end], traj.goal[n]) for n ∈ state_names]
        final_fidelity = sum(vals) / length(vals)
    end

    for state_name in state_names
        fidelity_constraint = FinalQuantumStateFidelityConstraint(
            state_name,
            final_fidelity,
            traj,
            eval_hessian=piccolo_options.eval_hessian
        )

        push!(constraints, fidelity_constraint)
    end

    return QuantumControlProblem(
        traj,
        obj,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        control_name=control_name,
        kwargs...
    )
end

function QuantumStateMinimumTimeProblem(
    prob::QuantumControlProblem;
    obj::Objective=get_objective(prob),
    constraints::AbstractVector{<:AbstractConstraint}=get_constraints(prob),
    ipopt_options::IpoptOptions=deepcopy(prob.ipopt_options),
    piccolo_options::PiccoloOptions=deepcopy(prob.piccolo_options),
    build_trajectory_constraints=false,
    kwargs...
)
    piccolo_options.build_trajectory_constraints = build_trajectory_constraints

    return QuantumStateMinimumTimeProblem(
        copy(prob.trajectory),
        obj,
        prob.integrators,
        constraints;
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Test quantum state minimum time" begin
        using NamedTrajectories
        using PiccoloQuantumObjects

        # System
        T = 50
        Δt = 0.2
        sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
        ψ_init = Vector{ComplexF64}[[1.0, 0.0]]
        ψ_target = Vector{ComplexF64}[[0.0, 1.0]]

        prob = QuantumStateSmoothPulseProblem(
            sys, ψ_init, ψ_target, T, Δt;
            ipopt_options=IpoptOptions(print_level=1),
            piccolo_options=PiccoloOptions(verbose=false)
        )
        initial = sum(get_timesteps(prob.trajectory))
        mintime_prob = QuantumStateMinimumTimeProblem(prob)
        solve!(mintime_prob, max_iter=20)
        final = sum(get_timesteps(mintime_prob.trajectory))
        @test final < initial

        # Test with final fidelity
        QuantumStateMinimumTimeProblem(prob, final_fidelity=0.99)
end
