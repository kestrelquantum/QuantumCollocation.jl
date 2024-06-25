"""
    QuantumStateMinimumTimeProblem

TODO: Add documentation
"""
function QuantumStateMinimumTimeProblem end

function QuantumStateMinimumTimeProblem(
    traj::NamedTrajectory,
    sys::QuantumSystem,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    state_symbol::Symbol=:ψ̃,
    final_fidelity::Union{Real, Nothing}=nothing,
    D=1.0,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    kwargs...
)
    state_names = [name for name in traj.names if startswith(name, state_symbol)]
    @assert length(state_names) ≥ 1 "No matching states found in trajectory"

    obj += MinimumTimeObjective(traj; D=D, eval_hessian=piccolo_options.eval_hessian)

    # Default to average state fidelity
    if isnothing(final_fidelity)
        vals = [fidelity(traj[n][:, end], traj.goal[n]) for n ∈ state_names]
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
        sys,
        traj,
        obj,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
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
        prob.system,
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

        # System
        T = 50
        Δt = 0.2
        sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
        ψ_init = [1.0, 0.0]
        ψ_target = [0.0, 1.0]
        
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
