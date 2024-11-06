export QuantumStateSamplingProblem

function QuantumStateSamplingProblem end

function QuantumStateSamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    ψ_inits::Vector{<:AbstractVector{<:ComplexF64}},
    ψ_goals::Vector{<:AbstractVector{<:ComplexF64}},
    T::Int,
    Δt::Union{Float64,Vector{Float64}};
    system_weights=fill(1.0, length(systems)),
    init_trajectory::Union{NamedTrajectory,Nothing}=nothing,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:a,
    timestep_name::Symbol=:Δt,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, systems[1].n_drives),
    a_guess::Union{Matrix{Float64},Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, systems[1].n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, systems[1].n_drives),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64,Vector{Float64}}=R,
    R_da::Union{Float64,Vector{Float64}}=R,
    R_dda::Union{Float64,Vector{Float64}}=R,
    leakage_operator::Union{Nothing, EmbeddedOperator}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    kwargs...
)
    @assert length(ψ_inits) == length(ψ_goals)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_trajectory(
            ψ_goals,
            ψ_inits,
            T,
            Δt,
            systems[1].n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            bound_state=piccolo_options.bound_state,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=systems,
            rollout_integrator=piccolo_options.rollout_integrator,
        )
    end

    # Outer dimension is the system, inner dimension is the initial state
    state_names = [
        name for name ∈ traj.names
        if startswith(string(name), string(state_name))
    ]
    @assert length(ψ_inits) * length(systems) == length(state_names) "State names do not match number of systems and initial states"
    state_names = reshape(state_names, length(ψ_inits), length(systems))

    control_names = [
        name for name ∈ traj.names
        if endswith(string(name), string(control_name))
    ]

    # Objective
    J = QuadraticRegularizer(control_names[1], traj, R_a; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[2], traj, R_da; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[3], traj, R_dda; timestep_name=timestep_name)

    for (weight, names) in zip(system_weights, eachcol(state_names))
        for name in names
            J += weight * QuantumStateObjective(name, traj, Q)
        end
    end

    # Integrators
    state_integrators = []
    for (system, names) ∈ zip(systems, eachcol(state_names))
        for name ∈ names
            if piccolo_options.integrator == :pade
                state_integrator = QuantumStatePadeIntegrator(
                    name, control_name, system, traj;
                    order=piccolo_options.pade_order
                )
            elseif piccolo_options.integrator == :exponential
                state_integrator = QuantumStateExponentialIntegrator(
                    name, control_name, system, traj
                )
            else
                error("integrator must be one of (:pade, :exponential)")
            end
            push!(state_integrators, state_integrator)
        end
    end

    integrators = [
        state_integrators...,
        DerivativeIntegrator(control_name, control_names[2], traj),
        DerivativeIntegrator(control_names[2], control_names[3], traj),
    ]

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, leakage_operator, state_name, timestep_name
    )

    return QuantumControlProblem(
        direct_sum(systems),
        traj,
        J,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

function QuantumStateSamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    ψ_init::AbstractVector{<:ComplexF64},
    ψ_goal::AbstractVector{<:ComplexF64},
    args...;
    kwargs...
)
    return QuantumStateSamplingProblem(systems, [ψ_init], [ψ_goal], args...; kwargs...)
end


# =============================================================================

@testitem "Sample systems with single initial, target" begin
    # System
    T = 50
    Δt = 0.2
    sys1 = QuantumSystem(0.3 * GATES[:Z], [GATES[:X], GATES[:Y]])
    sys2 = QuantumSystem(0.0 * GATES[:Z], [GATES[:X], GATES[:Y]])

    # Single initial and target states
    # --------------------------------
    ψ_init = Vector{ComplexF64}([1.0, 0.0])
    ψ_target = Vector{ComplexF64}([0.0, 1.0])

    prob = QuantumStateSamplingProblem(
        [sys1, sys2], ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )

    state_names = [n for n ∈ prob.trajectory.names if startswith(string(n), "ψ̃")]

    init = [rollout_fidelity(prob.trajectory, sys1, state_symb=n) for n in state_names]
    solve!(prob, max_iter=20)
    final = [rollout_fidelity(prob.trajectory, sys1, state_symb=n) for n in state_names]
    @test all(final .> init)

    # Compare a solution without robustness
    # -------------------------------------
    prob_default = QuantumStateSmoothPulseProblem(
        sys2, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false),
        robustness=false
    )
    solve!(prob_default, max_iter=20)
    final_default = rollout_fidelity(prob_default.trajectory, sys1)
    final_robust = rollout_fidelity(prob.trajectory, sys1, state_symb=state_names[1])
    @test final_robust > final_default
end

@testitem "Sample systems with multiple initial, target" begin
    # System
    T = 50
    Δt = 0.2
    sys1 = QuantumSystem(0.3 * GATES[:Z], [GATES[:X], GATES[:Y]])
    sys2 = QuantumSystem(0.0 * GATES[:Z], [GATES[:X], GATES[:Y]])

    # Multiple initial and target states
    # ----------------------------------
    ψ_inits = Vector{ComplexF64}.([[1.0, 0.0], [0.0, 1.0]])
    ψ_targets = Vector{ComplexF64}.([[0.0, 1.0], [1.0, 0.0]])

    prob = QuantumStateSamplingProblem(
        [sys1, sys2], ψ_inits, ψ_targets, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )

    state_names = [n for n ∈ prob.trajectory.names if startswith(string(n), "ψ̃")]

    init = [rollout_fidelity(prob.trajectory, sys1, state_symb=n) for n in state_names]
    solve!(prob, max_iter=20)
    final = [rollout_fidelity(prob.trajectory, sys1, state_symb=n) for n in state_names]
    @test all(final .> init)
end
