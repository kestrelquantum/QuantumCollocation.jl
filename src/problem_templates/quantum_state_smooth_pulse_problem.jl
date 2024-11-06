export QuantumStateSmoothPulseProblem


"""
    QuantumStateSmoothPulseProblem(system, ψ_inits, ψ_goals, T, Δt; kwargs...)
    QuantumStateSmoothPulseProblem(system, ψ_init, ψ_goal, T, Δt; kwargs...)
    QuantumStateSmoothPulseProblem(H_drift, H_drives, args...; kwargs...)

Create a quantum state smooth pulse problem. The goal is to find a control pulse
`a(t)` that drives all of the initial states `ψ_inits` to the corresponding
target states `ψ_goals` using `T` timesteps of size `Δt`. This problem also controls the first and second derivatives of the control pulse, `da(t)` and `dda(t)`, to ensure smoothness.

# Arguments
- `system::AbstractQuantumSystem`: The quantum system.
or
- `H_drift::AbstractMatrix{<:Number}`: The drift Hamiltonian.
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: The control Hamiltonians.
with
- `ψ_inits::Vector{<:AbstractVector{<:ComplexF64}}`: The initial states.
- `ψ_goals::Vector{<:AbstractVector{<:ComplexF64}}`: The target states.
or
- `ψ_init::AbstractVector{<:ComplexF64}`: The initial state.
- `ψ_goal::AbstractVector{<:ComplexF64}`: The target state.
with
- `T::Int`: The number of timesteps.
- `Δt::Float64`: The timestep size.


# Keyword Arguments
- `ipopt_options::IpoptOptions=IpoptOptions()`: The IPOPT options.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The Piccolo options.
- `state_name::Symbol=:ψ̃`: The name of the state variable.
- `control_name::Symbol=:a`: The name of the control variable.
- `timestep_name::Symbol=:Δt`: The name of the timestep variable.
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: The initial trajectory.
- `a_bound::Float64=1.0`: The bound on the control pulse.
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: The bounds on the control pulse.
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: The initial guess for the control pulse.
- `da_bound::Float64=Inf`: The bound on the first derivative of the control pulse.
- `da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives))`: The bounds on the first derivative of the control pulse.
- `dda_bound::Float64=1.0`: The bound on the second derivative of the control pulse.
- `dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives))`: The bounds on the second derivative of the control pulse.
- `Δt_min::Float64=0.5 * Δt`: The minimum timestep size.
- `Δt_max::Float64=1.5 * Δt`: The maximum timestep size.
- `drive_derivative_σ::Float64=0.01`: The standard deviation of the drive derivative random initialization.
- `Q::Float64=100.0`: The weight on the state objective.
- `R=1e-2`: The weight on the control pulse and its derivatives.
- `R_a::Union{Float64, Vector{Float64}}=R`: The weight on the control pulse.
- `R_da::Union{Float64, Vector{Float64}}=R`: The weight on the first derivative of the control pulse.
- `R_dda::Union{Float64, Vector{Float64}}=R`: The weight on the second derivative of the control pulse.
- `leakage_operator::Union{Nothing, EmbeddedOperator}=nothing`: The leakage operator, if leakage suppression is desired.
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: The constraints.
"""
function QuantumStateSmoothPulseProblem end

function QuantumStateSmoothPulseProblem(
    system::AbstractQuantumSystem,
    ψ_inits::Vector{<:AbstractVector{<:ComplexF64}},
    ψ_goals::Vector{<:AbstractVector{<:ComplexF64}},
    T::Int,
    Δt::Float64;
    G::Function=a -> G_bilinear(a, system.G_drift, system.G_drives),
    ∂G::Function=a -> system.G_drives,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:a,
    timestep_name::Symbol=:Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives)),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    leakage_operator::Union{Nothing, EmbeddedOperator}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    kwargs...
)
    @assert length(ψ_inits) == length(ψ_goals)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        n_drives = length(system.G_drives)

        traj = initialize_trajectory(
            ψ_goals,
            ψ_inits,
            T,
            Δt,
            n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            bound_state=piccolo_options.bound_state,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
        )
    end

    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_name))
    ]
    @assert length(state_names) == length(ψ_inits) "Number of states must match number of initial states"

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    # Objective
    J = QuadraticRegularizer(control_names[1], traj, R_a; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[2], traj, R_da; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[3], traj, R_dda; timestep_name=timestep_name)

    for name ∈ state_names
        J += QuantumStateObjective(name, traj, Q)
    end

    # Integrators
    state_integrators = []
    for name ∈ state_names
        if piccolo_options.integrator == :pade
            state_integrators = [QuantumStatePadeIntegrator(
                state_name,
                control_name,
                G,
                ∂G,
                traj;
                order=piccolo_options.pade_order
            )]
        elseif piccolo_options.integrator == :exponential
            state_integrators = [QuantumStateExponentialIntegrator(
                state_name,
                control_name,
                G,
                traj
            )]
        else
            error("integrator must be one of (:pade, :exponential)")
        end
    else
        state_names = [
            name for name ∈ traj.names
                if startswith(string(name), string(state_name))
        ]
        state_integrators = []
        for i = 1:length(ψ_inits)
            if piccolo_options.integrator == :pade
                state_integrator = QuantumStatePadeIntegrator(
                    state_names[i],
                    control_name,
                    G,
                    ∂G,
                    traj;
                    order=piccolo_options.pade_order
                )
            elseif piccolo_options.integrator == :exponential
                state_integrator = QuantumStateExponentialIntegrator(
                    state_names[i],
                    control_name,
                    G,
                    traj
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
        DerivativeIntegrator(control_names[2], control_names[3], traj)
    ]

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, leakage_operator, state_name, timestep_name
    )

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end

function QuantumStateSmoothPulseProblem(
    system::AbstractQuantumSystem,
    ψ_init::AbstractVector{<:ComplexF64},
    ψ_goal::AbstractVector{<:ComplexF64},
    args...;
    kwargs...
)
    return QuantumStateSmoothPulseProblem(system, [ψ_init], [ψ_goal], args...; kwargs...)
end

function QuantumStateSmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return QuantumStateSmoothPulseProblem(system, args...; kwargs...)
end

# *************************************************************************** #

@testitem "Test quantum state smooth pulse" begin
    # System
    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = Vector{ComplexF64}([1.0, 0.0])
    ψ_target = Vector{ComplexF64}([0.0, 1.0])

    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )
    initial = rollout_fidelity(prob)
    solve!(prob, max_iter=20)
    final = rollout_fidelity(prob)
    @test final > initial

    # Multiple initial and target states
    # ----------------------------------
    ψ_inits = Vector{ComplexF64}.([[1.0, 0.0], [0.0, 1.0]])
    ψ_targets = Vector{ComplexF64}.([[0.0, 1.0], [1.0, 0.0]])
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_inits, ψ_targets, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )
    initial = rollout_fidelity(prob)
    solve!(prob, max_iter=20)
    final = rollout_fidelity(prob)
    @test all(final .> initial)
end

@testitem "Test quantum state smooth pulse w/ exponential integrator" begin
    # System
    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = Vector{ComplexF64}([1.0, 0.0])
    ψ_target = Vector{ComplexF64}([0.0, 1.0])
    integrator=:exponential

    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, integrator=integrator)
    )
    initial = rollout_fidelity(prob)
    solve!(prob, max_iter=20)
    final = rollout_fidelity(prob)
    @test final > initial

    # Multiple initial and target states
    # ----------------------------------
    ψ_inits = Vector{ComplexF64}.([[1.0, 0.0], [0.0, 1.0]])
    ψ_targets = Vector{ComplexF64}.([[0.0, 1.0], [1.0, 0.0]])
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_inits, ψ_targets, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, integrator=integrator)
    )
    initial = rollout_fidelity(prob)
    solve!(prob, max_iter=20)
    final = rollout_fidelity(prob)
    @test all(final .> initial)
end

@testitem "Test quantum state with multiple initial states and final states" begin
    # System
    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_inits = Vector{ComplexF64}.([[1.0, 0.0], [0.0, 1.0]])
    ψ_targets = Vector{ComplexF64}.([[0.0, 1.0], [1.0, 0.0]])

    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_inits, ψ_targets, T, Δt;
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false),
        state_name=:psi,
        control_name=:u,
        timestep_name=:dt
    )
    initial = rollout_fidelity(prob)
    solve!(prob, max_iter=20)
    final = rollout_fidelity(prob)
    @test all(final .> initial)
end
