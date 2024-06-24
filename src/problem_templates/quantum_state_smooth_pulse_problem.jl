"""
    QuantumStateSmoothPulseProblem(
        system::AbstractQuantumSystem,
        ψ_init::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
        ψ_goal::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
        T::Int,
        Δt::Float64;
        kwargs...
    )

    QuantumStateSmoothPulseProblem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        args...;
        kwargs...
    )

Create a quantum control problem for smooth pulse optimization of a quantum state trajectory.

# Keyword Arguments

# TODO: clean up this whole constructor

"""
function QuantumStateSmoothPulseProblem end

function QuantumStateSmoothPulseProblem(
    system::AbstractQuantumSystem,
    ψ_init::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
    ψ_goal::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
    T::Int,
    Δt::Float64;
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    rollout_integrator=exp,
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
    R_L1::Float64=20.0,
    max_iter::Int=1000,
    linear_solver::String="mumps",
    ipopt_options::Options=Options(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    timesteps_all_equal::Bool=true,
    L1_regularized_names=Symbol[],
    L1_regularized_indices::NamedTuple=NamedTuple(),
    leakage_indcies=nothing,
    integrator::Symbol=:pade,
    pade_order::Int=4,
    autodiff::Bool=pade_order != 4,
    rollout_integrator=exp,
    bound_state=integrator == :exponential,
    # TODO: control modulus norm, advanced feature, needs documentation
    control_norm_constraint=false,
    control_norm_constraint_components=nothing,
    control_norm_R=nothing,
    verbose=false,
    kwargs...
)
    @assert all(name ∈ L1_regularized_names for name in keys(L1_regularized_indices) if !isempty(L1_regularized_indices[name]))
    if !isnothing(a_guess)
        @assert size(a_guess) == (length(system.G_drives), T) "a_guess (size = $(size(a_guess))) must have size (length(system.G_drives), T)"
    end

    if ψ_init isa AbstractVector{<:Number} && ψ_goal isa AbstractVector{<:Number}
        ψ_inits = [ψ_init]
        ψ_goals = [ψ_goal]
    else
        @assert length(ψ_init) == length(ψ_goal)
        ψ_inits = ψ_init
        ψ_goals = ψ_goal
    end

    ψ_inits = Vector{ComplexF64}.(ψ_inits)
    ψ̃_inits = ket_to_iso.(ψ_inits)

    ψ_goals = Vector{ComplexF64}.(ψ_goals)
    ψ̃_goals = ket_to_iso.(ψ_goals)

    n_drives = length(system.G_drives)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_state_trajectory(
            ψ̃_goals,
            ψ̃_inits,
            T,
            Δt,
            n_drives,
            a_bounds,
            dda_bounds;
            free_time=free_time,
            Δt_bounds=(Δt_min, Δt_max),
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
            rollout_integrator=rollout_integrator,
        )
    end

    # Objective
    J = QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    for i = 1:length(ψ_inits)
        J += QuantumStateObjective(Symbol("ψ̃$i"), traj, Q)
    end

    # Constraints
    for name in L1_regularized_names
        if name in keys(L1_regularized_indices)
            J += L1Regularizer!(
                constraints, name, traj,
                R_value=R_L1,
                indices=L1_regularized_indices[name],
                eval_hessian=!hessian_approximation
            )
        else
            J += L1Regularizer!(
                constraints, name, traj;
                R_value=R_L1,
                eval_hessian=!hessian_approximation
            )
        end
    end



    integrators = [
        ψ̃_integrators...,
        DerivativeIntegrator(:a, :da, traj),
        DerivativeIntegrator(:da, :dda, traj)
    ]

    if free_time
        if timesteps_all_equal
            push!(constraints, TimeStepsAllEqualConstraint(:Δt, traj))
        end
    end

    # Integrators
    ψ̃_integrators = [
        QuantumStatePadeIntegrator(system, Symbol("ψ̃$i"), :a)
            for i = 1:length(ψ_inits)
    ]

    integrators = [
        ψ̃_integrators...,
        DerivativeIntegrator(:a, :da, traj),
        DerivativeIntegrator(:da, :dda, traj)
    ]

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators;
        constraints=constraints,
        max_iter=max_iter,
        linear_solver=linear_solver,
        verbose=verbose,
        ipopt_options=ipopt_options,
        kwargs...
    )
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
    ψ_init = [1.0, 0.0]
    ψ_target = [0.0, 1.0]

    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=Options(print_level=1), verbose=false
    )
    initial = fidelity(prob)
    solve!(prob, max_iter=20)
    final = fidelity(prob)
    @test final > initial

    # Multiple initial and target states
    ψ_inits = [[1.0, 0.0], [0.0, 1.0]]
    ψ_targets = [[0.0, 1.0], [1.0, 0.0]]
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_inits, ψ_targets, T, Δt;
        ipopt_options=Options(print_level=1), verbose=false
    )
    initial = fidelity(prob)
    solve!(prob, max_iter=20)
    final = fidelity(prob)
    @test all(final .> initial)
end
