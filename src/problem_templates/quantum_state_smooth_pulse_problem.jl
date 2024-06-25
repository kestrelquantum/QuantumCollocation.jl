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

TODO: Document args
"""
function QuantumStateSmoothPulseProblem end

function QuantumStateSmoothPulseProblem(
    system::AbstractQuantumSystem,
    ψ_init::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
    ψ_goal::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
    T::Int,
    Δt::Float64;
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
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
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    L1_regularized_names=Symbol[],
    L1_regularized_indices::NamedTuple=NamedTuple(),
    kwargs...
)
    @assert all(name ∈ L1_regularized_names for name in keys(L1_regularized_indices) if !isempty(L1_regularized_indices[name]))

    if ψ_init isa AbstractVector{<:Number} && ψ_goal isa AbstractVector{<:Number}
        ψ_inits = [ψ_init]
        ψ_goals = [ψ_goal]
    else
        @assert length(ψ_init) == length(ψ_goal)
        ψ_inits = ψ_init
        ψ_goals = ψ_goal
    end

    ψ̃_inits = ket_to_iso.(Vector{ComplexF64}.(ψ_inits))
    ψ̃_goals = ket_to_iso.(Vector{ComplexF64}.(ψ_goals))

    n_drives = length(system.G_drives)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_quantum_state_trajectory(
            ψ̃_goals,
            ψ̃_inits,
            T,
            Δt,
            n_drives,
            a_bounds,
            dda_bounds;
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
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
                eval_hessian=piccolo_options.eval_hessian
            )
        else
            J += L1Regularizer!(
                constraints, name, traj; 
                R_value=R_L1,
                eval_hessian=piccolo_options.eval_hessian
            )
        end
    end

    if piccolo_options.free_time
        if piccolo_options.timesteps_all_equal
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
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
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

    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        ipopt_options=IpoptOptions(print_level=1), 
        piccolo_options=PiccoloOptions(verbose=false)
    )
    initial = fidelity(prob)
    solve!(prob, max_iter=20)
    final = fidelity(prob)
    @test final > initial

    # Multiple initial and target states
    # ----------------------------------
    ψ_inits = [[1.0, 0.0], [0.0, 1.0]]
    ψ_targets = [[0.0, 1.0], [1.0, 0.0]]
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_inits, ψ_targets, T, Δt;
        ipopt_options=IpoptOptions(print_level=1), 
        piccolo_options=PiccoloOptions(verbose=false)
    )
    initial = fidelity(prob)
    solve!(prob, max_iter=20)
    final = fidelity(prob)
    @test all(final .> initial)
end
