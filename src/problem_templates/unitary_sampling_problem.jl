export UnitarySamplingProblem


@doc raw"""
    UnitarySamplingProblem(systemns, operator, T, Δt; kwargs...)

A `UnitarySamplingProblem` is a quantum control problem where the goal is to find a control pulse that generates a target unitary operator for a set of quantum systems.
The controls are shared among all systems, and the optimization seeks to find the control pulse that achieves the operator for each system. The idea is to enforce a
robust solution by including multiple systems reflecting the problem uncertainty.

# Arguments
- `systems::AbstractVector{<:AbstractQuantumSystem}`: A vector of quantum systems.
- `operator::AbstractPiccoloOperator`: The target unitary operator.
- `T::Int`: The number of time steps.
- `Δt::Union{Float64, Vector{Float64}}`: The time step value or vector of time steps.

# Keyword Arguments
- `system_labels::Vector{String} = string.(1:length(systems))`: The labels for each system.
- `system_weights::Vector{Float64} = fill(1.0, length(systems))`: The weights for each system.
- `init_trajectory::Union{NamedTrajectory, Nothing} = nothing`: The initial trajectory.
- `ipopt_options::IpoptOptions = IpoptOptions()`: The IPOPT options.
- `piccolo_options::PiccoloOptions = PiccoloOptions()`: The Piccolo options.
- `state_name::Symbol = :Ũ⃗`: The name of the state variable.
- `control_name::Symbol = :a`: The name of the control variable.
- `timestep_name::Symbol = :Δt`: The name of the timestep variable.
- `constraints::Vector{<:AbstractConstraint} = AbstractConstraint[]`: The constraints.
- `a_bound::Float64 = 1.0`: The bound for the control amplitudes.
- `a_bounds::Vector{Float64} = fill(a_bound, length(systems[1].G_drives))`: The bounds for the control amplitudes.
- `a_guess::Union{Matrix{Float64}, Nothing} = nothing`: The initial guess for the control amplitudes.
- `da_bound::Float64 = Inf`: The bound for the control first derivatives.
- `da_bounds::Vector{Float64} = fill(da_bound, length(systems[1].G_drives))`: The bounds for the control first derivatives.
- `dda_bound::Float64 = 1.0`: The bound for the control second derivatives.
- `dda_bounds::Vector{Float64} = fill(dda_bound, length(systems[1].G_drives))`: The bounds for the control second derivatives.
- `Δt_min::Float64 = 0.5 * Δt`: The minimum time step size.
- `Δt_max::Float64 = 1.5 * Δt`: The maximum time step size.
- `Q::Float64 = 100.0`: The fidelity weight.
- `R::Float64 = 1e-2`: The regularization weight.
- `R_a::Union{Float64, Vector{Float64}} = R`: The regularization weight for the control amplitudes.
- `R_da::Union{Float64, Vector{Float64}} = R`: The regularization weight for the control first derivatives.
- `R_dda::Union{Float64, Vector{Float64}} = R`: The regularization weight for the control second derivatives.
- `kwargs...`: Additional keyword arguments.

"""
function UnitarySamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    operators::AbstractVector{<:AbstractPiccoloOperator},
    T::Int,
    Δt::Union{Float64,Vector{Float64}};
    system_weights=fill(1.0, length(systems)),
    init_trajectory::Union{NamedTrajectory,Nothing}=nothing,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    state_name::Symbol=:Ũ⃗,
    control_name::Symbol=:a,
    timestep_name::Symbol=:Δt,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, systems[1].n_drives),
    a_guess::Union{Matrix{Float64},Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, systems[1].n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, systems[1].n_drives),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64,Vector{Float64}}=R,
    R_da::Union{Float64,Vector{Float64}}=R,
    R_dda::Union{Float64,Vector{Float64}}=R,
    kwargs...
)
    @assert length(systems) == length(operators)

    # Trajectory
    state_names = [
        Symbol(string(state_name, "_system_", label)) for label ∈ 1:length(systems)
    ]

    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        trajs = map(zip(systems, operators, state_names)) do (sys, op, s)
            initialize_trajectory(
                op,
                T,
                Δt,
                sys.n_drives,
                (a_bounds, da_bounds, dda_bounds);
                state_name=s,
                control_name=control_name,
                timestep_name=timestep_name,
                free_time=piccolo_options.free_time,
                Δt_bounds=(Δt_min, Δt_max),
                geodesic=piccolo_options.geodesic,
                bound_state=piccolo_options.bound_state,
                a_guess=a_guess,
                system=sys,
                rollout_integrator=piccolo_options.rollout_integrator,
            )
        end

        traj = merge(
            trajs, 
            merge_names=(; a=1, da=1, dda=1, Δt=1),
            free_time=piccolo_options.free_time
        )
    end    

    control_names = [
        name for name ∈ traj.names
        if endswith(string(name), string(control_name))
    ]

    # Objective
    J = QuadraticRegularizer(control_name, traj, R_a; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[2], traj, R_da; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[3], traj, R_dda; timestep_name=timestep_name)

    for (weight, op, name) in zip(system_weights, operators, state_names)
        J += weight * UnitaryInfidelityObjective(
            name, traj, Q;
            subspace=op isa EmbeddedOperator ? op.subspace_indices : nothing
        )
    end

    # Integrators
    unitary_integrators = AbstractIntegrator[]
    for (sys, Ũ⃗_name) in zip(systems, state_names)
        if piccolo_options.integrator == :pade
            push!(
                unitary_integrators,
                UnitaryPadeIntegrator(Ũ⃗_name, control_name, sys, traj; order=piccolo_options.pade_order)
            )
        elseif piccolo_options.integrator == :exponential
            push!(
                unitary_integrators,
                UnitaryExponentialIntegrator(Ũ⃗_name, control_name, sys, traj)
            )
        else
            error("integrator must be one of (:pade, :exponential)")
        end
    end

    integrators = [
        unitary_integrators...,
        DerivativeIntegrator(control_name, control_names[2], traj),
        DerivativeIntegrator(control_names[2], control_names[3], traj),
    ]

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, operators, state_names, timestep_name
    )

    return QuantumControlProblem(
        traj,
        J,
        integrators;
        constraints=constraints,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        control_name=control_name,
        kwargs...
    )
end

function UnitarySamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    operator::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Float64,Vector{Float64}};
    kwargs...
)
    # Broadcast the operator to all systems
    return UnitarySamplingProblem(
        systems,
        fill(operator, length(systems)),
        T,
        Δt;
        kwargs...
    )
end

function UnitarySamplingProblem(
    system::Function,
    distribution::Sampleable,
    num_samples::Int,
    args...;
    kwargs...
)
    samples = rand(distribution, num_samples)
    systems = [system(x) for x in samples]
    return UnitarySamplingProblem(
        systems,
        args...;
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Sample robustness test" begin
    using Distributions
    using Random
    Random.seed!(1234)

    n_samples = 5
    T = 50
    Δt = 0.2
    timesteps = fill(Δt, T)
    operator = GATES[:H]
    systems(ζ) = QuantumSystem(ζ * GATES[:Z], [GATES[:X], GATES[:Y]])

    ip_ops = IpoptOptions(print_level=1, recalc_y="yes", recalc_y_feas_tol=1e1)
    pi_ops = PiccoloOptions(verbose=false)

    prob = UnitarySamplingProblem(
        systems, Normal(0, 0.05), n_samples, operator, T, Δt,
        ipopt_options=ip_ops, piccolo_options=pi_ops
    )
    solve!(prob, max_iter=20)

    d_prob = UnitarySmoothPulseProblem(
        systems(0), operator, T, Δt,
        ipopt_options=ip_ops, piccolo_options=pi_ops
    )
    solve!(prob, max_iter=20)

    # Check that the solution improves over the default
    # -------------------------------------------------
    ζ_tests = -0.05:0.01:0.05
    Ũ⃗_goal = operator_to_iso_vec(operator)
    fids = []
    default_fids = []
    for ζ in ζ_tests
        Ũ⃗_end = unitary_rollout(prob.trajectory.a, timesteps, systems(ζ))[:, end]
        push!(fids, iso_vec_unitary_fidelity(Ũ⃗_end, Ũ⃗_goal))

        d_Ũ⃗_end = unitary_rollout(d_prob.trajectory.a, timesteps, systems(ζ))[:, end]
        push!(default_fids, iso_vec_unitary_fidelity(d_Ũ⃗_end, Ũ⃗_goal))
    end
    @test sum(fids) > sum(default_fids)

    # Check initial guess initialization
    # ----------------------------------
    a_guess = prob.trajectory.a

    g1_prob = UnitarySamplingProblem(
        [systems(0), systems(0)], operator, T, Δt,
        a_guess=a_guess,
        piccolo_options=pi_ops
    )

    @test g1_prob.trajectory.Ũ⃗_system_1 ≈ g1_prob.trajectory.Ũ⃗_system_2

    g2_prob = UnitarySamplingProblem(
        [systems(0), systems(0.05)], operator, T, Δt;
        a_guess=a_guess,
        piccolo_options=pi_ops
    )

    @test ~(g2_prob.trajectory.Ũ⃗_system_1 ≈ g2_prob.trajectory.Ũ⃗_system_2)
end
