export UnitarySamplingProblem


@doc raw"""
    UnitarySamplingProblem

A `UnitarySamplingProblem` is a quantum control problem where the goal is to find a control pulse that generates a target unitary operator for a set of quantum systems.
The controls are shared among all systems, and the optimization seeks to find the control pulse that achieves the operator for each system. The idea is to enforce a
robust solution by including multiple systems reflecting the problem uncertainty.

# Arguments
- `systems::AbstractVector{<:AbstractQuantumSystem}`: A vector of quantum systems.
- `operator::OperatorType`: The target unitary operator.
- `T::Int`: The number of time steps.
- `Δt::Union{Float64, Vector{Float64}}`: The time step size.
- `system_labels::Vector{String}`: The labels for each system.
- `system_weights::Vector{Float64}`: The weights for each system.
- `init_trajectory::Union{NamedTrajectory, Nothing}`: The initial trajectory.
- `ipopt_options::IpoptOptions`: The IPOPT options.
- `piccolo_options::PiccoloOptions`: The Piccolo options.
- `constraints::Vector{<:AbstractConstraint}`: The constraints.
- `a_bound::Float64`: The bound for the control amplitudes.
- `a_bounds::Vector{Float64}`: The bounds for the control amplitudes.
- `a_guess::Union{Matrix{Float64}, Nothing}`: The initial guess for the control amplitudes.
- `da_bound::Float64`: The bound for the control first derivatives.
- `da_bounds::Vector{Float64}`: The bounds for the control first derivatives.
- `dda_bound::Float64`: The bound for the control second derivatives.
- `dda_bounds::Vector{Float64}`: The bounds for the control second derivatives.
- `Δt_min::Float64`: The minimum time step size.
- `Δt_max::Float64`: The maximum time step size.
- `drive_derivative_σ::Float64`: The standard deviation for the drive derivative noise.
- `Q::Float64`: The fidelity weight.
- `R::Float64`: The regularization weight.
- `R_a::Union{Float64, Vector{Float64}}`: The regularization weight for the control amplitudes.
- `R_da::Union{Float64, Vector{Float64}}`: The regularization weight for the control first derivatives.
- `R_dda::Union{Float64, Vector{Float64}}`: The regularization weight for the control second derivatives.
- `leakage_suppression::Bool`: Whether to suppress leakage.
- `R_leakage::Float64`: The regularization weight for the leakage.
- `bound_unitary::Bool`: Whether to bound the unitary.
- `control_norm_constraint::Bool`: Whether to enforce a control norm constraint.
- `control_norm_constraint_components`: The components for the control norm constraint.
- `control_norm_R`: The regularization weight for the control norm constraint.
- `kwargs...`: Additional keyword arguments.

"""
function UnitarySamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    operator::OperatorType,
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    system_labels=string.(1:length(systems)),
    system_weights=fill(1.0, length(systems)),
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, length(systems[1].G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, length(systems[1].G_drives)),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(systems[1].G_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    leakage_suppression=false,
    R_leakage=1e-1,
    bound_unitary=piccolo_options.integrator == :exponential,
    control_norm_constraint=false,
    control_norm_constraint_components=nothing,
    control_norm_R=nothing,
    kwargs...
)
    # Create keys for multiple systems
    Ũ⃗_keys = [add_suffix(:Ũ⃗, ℓ) for ℓ ∈ system_labels]

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        n_drives = length(systems[1].G_drives)
        traj = initialize_unitary_trajectory(
            operator,
            T,
            Δt,
            n_drives,
            (a = a_bounds, da = da_bounds, dda = dda_bounds);
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            geodesic=piccolo_options.geodesic,
            bound_unitary=bound_unitary,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=systems,
            rollout_integrator=piccolo_options.rollout_integrator,
            Ũ⃗_keys=Ũ⃗_keys
        )
    end

    # Objective
    J = NullObjective()
    for (wᵢ, Ũ⃗_key) in zip(system_weights, Ũ⃗_keys)
        J += wᵢ * UnitaryInfidelityObjective(
            Ũ⃗_key, traj, Q;
            subspace=operator isa EmbeddedOperator ? operator.subspace_indices : nothing
        )
    end
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    # Constraints
    if leakage_suppression
        if operator isa EmbeddedOperator
            leakage_indices = get_iso_vec_leakage_indices(operator)
            for Ũ⃗_key in Ũ⃗_keys
                J += L1Regularizer!(
                    constraints, Ũ⃗_key, traj,
                    R_value=R_leakage,
                    indices=leakage_indices,
                    eval_hessian=piccolo_options.eval_hessian
                )
            end
        else
            @warn "leakage_suppression is not supported for non-embedded operators, ignoring."
        end
    end

    if piccolo_options.free_time
        if piccolo_options.timesteps_all_equal
            push!(constraints, TimeStepsAllEqualConstraint(:Δt, traj))
        end
    end

    if control_norm_constraint
        @assert !isnothing(control_norm_constraint_components) "control_norm_constraint_components must be provided"
        @assert !isnothing(control_norm_R) "control_norm_R must be provided"
        norm_con = ComplexModulusContraint(
            :a,
            control_norm_R,
            traj;Constraint
            name_comps=control_norm_constraint_components,
        )
        push!(constraints, norm_con)
    end

    # Integrators
    unitary_integrators = AbstractIntegrator[]
    for (sys, Ũ⃗_key) in zip(systems, Ũ⃗_keys)
        if piccolo_options.integrator == :pade
            push!(
                unitary_integrators,
                UnitaryPadeIntegrator(sys, Ũ⃗_key, :a, traj; order=piccolo_options.pade_order)
            )
        elseif piccolo_options.integrator == :exponential
            push!(
                unitary_integrators,
                UnitaryExponentialIntegrator(sys, Ũ⃗_key, :a, traj)
            )
        else
            error("integrator must be one of (:pade, :exponential)")
        end
    end

    integrators = [
        unitary_integrators...,
        DerivativeIntegrator(:a, :da, traj),
        DerivativeIntegrator(:da, :dda, traj),
    ]

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

function UnitarySamplingProblem(
    system::Function,
    distribution::Sampleable,
    num_samples::Int,
    operator::OperatorType,
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    kwargs...
)
    samples = rand(distribution, num_samples)
    systems = [system(x) for x in samples]
    return UnitarySamplingProblem(
        systems,
        operator,
        T,
        Δt;
        kwargs...
    )
end

# =============================================================================

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

    ip_ops = IpoptOptions(print_level=1, recalc_y = "yes", recalc_y_feas_tol = 1e1)
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

    @test g1_prob.trajectory.Ũ⃗1 ≈ g1_prob.trajectory.Ũ⃗2

    g2_prob = UnitarySamplingProblem(
        [systems(0), systems(0.05)], operator, T, Δt;
        a_guess=a_guess,
        piccolo_options=pi_ops
    )

    @test ~(g2_prob.trajectory.Ũ⃗1 ≈ g2_prob.trajectory.Ũ⃗2)
end
