@doc raw"""
    UnitarySamplingProblem

A `UnitarySamplingProblem` is a quantum control problem where the goal is to find a control pulse that generates a target unitary operator for a set of quantum systems. 
The controls are shared among all systems, and the optimization seeks to find the control pulse that achieves the operator for each system. The idea is to enforce a 
robust solution by including multiple systems reflecting the problem uncertainty.

# Arguments
- `systems::AbstractVector{<:AbstractQuantumSystem}`: A vector of quantum systems.
- `operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}}`: The target unitary operator.
- `T::Int`: The number of time steps.
- `Δt::Union{Float64, Vector{Float64}}`: The time step size.
- `system_labels::Vector{String}`: The labels for each system.
- `system_weights::Vector{Float64}`: The weights for each system.
- `free_time::Bool`: Whether to optimize the time steps.
- `init_trajectory::Union{NamedTrajectory, Nothing}`: The initial trajectory.
- `a_bound::Float64`: The bound for the control amplitudes.
- `a_bounds::Vector{Float64}`: The bounds for the control amplitudes.
- `a_guess::Union{Matrix{Float64}, Nothing}`: The initial guess for the control amplitudes.
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
- `max_iter::Int`: The maximum number of iterations.
- `linear_solver::String`: The linear solver.
- `ipopt_options::Options`: The IPOPT options.
- `constraints::Vector{<:AbstractConstraint}`: The constraints.
- `timesteps_all_equal::Bool`: Whether to enforce equal time steps.
- `verbose::Bool`: Whether to print verbose output.
- `integrator::Symbol`: The integrator to use.
- `rollout_integrator`: The integrator for the rollout.
- `bound_unitary::Bool`: Whether to bound the unitary.
- `control_norm_constraint::Bool`: Whether to enforce a control norm constraint.
- `control_norm_constraint_components`: The components for the control norm constraint.
- `control_norm_R`: The regularization weight for the control norm constraint.
- `geodesic::Bool`: Whether to use the geodesic.
- `pade_order::Int`: The order of the Pade approximation.
- `autodiff::Bool`: Whether to use automatic differentiation.
- `jacobian_structure::Bool`: Whether to evaluate the Jacobian structure.
- `hessian_approximation::Bool`: Whether to approximate the Hessian.
- `blas_multithreading::Bool`: Whether to use BLAS multithreading.
- `kwargs...`: Additional keyword arguments.

"""
function UnitarySamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    system_labels=string.(1:length(systems)),
    system_weights=fill(1.0, length(systems)),
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, length(systems[1].G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=1.0,
    dda_bounds=fill(dda_bound, length(systems[1].G_drives)),
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
    max_iter::Int=1000,
    linear_solver::String="mumps",
    ipopt_options::Options=Options(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    timesteps_all_equal::Bool=true,
    verbose::Bool=false,
    integrator::Symbol=:pade,
    rollout_integrator=exp,
    bound_unitary=integrator == :exponential,
    control_norm_constraint=false,
    control_norm_constraint_components=nothing,
    control_norm_R=nothing,
    geodesic=true,
    pade_order=4,
    autodiff=pade_order != 4,
    jacobian_structure=true,
    hessian_approximation=false,
    blas_multithreading=true,
    kwargs...
)
    if !blas_multithreading
        BLAS.set_num_threads(1)
    end

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    # Create keys for multiple systems
    Ũ⃗_keys = [add_suffix(:Ũ⃗, ℓ) for ℓ ∈ system_labels]

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        n_drives = length(systems[1].G_drives)
        # TODO: Initial system?
        traj = initialize_unitary_trajectory(
            operator,
            T,
            Δt,
            n_drives,
            a_bounds,
            dda_bounds;
            free_time=free_time,
            Δt_bounds=(Δt_min, Δt_max),
            geodesic=geodesic,
            bound_unitary=bound_unitary,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=systems,
            rollout_integrator=rollout_integrator,
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
            leakage_indices = get_unitary_isomorphism_leakage_indices(operator)
            for Ũ⃗_key in Ũ⃗_keys
                J += L1Regularizer!(
                    constraints, Ũ⃗_key, traj, 
                    R_value=R_leakage, 
                    indices=leakage_indices,
                    eval_hessian=!hessian_approximation
                )
            end
        else
            @warn "leakage_suppression is not supported for non-embedded operators, ignoring."
        end
    end

    if free_time
        if timesteps_all_equal
            push!(constraints, TimeStepsAllEqualConstraint(:Δt, traj))
        end
    end

    if control_norm_constraint
        @assert !isnothing(control_norm_constraint_components) "control_norm_constraint_components must be provided"
        @assert !isnothing(control_norm_R) "control_norm_R must be provided"
        norm_con = ComplexModulusContraint(
            :a,
            control_norm_R,
            traj;
            name_comps=control_norm_constraint_components,
        )
        push!(constraints, norm_con)
    end

    # Integrators
    unitary_integrators = AbstractIntegrator[]
    for (sys, Ũ⃗_key) in zip(systems, Ũ⃗_keys)
        if integrator == :pade
            push!(
                unitary_integrators,
                UnitaryPadeIntegrator(sys, Ũ⃗_key, :a; order=pade_order, autodiff=autodiff)
            )
        elseif integrator == :exponential
            push!(
                unitary_integrators,
                UnitaryExponentialIntegrator(sys, Ũ⃗_key, :a)
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
        max_iter=max_iter,
        linear_solver=linear_solver,
        verbose=verbose,
        ipopt_options=ipopt_options,
        jacobian_structure=jacobian_structure,
        hessian_approximation=hessian_approximation,
        eval_hessian=!hessian_approximation,
        kwargs...
    )
end

function UnitarySamplingProblem(
    system::Function,
    distribution::Sampleable,
    num_samples::Int,
    operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
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

    n_samples = 3
    T = 50
    Δt = 0.2
    timesteps = fill(Δt, T)
    operator = GATES[:H]
    systems(ζ) = QuantumSystem(ζ * GATES[:Z], [GATES[:X], GATES[:Y]])

    prob = UnitarySamplingProblem(
        systems,
        Normal(0, 0.1),
        n_samples,
        operator,
        T,
        Δt;
        verbose=false,
        ipopt_options=Options(print_level=1, recalc_y = "yes", recalc_y_feas_tol = 1e1)
    )

    solve!(prob, max_iter=20)

    d_prob = UnitarySmoothPulseProblem(
        systems(0),
        operator,
        T,
        Δt;
        verbose=false,
        ipopt_options=Options(print_level=1, recalc_y = "yes", recalc_y_feas_tol = 1e1)
    )
    solve!(prob, max_iter=20)

    # Check that the solution improves over the default
    ζ_test = 0.02
    Ũ⃗_goal = operator_to_iso_vec(operator)

    Ũ⃗_end = unitary_rollout(prob.trajectory.a, timesteps, systems(ζ_test))[:, end]
    fid = unitary_fidelity(Ũ⃗_end, Ũ⃗_goal)
    
    d_Ũ⃗_end = unitary_rollout(d_prob.trajectory.a, timesteps, systems(ζ_test))[:, end]
    default_fid = unitary_fidelity(d_Ũ⃗_end, Ũ⃗_goal)

    @test fid > default_fid

    # Check initial guess initialization
    a_guess = prob.trajectory.a
    
    g1_prob = UnitarySamplingProblem(
        [systems(0), systems(0)],
        operator,
        T,
        Δt;
        verbose=false,
        a_guess=a_guess,
    )

    @test g1_prob.trajectory.Ũ⃗1 ≈ g1_prob.trajectory.Ũ⃗2

    g2_prob = UnitarySamplingProblem(
        [systems(0), systems(0.1)],
        operator,
        T,
        Δt;
        verbose=false,
        a_guess=a_guess,
    )

    @test ~(g2_prob.trajectory.Ũ⃗1 ≈ g2_prob.trajectory.Ũ⃗2)
end