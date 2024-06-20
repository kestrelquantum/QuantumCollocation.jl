@doc raw"""
    UnitarySamplingProblem

    TODO: systems might need flexible bounds.

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
        traj = initialize_trajectory(
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
            system=systems[1],
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


# ζs = range(-.05, .05, length=5)
# systems = [system(ζ) for ζ ∈ ζs];
# system_labels = string.(1:length(systems))
# system_weights = fill(1.0, length(systems))

