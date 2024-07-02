@doc raw"""
    UnitaryBangBangProblem(system::QuantumSystem, operator, T, Δt; kwargs...)

TODO: Docstring
"""
function UnitaryBangBangProblem(
    system::AbstractQuantumSystem,
    operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=1.0,
    da_bounds=fill(da_bound, length(system.G_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
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
    # TODO: control modulus norm, advanced feature, needs documentation
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

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        n_drives = length(system.G_drives)
        traj = initialize_unitary_trajectory(
            operator,
            T,
            Δt,
            n_drives,
            (a = a_bounds, da = da_bounds);
            n_derivatives=1,
            free_time=free_time,
            Δt_bounds=(Δt_min, Δt_max),
            geodesic=geodesic,
            bound_unitary=bound_unitary,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
            rollout_integrator=rollout_integrator,
        )
    end

    # Objective
    J = UnitaryInfidelityObjective(:Ũ⃗, traj, Q;
        subspace=operator isa EmbeddedOperator ? operator.subspace_indices : nothing,
    )
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)

    # Constraints
    if leakage_suppression
        if operator isa EmbeddedOperator
            leakage_indices = get_unitary_isomorphism_leakage_indices(operator)
            J += L1Regularizer!(
                constraints, :Ũ⃗, traj, 
                R_value=R_leakage, 
                indices=leakage_indices,
                eval_hessian=!hessian_approximation
            )
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
    if integrator == :pade
        unitary_integrator =
            UnitaryPadeIntegrator(system, :Ũ⃗, :a; order=pade_order, autodiff=autodiff)
    elseif integrator == :exponential
        unitary_integrator =
            UnitaryExponentialIntegrator(system, :Ũ⃗, :a)
    else
        error("integrator must be one of (:pade, :exponential)")
    end

    integrators = [
        unitary_integrator,
        DerivativeIntegrator(:a, :da, traj),
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
        jacobian_structure=jacobian_structure,
        hessian_approximation=hessian_approximation,
        eval_hessian=!hessian_approximation,
        kwargs...
    )
end


"""
    UnitarySmoothPulseProblem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        operator,
        T,
        Δt;
        kwargs...
    )

Constructor for a `UnitarySmoothPulseProblem` from a drift Hamiltonian and a set of control Hamiltonians.
"""
function UnitarySmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return UnitarySmoothPulseProblem(system, args...; kwargs...)
end

# *************************************************************************** #
