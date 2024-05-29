@doc raw"""
    UnitarySmoothPulseProblem(system::QuantumSystem, operator, T, Δt; kwargs...)

Construct a `QuantumControlProblem` for a free-time unitary gate problem with smooth control pulses enforced by constraining the second derivative of the pulse trajectory, i.e.,

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) + \frac{1}{2} \sum_t \qty(R_a a_t^2 + R_{\dot{a}} \dot{a}_t^2 + R_{\ddot{a}} \ddot{a}_t^2) \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
& \quad \dot{a}_{t+1} - \dot{a}_t - \ddot{a}_t \Delta t_t = 0 \\
& \quad |a_t| \leq a_{\text{bound}} \\
& \quad |\ddot{a}_t| \leq \ddot{a}_{\text{bound}} \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

where, for $U \in SU(N)$,

```math
\ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) =
\abs{1 - \frac{1}{N} \abs{ \tr \qty(U_{\text{goal}}, U_T)} }
```

is the *infidelity* objective function, $Q$ is a weight, $R_a$, $R_{\dot{a}}$, and $R_{\ddot{a}}$ are weights on the regularization terms, and $\vb{P}^{(n)}$ is the $n$th-order Pade integrator.

# Arguments

- `H_drift::AbstractMatrix{<:Number}`: the drift hamiltonian
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: the control hamiltonians
or
- `system::QuantumSystem`: the system to be controlled
with
- `operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}}`: the target unitary, either in the form of an `EmbeddedOperator` or a `Matrix{ComplexF64}
- `T::Int`: the number of timesteps
- `Δt::Float64`: the (initial) time step size

# Keyword Arguments
- `free_time::Bool=true`: whether or not to allow the time steps to vary
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `dda_bound::Float64=1.0`: the bound on the control pulse derivative
- `dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `Δt_min::Float64=0.5 * Δt`: the minimum time step size
- `Δt_max::Float64=1.5 * Δt`: the maximum time step size
- `drive_derivative_σ::Float64=0.01`: the standard deviation of the initial guess for the control pulse derivatives
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `leakage_suppression::Bool=false`: whether or not to suppress leakage to higher energy states
- `R_leakage=1e-1`: the weight on the leakage suppression term
- `max_iter::Int=1000`: the maximum number of iterations for the solver
- `linear_solver::String="mumps"`: the linear solver to use
- `ipopt_options::Options=Options()`: the options for the Ipopt solver
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: additional constraints to add to the problem
- `timesteps_all_equal::Bool=true`: whether or not to enforce that all time steps are equal
- `verbose::Bool=false`: whether or not to print constructor output
- `integrator=:pade`: the integrator to use for the unitary, either `:pade` or `:exponential`
- `rollout_integrator=exp`: the integrator to use for the rollout
- `bound_unitary=integrator == :exponential`: whether or not to bound the unitary
- `control_norm_constraint=false`: whether or not to enforce a constraint on the control pulse norm
- `control_norm_constraint_components=nothing`: the components of the control pulse to use for the norm constraint
- `control_norm_R=nothing`: the weight on the control pulse norm constraint
- `geodesic=true`: whether or not to use the geodesic as the initial guess for the unitary
- `pade_order=4`: the order of the Pade approximation to use for the unitary integrator
- `autodiff=pade_order != 4`: whether or not to use automatic differentiation for the unitary integrator
- `subspace=nothing`: the subspace to use for the unitary integrator
- `jacobian_structure=true`: whether or not to use the jacobian structure
- `hessian_approximation=false`: whether or not to use L-BFGS hessian approximation in Ipopt
- `blas_multithreading=true`: whether or not to use multithreading in BLAS
"""
function UnitarySmoothPulseProblem(
    system::AbstractQuantumSystem,
    operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=1.0,
    dda_bounds=fill(dda_bound, length(system.G_drives)),
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
    if operator isa EmbeddedOperator
        U_goal = operator.operator
        U_init = get_subspace_identity(operator)
    else
        U_goal = Matrix{ComplexF64}(operator)
        U_init = Matrix{ComplexF64}(I(size(U_goal, 1)))
    end

    if !blas_multithreading
        BLAS.set_num_threads(1)
    end

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end


    n_drives = length(system.G_drives)

    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        if free_time
            if Δt isa Float64
                Δt = fill(Δt, 1, T)
            end
        end

        if isnothing(a_guess)
            if geodesic
                if operator isa EmbeddedOperator
                    Ũ⃗ = unitary_geodesic(operator, T)
                else
                    Ũ⃗ = unitary_geodesic(U_goal, T)
                end
            else
                Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, T)
            end

            if a_bounds isa AbstractVector
                a_dists = [Uniform(-a_bounds[i], a_bounds[i]) for i = 1:n_drives]
            elseif a_bounds isa Tuple
                a_dists = [Uniform(aᵢ_lb, aᵢ_ub) for (aᵢ_lb, aᵢ_ub) ∈ zip(a_bounds...)]
            else
                error("a_bounds must be a Vector or Tuple")
            end

            a = hcat([
                zeros(n_drives),
                vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
                zeros(n_drives)
            ]...)

            da = randn(n_drives, T) * drive_derivative_σ
            dda = randn(n_drives, T) * drive_derivative_σ
        else
            Ũ⃗ = unitary_rollout(
                operator_to_iso_vec(U_init),
                a_guess,
                Δt,
                system;
                integrator=rollout_integrator
            )
            a = a_guess
            da = derivative(a, Δt)
            dda = derivative(da, Δt)

            # to avoid constraint violation error at initial iteration
            da[:, end] = da[:, end-1] + Δt[end-1] * dda[:, end-1]
        end

        initial = (
            Ũ⃗ = operator_to_iso_vec(U_init),
            a = zeros(n_drives),
        )

        final = (
            a = zeros(n_drives),
        )

        goal = (
            Ũ⃗ = operator_to_iso_vec(U_goal),
        )

        bounds = (
            a = a_bounds,
            dda = dda_bounds,
        )

        if bound_unitary
            Ũ⃗_dim = size(Ũ⃗, 1)
            Ũ⃗_bound = (
                Ũ⃗ = (-ones(Ũ⃗_dim), ones(Ũ⃗_dim)),
            )
            bounds = merge(bounds, Ũ⃗_bound)
        end

        if free_time
            components = (
                Ũ⃗ = Ũ⃗,
                a = a,
                da = da,
                dda = dda,
                Δt = Δt,
            )

            bounds = merge(bounds, (Δt = (Δt_min, Δt_max),))

            traj = NamedTrajectory(
                components;
                controls=(:dda, :Δt),
                timestep=:Δt,
                bounds=bounds,
                initial=initial,
                final=final,
                goal=goal
            )
        else
            components = (
                Ũ⃗ = Ũ⃗,
                a = a,
                da = da,
                dda = dda,
            )

            traj = NamedTrajectory(
                components;
                controls=(:dda,),
                timestep=Δt,
                bounds=bounds,
                initial=initial,
                final=final,
                goal=goal
            )
        end
    end

    J = UnitaryInfidelityObjective(:Ũ⃗, traj, Q;
        subspace=operator isa EmbeddedOperator ? operator.subspace_indices : nothing,
    )
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    if leakage_suppression
        if operator isa EmbeddedOperator
            leakage_indices = get_unitary_isomorphism_leakage_indices(operator)
            J_leakage, slack_con = L1Regularizer(
                :Ũ⃗,
                traj;
                R_value=R_leakage,
                indices=leakage_indices
            )
            push!(constraints, slack_con)
            J += J_leakage
        else
            @warn "leakage_suppression is not supported for non-embedded operators, ignoring."
        end
    end

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
        DerivativeIntegrator(:da, :dda, traj),
    ]

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
function randomly_fill_drives(
    T::Int,
    drive_bounds_lower::AbstractVector{<:Real},
    drive_bounds_upper::AbstractVector{<:Real},
    drive_derivatives_σ::AbstractVector{<:Real}
)
    data = Matrix{Float64}[]

    # Drive
    n_drives = length(drive_bounds_lower)
    a_dists =  [Uniform(drive_bounds_lower[i], drive_bounds_upper[i]) for i = 1:n_drives]
    push!(
        data,
        hcat([
            zeros(n_drives),
            vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
            zeros(n_drives)
        ]...)
    )

    # Drive derivatives
    for σ ∈ drive_derivatives_σ
        push!(
            data,
            randn(n_drives, T) * σ
        )
    end
    return data
end

function randomly_fill_drives(
    T::Int,
    drive_bounds::AbstractVector{<:Real},
    drive_derivatives_σ::AbstractVector{<:Real}
)
    return randomly_fill_drives(
        T,
        -drive_bounds,
        drive_bounds,
        drive_derivatives_σ
    )
end

# *************************************************************************** #

@testitem "Random drive initialization" begin
    T = 10
    n_drives = 2
    drive_bounds = [1.0, 2.0]
    drive_derivatives_σ = repeat([0.01], 2)
    
    a, da, dda = ProblemTemplates.randomly_fill_drives(T, drive_bounds, drive_derivatives_σ)

    @test size(a) == (n_drives, T)
    @test size(da) == (n_drives, T)
    @test size(dda) == (n_drives, T)
    @test all([-drive_bounds[i] < minimum(a[i, :]) < drive_bounds[i] for i in 1:n_drives])
end
