@doc raw"""
    UnitarySmoothPulseProblem(system::QuantumSystem, operator, T, Δt; kwargs...)

Construct a `QuantumControlProblem` for a free-time unitary gate problem with bang-bang control pulses.

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) + \frac{1}{2} \sum_t \qty(R_a a_t^2 + R_{\dot{a}} \dot{a}_t^2) \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
& \quad |a_t| \leq a_{\text{bound}} \\
& \quad |\dot{a}_t| \leq da_{\text{bound}} \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

where, for $U \in SU(N)$,

```math
\ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) =
\abs{1 - \frac{1}{N} \abs{ \tr \qty(U_{\text{goal}}, U_T)} }
```

is the *infidelity* objective function, $Q$ is a weight, $R_a$, and $R_{\dot{a}}$ are weights on the regularization terms, and $\vb{P}^{(n)}$ is the $n$th-order Pade integrator.

TODO: Document bang-bang modification.

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
- `ipopt_options::IpoptOptions=IpoptOptions()`: the options for the Ipopt solver
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: the options for the Piccolo solver
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: the constraints to enforce
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `da_bound::Float64=1.0`: the bound on the control pulse derivative
- `da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `Δt_min::Float64=0.5 * Δt`: the minimum time step size
- `Δt_max::Float64=1.5 * Δt`: the maximum time step size
- `drive_derivative_σ::Float64=0.01`: the standard deviation of the initial guess for the control pulse derivatives
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `leakage_suppression::Bool=false`: whether or not to suppress leakage to higher energy states
- `R_leakage=1e-1`: the weight on the leakage suppression term
- `bound_unitary=integrator == :exponential`: whether or not to bound the unitary
- `control_norm_constraint=false`: whether or not to enforce a constraint on the control pulse norm
- `control_norm_constraint_components=nothing`: the components of the control pulse to use for the norm constraint
- `control_norm_R=nothing`: the weight on the control pulse norm constraint


TODO: control modulus norm, advanced feature, needs documentation

"""
function UnitaryBangBangProblem(
    system::AbstractQuantumSystem,
    operator::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=1.0,
    da_bounds=fill(da_bound, length(system.G_drives)),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_bang_bang::Union{Float64, Vector{Float64}}=1.0,
    leakage_suppression=false,
    R_leakage=1e-1,
    control_norm_constraint=false,
    control_norm_constraint_components=nothing,
    control_norm_R=nothing,
    bound_unitary=piccolo_options.integrator == :exponential,
    kwargs...
)
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
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            geodesic=piccolo_options.geodesic,
            bound_unitary=bound_unitary,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
        )
    end

    # Objective
    J = UnitaryInfidelityObjective(:Ũ⃗, traj, Q;
        subspace=operator isa EmbeddedOperator ? operator.subspace_indices : nothing,
    )
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)

    # Constraints
    if R_bang_bang isa Float64
        R_bang_bang = fill(R_bang_bang, length(system.G_drives))
    end
    J += L1Regularizer!(
        constraints, :da, traj, 
        R=R_bang_bang, eval_hessian=piccolo_options.eval_hessian
    )

    if leakage_suppression
        if operator isa EmbeddedOperator
            leakage_indices = get_unitary_isomorphism_leakage_indices(operator)
            J += L1Regularizer!(
                constraints, :Ũ⃗, traj, 
                R_value=R_leakage, 
                indices=leakage_indices,
                eval_hessian=piccolo_options.eval_hessian
            )
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
            traj;
            name_comps=control_norm_constraint_components,
        )
        push!(constraints, norm_con)
    end

    # Integrators
    if piccolo_options.integrator == :pade
        unitary_integrator =
            UnitaryPadeIntegrator(system, :Ũ⃗, :a; order=piccolo_options.pade_order)
    elseif piccolo_options.integrator == :exponential
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
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        kwargs...
    )
end


"""
    UnitaryBangBangProblem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        operator,
        T,
        Δt;
        kwargs...
    )

Constructor for a `UnitaryBangBangProblem` from a drift Hamiltonian and a set of control Hamiltonians.
"""
function UnitaryBangBangProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return UnitaryBangBangProblem(system, args...; kwargs...)
end

# *************************************************************************** #

@testitem "Hadamard gate" begin
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2
    
    prob = UnitaryBangBangProblem(
        sys, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )
    
    initial = unitary_fidelity(prob)
    solve!(prob, max_iter=20)
    final = unitary_fidelity(prob)
    @test final > initial
end
