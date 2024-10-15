export UnitarySmoothPulseProblem


@doc raw"""
    UnitarySmoothPulseProblem(system::AbstractQuantumSystem, operator, T, Δt; kwargs...)
    UnitarySmoothPulseProblem(H_drift, H_drives, operator, T, Δt; kwargs...)

Construct a `QuantumControlProblem` for a free-time unitary gate problem with smooth control pulses enforced by constraining the second derivative of the pulse trajectory, i.e.,

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) + \frac{1}{2} \sum_t \qty(R_a a_t^2 + R_{\dot{a}} \dot{a}_t^2 + R_{\ddot{a}} \ddot{a}_t^2) \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& \quad a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
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

- `system::AbstractQuantumSystem`: the system to be controlled
or
- `H_drift::AbstractMatrix{<:Number}`: the drift hamiltonian
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: the control hamiltonians
with
- `operator::OperatorType`: the target unitary, either in the form of an `EmbeddedOperator` or a `Matrix{ComplexF64}
- `T::Int`: the number of timesteps
- `Δt::Float64`: the (initial) time step size

# Keyword Arguments
- `ipopt_options::IpoptOptions=IpoptOptions()`: the options for the Ipopt solver
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: the options for the Piccolo solver
- `state_name::Symbol = :Ũ⃗`: the name of the state
- `control_name::Symbol = :a`: the name of the control
- `timestep_name::Symbol = :Δt`: the name of the timestep
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `da_bound::Float64=Inf`: the bound on the control pulse derivative
- `da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `dda_bound::Float64=1.0`: the bound on the control pulse second derivative
- `dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives))`: the bounds on the control pulse second derivatives, one for each drive
- `Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt)`: the minimum time step size
- `Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt)`: the maximum time step size
- `drive_derivative_σ::Float64=0.01`: the standard deviation of the initial guess for the control pulse derivatives
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `global_data::Union{NamedTuple, Nothing}=nothing`: global data for the problem
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: the constraints to enforce

"""
function UnitarySmoothPulseProblem(
    system::AbstractQuantumSystem,
    operator::OperatorType,
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives)),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives)),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    global_data::Union{NamedTuple, Nothing}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    kwargs...
)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        n_drives = length(system.G_drives)

        traj = initialize_trajectory(
            operator,
            T,
            Δt,
            n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            geodesic=piccolo_options.geodesic,
            bound_state=piccolo_options.bound_state,
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
            global_data=global_data
        )
    end

    # Objective
    if isnothing(global_data)
        J = UnitaryInfidelityObjective(state_name, traj, Q;
            subspace=operator isa EmbeddedOperator ? operator.subspace_indices : nothing,
        )
    else
        # TODO: remove hardcoded args
        J = UnitaryFreePhaseInfidelityObjective(
            name=state_name,
            phase_name=:ϕ,
            goal=operator_to_iso_vec(
                operator isa EmbeddedOperator ?
                operator.operator :
                operator
            ),
            phase_operators=fill(GATES[:Z], length(traj.global_components[:ϕ])),
            Q=Q,
            eval_hessian=piccolo_options.eval_hessian,
            subspace=operator isa EmbeddedOperator ? operator.subspace_indices : nothing
        )
    end

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J += QuadraticRegularizer(control_names[1], traj, R_a; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[2], traj, R_da; timestep_name=timestep_name)
    J += QuadraticRegularizer(control_names[3], traj, R_dda; timestep_name=timestep_name)

    # Integrators
    if piccolo_options.integrator == :pade
        unitary_integrator =
            UnitaryPadeIntegrator(system, state_name, control_name, traj;
                order=piccolo_options.pade_order
            )
    elseif piccolo_options.integrator == :exponential
        unitary_integrator =
            UnitaryExponentialIntegrator(system, state_name, control_name, traj)
    else
        error("integrator must be one of (:pade, :exponential)")
    end

    integrators = [
        unitary_integrator,
        DerivativeIntegrator(control_name, control_names[2], traj),
        DerivativeIntegrator(control_names[2], control_names[3], traj),
    ]

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(J, constraints, piccolo_options, traj, operator, state_name, timestep_name)

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

@testitem "Hadamard gate" begin
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt;
        da_bound=1.0,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )

    initial = unitary_fidelity(prob)
    solve!(prob, max_iter=20)
    final = unitary_fidelity(prob)
    @test final > initial
end

@testitem "Hadamard gate with exponential integrator" begin
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, integrator=:exponential, jacobian_structure=false)
    )

    initial = unitary_fidelity(prob)
    solve!(prob, max_iter=20)
    final = unitary_fidelity(prob)
    @test final > initial
end

@testitem "Hadamard gate with exponential integrator, bounded states, and control norm constraint" begin
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    piccolo_options = PiccoloOptions(
        verbose=false,
        integrator=:exponential,
        # jacobian_structure=false,
        bound_state=true,
        complex_control_norm_constraint_name=:a
    )

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=piccolo_options
    )

    initial = unitary_fidelity(prob)
    solve!(prob, max_iter=20)
    final = unitary_fidelity(prob)
    @test final > initial
end



@testitem "EmbeddedOperator Hadamard gate" begin
    a = annihilate(3)
    sys = QuantumSystem([(a + a')/2, (a - a')/(2im)])
    U_goal = EmbeddedOperator(GATES[:H], sys)
    T = 51
    Δt = 0.2

    # Test embedded operator
    # ----------------------
    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )

    initial = unitary_fidelity(prob, subspace=U_goal.subspace_indices)
    solve!(prob, max_iter=20)
    final = unitary_fidelity(prob, subspace=U_goal.subspace_indices)
    @test final > initial

    # Test leakage suppression
    # ------------------------
    a = annihilate(4)
    sys = QuantumSystem([(a + a')/2, (a - a')/(2im)])
    U_goal = EmbeddedOperator(GATES[:H], sys)
    T = 50
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        leakage_suppression=true, R_leakage=1e-1,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false)
    )

    initial = unitary_fidelity(prob, subspace=U_goal.subspace_indices)
    solve!(prob, max_iter=20)
    final = unitary_fidelity(prob, subspace=U_goal.subspace_indices)
    @test final > initial
end
