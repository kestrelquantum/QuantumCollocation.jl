export UnitaryBangBangProblem


@doc raw"""
    UnitaryBangBangProblem(system::QuantumSystem, operator, T, Δt; kwargs...)
    UnitaryBangBangProblem(H_drift, H_drives, operator, T, Δt; kwargs...)

Construct a `QuantumControlProblem` for a free-time unitary gate problem with bang-bang control pulses.

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) + R_{\text{bang-bang}} \cdot \sum_t |\dot{a}_t| \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& \quad a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
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
- `operator::OperatorType`: the target unitary, either in the form of an `EmbeddedOperator` or a `Matrix{ComplexF64}
- `T::Int`: the number of timesteps
- `Δt::Float64`: the (initial) time step size

# Keyword Arguments
- `ipopt_options::IpoptOptions=IpoptOptions()`: the options for the Ipopt solver
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: the options for the Piccolo solver
- `state_name::Symbol = :Ũ⃗`: the name of the state variable
- `control_name::Symbol = :a`: the name of the control variable
- `timestep_name::Symbol = :Δt`: the name of the timestep variable
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `da_bound::Float64=1.0`: the bound on the control pulse derivative
- `da_bounds=fill(da_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt)`: the minimum time step size
- `Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt)`: the maximum time step size
- `drive_derivative_σ::Float64=0.01`: the standard deviation of the initial guess for the control pulse derivatives
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `quadratic_control_regularization=false`: whether or not to use quadratic regularization for the control pulses
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_bang_bang::Union{Float64, Vector{Float64}}=1e-1`: the weight on the bang-bang regularization term
- `phase_operators::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing`: the phase operators for free phase corrections
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: the constraints to enforce
"""
function UnitaryBangBangProblem end

function UnitaryBangBangProblem(
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
    a_bounds=fill(a_bound, system.n_drives),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=1.0,
    da_bounds=fill(da_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=0.0,
    R_bang_bang::Union{Float64, Vector{Float64}}=1e-1,
    phase_name::Symbol=:ϕ,
    phase_operators::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    kwargs...
)
    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_trajectory(
            operator,
            T,
            Δt,
            system.n_drives,
            (a_bounds, da_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            geodesic=piccolo_options.geodesic,
            bound_state=piccolo_options.bound_state,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
            phase_name=phase_name,
            phase_operators=phase_operators
        )
    end

    subspace = operator isa EmbeddedOperator ? operator.subspace_indices : nothing

    # Objective
    if isnothing(phase_operators)
        J = UnitaryInfidelityObjective(state_name, traj, Q;
            subspace=subspace,
        )
    else
        J = UnitaryFreePhaseInfidelityObjective(
            state_name, phase_name, phase_operators, traj, Q;
            subspace=subspace,
            eval_hessian=piccolo_options.eval_hessian,
        )
    end

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J += QuadraticRegularizer(control_names[1], traj, R_a; timestep_name=timestep_name)
    # Default to R_da = 0.0 because L1 is turned on by this problem.
    J += QuadraticRegularizer(control_names[2], traj, R_da; timestep_name=timestep_name)

    # Constraints
    if R_bang_bang isa Float64
        R_bang_bang = fill(R_bang_bang, system.n_drives)
    end
    J += L1Regularizer!(
        constraints, control_names[2], traj,
        R=R_bang_bang, eval_hessian=piccolo_options.eval_hessian
    )

    # Integrators
    if piccolo_options.integrator == :pade
        unitary_integrator =
            UnitaryPadeIntegrator(state_name, control_names[1], system, traj; order=piccolo_options.pade_order)
    elseif piccolo_options.integrator == :exponential
        unitary_integrator =
            UnitaryExponentialIntegrator(state_name, control_names[1], system, traj)
    else
        error("integrator must be one of (:pade, :exponential)")
    end

    integrators = [
        unitary_integrator,
        DerivativeIntegrator(control_names[1], control_names[2], traj),
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

@testitem "Bang-bang Hadamard gate" begin
    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    ipopt_options = IpoptOptions(print_level=1, max_iter=25)
    piccolo_options = PiccoloOptions(verbose=false, pade_order=12)

    prob = UnitaryBangBangProblem(
        sys, U_goal, T, Δt,
        R_bang_bang=10.,
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        control_name=:u
    )

    smooth_prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt;
        ipopt_options=ipopt_options,
        piccolo_options=piccolo_options,
        control_name=:u
    )
    initial = unitary_rollout_fidelity(prob; drive_name=:u)
    solve!(prob)
    final = unitary_rollout_fidelity(prob; drive_name=:u)
    @test final > initial
    solve!(smooth_prob)
    threshold = 1e-3
    a_sparse = sum(prob.trajectory.du .> 5e-2)
    a_dense = sum(smooth_prob.trajectory.du .> 5e-2)
    @test a_sparse < a_dense
end

@testitem "Free phase Y gate using X" begin
    using Random
    Random.seed!(1234)
    phase_name = :ϕ
    phase_operators = [PAULIS[:Z]]

    prob = UnitaryBangBangProblem(
        QuantumSystem([PAULIS[:X]]), GATES[:Y], 51, 0.2;
        phase_operators=phase_operators,
        phase_name=phase_name,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, free_time=false)
    )

    before = prob.trajectory.global_data[phase_name]
    solve!(prob, max_iter=20)
    after = prob.trajectory.global_data[phase_name]

    @test before ≠ after

    @test unitary_fidelity(
        prob, 
        phases=prob.trajectory.global_data[phase_name],
        phase_operators=phase_operators
    ) > 0.9

    @test unitary_fidelity(prob) < 0.9
end
