export DensityOperatorSmoothPulseProblem

function DensityOperatorSmoothPulseProblem(
    system::AbstractQuantumSystem,
    ρ_init::AbstractMatrix,
    ψ_goal::AbstractVector,
    T::Int,
    Δt::Union{Float64, Vector{Float64}};
    U_goal::Union{AbstractMatrix, Nothing}=nothing,
    ipopt_options::IpoptOptions=IpoptOptions(),
    piccolo_options::PiccoloOptions=PiccoloOptions(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    leakage_suppression=false,
    R_leakage=1e-1,
    control_norm_constraint=false,
    control_norm_constraint_components=nothing,
    control_norm_R=nothing,
    kwargs...
)
    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_trajectory(
            ρ_init,
            ψ_goal*ψ_goal',
            T,
            Δt,
            system.n_drives,
            (a_bounds, da_bounds, dda_bounds);
            free_time=piccolo_options.free_time,
            Δt_bounds=(Δt_min, Δt_max),
            drive_derivative_σ=drive_derivative_σ,
            a_guess=a_guess,
            system=system,
        )
    end

    # Objective
    J = DensityOperatorPureStateInfidelityObjective(:ρ⃗̃, ψ_goal; Q=Q)
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    # Constraints
    if leakage_suppression
        if operator isa EmbeddedOperator
            leakage_indices = get_iso_vec_leakage_indices(operator)
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
    # if piccolo_options.integrator == :pade
    #     unitary_integrator =
    #         UnitaryPadeIntegrator(system, :Ũ⃗, :a, traj; order=piccolo_options.pade_order)
    # elseif piccolo_options.integrator == :exponential
    #     unitary_integrator =
    #         UnitaryExponentialIntegrator(system, :Ũ⃗, :a, traj)
    # else
    #     error("integrator must be one of (:pade, :exponential)")
    # end

    density_operator_integrator = DensityOperatorExponentialIntegrator(
        :ρ⃗̃, :a, system, traj
    )

    integrators = [
        density_operator_integrator,
        DerivativeIntegrator(:a, :da, traj),
        DerivativeIntegrator(:da, :dda, traj),
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
