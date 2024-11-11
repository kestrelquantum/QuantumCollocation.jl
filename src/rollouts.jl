module Rollouts

export rollout
export open_rollout
export unitary_rollout
export lab_frame_unitary_rollout
export lab_frame_unitary_rollout_trajectory

using ..Isomorphisms
using ..QuantumSystems
using ..QuantumSystemUtils
using ..QuantumObjectUtils
using ..EmbeddedOperators
using ..Integrators
using ..Losses
using ..Problems
using ..DirectSums

using NamedTrajectories

using ExponentialAction
using LinearAlgebra
using ProgressMeter
using TestItemRunner


# ----------------------------------------------------------------------------- #

"""
    infer_is_evp(integrator::Function)

Infer whether the integrator is a exponential-vector product (EVP) function.

If `true`, the integrator is expected to have a signature like the exponential action,
`expv`. Otherwise, it is expected to have a signature like `exp`.
"""
function infer_is_evp(integrator::Function)
    # name + args
    ns = fieldcount.([m.sig for m ∈ methods(integrator)])
    is_exp = 2 ∈ ns
    is_expv = 4 ∈ ns
    if is_exp && is_expv
        throw(ErrorException("Ambiguous rollout integrator signature. Please specify manually."))
    elseif is_exp
        return false
    elseif is_expv
        return true
    else
        throw(ErrorException("No valid rollout integrator signature found."))
    end
end

# ----------------------------------------------------------------------------- #
# Quantum state rollouts
# ----------------------------------------------------------------------------- #

@doc raw"""
    rollout(
        ψ̃_init::AbstractVector{<:Float64},
        controls::AbstractMatrix,
        Δt::AbstractVector,
        system::AbstractQuantumSystem
    )

Rollout a quantum state `ψ̃_init` under the control `controls` for a time `Δt`
using the system `system`.

If `exp_vector_product` is `true`, the integrator is expected to have a signature like
the exponential action, `expv`. Otherwise, it is expected to have a signature like `exp`.

Types should allow for autodifferentiable controls and times.
"""
function rollout(
    ψ̃_init::AbstractVector{<:Real},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
    G=Integrators.G_bilinear
)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(ψ̃_init, controls, Δt)
    Ψ̃ = zeros(R, length(ψ̃_init), T)

    Ψ̃[:, 1] .= ψ̃_init

    G_drift = Matrix{Float64}(system.G_drift)
    G_drives = Matrix{Float64}.(system.G_drives)

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = G(aₜ₋₁, G_drift, G_drives)
        if exp_vector_product
            Ψ̃[:, t] .= integrator(Δt[t - 1], Gₜ, Ψ̃[:, t - 1])
        else
            Ψ̃[:, t] .= integrator(Gₜ * Δt[t - 1]) * Ψ̃[:, t - 1]
        end
        next!(p)
    end

    return Ψ̃
end

rollout(ψ::Vector{<:Complex}, args...; kwargs...) =
    rollout(ket_to_iso(ψ), args...; kwargs...)

function rollout(
    inits::AbstractVector{<:AbstractVector}, args...; kwargs...
)
    return vcat([rollout(state, args...; kwargs...) for state ∈ inits]...)
end

function Losses.iso_fidelity(
    ψ̃_init::AbstractVector{<:Real},
    ψ̃_goal::AbstractVector{<:Real},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ψ̃ = rollout(ψ̃_init, controls, Δt, system; kwargs...)
    return iso_fidelity(Ψ̃[:, end], ψ̃_goal)
end

function Losses.fidelity(
    ψ_init::AbstractVector{<:Complex},
    ψ_goal::AbstractVector{<:Complex},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    return iso_fidelity(ket_to_iso(ψ_init), ket_to_iso(ψ_goal), controls, Δt, system; kwargs...)
end

function Losses.fidelity(
    trajectory::NamedTrajectory,
    system::AbstractQuantumSystem;
    state_symb::Symbol=:ψ̃,
    control_symb=:a,
    kwargs...
)
    fids = []
    for symb in trajectory.names
        if startswith(symb, state_symb)
            controls = trajectory[control_symb]
            init = trajectory.initial[symb]
            goal = trajectory.goal[symb]
            fid = iso_fidelity(init, goal, controls, get_timesteps(trajectory), system; kwargs...)
            push!(fids, fid)
        end
    end
    return length(fids) == 1 ? fids[1] : fids
end

function Losses.fidelity(
    prob::QuantumControlProblem;
    kwargs...
)
    return fidelity(prob.trajectory, prob.system; kwargs...)
end

# ----------------------------------------------------------------------------- #
# Open quantum system rollouts
# ----------------------------------------------------------------------------- #

function open_rollout(
    ρ⃗₁::AbstractVector{<:Complex},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
    H=a -> Integrators.G_bilinear(a, system.H_drift, system.H_drives),
)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(ρ⃗₁, controls, Δt)
    ρ⃗̃ = zeros(R, 2length(ρ⃗₁), T)

    ρ⃗̃[:, 1] .= ket_to_iso(ρ⃗₁)

    if isnothing(system.dissipation_operators)
        @error "No dissipation operators found in system"
        L = iso(zeros(Float64, size(system.H_drift)))
    else
        L_dissipators = system.dissipation_operators
        L = Integrators.L_function(L_dissipators)
    end

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        adGₜ = Isomorphisms.G(ad_vec(H(aₜ₋₁)))
        if exp_vector_product
            ρ⃗̃[:, t] = integrator(Δt[t - 1], adGₜ + iso(L), ρ⃗̃[:, t - 1])
        else
            ρ⃗̃[:, t] = integrator(Δt[t - 1], adGₜ + iso(L)) * ρ⃗̃[:, t - 1]
        end
        next!(p)
    end

    return ρ⃗̃
end

# ----------------------------------------------------------------------------- #
# Unitary rollouts
# ----------------------------------------------------------------------------- #

function unitary_rollout(
    Ũ⃗_init::AbstractVector{<:Real},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    show_progress=false,
    integrator=expv,
    exp_vector_product=infer_is_evp(integrator),
    G=Integrators.G_bilinear,
)
    T = size(controls, 2)

    # Enable ForwardDiff
    R = Base.promote_eltype(Ũ⃗_init, controls, Δt)
    Ũ⃗ = zeros(R, length(Ũ⃗_init), T)

    Ũ⃗[:, 1] .= Ũ⃗_init

    G_drift = Matrix{Float64}(system.G_drift)
    G_drives = Matrix{Float64}.(system.G_drives)

    p = Progress(T-1; enabled=show_progress)
    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = G(aₜ₋₁, G_drift, G_drives)
        Ũₜ₋₁ = iso_vec_to_iso_operator(Ũ⃗[:, t - 1])
        if exp_vector_product
            Ũₜ = integrator(Δt[t - 1], Gₜ, Ũₜ₋₁)
        else
            Ũₜ = integrator(Gₜ * Δt[t - 1]) * Ũₜ₋₁
        end
        Ũ⃗[:, t] .= iso_operator_to_iso_vec(Ũₜ)
        next!(p)
    end

    return Ũ⃗
end

function unitary_rollout(
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ĩ⃗ = operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drift, 1))))
    return unitary_rollout(Ĩ⃗, controls, Δt, system; kwargs...)
end

function unitary_rollout(
    traj::NamedTrajectory,
    system::AbstractQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    drive_name::Symbol=:a,
    kwargs...
)
    return unitary_rollout(
        traj.initial[unitary_name],
        traj[drive_name],
        get_timesteps(traj),
        system;
        kwargs...
    )
end

"""
Compute the rollout fidelity.
"""
function Losses.iso_vec_unitary_fidelity(
    Ũ⃗_init::AbstractVector{<:Real},
    Ũ⃗_goal::AbstractVector{<:Real},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1),
    phases::Union{Nothing, AbstractVector{<:Real}}=nothing,
    phase_operators::Union{Nothing, AbstractVector{<:AbstractMatrix{<:Complex}}}=nothing,
    kwargs...
)
    Ũ⃗_T = unitary_rollout(Ũ⃗_init, controls, Δt, system; kwargs...)[:, end]
    if !isnothing(phases)
        return iso_vec_unitary_free_phase_fidelity(Ũ⃗_T, Ũ⃗_goal, phases, phase_operators; subspace=subspace)
    else
        return iso_vec_unitary_fidelity(Ũ⃗_T, Ũ⃗_goal; subspace=subspace)
    end
end

function Losses.iso_vec_unitary_fidelity(
    Ũ⃗_goal::AbstractVector{<:Real},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ĩ⃗ = operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drift, 1))))
    return iso_vec_unitary_fidelity(Ĩ⃗, Ũ⃗_goal, controls, Δt, system; kwargs...)
end

function Losses.unitary_fidelity(
    U_init::AbstractMatrix{<:Complex},
    U_goal::AbstractMatrix{<:Complex},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    return iso_vec_unitary_fidelity(Ũ⃗_init, Ũ⃗_goal, controls, Δt, system; kwargs...)
end

Losses.unitary_fidelity(
    U_goal::AbstractMatrix{<:Complex},
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    kwargs...
) = iso_vec_unitary_fidelity(operator_to_iso_vec(U_goal), controls, Δt, system; kwargs...)

Losses.unitary_fidelity(
    U_goal::EmbeddedOperator,
    controls::AbstractMatrix,
    Δt::AbstractVector,
    system::AbstractQuantumSystem;
    subspace::AbstractVector{Int}=U_goal.subspace_indices,
    kwargs...
) = unitary_fidelity(U_goal.operator, controls, Δt, system; subspace=subspace, kwargs...)

function Losses.unitary_fidelity(
    traj::NamedTrajectory,
    sys::AbstractQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    drive_name::Symbol=:a,
    kwargs...
)
    Ũ⃗_init = traj.initial[unitary_name]
    Ũ⃗_goal = traj.goal[unitary_name]
    controls = traj[drive_name]
    Δt = get_timesteps(traj)
    return iso_vec_unitary_fidelity(Ũ⃗_init, Ũ⃗_goal, controls, Δt, sys; kwargs...)
end

Losses.unitary_fidelity(prob::QuantumControlProblem; kwargs...) =
    unitary_fidelity(prob.trajectory, prob.system; kwargs...)


# ----------------------------------------------------------------------------- #
# Experimental rollouts
# ----------------------------------------------------------------------------- #

Losses.unitary_fidelity(
    U_goal::EmbeddedOperator,
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    sys::AbstractQuantumSystem;
    subspace=U_goal.subspace_indices,
    kwargs...
) = unitary_fidelity(U_goal.operator, controls, Δt, sys; subspace=subspace, kwargs...)

"""
    lab_frame_unitary_rollout(
        sys::AbstractQuantumSystem,
        controls::AbstractMatrix{Float64};
        duration=nothing,
        timestep=nothing,
        ω=nothing,
        timestep_nyquist=1 / (50 * ω)
    )
"""
function lab_frame_unitary_rollout(
    sys::AbstractQuantumSystem,
    controls::AbstractMatrix{Float64};
    duration=nothing,
    timestep=nothing,
    ω=nothing,
    timestep_nyquist=1 / (50 * ω)
)
    @assert !isnothing(duration) "must specify duration"
    @assert !isnothing(timestep) "must specify timestep"
    @assert !isnothing(ω) "must specify ω"

    controls_upsampled, times_upsampled =
        upsample(controls, duration, timestep, timestep_nyquist)

    u = controls_upsampled[1, :]
    v = controls_upsampled[2, :]

    c = cos.(2π * ω * times_upsampled)
    s = sin.(2π * ω * times_upsampled)

    pulse = stack([u .* c - v .* s, u .* s + v .* c], dims=1)

    return unitary_rollout(pulse, timestep_nyquist, sys), pulse
end

function lab_frame_unitary_rollout_trajectory(
    sys_lab_frame::AbstractQuantumSystem,
    traj_rotating_frame::NamedTrajectory,
    op_lab_frame::EmbeddedOperator;
    timestep_nyquist=1/(400 * sys_lab_frame.params[:ω]),
    control_name::Symbol=:a,
)::NamedTrajectory
    @assert sys_lab_frame.params[:lab_frame] "QuantumSystem must be in the lab frame"

    Ũ⃗_labframe, pulse = lab_frame_unitary_rollout(
        sys_lab_frame,
        traj_rotating_frame[control_name];
        duration=get_times(traj_rotating_frame)[end],
        timestep=traj_rotating_frame.Δt[end],
        ω=sys_lab_frame.params[:ω],
        timestep_nyquist=timestep_nyquist
    )

    return NamedTrajectory(
        (
            Ũ⃗=Ũ⃗_labframe,
            a=pulse
        );
        timestep=timestep_nyquist,
        controls=(control_name,),
        initial=(
            Ũ⃗=Ũ⃗_labframe[:, 1],
        ),
        goal=(
            Ũ⃗=operator_to_iso_vec(op_lab_frame.operator),
        )
    )
end

# ============================================================================= #

@testitem "Test rollouts using fidelities" begin
    using ExponentialAction

    sys = QuantumSystem(0 * GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:X]
    embedded_U_goal = EmbeddedOperator(U_goal, sys)
    T = 51
    Δt = 0.2
    ts = fill(Δt, T)
    as = collect([π/(T-1)/Δt * sin.(π*(0:T-1)/(T-1)).^2 zeros(T)]')

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt, a_guess=as,
        ipopt_options=IpoptOptions(print_level=1),
        piccolo_options=PiccoloOptions(verbose=false, free_time=false)
    )

    ψ = ComplexF64[1, 0]
    ψ_goal = U_goal * ψ
    ψ̃ = ket_to_iso(ψ)
    ψ̃_goal = ket_to_iso(ψ_goal)

    # Default integrator
    # State fidelity
    @test fidelity(ψ, ψ_goal, as, ts, sys) ≈ 1
    @test iso_fidelity(ψ̃, ψ̃_goal, as, ts, sys) ≈ 1

    # Unitary fidelity
    @test unitary_fidelity(U_goal, as, ts, sys) ≈ 1
    @test unitary_fidelity(prob.trajectory, sys) ≈ 1
    @test unitary_fidelity(prob) ≈ 1
    @test unitary_fidelity(embedded_U_goal, as, ts, sys) ≈ 1

    # Free phase unitary
    @test unitary_fidelity(prob, phases=[0.0], phase_operators=[PAULIS[:Z]]) ≈ 1

    # Expv explicit
    # State fidelity
    @test fidelity(ψ, ψ_goal, as, ts, sys, integrator=expv) ≈ 1
    @test iso_fidelity(ψ̃, ψ̃_goal, as, ts, sys, integrator=expv) ≈ 1

    # Unitary fidelity
    @test unitary_fidelity(U_goal, as, ts, sys, integrator=expv) ≈ 1
    @test unitary_fidelity(prob.trajectory, sys, integrator=expv) ≈ 1
    @test unitary_fidelity(prob, integrator=expv) ≈ 1
    @test unitary_fidelity(embedded_U_goal, as, ts, sys, integrator=expv) ≈ 1

    # Exp explicit
    # State fidelity
    @test fidelity(ψ, ψ_goal, as, ts, sys, integrator=exp) ≈ 1
    @test iso_fidelity(ψ̃, ψ̃_goal, as, ts, sys, integrator=exp) ≈ 1

    # Unitary fidelity
    @test unitary_fidelity(U_goal, as, ts, sys, integrator=exp) ≈ 1
    @test unitary_fidelity(prob.trajectory, sys, integrator=exp) ≈ 1
    @test unitary_fidelity(prob, integrator=exp) ≈ 1
    @test unitary_fidelity(embedded_U_goal, as, ts, sys, integrator=exp) ≈ 1

    # Bad integrator
    @test_throws ErrorException unitary_fidelity(U_goal, as, ts, sys, integrator=(a,b) -> 1) ≈ 1
end

@testitem "Foward diff rollout" begin
    using ForwardDiff
    using ExponentialAction

    sys = QuantumSystem(0 * GATES[:Z], [GATES[:X], GATES[:Y]])
    T = 51
    Δt = 0.2
    ts = fill(Δt, T)
    as = collect([π/(T-1)/Δt * sin.(π*(0:T-1)/(T-1)).^2 zeros(T)]')

    # Control derivatives
    ψ = ComplexF64[1, 0]
    result1 = ForwardDiff.jacobian(
        as -> rollout(ψ, as, ts, sys, integrator=expv)[:, end], as
    )
    iso_ket_dim = length(ket_to_iso(ψ))
    @test size(result1) == (iso_ket_dim, T * length(sys.H_drives))

    result2 = ForwardDiff.jacobian(
        as -> unitary_rollout(as, ts, sys, integrator=expv)[:, end], as
    )
    iso_vec_dim = length(operator_to_iso_vec(sys.H_drift))
    @test size(result2) == (iso_vec_dim, T * length(sys.H_drives))

    # Time derivatives
    ψ = ComplexF64[1, 0]
    result1 = ForwardDiff.jacobian(
        ts -> rollout(ψ, as, ts, sys, integrator=expv)[:, end], ts
    )
    iso_ket_dim = length(ket_to_iso(ψ))
    @test size(result1) == (iso_ket_dim, T)

    result2 = ForwardDiff.jacobian(
        ts -> unitary_rollout(as, ts, sys, integrator=expv)[:, end], ts
    )
    iso_vec_dim = length(operator_to_iso_vec(sys.H_drift))
    @test size(result2) == (iso_vec_dim, T)
end
end
