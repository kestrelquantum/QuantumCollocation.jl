module Rollouts

export rollout
export unitary_rollout
export lab_frame_unitary_rollout
export lab_frame_unitary_rollout_trajectory
export sample_at
export resample

using ..QuantumUtils
using ..QuantumSystems
using ..EmbeddedOperators
using ..Integrators
using ..Problems
using ..DirectSums

using LinearAlgebra
using NamedTrajectories

function rollout(
    ψ̃₁::AbstractVector{Float64},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    integrator=exp,
    G=Integrators.G_bilinear
)
    if Δt isa AbstractMatrix
        @assert size(Δt, 1) == 1
        Δt = vec(Δt)
    elseif Δt isa Float64
        Δt = fill(Δt, size(controls, 2))
    end

    T = size(controls, 2)

    Ψ̃ = zeros(length(ψ̃₁), T)

    Ψ̃[:, 1] .= ψ̃₁

    G_drift = Matrix{Float64}(system.G_drift)
    G_drives = Matrix{Float64}.(system.G_drives)

    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = G(aₜ₋₁, G_drift, G_drives)
        Ψ̃[:, t] .= integrator(Gₜ * Δt[t - 1]) * Ψ̃[:, t - 1]
    end

    return Ψ̃
end

rollout(ψ::Vector{<:Complex}, args...; kwargs...) =
    rollout(ket_to_iso(ψ), args...; kwargs...)

function rollout(
    ψ̃₁s::AbstractVector{<:AbstractVector}, args...; kwargs...
)
    return vcat([rollout(ψ̃₁, args...; kwargs...) for ψ̃₁ ∈ ψ̃₁s]...)
end

function QuantumUtils.fidelity(
    ψ̃₁::AbstractVector{Float64},
    ψ̃_goal::AbstractVector{Float64},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    kwargs...
)
    Ψ̃ = rollout(ψ̃₁, controls, Δt, system; kwargs...)
    return iso_fidelity(Ψ̃[:, end], ψ̃_goal)
end

function QuantumUtils.fidelity(
    ψ₁::AbstractVector{<:Complex},
    ψ_goal::AbstractVector{<:Complex},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    kwargs...
)
    return fidelity(ket_to_iso(ψ₁), ket_to_iso(ψ_goal), controls, Δt, system; kwargs...)
end

function QuantumUtils.fidelity(
    trajectory::NamedTrajectory,
    system::AbstractQuantumSystem;
    state_symb::Symbol=:ψ̃,
    control_symb=:a,
    kwargs...
)
    fids = []
    for symb in trajectory.names
        if startswith(symb, state_symb)
            init = trajectory.initial[symb]
            goal = trajectory.goal[symb]
            push!(
                fids, 
                fidelity(init, goal, trajectory[control_symb], get_timesteps(trajectory), system; kwargs...)
            )
        end
    end
    return fids
end

function QuantumUtils.fidelity(
    prob::QuantumControlProblem;
    kwargs...
)
    return fidelity(prob.trajectory, prob.system; kwargs...)
end

# ============================================================================= #

function unitary_rollout(
    Ũ⃗₁::AbstractVector{<:Real},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    integrator=exp,
    G=Integrators.G_bilinear
)
    if Δt isa AbstractMatrix
        @assert size(Δt, 1) == 1
        Δt = vec(Δt)
    elseif Δt isa Float64
        Δt = fill(Δt, size(controls, 2))
    end

    T = size(controls, 2)

    Ũ⃗ = zeros(length(Ũ⃗₁), T)

    Ũ⃗[:, 1] .= Ũ⃗₁

    G_drift = Matrix{Float64}(system.G_drift)
    G_drives = Matrix{Float64}.(system.G_drives)

    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = G(aₜ₋₁, G_drift, G_drives)
        Ũ⃗ₜ₋₁ = Ũ⃗[:, t - 1]
        Ũₜ₋₁ = iso_vec_to_iso_operator(Ũ⃗ₜ₋₁)
        Ũₜ = integrator(Gₜ * Δt[t - 1]) * Ũₜ₋₁
        Ũ⃗ₜ = iso_operator_to_iso_vec(Ũₜ)
        Ũ⃗[:, t] .= Ũ⃗ₜ
    end

    return Ũ⃗
end

function unitary_rollout(
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    integrator=exp
)
    return unitary_rollout(
        operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drift, 1)))),
        controls,
        Δt,
        system;
        integrator=integrator
    )
end

function unitary_rollout(
    traj::NamedTrajectory,
    system::AbstractQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    drive_name::Symbol=:a,
    integrator=exp,
    only_drift=false
)
    Ũ⃗₁ = traj.initial[unitary_name]
    if only_drift
        controls = zeros(size(traj[drive_name]))
    else
        controls = traj[drive_name]
    end
    Δt = get_timesteps(traj)
    return unitary_rollout(
        Ũ⃗₁,
        controls,
        Δt,
        system;
        integrator=integrator
    )
end

function QuantumUtils.unitary_fidelity(
    traj::NamedTrajectory,
    system::AbstractQuantumSystem;
    unitary_name::Symbol=:Ũ⃗,
    subspace=nothing,
    kwargs...
)
    Ũ⃗_final = unitary_rollout(
        traj,
        system;
        unitary_name=unitary_name,
        kwargs...
    )[:, end]
    return unitary_fidelity(
        Ũ⃗_final,
        traj.goal[unitary_name];
        subspace=subspace
    )
end

function QuantumUtils.unitary_fidelity(
    prob::QuantumControlProblem;
    kwargs...
)
    return unitary_fidelity(prob.trajectory, prob.system; kwargs...)
end

function QuantumUtils.unitary_fidelity(
    U_goal::AbstractMatrix{ComplexF64},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    subspace=nothing,
    integrator=exp
)
    Ũ⃗_final = unitary_rollout(controls, Δt, system; integrator=integrator)[:, end]
    return unitary_fidelity(
        Ũ⃗_final,
        operator_to_iso_vec(U_goal);
        subspace=subspace
    )
end

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

"""
    sample_at(controls::AbstractMatrix{Float64}, t::Float64, ts::AbstractVector{Float64}; interp_method::Symbol=:linear)

Sample the controls at time `t` using linear interpolation.

# Keyword Arguments
- `interp_method::Symbol=:linear`: Interpolation method. Either `:linear` or `:constant`.
"""
function sample_at(
    t::Float64,
    controls::AbstractMatrix{Float64},
    ts::AbstractVector{Float64};
    interp_method::Symbol=:linear
)
    @assert interp_method ∈ (:constant, :linear) "interp_method must be either :constant or :linear"
    @assert t >= 0 "t must be non-negative"
    @assert t <= ts[end] "t must be less than the duration of the controls"
    if interp_method == :constant
        return controls[:, searchsortedfirst(ts, t)]
    else
        i = searchsortedfirst(ts, t)
        if i == 1
            return controls[:, 1]
        else
            t_prev = ts[i - 1]
            t_next = ts[i]
            α = (t - t_prev) / (t_next - t_prev)
            return (1 - α) * controls[:, i - 1] + α * controls[:, i]
        end
    end
end



function resample(
    controls::AbstractMatrix{Float64},
    ts::AbstractVector{Float64},
    ts_new::AbstractVector{Float64};
    kwargs...
)
    controls_new = stack([sample_at(t, controls, ts; kwargs...) for t ∈ ts_new])
    controls_new[:, end] = controls[:, end]
    return controls_new
end

function resample(controls, ts, Δt::Float64; kwargs...)
    n_samples = Int(ts[end] / Δt)
    ts_new = range(0, step=Δt, length=n_samples)
    return resample(controls, ts, ts_new)
end

function resample(controls, ts, N::Int; kwargs...)
    Δt = ts[end] / N
    ts_new = range(0, step=Δt, length=N)
    return resample(controls, ts, ts_new; kwargs...)
end

"""
    resample(controls::AbstractMatrix{Float64}, ts::AbstractVector{Float64}, ts_new::AbstractVector{Float64})
    resample(controls::AbstractMatrix{Float64}, ts::AbstractVector{Float64}, Δt::Float64)
    resample(controls::AbstractMatrix{Float64}, ts::AbstractVector{Float64}, N::Int)

Resample the controls at new time points.

# Keyword Arguments
- `interp_method::Symbol=:linear`: Interpolation method. Either `:linear` or `:constant`.
"""
function resample end

end
