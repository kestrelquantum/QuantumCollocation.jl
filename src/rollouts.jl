module Rollouts

export rollout
export unitary_rollout
export lab_frame_unitary_rollout
export lab_frame_unitary_rollout_trajectory
export sample_at
export upsample

using ..QuantumUtils
using ..QuantumSystems
using ..EmbeddedOperators
using ..Integrators
using ..Problems

using LinearAlgebra
using NamedTrajectories

function rollout(
    ψ̃₁::AbstractVector{Float64},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    integrator=Integrators.fourth_order_pade
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
        Gₜ = Integrators.G(
            aₜ₋₁,
            G_drift,
            G_drives
        )
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
    return fidelity(ket_to_iso(ψ₁), ket_to_iso(ψ_goal), args...; kwargs...)
end


function unitary_rollout(
    Ũ⃗₁::AbstractVector{<:Real},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}, Float64},
    system::AbstractQuantumSystem;
    integrator=exp
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
        Gₜ = Integrators.G(
            aₜ₋₁,
            G_drift,
            G_drives
        )
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

function sample_at(t::Float64, controls::AbstractMatrix{Float64}, Δt::Float64)
    @assert t >= 0 "t must be non-negative"
    @assert Δt > 0 "Δt must be positive"
    @assert t <= size(controls, 2) * Δt "t must be less than the duration of the controls"

    i = floor(Int, t / Δt) + 1
    controls[:, i]
end

function upsample(
    controls::AbstractMatrix{Float64},
    duration::Float64,
    Δt::Float64,
    Δt_new::Float64
)
    @assert Δt > 0 "Δt must be positive"
    @assert Δt_new > 0 "Δt_new must be positive"
    @assert Δt_new < Δt "Δt_new must be less than Δt"

    times_upsampled = LinRange(0, duration, floor(Int, duration / Δt_new) + 1)

    controls_upsampled = stack([sample_at(t, controls, Δt) for t in times_upsampled])

    return controls_upsampled, times_upsampled
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
    sys::QuantumSystem,
    controls::AbstractMatrix{Float64};
    duration=nothing,
    timestep=nothing,
    ω=:ω ∈ keys(sys.params) ? sys.params[:ω] : nothing,
    timestep_nyquist=1 / (50 * ω)
)
    @assert !isnothing(duration) "must specify duration"
    @assert !isnothing(timestep) "must specify timestep"
    @assert !isnothing(ω) "must specify ω"

    # TODO: generalize to more than 2 drives
    @assert length(sys.H_drives) == 2 "must have 2 drives"


    controls_upsampled, times_upsampled =
        upsample(controls, duration, timestep, timestep_nyquist)

    u = controls_upsampled[1, :]
    v = controls_upsampled[2, :]

    c = cos.(2π * ω * times_upsampled)
    s = sin.(2π * ω * times_upsampled)

    pulse = stack([u .* c - v .* s, u .* s + v .* c], dims=1)

    return unitary_rollout(pulse, timestep_nyquist, sys), pulse
end

function lab_frame_unitary_rollout(
    sys::CompositeQuantumSystem,
    controls::AbstractMatrix{Float64};
    duration=nothing,
    timestep=nothing,
    ω_index=1,
    ω=:ω ∈ keys(sys.subsystems[ω_index].params) ?
        sys.subsystems[ω_index].params[:ω] :
        nothing,
    timestep_nyquist=1 / (50 * ω),
    drive_frequencies_all_equal=false,
)
    @assert !isnothing(duration) "must specify duration"
    @assert !isnothing(timestep) "must specify timestep"
    @assert !isnothing(ω) "must specify ω"

    # TODO: generalize to more than 2 drives per subsystem
    n_drives = length(sys.H_drives)

    controls_upsampled, times_upsampled =
        upsample(controls, duration, timestep, timestep_nyquist)

    U = controls_upsampled[1:2:n_drives, :]
    V = controls_upsampled[2:2:n_drives, :]

    if drive_frequencies_all_equal
        ωs = fill(ω, n_drives ÷ 2)
    else
        ωs = [subsys.params[:ω] for subsys ∈ sys.subsystems if !isempty(subsys.H_drives)]
    end

    C = cos.(2π * ωs * times_upsampled')
    S = sin.(2π * ωs * times_upsampled')

    pulse = vcat([
        stack([u .* c - v .* s, u .* s + v .* c]; dims=1)
            for (u, v, c, s) ∈ zip(eachrow.([U, V, C, S])...)
    ]...)

    return unitary_rollout(pulse, timestep_nyquist, sys), pulse
end



function lab_frame_unitary_rollout_trajectory(
    sys_lab_frame::AbstractQuantumSystem,
    traj_rotating_frame::NamedTrajectory,
    op_lab_frame::EmbeddedOperator;
    timestep_nyquist=sys_lab_frame isa QuantumSystem ?
        1/(400 * sys_lab_frame.params[:ω]) :
        1/(400 * sys_lab_frame.subsystems[1].params[:ω]),
    control_name::Symbol=:a,
    kwargs...
)::NamedTrajectory
    @assert sys_lab_frame.params[:lab_frame] "QuantumSystem must be in the lab frame"

    Ũ⃗_labframe, pulse = lab_frame_unitary_rollout(
        sys_lab_frame,
        traj_rotating_frame[control_name];
        duration=get_times(traj_rotating_frame)[end],
        timestep=traj_rotating_frame.Δt[end],
        timestep_nyquist=timestep_nyquist,
        kwargs...
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

end
