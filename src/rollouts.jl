module Rollouts

export rollout
export unitary_rollout
export unitary_geodesic
export skew_symmetric
export skew_symmetric_vec
export linear_interpolation

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

    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = Integrators.G(
            aₜ₋₁,
            system.G_drift,
            system.G_drives
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
        operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drift_real, 1)))),
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
    Δt = timesteps(traj)
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


function skew_symmetric(v::AbstractVector, n::Int)
    M = zeros(eltype(v), n, n)
    k = 1
    for j = 1:n
        for i = 1:j-1
            vᵢⱼ = v[k]
            M[i, j] = vᵢⱼ
            M[j, i] = -vᵢⱼ
            k += 1
        end
    end
    return M
end

function skew_symmetric_vec(M::AbstractMatrix)
    n = size(M, 1)
    v = zeros(eltype(M), n * (n - 1) ÷ 2)
    k = 1
    for j = 1:n
        for i = 1:j-1
            v[k] = M[i, j]
            k += 1
        end
    end
    return v
end

function unitary_geodesic(U_goal, samples; kwargs...)
    N = size(U_goal, 1)
    U_init = Matrix{ComplexF64}(I(N))
    return unitary_geodesic(U_init, U_goal, 1:samples; kwargs...)
end

unitary_geodesic(
    U₀::AbstractMatrix{<:Number},
    U₁::AbstractMatrix{<:Number},
    samples::Number;
    kwargs...
) = unitary_geodesic(U₀, U₁, 1:samples; kwargs...)

function unitary_geodesic(
    U₀::AbstractMatrix{<:Number},
    U₁::AbstractMatrix{<:Number},
    timesteps::AbstractVector{<:Number};
    return_generator=false
)
    """
    Compute the effective generator of the geodesic connecting U₀ and U₁.
        U₁ = exp(-im * H * T) U₀
        log(U₁ * U₀') = -im * H * T

    Allow for the possibiltiy of unequal timesteps and ranges outside [0,1].

    Returns the geodesic.
    Optionally returns the effective Hamiltonian generating the geodesic.
    """
    t₀ = timesteps[1]
    T = timesteps[end] - t₀
    H = im * log(U₁ * U₀') / T
    # -im prefactor is not included in H
    U_geo = [exp(-im * H * (t - t₀)) * U₀ for t ∈ timesteps]
    Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
    if return_generator
        return Ũ⃗_geo, H
    else
        return Ũ⃗_geo
    end
end

function linear_interpolation(ψ̃₁::AbstractVector, ψ̃₂::AbstractVector, samples::Int)
    ts = range(0, 1; length=samples)
    ψ̃s = [ψ̃₁ + t * (ψ̃₂ - ψ̃₁) for t ∈ ts]
    return hcat(ψ̃s...)
end

end
