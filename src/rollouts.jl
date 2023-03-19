module Rollouts

export rollout
export unitary_rollout

using ..QuantumUtils
using ..QuantumSystems
using ..Integrators

function rollout(
    ψ̃₁::AbstractVector,
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}},
    system::QuantumSystem;
    integrator=Integrators.fourth_order_pade
)
    if Δt isa AbstractMatrix
        @assert size(Δt, 1) == 1
        Δt = vec(Δt)
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

function unitary_rollout(
    Ũ⃗₁::AbstractVector{<:Real},
    controls::AbstractMatrix{Float64},
    Δt::Union{AbstractVector{Float64}, AbstractMatrix{Float64}},
    system::QuantumSystem;
    integrator=Integrators.fourth_order_pade
)
    if Δt isa AbstractMatrix
        @assert size(Δt, 1) == 1
        Δt = vec(Δt)
    end

    T = size(controls, 2)

    Ũ⃗ = zeros(length(Ũ⃗₁), T)

    Ũ⃗[:, 1] .= Ũ⃗₁

    for t = 2:T
        aₜ₋₁ = controls[:, t - 1]
        Gₜ = Integrators.G(
            aₜ₋₁,
            system.G_drift,
            system.G_drives
        )
        Ũ⃗ₜ₋₁ = Ũ⃗[:, t - 1]
        Ũₜ₋₁ = iso_vec_to_iso_unitary(Ũ⃗ₜ₋₁)
        Ũₜ = integrator(Gₜ * Δt[t - 1]) * Ũₜ₋₁
        Ũ⃗ₜ = iso_unitary_to_iso_vec(Ũₜ)
        Ũ⃗[:, t] .= Ũ⃗ₜ
    end

    return Ũ⃗
end


end
