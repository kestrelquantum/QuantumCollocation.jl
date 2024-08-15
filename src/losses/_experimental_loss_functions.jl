#
# experimental loss functions
#
# TODO: renormalize vectors in place of abs
#       ⋅ penalize loss to remain near unit norm
#       ⋅ Σ α * (1 - ψ̃'ψ̃), α = 1e-3

function energy_loss(
    ψ̃::AbstractVector,
    H::AbstractMatrix
)
    ψ = iso_to_ket(ψ̃)
    return real(ψ' * H * ψ)
end

# TODO: figure out a way to implement this without erroring and Von Neumann entropy being always 0 for a pure state
function neg_entropy_loss(
    ψ̃::AbstractVector
)
    ψ = iso_to_ket(ψ̃)
    ρ = ψ * ψ'
    ρ = Hermitian(ρ)
    return tr(ρ * log(ρ))
end

function pure_real_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    return -(ψ'ψgoal)
end

function geodesic_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    return min(abs(1 - amp), abs(1 + amp))
end

function real_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    return min(abs(1 - real(amp)), abs(1 + real(amp)))
end

function quaternionic_loss(ψ̃, ψ̃goal)
    return min(
        abs(1 - dot(ψ̃, ψ̃goal)),
        abs(1 + dot(ψ̃, ψ̃goal))
    )
end