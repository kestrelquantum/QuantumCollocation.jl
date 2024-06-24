# G_bilinear(a) helper function
function G_bilinear(
    a::AbstractVector,
    G_drift::AbstractMatrix,
    G_drives::AbstractVector{<:AbstractMatrix}
)
    return G_drift + sum(a .* G_drives)
end

const Id2 = 1.0 * I(2)
const Im2 = 1.0 * [0 -1; 1 0]

anticomm(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Number = A * B + B * A

function anticomm(
    A::AbstractMatrix{R},
    Bs::AbstractVector{<:AbstractMatrix{R}}
) where R <: Number
    return [anticomm(A, B) for B in Bs]
end

function anticomm(
    As::AbstractVector{<:AbstractMatrix{R}},
    Bs::AbstractVector{<:AbstractMatrix{R}}
) where R <: Number
    @assert length(As) == length(Bs)
    n = length(As)
    anticomms = Matrix{Matrix{R}}(undef, n, n)
    for i = 1:n
        for j = 1:n
            anticomms[i, j] = anticomm(As[i], Bs[j])
        end
    end
    return anticomms
end

pade(n, k) = (factorial(n + k) // (factorial(n - k) * factorial(k) * 2^n))
pade_coeffs(n) = [pade(n, k) for k = n:-1:0][2:end] // pade(n, n)

@inline function operator(
    a::AbstractVector{<:Real},
    A_drift::Matrix{<:Real},
    A_drives::Vector{<:Matrix{<:Real}}
)
    return A_drift + sum(a .* A_drives)
end

@inline function operator_anticomm_operator(
    a::AbstractVector{<:Real},
    A_drift_anticomm_B_drift::Matrix{<:Real},
    A_drift_anticomm_B_drives::Vector{<:Matrix{<:Real}},
    B_drift_anticomm_A_drives::Vector{<:Matrix{<:Real}},
    A_drives_anticomm_B_drives::Matrix{<:Matrix{<:Real}},
    n_drives::Int
)
    A_anticomm_B = A_drift_anticomm_B_drift
    for i = 1:n_drives
        aⁱ = a[i]
        A_anticomm_B += aⁱ * A_drift_anticomm_B_drives[i]
        A_anticomm_B += aⁱ * B_drift_anticomm_A_drives[i]
        A_anticomm_B += aⁱ^2 * A_drives_anticomm_B_drives[i, i]
        for j = i+1:n_drives
            aʲ = a[j]
            A_anticomm_B += 2 * aⁱ * aʲ * A_drives_anticomm_B_drives[i, j]
        end
    end
    return A_anticomm_B
end

@inline function operator_anticomm_term(
    a::AbstractVector{<:Real},
    A_drift_anticomm_B_drives::Vector{<:Matrix{<:Real}},
    A_drives_anticomm_B_drives::Matrix{<:Matrix{<:Real}},
    n_drives::Int,
    j::Int
)
    A_anticomm_Bⱼ = A_drift_anticomm_B_drives[j]
    for i = 1:n_drives
        aⁱ = a[i]
        A_anticomm_Bⱼ += aⁱ * A_drives_anticomm_B_drives[i, j]
    end
    return A_anticomm_Bⱼ
end

function build_anticomms(
    G_drift::AbstractMatrix{R},
    G_drives::Vector{<:AbstractMatrix{R}},
    n_drives::Int) where R <: Number

    drive_anticomms = fill(
            zeros(size(G_drift)),
            n_drives,
            n_drives
        )

        for j = 1:n_drives
            for k = 1:j
                if k == j
                    drive_anticomms[k, k] = 2 * G_drives[k]^2
                else
                    drive_anticomms[k, j] =
                        anticomm(G_drives[k], G_drives[j])
                end
            end
        end

        drift_anticomms = [
            anticomm(G_drive, G_drift)
                for G_drive in G_drives
        ]

    return Symmetric(drive_anticomms), drift_anticomms
end
