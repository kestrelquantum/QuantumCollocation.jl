module Isomorphisms

export mat
export ket_to_iso
export iso_to_ket
export iso_vec_to_operator
export iso_vec_to_iso_operator
export operator_to_iso_vec
export iso_operator_to_iso_vec
export iso_operator_to_operator
export operator_to_iso_operator
export iso
export iso_dm
export ad_vec
export ⊗

using LinearAlgebra
using TestItemRunner

⊗(xs::AbstractVecOrMat...) = kron(xs...)

@doc raw"""
    mat(x::AbstractVector)

Convert a vector `x` into a square matrix. The length of `x` must be a perfect square.
"""
function mat(x::AbstractVector)
    n = isqrt(length(x))
    @assert n^2 == length(x) "Vector length must be a perfect square"
    return reshape(x, n, n)
end


# ----------------------------------------------------------------------------- #
#                                Kets                                           #
# ----------------------------------------------------------------------------- #

@doc raw"""
    ket_to_iso(ψ)

Convert a ket vector `ψ` into a complex vector with real and imaginary parts.
"""
ket_to_iso(ψ) = [real(ψ); imag(ψ)]

@doc raw"""
    iso_to_ket(ψ̃)

Convert a complex vector `ψ̃` with real and imaginary parts into a ket vector.
"""
iso_to_ket(ψ̃) = ψ̃[1:div(length(ψ̃), 2)] + im * ψ̃[(div(length(ψ̃), 2) + 1):end]

# ----------------------------------------------------------------------------- #
#                             Unitaries                                         #
# ----------------------------------------------------------------------------- #

@doc raw"""
    iso_vec_to_operator(Ũ⃗::AbstractVector)

Convert a real vector `Ũ⃗` into a complex matrix representing an operator.

Must be differentiable.
"""
function iso_vec_to_operator(Ũ⃗::AbstractVector{R}) where R
    Ũ⃗_dim = div(length(Ũ⃗), 2)
    N = Int(sqrt(Ũ⃗_dim))
    U = Matrix{complex(R)}(undef, N, N)
    for i=0:N-1
        U[:, i+1] .= @view(Ũ⃗[i * 2N .+ (1:N)]) + one(R) * im * @view(Ũ⃗[i * 2N .+ (N+1:2N)])
    end
    return U
end

@doc raw"""
    iso_vec_to_iso_operator(Ũ⃗::AbstractVector)

Convert a real vector `Ũ⃗` into a real matrix representing an isomorphism operator.

Must be differentiable.
"""
function iso_vec_to_iso_operator(Ũ⃗::AbstractVector{R}) where R
    N = Int(sqrt(length(Ũ⃗) ÷ 2))
    Ũ = Matrix{R}(undef, 2N, 2N)
    U_real = Matrix{R}(undef, N, N)
    U_imag = Matrix{R}(undef, N, N)
    for i=0:N-1
        U_real[:, i+1] .= @view(Ũ⃗[i*2N .+ (1:N)])
        U_imag[:, i+1] .= @view(Ũ⃗[i*2N .+ (N+1:2N)])
    end
    Ũ[1:N, 1:N] .= U_real
    Ũ[1:N, (N + 1):end] .= -U_imag
    Ũ[(N + 1):end, 1:N] .= U_imag
    Ũ[(N + 1):end, (N + 1):end] .= U_real
    return Ũ
end

@doc raw"""
    operator_to_iso_vec(U::AbstractMatrix{<:Complex})

Convert a complex matrix `U` representing an operator into a real vector.

Must be differentiable.
"""
function operator_to_iso_vec(U::AbstractMatrix{R}) where R
    N = size(U,1)
    Ũ⃗ = Vector{real(R)}(undef, N^2 * 2)
    for i=0:N-1
        Ũ⃗[i*2N .+ (1:N)] .= real(@view(U[:, i+1]))
        Ũ⃗[i*2N .+ (N+1:2N)] .= imag(@view(U[:, i+1]))
    end
    return Ũ⃗
end

@doc raw"""
    iso_operator_to_iso_vec(Ũ::AbstractMatrix)

Convert a real matrix `Ũ` representing an isomorphism operator into a real vector.

Must be differentiable.
"""
function iso_operator_to_iso_vec(Ũ::AbstractMatrix{R}) where R
    N = size(Ũ, 1) ÷ 2
    Ũ⃗ = Vector{R}(undef, N^2 * 2)
    for i=0:N-1
        Ũ⃗[i*2N .+ (1:2N)] .= @view Ũ[:, i+1]
    end
    return Ũ⃗
end

iso_operator_to_operator(Ũ) = iso_vec_to_operator(iso_operator_to_iso_vec(Ũ))

operator_to_iso_operator(U) = iso_vec_to_iso_operator(operator_to_iso_vec(U))

# ----------------------------------------------------------------------------- #
# Open systems
# ----------------------------------------------------------------------------- #

function ad_vec(H::AbstractMatrix{<:Number}; anti::Bool=false)
    Id = sparse(eltype(H), I, size(H)...)
    return Id ⊗ H - (-1)^anti * conj(H)' ⊗ Id
end

# ----------------------------------------------------------------------------- #
# Hamiltonians
# ----------------------------------------------------------------------------- #

const Im2 = [
    0 -1;
    1  0
]

@doc raw"""
    G(H::AbstractMatrix)::Matrix{Float64}

Returns the isomorphism of ``-iH``:

```math
G(H) = \widetilde{- i H} = \mqty(1 & 0 \\ 0 & 1) \otimes \Im(H) - \mqty(0 & -1 \\ 1 & 0) \otimes \Re(H)
```

where ``\Im(H)`` and ``\Re(H)`` are the imaginary and real parts of ``H`` and the tilde indicates the standard isomorphism of a complex valued matrix:

```math
\widetilde{H} = \mqty(1 & 0 \\ 0 & 1) \otimes \Re(H) + \mqty(0 & -1 \\ 1 & 0) \otimes \Im(H)
```
"""
G(H::AbstractMatrix{<:Number}) = kron(I(2), imag(H)) - kron(Im2, real(H))

iso(H::AbstractMatrix{<:Number}) = kron(I(2), real(H)) + kron(Im2, imag(H))


"""
    H(G::AbstractMatrix{<:Number})::Matrix{ComplexF64}

Returns the inverse of `G(H) = iso(-iH)`, i.e. returns H

"""
function H(G::AbstractMatrix{<:Number})
    dim = size(G, 1) ÷ 2
    H_imag = G[1:dim, 1:dim]
    H_real = -G[dim+1:end, 1:dim]
    return H_real + 1.0im * H_imag
end

"""
    iso_dm(ρ::AbstractMatrix)

returns the isomorphism `ρ⃗̃ = ket_to_iso(vec(ρ))` of a density matrix `ρ`
"""
iso_dm(ρ::AbstractMatrix) = ket_to_iso(vec(ρ))



# =========================================================================== #

@testitem "Test isomorphism utilities" begin
    using LinearAlgebra
    iso_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test mat([1.0, 2.0, 3.0, 4.0]) == [1.0 3.0; 2.0 4.0]
    @test ket_to_iso([1.0, 2.0]) == [1.0, 2.0, 0.0, 0.0]
    @test iso_to_ket([1.0, 2.0, 0.0, 0.0]) == [1.0, 2.0]
    @test iso_vec_to_operator(iso_vec) == [1.0 0.0; 0.0 1.0]
    @test iso_vec_to_iso_operator(iso_vec) == [1.0 0.0 -0.0 -0.0; 0.0 1.0 -0.0 -0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    @test operator_to_iso_vec(Complex[1.0 0.0; 0.0 1.0]) == iso_vec
    @test iso_operator_to_iso_vec(iso_vec_to_iso_operator(iso_vec)) == iso_vec
end

end
