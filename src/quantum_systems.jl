module QuantumSystems

export AbstractSystem
export QuantumSystem
export MultiModeSystem
export operator_algebra

export iso

using ..QuantumUtils

using LinearAlgebra
using SparseArrays

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
G(H::AbstractMatrix{<:Number}) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

iso(H::AbstractMatrix{<:Number}) = I(2) ⊗ real(H) + Im2 ⊗ imag(H)


"""
    H(G::AbstractMatrix{<:Number})::Matrix{ComplexF64}

Returns the inverse of `G(H) = iso(-iH)`, i.e. returns H

"""
function H(G::AbstractMatrix{<:Number})
    dim = size(G, 1) ÷ 2
    H_imag = G[1:dim, 1:dim]
    H_real = -G[dim+1:end, 1:dim]
    return H_real + 1im * H_imag
end

"""
```julia
AbstractSystem
```

Abstract type for defining systems.
"""
abstract type AbstractSystem end

# TODO: make subtypes: SingleQubitSystem, TwoQubitSystem, TransmonSystem, MultimodeSystem, etc.

"""
    QuantumSystemNew <: AbstractSystem

A struct for storing the isomorphisms of the system's drift and drive Hamiltonians,
as well as the system's parameters.
"""
struct QuantumSystem{R} <: AbstractSystem
    H_drift_real::Matrix{R}
    H_drift_imag::Matrix{R}
    H_drives_real::Vector{Matrix{R}}
    H_drives_imag::Vector{Matrix{R}}
    G_drift::Matrix{R}
    G_drives::Vector{Matrix{R}}
    params::Dict{Symbol, Any}
end

"""
    QuantumSystem(
        H_drift::Matrix{<:Number},
        H_drives::Vector{Matrix{<:Number}};
        params=Dict{Symbol, Any}(),
        kwargs...
    )::QuantumSystem

Constructs a `QuantumSystem` object from the drift and drive Hamiltonian terms.
"""
function QuantumSystem(
    H_drift::Matrix{<:Number},
    H_drives::Vector{<:Matrix{<:Number}};
    params=Dict{Symbol, Any}(),
    R::DataType=Float64,
    kwargs...
)
    H_drift_real = real(H_drift)
    H_drift_imag = imag(H_drift)
    H_drives_real = real.(H_drives)
    H_drives_imag = imag.(H_drives)
    G_drift = G(H_drift)
    G_drives = G.(H_drives)
    params = merge(params, Dict(kwargs...))
    return QuantumSystem{R}(
        H_drift_real,
        H_drift_imag,
        H_drives_real,
        H_drives_imag,
        G_drift,
        G_drives,
        params
    )
end

function QuantumSystem(H_drives::Vector{<:Matrix{<:Number}}; kwargs...)
    return QuantumSystem(zeros(eltype(H_drives[1]), size(H_drives[1])), H_drives; kwargs...)
end


@doc raw"""
    MultiModeSystem(
        transmon_levels::Int,
        cavity_levels::Int;
        χ=2π * -0.5459e-3,
        κ=2π * 4e-6,
        χGF=2π * -1.01540302914e-3,
        α=-2π * 0.143,
        n_cavities=1
    )::QuantumSystem

Create a new `QuantumSystemNew` object for a transmon qubit with `transmon_levels` levels
coupled to a single cavity with `cavity_levels` levels.

The Hamiltonian for this system is given by

```math
\hat H =
    \frac{\kappa}{2} \hat{a}^{ \dagger} \hat{a} \left( \hat{a}^{\dagger}\hat{a}-1\right) +
    2 \chi \dyad{e} \hat{a}^{\dagger}\hat{a} +
    \left(
        \epsilon_{c}(t) +
        \epsilon_{q}(t) +
        \mathrm{c.c.}
    \right)
```
"""
function MultiModeSystem(
    transmon_levels::Int,
    cavity_levels::Int;
    lab_frame=false,
    ωq=2π * 4.9613896,
    ωc=2π * 6.2230641,
    χ=2π * -0.5459e-3,
    κ=2π * 4e-6,
    χGF=2π * -1.01540302914e-3,
    α=-2π * 0.143,
    α_ef = -143.277e-3 * 2π,
    α_fh = -162.530e-3 * 2π,
    χ₂  = -0.63429e-3 * 2π,
    χ₂_gf = -1.12885e-3 * 2π,
    χ₂_gh = -1.58878e-3 * 2π,
    χ₃ = -0.54636e-3 * 2π,
    χ₃_gf = -1.017276e-3 * 2π,
    χ₃_gh = -1.39180e-3 * 2π,
    κ₂ = 5.23e-6 * 2π,
    κ₃ = 4.19e-6 * 2π,
    κ_cross = 3.6e-6 * 2π,
    n_cavities=1 # TODO: add functionality for multiple cavities
)
    @assert transmon_levels ∈ 2:4
    @assert n_cavities ∈ 1:2

    params = Dict(
        :lab_frame => lab_frame,
        :ωq => ωq,
        :ωc => ωc,
        :χ => χ,
        :κ => κ,
        :χGF => χGF,
        :α => α,
        :transmon_levels => transmon_levels,
        :cavity_levels => cavity_levels,
        :n_cavities => n_cavities,
    )

    transmon_g = [1; zeros(transmon_levels - 1)]
    transmon_e = [0; 1; zeros(transmon_levels - 2)]

    A = Diagonal([0., 0., α_ef, α_ef + α_fh])

    # --------------------------------------------------
    # Drift Hamiltonian Term
    # --------------------------------------------------

    if transmon_levels == 2

        H_drift =
            2*χ₃ * kron(transmon_e * transmon_e', number(cavity_levels)) +
            κ₃/2 * kron(I(transmon_levels), quad(cavity_levels))

        # H_drift =
        #     2χ * kron(
        #         transmon_e * transmon_e',
        #         number(cavity_levels)
        #     ) + κ / 2 * kron(
        #         I(transmon_levels),
        #         quad(cavity_levels)
        #     )

        if lab_frame
            H_drift += ωq * kron(
                number(transmon_levels),
                I(cavity_levels)
            ) + ωc * kron(
                I(transmon_levels),
                number(cavity_levels)
            )
        end
    elseif transmon_levels == 3

        transmon_f = [0, 0, 1]

        H_drift =
            2*χ₃ * kron(transmon_e*transmon_e', number(cavity_levels)) +
            2*χ₃_gf * kron(transmon_f*transmon_f', number(cavity_levels)) +
            κ₃/2 * kron(I(transmon_levels), quad(cavity_levels)) +
            kron(A[1:transmon_levels, 1:transmon_levels], I(cavity_levels))



        # H_drift =
        #     α / 2 * kron(
        #         quad(transmon_levels),
        #         I(cavity_levels)
        #     ) +
        #     2χ * kron(
        #         transmon_e * transmon_e',
        #         number(cavity_levels)
        #     ) +
        #     2χGF * kron(
        #         transmon_f * transmon_f',
        #         number(cavity_levels)
        #     ) +
        #     κ / 2 * kron(
        #         I(transmon_levels),
        #         quad(cavity_levels)
        #     )

        if lab_frame
            H_drift += ωq * kron(
                number(transmon_levels),
                I(cavity_levels)
            ) + ωc * kron(
                I(transmon_levels),
                number(cavity_levels)
            )
        end
    elseif transmon_levels == 4

        transmon_f = [0, 0, 1, 0]
        transmon_h = [0, 0, 0, 1]

        H_drift =
            2*χ₃ * kron(transmon_e * transmon_e', number(cavity_levels)) +
            2*χ₃_gf * kron(transmon_f * transmon_f', number(cavity_levels)) +
            2*χ₃_gh * kron(transmon_h * transmon_h', number(cavity_levels)) +
            κ₃/2 * kron(I(transmon_levels), quad(cavity_levels)) +
            κ_cross* kron(I(transmon_levels), number(cavity_levels)) +
            kron(A[1:transmon_levels, 1:transmon_levels], I(cavity_levels))
    end

    # --------------------------------------------------
    # Drive Hamiltonian Terms
    # --------------------------------------------------

    if lab_frame
        H_drift += ωq * kron(
            number(transmon_levels),
            I(cavity_levels)
        ) +
        ωc * kron(
            I(transmon_levels),
            number(cavity_levels)
        )

        H_drive_transmon = kron(
            create(transmon_levels) + annihilate(transmon_levels),
            I(cavity_levels)
        )

        H_drive_cavity = kron(
            I(transmon_levels),
            create(cavity_levels) + annihilate(cavity_levels)
        )

        H_drives = [
            H_drive_transmon,
            H_drive_cavity,
        ]
    else
        H_drive_transmon_R = kron(
            create(transmon_levels) + annihilate(transmon_levels),
            I(cavity_levels)
        )

        H_drive_transmon_I = kron(
            1im * (annihilate(transmon_levels) - create(transmon_levels)),
            I(cavity_levels)
        )

        H_drive_cavity_R = kron(
            I(transmon_levels),
            create(cavity_levels) + annihilate(cavity_levels)
        )

        H_drive_cavity_I = kron(
            I(transmon_levels),
            1im * (annihilate(cavity_levels) - create(cavity_levels))
        )

        H_drives = [
            H_drive_transmon_R,
            H_drive_transmon_I,
            H_drive_cavity_R,
            H_drive_cavity_I
        ]
    end

    return QuantumSystem(
        H_drift,
        H_drives;
        params=params
    )
end


commutator(A::AbstractMatrix, B::AbstractMatrix) = A * B - B * A

is_hermitian(H::AbstractMatrix; atol=eps(Float32)) =
    all(isapprox.(H - H', 0.0, atol=atol))

function is_linearly_dependent(
    basis::Vector{<:AbstractMatrix},
    op::AbstractMatrix; 
    eps=eps(Float32)
)        
    # Note: basis is assumed to be linearly independent
    M = hcat(vec.(basis)..., vec(op))
    return is_linearly_dependent(M, eps=eps)
end

function is_linearly_dependent(M::AbstractMatrix; eps=eps(Float32))
    if size(M, 2) > size(M, 1)
        println("Linearly dependent because columns > rows.")
        return true
    end
    # QR decomposition has a zero R on diagonal if linearly dependent
    val = minimum(abs.(diag(qr(M).R)))
    return isapprox(val, 0.0, atol=eps)
end

function operator_algebra(
    generators::Vector{<:AbstractMatrix}; return_layers=false
)
    """
    operator_algebra(generators; return_layers=false)

        Compute the Lie algebra basis for the given generators.
        If return_layers is true, the Lie tree layers are also returned.
    """
    basis = copy(generators)
    current_layer = copy(generators)
    if return_layers
        all_layers = Vector{Matrix{ComplexF64}}[copy(generators)]
    end

    if is_linearly_dependent(stack(vec.(basis)))
        println("Linearly dependent generators.")
    else
        # Note: Use left normalized commutators
        # Note: Jacobi identity is not checked
        ℓ = 1    
        while length(basis) < size(first(generators), 1)^2 - 1
            println("ℓ = $ℓ")
            layer = Matrix{ComplexF64}[]
            # Repeat commutators until no new operators are found.
            for op ∈ current_layer
                for gen ∈ generators
                    test = commutator(gen, op)
                    if all(test .≈ 0) || is_linearly_dependent(basis, test)
                        continue
                    else
                        # Store as Hermitian operator
                        test = is_hermitian(test) ? test : im * test
                        push!(layer, test)
                        push!(basis, test)
                    end
                end
            end

            if isempty(layer)
                println("Subspace termination.")
                break
            else
                current_layer = layer
                ℓ += 1
            end

            if return_layers
                append!(all_layers, [current_layer])
            end
        end
    end

    if return_layers
        return basis, all_layers
    else
        return basis
    end
end






end
