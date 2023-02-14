module QuantumSystems

export AbstractSystem
export QuantumSystem
export MultiModeSystem

using ..QuantumUtils

using LinearAlgebra

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
function G(H::AbstractMatrix{<:Number})
    println("howdy!!!!!")
    # return I(2) ⊗ imag(H) - Im2 ⊗ real(H)
    H_real = real(H)
    H_real_iso = Im2 ⊗ H_real
    return H, Im2 ⊗ I(2), H_real, H_real_iso, Im2 ⊗ H_real, I(2) ⊗ imag(H) - Im2 ⊗ real(H)
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
struct QuantumSystem <: AbstractSystem
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
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
    H_drift::Matrix{T},
    H_drives::Vector{Matrix{T}};
    params=Dict{Symbol, Any}(),
    kwargs...
) where T <: Number
    println("howdy!!!")
    G_drift = G(H_drift)
    G_drives = G.(H_drives)
    params = merge(params, Dict(kwargs...))
    # return QuantumSystem(G_drift, G_drives, params)
    return H_drift, G_drift, H_drives, G_drives
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

    if transmon_levels == 2

        transmon_g = [1, 0]
        transmon_e = [0, 1]

        H_drift =
            2χ * kron(
                transmon_e * transmon_e',
                number(cavity_levels)
            ) + κ / 2 * kron(
                I(transmon_levels),
                quad(cavity_levels)
            )

        if lab_frame
            H_drift += ωq * kron(
                number(transmon_levels),
                I(cavity_levels)
            ) + ωc * kron(
                I(transmon_levels),
                number(cavity_levels)
            )
        end


        if lab_frame
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
                im * (create(transmon_levels) -
                    annihilate(transmon_levels)),
                I(cavity_levels)
            )

            H_drive_cavity_R = kron(
                I(transmon_levels),
                create(cavity_levels) + annihilate(cavity_levels)
            )

            H_drive_cavity_I = kron(
                I(transmon_levels),
                im * (create(cavity_levels) -
                    annihilate(cavity_levels))
            )

            H_drives = [
                H_drive_transmon_R,
                H_drive_transmon_I,
                H_drive_cavity_R,
                H_drive_cavity_I
            ]
        end

    elseif transmon_levels == 3

        transmon_g = [1, 0, 0]
        transmon_e = [0, 1, 0]
        transmon_f = [0, 0, 1]

        H_drift =
            α / 2 * kron(
                quad(transmon_levels),
                I(cavity_levels)
            ) +
            2χ * kron(
                transmon_e * transmon_e',
                number(cavity_levels)
            ) +
            2χGF * kron(
                transmon_f * transmon_f',
                number(cavity_levels)
            ) +
            κ / 2 * kron(
                I(transmon_levels),
                quad(cavity_levels)
            )

        if lab_frame
            H_drift += ωq * kron(
                number(transmon_levels),
                I(cavity_levels)
            ) + ωc * kron(
                I(transmon_levels),
                number(cavity_levels)
            )
        end

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
            println("howdy!")
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
    end

    return QuantumSystem(
        H_drift,
        H_drives;
        params=params
    )
end






end
