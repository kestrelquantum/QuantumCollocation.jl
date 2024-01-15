module QuantumSystemTemplates

export TransmonSystem
export TransmonDipoleCoupling

using ..QuantumUtils
using ..QuantumSystems

@doc raw"""
    TransmonSystem(;
        ω::Float64=4.4153,  # GHz
        δ::Float64=0.17215, # GHz
        levels::Int=3,
        lab_frame::Bool=false,
        frame_ω::Float64=ω,
    ) -> QuantumSystem

Returns a `QuantumSystem` object for a transmon qubit, with the Hamiltonian

```math
H = \omega a^\dagger a + \frac{\delta}{2} a^\dagger a^\dagger a a
```

where `a` is the annihilation operator.

# Keyword Arguments
- `ω`: The frequency of the transmon, in GHz.
- `δ`: The anharmonicity of the transmon, in GHz.
- `levels`: The number of levels in the transmon.
- `lab_frame`: Whether to use the lab frame Hamiltonian, or an ω-rotating frame.
- `frame_ω`: The frequency of the rotating frame, in GHz.
- `mutiply_by_2π`: Whether to multiply the Hamiltonian by 2π, set to true by default because the frequency is in GHz.

"""
function TransmonSystem(;
    ω::Float64=4.0,  # GHz
    δ::Float64=-0.2, # GHz
    levels::Int=3,
    lab_frame::Bool=false,
    frame_ω::Float64=lab_frame ? 0.0 : ω,
    mutiply_by_2π::Bool=true,
)
    if lab_frame
        frame_ω = 0.0
    end

    if !(frame_ω ≈ 0.0)
        lab_frame = false
    end

    a = annihilate(levels)

    if lab_frame
        # TODO: add functionality for hamiltonian with cosine
        H_drift = ω * a' * a + δ / 12 * (a + a')^4
    else
        H_drift = (ω - frame_ω) * a' * a + δ / 2 * a' * a' * a * a
    end

    H_drives = [a + a', 1.0im * (a - a')]

    if mutiply_by_2π
        H_drift *= 2π
        H_drives .*= 2π
    end

    params = Dict{Symbol, Any}(
        :ω => ω,
        :δ => δ,
        :levels => levels,
        :lab_frame => lab_frame,
        :frame_ω => frame_ω,
        :mutiply_by_2π => mutiply_by_2π,
    )

    return QuantumSystem(
        H_drift,
        H_drives;
        constructor=TransmonSystem,
        params=params,
    )
end

@doc raw"""
    TransmonDipoleCoupling(
        g_ij::Float64,
        pair::Tuple{Int, Int},
        subsystem_levels::Vector{Int};
        lab_frame::Bool=false,
    ) -> QuantumSystemCoupling

    TransmonDipoleCoupling(
        g_ij::Float64,
        pair::Tuple{Int, Int},
        sub_systems::Vector{QuantumSystem};
        kwargs...
    ) -> QuantumSystemCoupling

Returns a `QuantumSystemCoupling` object for a transmon qubit. In the lab frame, the Hamiltonian coupling term is

```math
H = g_{ij} (a_i + a_i^\dagger) (a_j + a_j^\dagger)
```

In the rotating frame, the Hamiltonian coupling term is

```math
H = g_{ij} (a_i a_j^\dagger + a_i^\dagger a_j)
```

where `a_i` is the annihilation operator for the `i`th transmon.

"""
function TransmonDipoleCoupling end

function TransmonDipoleCoupling(
    g_ij::Float64,
    pair::Tuple{Int, Int},
    subsystem_levels::Vector{Int};
    lab_frame::Bool=false,
)
    i, j = pair
    a_i = lift(annihilate(subsystem_levels[i]), i, subsystem_levels)
    a_j = lift(annihilate(subsystem_levels[j]), j, subsystem_levels)

    if lab_frame
        op = g_ij * (a_i + a_i') * (a_j + a_j')
    else
        op = g_ij * (a_i * a_j' + a_i' * a_j)
    end

    params = Dict{Symbol, Any}(
        :lab_frame => lab_frame,
    )

    return QuantumSystemCoupling(
        op,
        g_ij,
        pair,
        subsystem_levels,
        TransmonDipoleCoupling,
        params
    )
end

function TransmonDipoleCoupling(
    g_ij::Float64,
    pair::Tuple{Int, Int},
    sub_systems::Vector{QuantumSystem};
    kwargs...
)
    subsystem_levels = [sys.levels for sys ∈ sub_systems]
    return TransmonDipoleCoupling(g_ij, pair, subsystem_levels; kwargs...)
end

end
