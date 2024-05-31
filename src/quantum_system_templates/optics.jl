@doc raw"""


Returns a `QuantumSystem` object for a quantum optics qubit, with the Hamiltonian
"""

using QuantumOptics


function OpticsSystem(;
    N_cutoff::Int=10,
    ωc::Float64=0.1,
    ωa::Float64=0.1.
    Ω::Float64=1.,
    ωd::Float64=0.1,
    ϵ::Float64=1.,  
    drives::Bool=true,
)
    # Bases
    b_fock = FockBasis(N_cutoff)
    b_spin = SpinBasis(1//2)
    b = b_fock ⊗ b_spin
    a = destroy(b_fock)
    at = create(b_fock)
    n = number(b_fock)

    # Operators
    sm = sigmam(b_spin)
    sp = sigmap(b_spin)
    sz = sigmaz(b_spin)

    # Hamiltonian
    Hatom = ωa*sz/2
    Hfield = ωc*n
    Hint = Ω*(at⊗sm + a⊗sp)
    H_drift = one(b_fock)⊗Hatom + Hfield⊗one(b_spin) + Hint 

    if drives
            H_drives = ϵ * a' + ϵ * a;
    else
            H_drives = Matrix{ComplexF64}[]
    end

    params = Dict{Symbol, Any}(
        :N_cutoff => N_cutoff,
        :ωc => ωc,
        :ωa => ωa,
        :Ω => Ω,
        :ωd => ωd,
        :ϵ  => ϵ,
    )

    return QuantumSystem(
        H_drift,
        H_drives;
        constructor=OpticSystem,
        params=params,
    )
end
