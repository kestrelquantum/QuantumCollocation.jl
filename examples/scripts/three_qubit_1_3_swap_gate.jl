using Pico
using Revise
using LinearAlgebra

U_goal = [
    1 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0;
    0 0 1 0 0 0 0 0;
    0 0 0 0 0 0 1 0;
    0 1 0 0 0 0 0 0;
    0 0 0 0 0 1 0 0;
    0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 0 1
]

a = create(2)
a_dag = annihilate(2)

ωs = [5.18, 5.12, 5.06]
ω_d = 5.12

ξs = [0.01, 0.01, 0.01]

J_12 = 5.0e-3
J_23 = 5.0e-3


function lift(U, q, n; l=2)
    Is = Matrix{Number}[I(l) for i in 1:n]
    Is[q] = U
    return foldr(kron, Is)
end

lift(number(2), 1, 3)
lift(a_dag, 1, 3)
lift(a, 1, 3)

H_drift = sum(
    (ωs[q] - ω_d) * lift(a_dag, q, 3) * lift(a, q, 3) -
    ξs[q] / 2 * lift(a_dag, q, 3) * lift(a_dag, q, 3) * lift(a, q, 3) * lift(a, q, 3)
        for q = 1:3
)

# dispersive coupling
H_drift +=
    J_12 * (lift(a_dag, 1, 3) * lift(a, 2, 3) + lift(a, 1, 3) * lift(a_dag, 2, 3)) +
    J_23 * (lift(a_dag, 2, 3) * lift(a, 3, 3) + lift(a, 2, 3) * lift(a_dag, 3, 3))

H_drives_Re = [lift(a, j, 3) + lift(a_dag, j, 3) for j = 1:3]
H_drives_Im = [lift(a, j, 3) - lift(a_dag, j, 3) for j = 1:3]

H_drives = vcat(H_drives_Re, H_drives_Im)

system = QuantumSystem(H_drift, H_drives)
