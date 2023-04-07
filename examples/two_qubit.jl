using QuantumCollocation
using NamedTrajectories
using LinearAlgebra

max_iter = 5000
linear_solver = "pardiso"


ω1 =  2π * 3.5 #GHz
ω2 = 2π * 3.9 #GHz
J = 0.1 * 2π
alpha = -2π * 0.225
levels = 2


U_goal = [
    1. 0. 0. 0.;
    0. 1. 0. 0.;
    0. 0. 0. 1.;
    0. 0. 1. 0.
]


H_drift = ω1*kron(number(levels), I(levels)) + ω2*kron(I(levels), number(levels)) 
          + 1/2*alpha * kron(quad(levels), I(levels)) + 1/2*alpha*kron(I(levels), quad(levels)) 
          + J*kron(create(levels) + annihilate(levels), create(levels) + annihilate(levels))

H_drive = [kron(create(levels) + annihilate(levels), I(levels)), 
           kron(I(levels), create(levels) + annihilate(levels)),
           kron(I(levels), number(levels))]

