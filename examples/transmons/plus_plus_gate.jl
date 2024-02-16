using QuantumCollocation
using NamedTrajectories

levels = 4

syss = [
    TransmonSystem(levels=levels),
    TransmonSystem(levels=levels, ω=5.0, δ=0.3),
]

csys = CompositeQuantumSystem(syss)

op = EmbeddedOperator([:H, :H], csys, 1:2)

T = 100
dt = 0.1

prob = UnitarySmoothPulseProblem(csys, op, T, dt)

solve!(prob; max_iter=100)

F = unitary_fidelity(prob; subspace=op.subspace_indices)

println("F = ", F)

plot_path = joinpath(@__DIR__, "plus_plus_gate.png")

unitary_populations_plot(plot_path, prob; subspace=op.subspace_indices)
