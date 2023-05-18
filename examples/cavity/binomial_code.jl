using QuantumCollocation
using NamedTrajectories

transmon_levels = 3
cavity_levels = 5

system = MultiModeSystem(transmon_levels, cavity_levels)

g0 = multimode_state("g0", transmon_levels, cavity_levels)
g1 = multimode_state("g1", transmon_levels, cavity_levels)
g4 = multimode_state("g4", transmon_levels, cavity_levels)

ψ_init = [g0, g1]
