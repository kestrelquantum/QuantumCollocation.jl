"""
    Testing problems

    - test problem creation
    - test problem solving
    - test problem saving
    - test problem loading
"""

@testset "Problems" begin
    # initializing test system
    T = 5
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    # test problem creation


end
