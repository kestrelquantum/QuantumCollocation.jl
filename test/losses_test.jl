"""
Tests: Losses submodule
"""

# @testset "Losses" begin

# end

# @testset "Isovec Unitary Infidelity" begin

# end


        # ∇²l = Ũ⃗ -> begin
        #     ∇²l = unitary_infidelity_hessian(Ũ⃗, Ũ⃗_goal)
        #     ∇²l_forwarddiff = ForwardDiff.hessian(l, Ũ⃗)
        #     ∇²l_forwarddiff_jacobian = ForwardDiff.jacobian(∇l, Ũ⃗)

        #     check = ∇²l .≈ ∇²l_forwarddiff
        #     for i ∈ axes(check, 1), j ∈ axes(check, 2)
        #         if !check[i, j]
        #             println("∇²l[$i, $j]    = $(∇²l[i, j])")
        #             println("∇²l_fd[$i, $j] = $(∇²l_forwarddiff[i, j])")
        #         end
        #     end

        #     jacobian_check = ∇²l_forwarddiff_jacobian .≈ ∇²l_forwarddiff
        #     for i ∈ axes(jacobian_check, 1), j ∈ axes(jacobian_check, 2)
        #         if !jacobian_check[i, j]
        #             println("∇²l_fd[$i, $j] = $(∇²l_forwarddiff[i, j])")
        #             println("∇²l_fd_j[$i, $j] = $(∇²l_forwarddiff_jacobian[i, j])")
        #         end
        #     end

        #     @assert all(check)
        #     return unitary_infidelity_hessian(Ũ⃗, Ũ⃗_goal)
        # end
