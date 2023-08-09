module IpoptOptions

export Options
export set!

using Ipopt
using Libdl
using Base: @kwdef

"""
    Solver options for Ipopt

    https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_print_options_documentation
"""
@kwdef mutable struct Options{T}
    tol::T = 1e-8
    s_max::T = 100.0
    max_iter::Int = 1_000
    max_cpu_time = 1_000_000.0
    dual_inf_tol::T = 1.0
    constr_viol_tol::T = 1.0e-6
    compl_inf_tol::T = 1.0e-3
    acceptable_tol::T = 1.0e-6
    acceptable_iter::Int = 15
    acceptable_dual_inf_tol::T = 1.0e10
    acceptable_constr_viol_tol::T = 1.0e-2
    acceptable_compl_inf_tol::T = 1.0e-2
    acceptable_obj_change_tol::T = 1.0e-5
    diverging_iterates_tol::T = 1.0e8
    mu_target::T = 1.0e-4
    print_level::Int = 5
    output_file = nothing
    print_user_options = "no"
    print_options_documentation = "no"
    print_timing_statistics = "no"
    print_options_mode = "text"
    print_advanced_options = "no"
    print_info_string = "no"
    inf_pr_output = "original"
    print_frequency_iter = 1
    print_frequency_time = 0.0
    skip_finalize_solution_call = "no"
    hsllib = nothing
    hessian_approximation = "exact"
    linear_solver = "mumps"
    watchdog_shortened_iter_trigger = 10
    watchdog_trial_iter_max = 3
end

function set!(optimizer::Ipopt.Optimizer, options::Options)
    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name == :linear_solver
            if value == "pardiso"
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
            end
        end
        if !isnothing(value)
           optimizer.options[String(name)] = value
        end
    end
end

end
