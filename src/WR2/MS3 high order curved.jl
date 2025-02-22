
using OrdinaryDiffEq
using Trixi
using Printf
using PrettyTables

######### EINSTELLUNGEN ############
N_vec = [3, 4]
Nq_vec = [2, 4, 8, 16, 32]
CFL = 0.8           
# timespan for periodic solution
tspan = (0.0, 2.0)
latex = false

 
surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) #flux_lax_friedrichs
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
# volume_integral = Trixi.VolumeIntegralPseudoStrongForm() 

# mapping as described in the worksheet
function mapping(xi_, eta_)

    xi = xi_ 
    eta = eta_
  
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3/2) * pi * eta)
    y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)

    return SVector(x, y)
  end

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# copied from EUlerEquation2D to do modifications 
function WR2_initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)
    c = 2
    A = 0.1
    L = 2
    f = 1 / L
    ω = 2 * pi * f
    ini = c + A * sin(ω * (x[1] + x[2] - t))
  
    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2
  
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = WR2_initial_condition_convergence_test


# coordinates neccessary for periodic solution with periodic boundaries
coordinates_min = (0.0, 0.0)
coordinates_max = (2.0,  2.0)

####
    

    # Table preparation
kwargs = Dict{Symbol,Any}(
        :formatters => (
            ft_printf("%.5e", [2]),
            ft_printf("%.2f", [3])
        )
    )

if latex
    kwargs[:backend] = :latex
    kwargs[:highlighters] = LatexHighlighter(
            (data, i, j) -> j > 1,
            (data, i, j, str) -> "\$\\num{$str}\$"
        )
end

error = Vector{Float64}(undef, length(Nq_vec))
for N in N_vec
      
    basis = LobattoLegendreBasis(N)
    solver = DGSEM(basis, surface_flux, volume_integral)
    println(typeof(solver.volume_integral))
    # solver = DGSEM(polydeg=N, surface_flux=surface_flux, volume_integral=volume_integral)
    for j in eachindex(Nq_vec)
        Nq = Nq_vec[j]

        cells_per_dimension = (Nq, Nq)

        mesh = CurvedMesh(cells_per_dimension, mapping)#(-1.0, -1.0), (1.0, 1.0))
        # mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
            
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms=source_terms_convergence_test)
            
            
            ###############################################################################
            # ODE solvers, callbacks etc.
            
            
        ode = semidiscretize(semi, tspan)
            
        summary_callback = SummaryCallback()
        analysis_interval = 1000
            
        analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                                #  extra_analysis_errors=(:conservation_error,),
                                                #  extra_analysis_integrals=(entropy, energy_total,
                                                #                            energy_kinetic, energy_internal)
                                                
                                                )
            
            # alive_callback = AliveCallback(analysis_interval=analysis_interval)
            
            # save_solution = SaveSolutionCallback(interval=100,
            #                                      save_initial_solution=true,
            #                                      save_final_solution=true,
            #                                      solution_variables=cons2prim)
            
        stepsize_callback = StepsizeCallback(cfl= CFL)
            
        callbacks = CallbackSet(
                                    # summary_callback,
                                    analysis_callback,
                                    # alive_callback,
                                    # save_solution,
                                    stepsize_callback)
            
            ###############################################################################
            # run the simulation
            
        sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                        dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                        save_everystep=false, callback=callbacks);
        summary_callback() # print the timer summary
            

        error[j] = maximum(abs.(sol.u[1] - sol.u[2])) # maximum(abs.(exact - sol))
    end
      
        # Compute EOC
    eoc = vcat("", [log(error[j + 1] / error[j]) /
            log(Nq_vec[j] / Nq_vec[j + 1]) for j in 1:length(Nq_vec) - 1])
    println("N = $N")

    pretty_table(hcat(Nq_vec, error, eoc), ["Nq", "Error", "EOC"]; kwargs...)

    mean = sum(eoc[2:end]) / (length(Nq_vec) - 1)
    @printf("Mean EOC: %.2f\n", mean)
end
####





# built in convergence analysis doesnt work yet
# default_example() = joinpath(examples_dir(), "2d", "elixir_advection_basic.jl")
# convergence_test(default_example(), 4)

#   "c:\\Users\\jmbr2\\git\\TrixiFork\\Trixi.jl\\src\\WR2\\milestone1.jl"
# convergence_test("c:\\Users\\jmbr2\\git\\TrixiFork\\Trixi.jl\\src\\WR2\\milestone1.jl", 4)

