
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# copied from EUlerEquation2D to do modifications 
function WR2_initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)
    c = 2
    A = 0.1
    L = 2
    f = 1/L
    ω = 2 * pi * f
    ini = c + A * sin(ω * (x[1] + x[2] - t))
  
    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2
  
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end
initial_condition = WR2_initial_condition_convergence_test


#coordinates neccessary for periodic solution with periodic boundaries
coordinates_min = (0.0, 0.0)
coordinates_max = ( 2.0,  2.0)
 # surface flux is actually the rusanov/local lax friedrichs flux
solver = DGSEM(polydeg = 3, surface_flux=flux_lax_friedrichs)


cells_per_dimension = (4, 4)

mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
            
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
            
            # semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
            #                                     source_terms=source_terms_convergence_test)
            
            
            ###############################################################################
# ODE solvers, callbacks etc.
            
# timespan for periodic solution
 tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)
# summary_callback = SummaryCallback()
analysis_interval = 1000
            
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                      extra_analysis_errors=(:conservation_error,),
                                     #  extra_analysis_integrals=(entropy, energy_total,
                                     #                            energy_kinetic, energy_internal)
                                     
                                     )           
# alive_callback = AliveCallback(analysis_interval=analysis_interval)
# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      solution_variables=cons2prim)
            
stepsize_callback = StepsizeCallback(cfl=0.5)
            
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
 # summary_callback() # print the timer summary
            


# built in convergence analysis doesnt work yet
# default_example() = joinpath(examples_dir(), "2d", "elixir_advection_basic.jl")
# convergence_test(default_example(), 4)

#   "c:\\Users\\jmbr2\\git\\TrixiFork\\Trixi.jl\\src\\WR2\\milestone1.jl"
# convergence_test("c:\\Users\\jmbr2\\git\\TrixiFork\\Trixi.jl\\src\\WR2\\milestone1.jl", 4)

