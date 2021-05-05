
using OrdinaryDiffEq
using Trixi


using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(3)
solver = DGSEM(basis, surface_flux, volume_integral)
tspan = (0.0, 0.4)

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
cells_per_dimension = (16, 16)
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.


ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.9)

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                                 extra_analysis_errors=(:conservation_error,),
                                                 extra_analysis_integrals=(entropy, energy_total,
                                                                           energy_kinetic, energy_internal)
                                                
                                                )

callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary



# ###############################################################################
# # semidiscretization of the compressible Euler equations
# equations = CompressibleEulerEquations2D(1.4)

# initial_condition = initial_condition_weak_blast_wave

# volume_flux = flux_chandrashekar
# solver = DGSEM(polydeg=3, surface_flux=flux_chandrashekar,
#                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# coordinates_min = (-2.0, -2.0)
# coordinates_max = ( 2.0,  2.0)
# cells_per_dimension = (16, 16)

# mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


# ###############################################################################
# # ODE solvers, callbacks etc.

# tspan = (0.0, 2)
# ode = semidiscretize(semi, tspan)

# summary_callback = SummaryCallback()

# analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
#                                                  extra_analysis_errors=(:conservation_error,),
#                                                  extra_analysis_integrals=(entropy, energy_total,
#                                                                            energy_kinetic, energy_internal)
                                                
#                                                 )

# alive_callback = AliveCallback(analysis_interval=analysis_interval)

# save_restart = SaveRestartCallback(interval=100,
#                                    save_final_restart=true)

# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      solution_variables=cons2prim)

# stepsize_callback = StepsizeCallback(cfl=0.8)

# callbacks = CallbackSet(summary_callback,
#                         analysis_callback, alive_callback,
#                         save_restart, save_solution,
#                         stepsize_callback)


# ###############################################################################
# # run the simulation

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks);
# summary_callback() # print the timer summary
