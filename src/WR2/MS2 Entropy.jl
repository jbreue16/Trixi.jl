
using OrdinaryDiffEq
using Trixi


# flux_central for conservation and lax friedrichs for stability ? 
# -> central flux doesnt even work for standard DGSEM

##########  EINSTELLUNGEN     #########
# Entropy conservation test with central surface flux and other initial conditions since it crashes with the blast wave
# initial_condition =  initial_condition_convergence_test
# surface_flux = flux_central

initial_condition = initial_condition_weak_blast_wave
CFL = 0.8
tspan = (0.0, 10)
N = 3
cells_per_dimension = (16, 16)

# CHandrashekar DGSEM Entropy STability
surface_flux = flux_chandrashekar # FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) #    
volume_flux  = flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(N)
solver = DGSEM(basis, surface_flux, volume_integral)

# Standard DGSEM Entropy STability
# polydeg = N
# surface_flux = flux_lax_friedrichs
# volume_integral = VolumeIntegralWeakForm()
# solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = volume_integral )

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)

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

stepsize_callback = StepsizeCallback(cfl=CFL)
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                                 extra_analysis_errors=(:conservation_error,),
                                                 extra_analysis_integrals=(entropy,)# energy_total,
                                                                           #energy_kinetic, energy_internal)
                                                
                                                )

callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

