
using OrdinaryDiffEq
using Trixi

##########  EINSTELLUNGEN     #########
# BLZ: we adjust the CFL and the terminal time to test the different methods for stability
# maximal setting for N = 3, Nq = 16: Chandrashekar without dissipation, CFL = 2.37, t = 100

initial_condition = initial_condition_weak_blast_wave
N = 3
cells_per_dimension = (16, 16)

CFL = 2.37
tspan = (0.0, 100)

# Standard DGSEM Entropy STability
# polydeg = N
# surface_flux = flux_lax_friedrichs # flux_chandrashekar # flux_central # 
# volume_integral = VolumeIntegralWeakForm()
# solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = volume_integral )

# CHandrashekar DGSEM Entropy STability
surface_flux = flux_chandrashekar # FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) #   
volume_flux  = flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(N)
solver = DGSEM(basis, surface_flux, volume_integral)


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

