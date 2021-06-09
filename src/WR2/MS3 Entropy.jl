
using OrdinaryDiffEq
using Trixi


# flux_central for conservation and lax friedrichs for stability ? 
# -> central flux doesnt even work for standard DGSEM

##########  EINSTELLUNGEN     #########
# for entropy conservation choose the central flux -> no surface dissipation.
#  also needs other initial conditions since central flux crashes for the blast wave
# surface_flux = flux_central
# initial_condition = initial_condition_convergence_test

initial_condition = initial_condition_convergence_test # initial_condition_weak_blast_wave

# Standard DGSEM Entropy STability
# polydeg = 3
# surface_flux = flux_lax_friedrichs
# volume_integral = VolumeIntegralWeakForm()
# solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = volume_integral )


# CHandrashekar DGSEM Entropy STability
# surface Flux can either be Lax Friedrichs or the chandrashekar flux with or without dissipation (Entropy Stability/Conservation) 
surface_flux = flux_chandrashekar # FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) # flux_lax_friedrichs #
volume_flux  = flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(4)
solver = DGSEM(basis, surface_flux, volume_integral)


CFL = 0.9
tspan = (0.0, 2)
###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)


coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
cells_per_dimension = (16, 16)
# mapping as described in the worksheet
function mapping(xi_, eta_)

    xi = xi_ 
    eta = eta_
  
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3/2) * pi * eta)
    y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)

    return SVector(x, y)
  end

  mesh = CurvedMesh(cells_per_dimension, mapping)


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

