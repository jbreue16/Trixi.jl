using OrdinaryDiffEq
using Trixi
using Plots


# Vgl. elixir_euler_free_stream_curved
###############################################################################
CFL = 0.01          # 2
tspan = (0.0, 0.2)
N = 3
c = 16
mu = 0.001

# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4, viscous = true, mu = mu)

initial_condition = initial_condition_constant


surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# surface_flux = flux_lax_friedrichs
# volume_integral = Trixi.VolumeIntegralWeakForm()
# solver = DGSEM(polydeg=3, surface_flux=surface_flux, volume_integral = volume_integral )

# mapping as described in the worksheet
function mappingCos(xi_, eta_)

    xi = xi_ 
    eta = eta_
  
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3/2) * pi * eta)
    y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)

    return SVector(x, y)
end

function mappingOmesh(xi_, eta_)

    ξ = xi_ 
    η = eta_

    x = 2 * (2 + ξ ) * cos(π * (η + 1))
    y = 2 * (2 + ξ ) * sin(π * (η + 1))

    return SVector(x, y)
end

function mapping1zu1(xi_, eta_)

    x = xi_ 
    y = eta_

    return SVector(x, y)
end

cells_per_dimension = (c, c)

# mesh = CurvedMesh(cells_per_dimension, mappingCos, periodicity = true)
mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.


ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                      # extra_analysis_errors=(:conservation_error,),
                                      # extra_analysis_integrals=(entropy, energy_total,
                                      # energy_kinetic, energy_internal)
                                      )

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        #alive_callback,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
