using OrdinaryDiffEq
using Trixi


# Vgl. elixir_euler_free_stream_curved
###############################################################################
CFL = 0.4           # 2
tspan = (0.0, 0.036)  # 0.4 split form läuft nur bis 0.036 mit cfl=0.4, standard DGSEM läuft 

# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

surface_flux = flux_lax_friedrichs
# basis = LobattoLegendreBasis(3)
# volume_flux  = flux_chandrashekar
# volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
# solver = DGSEM(basis, surface_flux, volume_integral)
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# mapping as described in the worksheet
function mapping(xi_, eta_)

    xi = xi_ 
    eta = eta_
  
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3/2) * pi * eta)
    y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)

    return SVector(x, y)
  end

cells_per_dimension = (16, 16)

mesh = CurvedMesh(cells_per_dimension, mapping)#(-1.0, -1.0), (1.0, 1.0))#, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.


ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
