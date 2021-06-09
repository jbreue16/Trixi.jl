
using OrdinaryDiffEq
using Trixi
using Plots


initial_condition = initial_condition_constant


surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) # flux_lax_friedrichs #  
volume_flux  = flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(4)
solver = DGSEM(basis, surface_flux, volume_integral)


CFL = 0.5
tspan = (0.0, 1)
###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)


coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
cells_per_dimension = (16, 16)

# mapping O-mesh
function mapping(xi_, eta_)

    ξ = xi_ 
    η = eta_

    x = 100 * (2 + ξ ) * cos(π * (η + 1))
    y = 100 * (2 + ξ ) * sin(π * (η + 1))

    return SVector(x, y)
end


function boundary_condition_constant_farfield(u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerEquations2D)
    # Far Field Conditions
    u_boundary = SVector(1, 0.1, -0.2, 10)
  
    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
  
    return flux
end
  
function boundary_condition_freeslip(u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerEquations2D)
    # Freeslip wall Conditions
    rho, v1, v2, p = cons2prim(u_inner, equations)
    u_boundary = prim2cons(SVector(rho, -v1, -v2, p), equations)
  
    # Calculate boundary flux
    if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
  
    return flux
end
        

mesh = CurvedMesh(cells_per_dimension, mapping,periodicity=(false,true))

#mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity=(false, true))

# Implement boundary conditions
# x neg and x_pos are not periodic 
# boundary_conditions = boundary_condition_periodic
boundary_conditions = (x_neg=boundary_condition_farfield,
                       x_pos=boundary_condition_farfield,
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,boundary_conditions=boundary_conditions)


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

pd=PlotData2D(sol)
b=plot(pd["rho"])
plot!(getmesh(pd))

# plot(sol)
# savefig(plotd, "test.png")