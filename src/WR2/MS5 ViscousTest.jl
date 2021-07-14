using OrdinaryDiffEq
using Trixi
using Plots


######################Functions that need to be loaded ###########
# mapping O-mesh
function mapping(xi_, eta_)
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
function boundary_condition_stream(u_inner, q1_inner, q2_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::Trixi.AbstractEquations)

        u_boundary = SVector(1, 0.38, 0, 25)
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
    equations::Trixi.AbstractEquations)
    # Freeslip wall Conditions
    rho, rho_v1, rho_v2, rho_e = u_inner
    u_boundary = SVector(rho, -rho_v1, -rho_v2, rho_e)

    # Calculate boundary flux
    if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
    return flux
end
function boundary_condition_noslip_isothermal(u_inner, q1_inner, q2_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::Trixi.AbstractEquations)
    # Freeslip wall Conditions
    rho, rho_v1, rho_v2, rho_e = u_inner
    u_boundary = SVector(rho, -rho_v1, -rho_v2, rho_e)
    
    if typeof(equations) != Trixi.AuxiliaryEquation
        ρ = rho # unclear how to choose/compute.. 
        p = ρ # 𝑝 = 𝜌* 𝑅* T perfect gas model temperature relation with R = 1 universal gas constant
        vector_dummy = SVector(ρ, 0, 0, p)
        u_viscous = prim2cons(vector_dummy, equations)
        viscous = Trixi.viscous_flux(u_viscous, q1_inner, q2_inner, orientation, equations)
    end


    # Calculate boundary flux
    if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
    jojo = flux
    if typeof(equations) != Trixi.AuxiliaryEquation
    jojo += viscous
    end
    return jojo
end

########################    Setting    ######################################
CFL = 0.01
tspan = (0.0, 2)
N = 2
c = 32
mu = 0.001

function WR2_initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    rho_v1 = 0.1
    rho_v2 = 0
    rho_e = 25.0
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end

# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4, viscous = true, mu = mu)

initial_condition = WR2_initial_condition_constant
boundary_conditions = (x_neg=boundary_condition_noslip_isothermal,#boundary_condition_freeslip,
                       x_pos=boundary_condition_stream,
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic)
cells_per_dimension = (c, c)
mesh = CurvedMesh(cells_per_dimension, mapping, periodicity = (false, true))
#, periodicity=(false, true))#(-1.0, -1.0), (1.0, 1.0))#, mapping)
# mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))

surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,boundary_conditions=boundary_conditions)



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


pd=PlotData2D(sol)
b=plot(pd["rho"])
plot!(getmesh(pd))