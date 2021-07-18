using OrdinaryDiffEq
using Trixi
using Plots


########## functions that need to be loaded ##########################
function WR2_initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    rho_v1 = 1.5
    rho_v2 = 0
    rho_e = 25.0
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
     
function boundary_condition_noslip_isothermal(u_inner, q1_inner, q2_inner, normal::AbstractVector,
        direction, x, t, surface_flux_function, equations::Trixi.AbstractEquations)
        # Freeslip wall Conditions
    rho, rho_v1, rho_v2, rho_e = u_inner
    u_boundary = SVector(rho, -rho_v1, -rho_v2, rho_e)

    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, normal, equations)
    end
    
    if typeof(equations) == Trixi.CompressibleEulerEquations2D{Float64} && equations.viscous

        p_0 = Trixi.cons2prim([1.0, 1.5, 0.0, 25], equations)[4] # initial pressure with constant initial condition, v1 !
        ρ_0 = 1.0 # initial density
        T_0 = p_0/ρ_0 # initial temperature	
        γ = equations.gamma
        # computation of the exact pressure solution of the riemann problem
        rho, v1, v2, p = cons2prim([rho, rho_v1, rho_v2, rho_e], equations) # inner pressure
        c = sqrt(γ * (p/rho)) # sound speed
        v_norm = sqrt((normal[1]* v1)^2 + (normal[2] * v2)^2) # norm of velocity in normal direction
        Mach_n = v_norm / c # Mach number in normal direction
        hmpf = Mach_n*(γ+1)/4 # hmpf
        if v_norm > 0
            pstar_p = 1 + γ* Mach_n* (hmpf + sqrt(hmpf^2 + 1)) # exact pressure solution of the riemann problem
        else
            pstar_p = (1 + 0.5* (γ - 1)* Mach_n)^(2* γ/(γ - 1)) # exact pressure solution of the riemann problem
        end

        rho_viscous = pstar_p / T_0 # 𝑝 = 𝜌* 𝑅* T perfect gas model temperature relation with R = 1 universal gas constant
        vector_dummy = SVector(rho_viscous, 0, 0, pstar_p)
        u_viscous = prim2cons(vector_dummy, equations) # u_boundary for the viscous computation
        viscous = Trixi.viscous_flux(u_viscous, q1_inner, q2_inner, normal, equations)
        flux += viscous # add viscous part to boundary flux computation
    end
    return flux
end

function boundary_condition_stream(u_inner, q1, q2, orientation, direction, x, t,
        surface_flux_function,
        equations::Trixi.AbstractEquations)
        
    u_boundary = SVector(1, 1.5, 0, 25)
        # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
      
    return flux
end
function mapping1zu1(xi_, eta_)
    x = xi_
    y = eta_
    return SVector(x, y)
end
    # mapping O-mesh
function mapping(xi_, eta_)
    ξ = xi_ 
    η = eta_
    size = 2    # gebietsgröße * 3
    d = 1.5       # relativ zum durchmesser
    x = size * (d + ξ ) * cos(π * (η + 1))
    y = size * (d + ξ ) * sin(π * (η + 1))
    return SVector(x, y)
end

#########################   Einstellungen  ############################################
"Note : if you change the velocity, it needs to be changed in the boundaries as well !"
CFL = 0.8       
tspan = (0.0, 500)
N = 3
c = 32
cells_per_dimension = (32, 64)
mu = 0.0008 # Re = ∣∣v0∣∣ D ρ0/µ = 100 -> μ = 0.002
#visualization = VisualizationCallback(interval = 500, plot_creator=Trixi.save_plot)# clims=(-0.5,0.5),
# variable_names=["v1","v2"],

initial_condition = WR2_initial_condition_constant
boundary_conditions = (x_neg = boundary_condition_noslip_isothermal,# boundary_condition_freeslip,
                       x_pos = boundary_condition_stream,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

mesh = CurvedMesh(cells_per_dimension, mapping, periodicity=(false, true))
# , periodicity=(false, true))#(-1.0, -1.0), (1.0, 1.0))#, mapping)
# mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))

###############################################################

equations = CompressibleEulerEquations2D(1.4, viscous=true, mu=mu)

surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)


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

visualization = VisualizationCallback(interval=150, variable_names=["v1", "v2"], plot_creator=Trixi.save_plot)                                      

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        # alive_callback,
                        # save_solution,
                        visualization,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


pd = PlotData2D(sol)
b = plot(pd["rho"])
plot!(getmesh(pd))