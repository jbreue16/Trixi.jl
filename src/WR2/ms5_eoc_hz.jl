
using OrdinaryDiffEq
using Trixi
using Printf
using PrettyTables

######### EINSTELLUNGEN ############
N_vec = [2]
Nq_vec = [2, 4, 8] # , 16] # , 32]
CFL = 0.001         
# timespan for periodic solution
tspan = (0.0, 2.0)
latex = false


surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) # flux_lax_friedrichs
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

# mapping as described in the worksheet
function mapping(xi_, eta_)

    xi = xi_ 
    eta = eta_
  
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3 / 2) * pi * eta)
    y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)

    return SVector(x, y)
end

###############################################################################
# semidiscretization of the compressible Euler equations

# equations = CompressibleEulerEquations2D(1.4)
equations = CompressibleEulerEquations2D(1.4, viscous=true, mu=0.01)

# copied from EUlerEquation2D to do modifications 
# Source terms and initial conditions for EOC tests
function initial_condition_convergence_test_viscous(x, t, equations::Trixi.AbstractEquations)
    # μ = equations.mu
    # Pr = equations.Pr
    # κ = equations.gamma
    # k = 2 # k = c_p * μ / Pr
    # k = 114 * μ / Pr
    k = 0
    c = 4 
    ω = 0.5
    ini = sin(k * (x[1] + x[2]) - ω * t) + c

    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2

    return SVector(rho, rho_v1, rho_v2, rho_e)
end

@inline function source_terms_convergence_test_viscous(u, x, t, equations::Trixi.AbstractEquations)
    # Same settings as in `initial_condition`
    μ = equations.mu
    Pr = equations.Pr
    κ = equations.gamma
    # k = 2 # k = c_p * μ / Pr
    k = 114 * μ / Pr
    # k = 0
    c = 4 
    ω = 0.5

    x, y = x
    co = cos(k * (x + y) - ω * t)
    si = sin(2 * k * (x + y) - 2 * ω * t)
    tmp1 = 2 * k - ω
    tmp2 = - ω + 2 * k * c * κ - k * κ + 3 * ω - 2 * k * c
    tmp3 = k * κ - k
    tmp4 = 4 * k * c * κ + 2 * k - 2 * k * κ - 2 * ω * c
    tmp5 = 2 * k * κ - ω

    du1 = co * tmp1
    du2 = co * tmp2 + si * tmp3
    du3 = co * tmp2 + si * tmp3
    du4 = co * tmp4 + si * tmp5 + sin(k * (x + y) - ω * t) * (2 * k^2 * μ * κ / Pr)

    return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_convergence_test_viscous # initial_condition_convergence_test # 
source_terms = source_terms_convergence_test_viscous # source_terms_convergence_test #


# coordinates neccessary for periodic solution with periodic boundaries
coordinates_min = (0.0, 0.0)
coordinates_max = (2.0,  2.0)

####
    

    # Table preparation
kwargs = Dict{Symbol,Any}(
        :formatters => (
            ft_printf("%.5e", [2]),
            ft_printf("%.2f", [3])
        )
    )

if latex
    kwargs[:backend] = :latex
    kwargs[:highlighters] = LatexHighlighter(
            (data, i, j) -> j > 1,
            (data, i, j, str) -> "\$\\num{$str}\$"
        )
end

error = Vector{Float64}(undef, length(Nq_vec))
for N in N_vec
      
    basis = LobattoLegendreBasis(N)
    solver = DGSEM(basis, surface_flux, volume_integral)
    println(typeof(solver.volume_integral))
    # solver = DGSEM(polydeg=N, surface_flux=surface_flux, volume_integral=volume_integral)
    for j in eachindex(Nq_vec)
        Nq = Nq_vec[j]

        cells_per_dimension = (Nq, Nq)

        # mesh = CurvedMesh(cells_per_dimension, mapping)#(-1.0, -1.0), (1.0, 1.0))
        mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
        boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)
            
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions, source_terms=source_terms_convergence_test)
            
            
            ###############################################################################
            # ODE solvers, callbacks etc.
            
            
        ode = semidiscretize(semi, tspan)
            
        summary_callback = SummaryCallback()
        analysis_interval = 1000
            
        analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true)
            
        stepsize_callback = StepsizeCallback(cfl=CFL)
            
        callbacks = CallbackSet(
                                    analysis_callback,
                                    stepsize_callback)
            
            ###############################################################################
            # run the simulation
            
        sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                        dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                        save_everystep=false, callback=callbacks);
        summary_callback() # print the timer summary
            

        error[j] = maximum(abs.(sol.u[1] - sol.u[2])) # maximum(abs.(exact - sol))
    end
      
        # Compute EOC
    eoc = vcat("", [log(error[j + 1] / error[j]) /
            log(Nq_vec[j] / Nq_vec[j + 1]) for j in 1:length(Nq_vec) - 1])
    println("N = $N")

    pretty_table(hcat(Nq_vec, error, eoc), ["Nq", "Error", "EOC"]; kwargs...)

    mean = sum(eoc[2:end]) / (length(Nq_vec) - 1)
    @printf("Mean EOC: %.2f\n", mean)
end


