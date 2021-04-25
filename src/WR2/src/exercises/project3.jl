# Linear advection and burgers equation for testing purposes
function convergence_analysis_linear_advection()
    equation = EquationLinearAdvection(-2)

    # Time-invariant manufactured solution in first variable and advection in second
    u0(x) = @SVector [sin(2 * pi * x), sin(2 * pi * x)]
    u_solution(x) = [sin(2 * pi * x), -sin(2 * pi * x)]
    source(u, x, t) = @SVector [-4 * pi * cos(2 * pi * x), 0]

    convergence_analysis(equation, u0, source, u_solution; tspan=(0, 0.75), CFL=0.1)
end


function convergence_analysis_linear_advection_in_shallow_water()
    equation = EquationShallowWater(0, x -> 0)

    # Time-invariant manufactured solution in first variable and advection in second
    u0(x) = @SVector [sin(2 * pi * x) + 2, 2 * (sin(2 * pi * x) + 2)]
    u_solution(x) = [sin(2 * pi * x) + 2, 2 * (sin(2 * pi * x) + 2)]
    source(u, x, t) = @SVector [4 * pi * cos(2 * pi * x), 2 * (4 * pi * cos(2 * pi * x))]

    convergence_analysis(equation, u0, source, u_solution; tspan=(0, 0.75), CFL=0.1)
end


function convergence_analysis_burgers()
    equation = EquationBurgers()

    u0(x) = sin(2 * pi * x) + 2
    u_solution = u0
    source(u, x, t) = 2 * pi * (sin(2 * pi * x) + 2) * cos(2 * pi * x)

    # u(x, t) = 5 + sin(2 * pi * (x - t))
    # u0(x) = u(x, 0)
    # u_solution(x) = u(x, 1)
    # source(u, x, t) = 8 * pi * cos(2 * pi * (x - t)) + pi * sin(4 * pi * (x - t))

    # godunov(uL, uR, equation) = godunov_scalar_minimum(uL, uR, equation, 0)

    convergence_analysis(equation, u0, source; CFL=1.0)
end


function p3_exercise1(; latex=false)
    p3_convergence_analysis(; latex=latex)
    p3_conservation_analysis()
    p3_well_balanced_analysis()
end


function p3_exercise2(; latex=false)
    p3_convergence_analysis(:ecDG; latex=latex)
    p3_conservation_analysis(:ecDG)
    p3_well_balanced_analysis(:ecDG)
end


function p3_exercise3(; latex=false)
    p3_convergence_analysis(:esDG; latex=latex)
    p3_conservation_analysis(:esDG)
    p3_well_balanced_analysis(:esDG)
end


function p3_entropy_analysis(; latex=false)
    riemann_solvers = Dict(:DG => rusanov, 
                           :ecDG => entropy_flux, 
                           :esDG => entropy_shock_flux)
    p = plot()

    kwargs = Dict{Symbol, Any}(
        :xlabel => "t",
        :ylabel => "Gesamtenergie",
        :title => "Zeitliche Änderung der Gesamtenergie"
    )
    if latex
        pgfplotsx()
        kwargs[:tex_output_standalone] = true
        kwargs[:xlabel] = "\$t\$"
    end

    for method in [:DG, :ecDG, :esDG]
        riemann_solver = riemann_solvers[method]

        N = 3
        Nq = 100
        g = 9.812

        equation = EquationShallowWater(g, b1)

        dg = DG(N, Nq, 2, equation, riemann_solver; xspan=(0, 20), method=method)

        Δx = (dg.xspan[2] - dg.xspan[1]) / nelements(dg)

        plot_values = 200
        energy_data = zeros(plot_values)
        t_vec = range(0, 1; length=plot_values)

        for i in 1:plot_values
            # This is extremely inefficient
            @timeit_debug "semidiscretize" ode = semidiscretize(dg, u0_entropy; CFL=0.1, tspan=(0, t_vec[i]))

            @timeit_debug "solve" sol = solve(ode)

            for element in 1:Nq
                for node in 1:N+1
                    @views energy_data[i] += energy(sol[:, node, element], dg.node_pos[node, element], g, b1) * dg.weights[node] * Δx * 0.5
                end
            end
        end

        println("\nEntropy analysis $(uppercase(string(method)))SEM:\n")
        println("Total energy before: $(energy_data[1])")
        println("Total energy after: $(energy_data[end])")
        println("Difference: $(energy_data[end] - energy_data[1])")

        plot!(p, t_vec, energy_data, label="$(uppercase(string(method)))SEM"; kwargs...)
    end

    return p
end


function p3_exercise4(t, method=:DG; latex=false)
    equation = EquationShallowWater(1, b1)

    function u0(x)
        h = 1.1 - b1(x)
        v = x <= 5.0 ? -0.525 : 0.525

        return @SVector [h, h*v]
    end

    boundary_condition(u, x, dg) = u0(x)

    if method == :DG
        riemann_solver = rusanov
    elseif method == :ecDG
        riemann_solver = entropy_flux
    elseif method == :esDG
        riemann_solver = entropy_shock_flux
    end

    dg = DG(3, 100, 2, equation, riemann_solver; xspan=(0, 20),
        boundary_condition=boundary_condition, method=method)

    @timeit_debug "semidiscretize" ode = semidiscretize(dg, u0; tspan=(0.0, t), CFL=0.5)

    @timeit_debug "solve" sol = solve(ode)

    # Compute H
    for element in 1:nelements(dg)
        for node in 1:polydeg(dg)+1
            sol[1, node, element] += b1(dg.node_pos[node, element])
        end
    end

    # Plot solution
    x, data = dg_to_plot_data(sol, dg)

    kwargs = Dict{Symbol, Any}(
        :xlabel => "x",
        :ylabel => "H(x)",
        :label => "H"
    )
    if latex
        pgfplotsx()
        kwargs[:tex_output_standalone] = true
        kwargs[:xlabel] = "\$x\$"
        kwargs[:ylabel] = "\$H(x)\$"
        kwargs[:label] = "\$H\$"
    end

    p = plot(x, data[1, :]; kwargs...)

    if latex
        kwargs[:label] = "\$b_1\$"
    else
        kwargs[:label] = "b1"
    end
    plot!(p, x, b1.(x); kwargs...)

    return p
end


function p3_convergence_analysis_stationary(method=:DG; latex=false)
    equation = EquationShallowWater(1, x -> 0)
    if method == :DG
        riemann_solver = rusanov
    elseif method == :ecDG
        riemann_solver = entropy_flux
    elseif method == :esDG
        riemann_solver = entropy_shock_flux
    end

    convergence_analysis(equation, u0_sin, source_manufactured_stationary; 
            riemann_solver=riemann_solver, xspan=(0, 20), CFL=0.1, latex=latex, method=method) 
end


function p3_convergence_analysis(method=:DG; latex=false)
    equation = EquationShallowWater(1, b2)

    if method == :DG
        riemann_solver = rusanov
    elseif method == :ecDG
        riemann_solver = entropy_flux
    elseif method == :esDG
        riemann_solver = entropy_shock_flux
    end

    convergence_analysis(equation, x -> u_sin(x, 0), source_manufactured_constant_speed, 
            x -> u_sin(x, 1); riemann_solver=riemann_solver, CFL=0.1, latex=latex, method=method) 
end


function p3_conservation_analysis(method=:DG)
    equation = EquationShallowWater(1, b1)

    if method == :DG
        riemann_solver = rusanov
    elseif method == :ecDG
        riemann_solver = entropy_flux
    elseif method == :esDG
        riemann_solver = entropy_shock_flux
    end

    dg = DG(3, 100, 2, equation, riemann_solver; xspan=(0, 20), method=method)
    
    Δx = (dg.xspan[2] - dg.xspan[1]) / nelements(dg)

    @timeit_debug "semidiscretize" ode = semidiscretize(dg, u0_sin; CFL=1.1)

    integral_before = zeros(2)
    for s in 1:nvariables(dg)
        for element in 1:100
            for node in 1:4
                @views integral_before[s] += ode.u0[s, node, element] * dg.weights[node] * Δx * 0.5
            end
        end
    end

    @timeit_debug "solve" sol = solve(ode)

    integral_after = zeros(2)
    for s in 1:nvariables(dg)
        for element in 1:100
            for node in 1:4
                @views integral_after[s] += sol[s, node, element] * dg.weights[node] * Δx * 0.5
            end
        end
    end

    println("\nConservation analysis:\n")
    println("Integral before: $integral_before")
    println("Integral after: $integral_after")
    println("Error: $(integral_after - integral_before)")
end


function p3_well_balanced_analysis(method=:DG)
    if method == :DG
        riemann_solver = rusanov
    elseif method == :ecDG
        riemann_solver = entropy_flux
    elseif method == :esDG
        riemann_solver = entropy_shock_flux
    end

    N = 3
    Nq = 100
    g = 9.812

    equation = EquationShallowWater(g, b1)
    u0(x) = @SVector [3 - b1(x), 0]

    dg = DG(N, Nq, 2, equation, riemann_solver; xspan=(0, 20), method=method)

    @timeit_debug "semidiscretize" ode = semidiscretize(dg, u0; CFL=0.1)

    @timeit_debug "solve" sol = solve(ode)

    # Compute error in [H, v]
    for element in 1:Nq
        for node in 1:N+1
            sol[2, node, element] /= sol[1, node, element]
            sol[1, node, element] += b1(dg.node_pos[node, element]) - 3
        end
    end

    error = maximum(abs.(sol))

    println("\nWell-balanced analysis:\n")
    println("Maximum error: $error")
end


function u0_sin(x)
    h = sin(pi/5 * x) + 4
    v = 1

    return @SVector [h, h * v]
end

function source_manufactured_stationary(u, x, t)
    c = pi/5 * cos(pi/5 * x)

    return @SVector [c, c * (sin(pi/5 * x) + 3)]
end

function u_sin(x, t)
    h = sin(2 * pi * (x - t)) + 5
    v = 1

    return @SVector [h, h * v]
end

source_manufactured_constant_speed(u, x, t) =  @SVector [0, 2 * pi * (sin(2 * pi * (x - t)) + 5) * cos(2 * pi * (x - t)) + u[1] * b2_x(x)]

b1_x(x) = abs(x - 10) <= 2 ? pi/4 * cos(pi/4 * x) : 0.0
b1(x) = abs(x - 10) <= 2 ? sin(pi/4 * x) : 0.0
b2(x) = sin(pi / 10 * x) + 1
b2_x(x) = pi / 10 * cos(pi / 10 * x)

u0_entropy(x) = x <= 10 ? @SVector([3.0 - b1(x), 0.0]) : @SVector([2.5 - b1(x), 0.0])
energy(u, x, g, b) = 0.5 * u[2]^2 / u[1] + 0.5 * u[1]^2 * g + g * u[1] * b(x)


# Generic convergence analysis
function convergence_analysis(equation, u0, source, u_solution=u0; riemann_solver=rusanov,
                                xspan=(0, 1), tspan=(0, 1), CFL=1.4, method=:DG, latex=false)
    reset_timer!()

    N_vec = [1, 3, 7]
    Nq_vec = [5, 10, 20, 40, 80, 160]
    
    # Table preparation
    kwargs = Dict{Symbol, Any}(
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
        if N == 7
            Nq_vec = div.(Nq_vec, 5)
        end

        for j in eachindex(Nq_vec)
            Nq = Nq_vec[j]
        
            dg = DG(N, Nq, length(u0(0.0)), equation, riemann_solver; source=source, xspan=xspan, method=method)

            @timeit_debug "Semidiscretization" ode = semidiscretize(dg, u0; tspan=tspan, CFL=CFL)

            @timeit_debug "Solve ODE" sol = solve(ode)

            # Compute exact solution vector
            exact = similar(sol)
            for element in 1:Nq
                for node in 1:N+1
                    node_vars = u_solution(dg.node_pos[node, element])
                    for s in 1:nvariables(dg)
                        exact[s, node, element] = node_vars[s]
                    end
                end
            end

            error[j] = maximum(abs.(exact - sol))
        end

        # Compute EOC
        eoc = vcat("", [log(error[j+1]/error[j]) /
            log(Nq_vec[j]/Nq_vec[j+1]) for j in 1:length(Nq_vec)-1])
        println("N = $N")

        pretty_table(hcat(Nq_vec, error, eoc), ["Nq", "Error", "EOC"]; kwargs...)

        mean = sum(eoc[2:end])/(length(Nq_vec) - 1)
        @printf("Mean EOC: %.2f\n", mean)
    end

    print_timer()
end