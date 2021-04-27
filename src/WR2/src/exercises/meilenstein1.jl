# Linear advection and burgers equation for testing purposes
function convergence_analysis_linear_advection_2d()
    equation = EquationLinearAdvection2D(1)

    # Time-invariant manufactured solution in first variable and advection in second
    u0(pos) = @SVector [sin(2 * pi * pos[1]), cos(2 * pi * pos[1])]
    u_solution(pos) = [-sin(2 * pi * pos[1]), -cos(2 * pi * pos[1])]

    convergence_analysis_2d(equation, u0, nothing, u_solution; tspan=(0, 0.5), CFL=0.9)
end


# Linear advection and burgers equation for testing purposes
function convergence_analysis_linear_advection_2d_constant()
    equation = EquationLinearAdvection2D(1)

    # Time-invariant manufactured solution in first variable and advection in second
    u0(pos) = @SVector [1, 1]
    u_solution(pos) = [1, 1]

    convergence_analysis_2d(equation, u0, nothing, u_solution; tspan=(0, 0.5), CFL=0.1)
end



function p4_exercise2(; latex=false)
    #equation = EquationLinearEuler2D(1,1)
    equation = EquationComprEuler2D(1.4)

    function u0(pos, t)
        x = pos[1]
        y = pos[2]

        @SVector [1.0,1.0,1.0,1.0]
    end

    convergence_analysis_2d(equation, pos->u0(pos,0), nothing, pos->u0(pos,1); CFL=0.1, latex=latex)
end


function p4_exercise3(; N=6, Nq_vec=[10, 50], t_end=0.5, animate=false, latex=false)
    reset_timer!()

    kwargs = Dict{Symbol, Any}()

    equation = EquationLinearEuler2D(1, 1)

    function boundary_condition(u, pos, element_x, element_y, node_x, node_y, orientation, dg)
        if orientation == 3 # Down
            u_boundary = get_node_vars(u, dg, element_x, 1, node_x, 1)
            return @SVector [u_boundary[1], -u_boundary[2], u_boundary[3]]
        end

        # Dirichlet
        return @SVector [0.0, 0.0, 2.0]
    end

    is_solid(element_x, element_y, Nq) = (10 * element_x <= 3 * Nq || 10 * element_x > 7 * Nq) && 10 * element_y <= 4 * Nq

    function u0(pos)
        x = pos[1]
        y = pos[2]

        if 0.4 <= x <= 0.6 && 0.1 <= y <= 0.4 # in W
            return @SVector [0.0, 1.0, 3 * exp(-((x-0.5)^2 + (y-0.25)^2) * 50) + 2]
        end

        return @SVector [0.0, 0.0, 2.0]
    end

    for Nq in Nq_vec
        dg = DG_2D(N, Nq, 3, equation, rusanov; boundary_condition=boundary_condition, is_solid=is_solid)

        @timeit_debug "Semidiscretization" ode = semidiscretize(dg, u0; tspan=(0, 0.0), CFL=0.6)

        t_end_vec = 0:0.005:t_end

        if Nq == 10
            mic_element = (7, 1)
        elseif Nq == 50
            mic_element = (33, 3)
        end

        plot_microphone = N == 6 && (Nq == 10 || Nq == 50)
        if plot_microphone 
            mic = Vector{Float64}(undef, length(t_end_vec))
        end

        plot()

        anim = @animate for i in 1:length(t_end_vec)
            if latex
                pgfplotsx()
                kwargs[:tex_output_standalone] = true
                
                kwargs[:xlabel] = "\$x\$"
                kwargs[:ylabel] = "\$y\$"
            else
                kwargs[:xlabel] = "x"
                kwargs[:ylabel] = "y"
                kwargs[:title] = @sprintf("Druck zum Zeitpunkt t=%.3f", t_end_vec[i])
            end

            # Solve ODE to t_end_vec[i]
            ode.tspan = (ode.tspan[1], t_end_vec[i])
            @timeit_debug "Solve ODE" sol = solve(ode)

            if plot_microphone
                mic[i] = sol[3, mic_element[1], mic_element[2], 4, 4]
            end

            if animate || i == length(t_end_vec)
                @timeit_debug "Interpolate solution data" x, y, data = dg_to_plot_data(sol, dg; n_out=200)

                @timeit_debug "Heatmap" p = heatmap(x, y, data[3, :, :]',
                    aspect_ratio=1, clims=(0, 4); kwargs...)
                display(p)
                latex && savefig(p, "heatmap.tex")
            end
        end

        animate && gif(anim, "out/test.gif", fps=30)

        if plot_microphone
            if latex
                pgfplotsx()
                kwargs[:tex_output_standalone] = true
                
                kwargs[:xlabel] = "\$t\$"
                kwargs[:ylabel] = "\$p\$"
                kwargs[:label] = "Druck am Mikrofon"
            else
                kwargs[:xlabel] = "t"
                kwargs[:ylabel] = "p"
                kwargs[:title] = "Druckverlauf am Mikrofon"
                kwargs[:label] = "Druck am Mikrofon"
            end

            # Plot microphone
            p = plot(t_end_vec, mic, ylims=(0, 4); kwargs...)
            display(p)
            latex && savefig(p, "microphone.tex")
        end
    end
    print_timer()
end


# Generic convergence analysis
function convergence_analysis_2d(equation, u0, source, u_solution=u0; riemann_solver=rusanov,
                                xspan=(0, 1), tspan=(0, 1), CFL=1.0, latex=false)
    reset_timer!()

    N_vec = [5, 6]
    Nq_vec = [1, 2, 4, 8, 16]

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
        for j in eachindex(Nq_vec)
            Nq = Nq_vec[j]

            dg = DG_2D(N, Nq, length(u0((0.0, 0.0))), equation, riemann_solver; source=source, xspan=xspan)

            @timeit_debug "Semidiscretization" ode = semidiscretize(dg, u0; tspan=tspan, CFL=CFL)

            @timeit_debug "Solve ODE" sol = solve(ode)

            # Compute exact solution vector
            exact = similar(sol)
            for element_x in 1:Nq, element_y in 1:Nq
                for node_x in 1:N+1, node_y in 1:N+1
                    node_vars = u_solution(dg.elements.node_pos[element_x, element_y, node_x, node_y])

                    for s in 1:nvariables(dg)
                        exact[s, element_x, element_y, node_x, node_y] = node_vars[s]
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
