function p2_exercise2(; latex=false)
    reset_timer!()

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

    # Problem to test
    f(u, p, t) = t .- u/t
    dt_vec = [1/2^i for i in 0:9]

    error = Vector{Float64}(undef, length(dt_vec))
    for i in eachindex(dt_vec)
        dt = dt_vec[i]

        ode = Numerik4.ODEProblem(f, [4/3], (1, 2))

        @timeit_debug "Solve ODE" result = solve(ode, dt)
        err = 11/6 .- result
        error[i] = abs(err[1])
    end

    # Compute EOC
    eoc = vcat("", [log(error[j+1]/error[j]) /
        log(dt_vec[j+1]/dt_vec[j]) for j in 1:length(dt_vec)-1])

    pretty_table(hcat(dt_vec, error, eoc), ["Δt", "Fehler", "EOC"]; kwargs...)

    print_timer()
end


function p2_exercise3a(; latex=false)
    reset_timer!()

    u0(x) = sin(2 * pi * x)
    N_vec = 1:7
    Nq_vec = [4, 8, 16, 32]
    CFL = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
    method = :lobatto

    energy_factor = Array{Float64}(undef, length(CFL), length(Nq_vec))
    for N in N_vec
        for j in eachindex(Nq_vec)
            Nq = Nq_vec[j]

            for i in eachindex(CFL)
                @timeit_debug "Semidiscretization" ode =
                    semidiscretize(2, N, Nq, u0; CFL=CFL[i], tspan=(0, 1), method=method)

                @timeit_debug "Solve ODE" dg_data = solve(ode)

                Δx = 1 / Nq
                dg_nodes = compute_dg_nodes(Δx, N, Nq, method)

                # Compute exact solution vector
                exact = u0.(dg_nodes)

                energy_factor[i, j] = sum(dg_data.^2) / sum(exact.^2)
            end
        end

        println("N = $N")

        if latex
            kwargs = Dict{Symbol, Any}(
                :backend => :latex,
                :highlighters => (
                    # Insert all data in math mode and as siunitx number
                    LatexHighlighter(
                        (data, i, j) -> j > 1 && data[i,j] > 1.05,
                        (data, i, j, str) -> "{\\color{myred}\$\\num{$str}\$}"
                    ),
                    LatexHighlighter(
                        (data, i, j) -> j > 1 && data[i,j] <= 1.000001,
                        (data, i, j, str) -> "{\\color{mygreen}\$\\num{$str}\$}"
                    ),
                    LatexHighlighter(
                        (data, i, j) -> j > 1,
                        (data, i, j, str) -> "\$\\num{$str}\$"
                    )
                )
            )
        else
            # Color highlighing
            kwargs = Dict{Symbol, Any}(
                :highlighters => (
                    Highlighter(
                        (data, i, j) -> j > 1 && data[i,j] > 1.05,
                        foreground = :red
                    ),
                    Highlighter(
                        (data, i, j) -> j > 1 && data[i,j] <= 1.000001,
                        foreground = :green
                    )
                )
            )
        end

        pretty_table(hcat(CFL, energy_factor), vcat(["CFL"], ["Nq = $Nq" for Nq in Nq_vec]); kwargs...)
    end

    print_timer()
end


function p2_exercise3a2(; latex=false, save=false)
    method = :lobatto

    N = 4
    Nq = 16
    CFL = [1.7, 1.6]

    for i in eachindex(CFL)
        xspan = (0, 1)
        Δx = (xspan[2] - xspan[1]) / Nq
        A = Array(semidiscretization_matrix(2, Δx, N, Nq; method=method))

        # Compute timestep
        dg_nodes = compute_dg_nodes(Δx, N, Nq, method)
        dt = timestep(CFL[i], dg_nodes, 2, xspan[1])

        values = eigvals(A)
        R = stability_function()
        scaled_eigenvals = dt * values

        f(x, y) = abs(R(x + im*y))

        kwargs = Dict(
            :xlabel => "Re",
            :ylabel => "Im",
            :colorbar => :none,
            :label => permutedims(["Approximation", "Exakte Ableitung", "f¹"])
        )
        if latex
            pgfplotsx()
            kwargs[:tex_output_standalone] = true
        end
        plot = contour(-5:0.1:0.5, -5:0.1:5, f, levels=[1.0]; kwargs...)
        scatter!(plot, real.(scaled_eigenvals), imag.(scaled_eigenvals), label="Eigenwerte", markersize=3)
        display(plot)
        save && savefig(plot, "plot_$i.tex")
    end
end


function stability_function()
    A = @SVector [0.0, -567301805773/1357537059087, -2404267990393/2016746695238,
        -3550918686646/2091501179385, -1275806237668/842570457699]
    B = @SVector [1432997174477/9575080441755, 5161836677717/13612068292357,
        1720146321549/2090206949498, 3134564353537/4481467310338,
        2277821191437/14882151754819]
    c = @SVector [0.0, 1432997174477/9575080441755, 2526269341429/6820363962896,
        2006345519317/3224310063776, 2802321613138/2924317926251]

    a = @SMatrix [0 0 0 0 0;
            B[1] 0 0 0 0;
            A[2]*B[2]+B[1] B[2] 0 0 0;
            A[2]*(A[3]*B[3]+B[2])+B[1] A[3]*B[3]+B[2] B[3] 0 0;
            A[2]*(A[3]*(A[4]*B[4]+B[3])+B[2])+B[1] A[3]*(A[4]*B[4]+B[3])+B[2] A[4]*B[4]+B[3] B[4] 0]
    b = Vector{Float64}(undef, 5)
    b[5] = B[5]
    for i in [4, 3, 2, 1]
        b[i] = A[i+1] * b[i+1] + B[i]
    end

    R = Polynomial([1, sum(b), sum(b .* c),
                       b' * a * c,
                       b' * a * a * c,
                       b' * a * a * a * c])

    # Determinant formula to verify stability function
    # Q(z) = det(I(5) - z*a + z*ones(5)*b')

    return R
end


function p2_exercise3b(; latex=false)
    reset_timer!()

    u0(x) = sin(2 * pi * x)
    N_vec = [1, 3, 7]
    Nq_vec = [1, 2, 4, 8, 16, 32, 64]

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

    for method in [:lobatto, :gauss]
        println(String(method))
        error = Vector{Float64}(undef, length(Nq_vec))
        for N in N_vec
            for j in eachindex(Nq_vec)
                Nq = Nq_vec[j]

                @timeit_debug "Semidiscretization" ode =
                    semidiscretize(2, N, Nq, u0; CFL=1.4, tspan=(0, 1), method=method)

                @timeit_debug "Solve ODE" dg_data = solve(ode)

                Δx = 1 / Nq
                dg_nodes = compute_dg_nodes(Δx, N, Nq, method)

                # Compute exact solution vector
                exact = u0.(dg_nodes)

                error[j] = maximum(abs.(exact - dg_data))
            end

            # Compute EOC
            eoc = vcat("", [log(error[j+1]/error[j]) /
                log(Nq_vec[j]/Nq_vec[j+1]) for j in 1:length(Nq_vec)-1])
            println("N = $N")

            pretty_table(hcat(Nq_vec, error, eoc), ["Nq", "Error", "EOC"]; kwargs...)
        end
    end

    print_timer()
end


function p2_exercise3c(; latex=false)
    u0(x) = sin(2 * pi * x)
    method = :lobatto

    p2_efficiency_table_constant_error(u0, method; latex=latex)
    p2_efficiency_table_1024(u0, method; latex=latex)
    p2_efficiency_table_constant_time(u0, method; latex=latex)
end


function p2_efficiency_table_constant_error(u0::T, method; latex=false) where T
    N_vec = [1, 3, 7]
    error = 1e-5
    elapsed_time = Vector{Float64}(undef, length(N_vec))
    cells_needed = Vector{Float64}(undef, length(N_vec))

    for i in eachindex(N_vec)
        N = N_vec[i]
        current_error = error + 10
        Nq = 5

        while current_error > error
            Nq *= 2

            before = time()

            # Semidiscretize
            ode = semidiscretize(2, N, Nq, u0; tspan=(0, 10), method=method)

            # Solve ODE
            dg_data = solve(ode)

            elapsed_time[i] = time() - before

            # Compute exact solution vector
            Δx = 1 / Nq
            dg_nodes = compute_dg_nodes(Δx, N, Nq, method)
            exact = u0.(dg_nodes)

            current_error = maximum(abs.(exact - dg_data))
        end
        cells_needed[i] = current_error === NaN ? NaN : Nq
    end

    kwargs = Dict{Symbol, Any}(
        :formatters => (
            ft_printf("%d", [1, 2]),
            ft_printf("%.4f", [3])
        )
    )
    if latex
        kwargs[:backend] = :latex
        kwargs[:highlighters] = LatexHighlighter(
            (data, i, j) -> j > 2,
            (data, i, j, str) -> "\$\\num{$str}\$"
        )
    end

    pretty_table(hcat(N_vec, cells_needed, elapsed_time), ["N", "Benötigte Zellen", "Laufzeit in s"]; kwargs...)
end


function p2_efficiency_table_1024(u0::T, method; latex=false) where T
    N_vec = [1, 3, 7]
    elapsed_time = Vector{Float64}(undef, length(N_vec))

    error = Vector{Float64}(undef, length(N_vec))
    for i in eachindex(N_vec)
        N = N_vec[i]
        Nq = div(1024, (N+1))

        before = time()

        # Semidiscretize
        ode = semidiscretize(2, N, Nq, u0; tspan=(0, 10), method=method)

        # Solve ODE
        dg_data = solve(ode)

        elapsed_time[i] = time() - before

        # Compute exact solution vector
        Δx = 1 / Nq
        dg_nodes = compute_dg_nodes(Δx, N, Nq, method)
        exact = u0.(dg_nodes)

        error[i] = maximum(abs.(exact - dg_data))
    end

    kwargs = Dict{Symbol, Any}(
        :formatters => (
            ft_printf("%d", [1, 2]),
            ft_printf("%.5e", [3]),
            ft_printf("%.4f", [4])
        )
    )
    if latex
        kwargs[:backend] = :latex
        kwargs[:highlighters] = LatexHighlighter(
            (data, i, j) -> j > 2,
            (data, i, j, str) -> "\$\\num{$str}\$"
        )
    end

    pretty_table(hcat(N_vec, div.(1024, (N_vec.+1)), error, elapsed_time),
        ["N", "Nq", "Fehler", "Laufzeit in s"]; kwargs...)
end


function p2_efficiency_table_constant_time(u0::T, method; latex=false) where T
    N_vec = [1, 3, 7]
    maximum_time = 0.5
    cells_needed = Vector{Float64}(undef, length(N_vec))
    error = Vector{Float64}(undef, length(N_vec))

    for i in eachindex(N_vec)
        N = N_vec[i]
        elapsed_time = 0
        Nq = 1
        dg_data = Vector{Float64}(undef, 0)

        while elapsed_time < maximum_time
            before = time()

            # Semidiscretize
            ode = semidiscretize(2, N, Nq, u0; tspan=(0, 10), method=method)

            # Solve ODE
            temp_dg_data = solve(ode)

            elapsed_time = time() - before

            if elapsed_time < maximum_time
                dg_data = temp_dg_data
                Nq *= 2
            end
        end

        Nq = div(Nq, 2)

        # Compute exact solution vector
        Δx = 1 / Nq
        dg_nodes = compute_dg_nodes(Δx, N, Nq, method)
        exact = u0.(dg_nodes)

        error[i] = maximum(abs.(exact - dg_data))

        cells_needed[i] = Nq
    end

    kwargs = Dict{Symbol, Any}(
        :formatters => (
            ft_printf("%d", [1, 2]),
            ft_printf("%.5e", [3])
        )
    )
    if latex
        kwargs[:backend] = :latex
        kwargs[:highlighters] = LatexHighlighter(
            (data, i, j) -> j > 2,
            (data, i, j, str) -> "\$\\num{$str}\$"
        )
    end

    pretty_table(hcat(N_vec, cells_needed, error), ["N", "Nq", "Fehler"]; kwargs...)
end


function p2_exercise3d(; latex=false, save=false)
    reset_timer!()
    function u0(x)
        if 0.3 <= x <= 0.7
            return 1.0
        end
        return 0.0
    end

    for method in [:lobatto, :gauss]
        for N in [1, 3, 7]
            Nq = 50

            @timeit_debug "Semidiscretize" ode =
                semidiscretize(2, N, Nq, u0; tspan=(0, 2), method=method)

            @timeit_debug "Solve ODE" dg_data = solve(ode)

            @timeit_debug "Plot" plot = plot_dg(dg_data, N, xspan=(0, 1); method=method, latex=latex, f_exact=u0)

            display(plot)
            method_string = String(method)
            save && savefig(plot, "plot_a3d_$(N)_$method_string.tex")
        end
    end

    print_timer()
end
