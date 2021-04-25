function p1_exercise2()
    p1_quadrature_table()
end


function p1_quadrature_table(;latex=false)
    f1(x, N) =  cos.(x)
    f1_int(N) = 2 * sin(1)
    p1_quadrature_table(f1, f1_int; latex=latex)

    f2(x, N) = @. 1/(1 + x^2)
    f2_int(N) = π/2
    p1_quadrature_table(f2, f2_int; latex=latex)

    f3(x, N) = @. x^(2 * N - 2)
    f3_int(N) = 2/(2*N-1)
    p1_quadrature_table(f3, f3_int; latex=latex)

    f4(x, N) = @. x^(2 * N)
    f4_int(N) = 2/(2*N+1)
    p1_quadrature_table(f4, f4_int; latex=latex)

    f5(x, N) = @. x^(2 * N + 2)
    f5_int(N) = 2/(2*N+3)
    p1_quadrature_table(f5, f5_int; latex=latex)
end


function p1_quadrature_table(f, f_int; latex=false)
    N = [5, 10, 20]

    result = zeros(2, length(N))
    ϵ = zeros(2, length(N))

    for i in 1:length(N)
        (x1, w1) = legendre_gauss_nodes_weights(N[i])
        (x2, w2) = legendre_gauss_lobatto_nodes_weights(N[i])

        result[1, i] = f(x1, N[i])' * w1
        result[2, i] = f(x2, N[i])' * w2
    end

    ϵ = @. abs([f_int(n) for n in N]' - result)
    data = hcat(["Legendre-Gauß", "Legendre-Gauß-Lobatto"], ϵ)

    column_names = Array{String}(undef, length(N) + 1)
    column_names[1] = "Method"
    column_names[2:end] = [string("N = ", n) for n in N]


    kwargs = Dict{Symbol, Any}(
        :formatters => ft_printf("%.5e", 2:length(N)+1)
    )
    if latex
        kwargs[:backend] = :latex
        # Insert all data in math mode and as siunitx number
        kwargs[:highlighters] = LatexHighlighter(
            (data, i, j) -> j > 1,
            (data, i, j, str) -> "\$\\num{" * str * "}\$")
    end

    pretty_table(data, column_names; kwargs...)
end


function p1_exercise3b()
    N = [1, 2, 3]

    for i in 1:length(N)
        n = N[i]
        (x1, w1) = legendre_gauss_nodes_weights(n)
        (x2, w2) = legendre_gauss_lobatto_nodes_weights(n)
        M1 = diagm(w1)
        M2 = diagm(w2)

        p1 = InterpolationPolynomial(x1)
        p2 = InterpolationPolynomial(x2)

        D1 = derivation_matrix(p1)
        D2 = derivation_matrix(p2)

        B = zeros(n+1, n+1)
        B[1, 1] = -1
        B[n+1, n+1] = 1
        @info string("Legendre-Gauß, N = ", n) M1*D1 + (M1*D1)'
        @info string("Legendre-Gauß-Lobatto N = ", n) M2*D2 + (M2*D2)'
    end
end


function p1_exercise4(;latex=false, save=false)
    x, w = legendre_gauss_nodes_weights(10)

    f1 = cos.(x)
    f2 = @. 1/(1 + x^2)

    p = InterpolationPolynomial(x)
    D = derivation_matrix(p)

    Df1 = D * f1
    Df2 = D * f2

    x_out = range(-1, 1, length=1001)
    VDf1 = [p(x_out[i], Df1) for i in 1:length(x_out)]
    VDf2 = [p(x_out[i], Df2) for i in 1:length(x_out)]

    kwargs = Dict(
        :xlabel => "x",
        :ylabel => "f¹'",
        :label => permutedims(["Approximation", "Exakte Ableitung", "f¹"])
    )

    if latex
        pgfplotsx()
        kwargs[:xlabel] = "\$x\$"
        kwargs[:ylabel] = "\$\\frac{\\mathrm{d}}{\\mathrm{d}x}f^1\$"
        kwargs[:label][3] = "\$f^1\$"
        kwargs[:tex_output_standalone] = true
    end
    p1 = plot(x_out, [VDf1, -sin.(x_out), cos.(x_out)]; kwargs...)
    display(p1)
    save && savefig(p1, "plot_1.tikz")

    if latex
        kwargs[:ylabel] = "\$\\frac{\\mathrm{d}}{\\mathrm{d}x}f^2\$"
        kwargs[:label][3] = "\$f^2\$"
    else
        kwargs[:ylabel] = "f²'"
        kwargs[:label][3] = "f²"
    end
    f2_exact = @. 1/(1 + x_out^2)
    VDf2_exact = @. -2 * x_out * 1/(1 + x_out^2)^2
    p2 = plot(x_out, [VDf2, VDf2_exact, f2_exact]; kwargs...)
    save && savefig(p2, "plot_2.tikz")
    display(p2)
end
