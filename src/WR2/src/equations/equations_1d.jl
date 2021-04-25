struct EquationLinearAdvection{A}
    a::A

    EquationLinearAdvection(a) = new{typeof(a)}(a)
end


calcflux(u, equation::EquationLinearAdvection) = equation.a * u
max_abs_eigenvalue(u, equation::EquationLinearAdvection) = abs(equation.a)


struct EquationBurgers end


@. calcflux(u, ::EquationBurgers) = 0.5 * u^2
max_abs_eigenvalue(u, ::EquationBurgers) = maximum(abs.(u))


struct EquationShallowWater{G, B}
    g::G
    b::B

    EquationShallowWater(g, b) = new{typeof(g), typeof(b)}(g, b)
end


calcflux(u, equation::EquationShallowWater) = @SVector [u[2], u[2]^2 / u[1] + 0.5 * equation.g * u[1]^2]
max_abs_eigenvalue(u, equation::EquationShallowWater) = abs(u[2] / u[1]) + sqrt(equation.g * u[1])