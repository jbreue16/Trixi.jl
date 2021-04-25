struct EquationLinearAdvection2D{A}
    a::A

    EquationLinearAdvection2D(a) = new{typeof(a)}(a)
end


calcflux(u, orientation, equation::EquationLinearAdvection2D) = equation.a * u
max_abs_eigenvalue(u, equation::EquationLinearAdvection2D) = abs(equation.a)


struct EquationLinearEuler2D{R, L}
    ρ::R
    λ::L

    EquationLinearEuler2D(ρ, λ) = new{typeof(ρ), typeof(λ)}(ρ, λ)
end


function calcflux(u, orientation, equation::EquationLinearEuler2D)
    if orientation == 1
        return @SVector [1 / equation.ρ * u[3], 0, equation.λ * u[1]]
    else
        return @SVector [0, 1 / equation.ρ * u[3], equation.λ * u[2]]
    end
end

max_abs_eigenvalue(u, equation::EquationLinearEuler2D) = sqrt(equation.λ / equation.ρ)