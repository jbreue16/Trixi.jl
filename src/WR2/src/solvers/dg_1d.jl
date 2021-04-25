struct DG_1D{S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B, METHOD} <: DG{S, N}
    # METHOD can be: DG, ecDG
    Nq::NQ # Number of elements
    equation::EQ # equation
    source::SRC
    riemann_solver::R # Riemann solver
    xspan::X
    D::T # Derivation matrix for the corresponding nodes
    node_pos::NP # global position of nodes (element and node)
    weights::W # Quadrature weights
    flux_buffer::FB
    boundary_condition::B

    function DG_1D(N, Nq, S, equation, riemann_solver; xspan=(0, 1), boundary_condition=boundary_condition_periodic, source=nothing, method=:DG)
        if N == 0
            nodes, weights = legendre_gauss_nodes_weights(N)
        else
            nodes, weights = legendre_gauss_lobatto_nodes_weights(N)
        end
        node_pos = init_mesh(N, Nq, xspan, nodes, Val(1))

        D_temp = derivation_matrix(InterpolationPolynomial(nodes))
        D = SMatrix{size(D_temp, 1), size(D_temp, 2)}(D_temp)
        flux_buffer = Array{Float64}(undef, S, Nq+1)

        new{S, N, typeof(Nq), typeof(equation), typeof(source), typeof(riemann_solver), typeof(xspan),
            typeof(D), typeof(node_pos), typeof(weights), typeof(flux_buffer), typeof(boundary_condition), method}(
            Nq, equation, source, riemann_solver, xspan, D, node_pos, weights, flux_buffer, boundary_condition)
    end
end


@inline nelements(dg::DG_1D{S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B, METHOD}) where {S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B, METHOD} = dg.Nq

@inline get_var_nodes(u, dg, s, element) = SVector(ntuple(node -> u[s, node, element], polydeg(dg)+1))
@inline get_node_pos_for_element(dg, element) = SVector(ntuple(node -> dg.node_pos[node, element], polydeg(dg)+1))


function rhs!(du::AbstractArray{<:Any,3}, u, dg, t)
    du .= zero(eltype(du))

    @timeit "calculate volume integral" calc_volume_integral!(du, u, dg)

    @timeit "calculate interface flux" calc_interface_flux!(u, dg)

    @timeit "calculate boundary flux" calc_boundary_flux!(u, dg)

    @timeit "calculate surface integral" calc_surface_integral!(du, u, dg)

    @timeit "shallow water source terms" calc_shallow_water_source_term!(du, u, t, dg)

    # Correction term, only for ecDG
    @timeit "correction term" calc_correction_term!(du, u, t, dg)

    @timeit "inverse jacobian" calc_inverse_jacobian!(du, dg)

    @timeit "source term" calc_source_term!(du, u, t, dg)
end


function init_mesh(N, Nq, xspan, nodes, ::Val{1})
    node_pos = Array{Float64}(undef, N+1, Nq)
    Δx = (xspan[2] - xspan[1]) / Nq
    for element in 1:Nq
        for node in 1:N+1
            # Transform to [0, 1]
            pos = nodes[node] * 0.5 + 0.5
            # Global position of node
            pos = xspan[1] + (element - 1) * Δx + pos * Δx
            node_pos[node, element] = pos
        end
    end

    return node_pos
end


# Compute timestep using Δt = CFL * Δx / λ_max for LGL nodes
function timestep(u, CFL, dg::DG_1D)
    # Compute Δx_eff as the smallest distance between interpolation nodes.
    if polydeg(dg) == 0
        # Δx_eff = Δx
        Δx_eff = (dg.xspan[2] - dg.xspan[1]) / nelements(dg)
    else
        # Smallest distance is between the first two nodes for LGL nodes
        Δx_eff = dg.node_pos[2, 1] - dg.node_pos[1, 1] # TODO different Δx and Δy
    end

    # Compute maximum eigenvalue over all values of u
    λ_max = 0.0
    for element in 1:dg.Nq, node in 1:polydeg(dg)+1
        λ = max_abs_eigenvalue(get_node_vars(u, dg, node, element), dg.equation)
        if λ > λ_max
            λ_max = λ
        end
    end

    if !isfinite(λ_max) || λ_max ≈ 0
        error("Something went horribly wrong!")
    end

    return CFL * Δx_eff / λ_max
end


function discretize_initial_condition(dg::DG_1D, initial_condition)
    u = Array{Float64}(undef, nvariables(dg), polydeg(dg)+1, nelements(dg))

    for element in 1:nelements(dg)
        for node in 1:polydeg(dg)+1
            node_vars = initial_condition(dg.node_pos[node, element])

            for s in 1:nvariables(dg)
                u[s, node, element] = node_vars[s]
            end
        end
    end

    return u
end


function calc_inverse_jacobian!(du::AbstractArray{<:Any,3}, dg)
    Δx = (dg.xspan[2] - dg.xspan[1]) / nelements(dg)
    @. du *= 2 / Δx
end


# Naive implementation for better readability and comparison.
# This implementation is about 20 times slower.
function calc_volume_integral_naive!(du::AbstractArray{<:Any,3}, u, dg)
    for element in 1:nelements(dg)
        for s in 1:nvariables(dg)
            du[s, :, element] -= dg.D * [calcflux(u[:, node, element], dg.equation)[s] for node in 1:polydeg(dg)+1]
        end
    end

    return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,3}, u, dg)
    #            | D(1,1)  .   .   .   D(1,N+1) |  | f_s(u_1) |                |  D(1,k)*f_s(u_1)  |
    #            |   .     .               .    |  |    .     |                |         .         |
    # -D*f_s = - |   .         .           .    |  |    .     | =      Σ     - |         .         |    , mit s={1,...,S}
    #            |   .             .       .    |  |    .     |  k={1,..,N+1}  |         .         |
    #            |D(N+1,1) .   .   .  D(N+1,N+1)|  |f_s(u_N+1)|                |D(N+1,k)*f_s(u_N+1)|
    for element in 1:nelements(dg)
        # Matrix multiplication D*f_s yields a sum over j
        for j in 1:polydeg(dg)+1
            # f_vec = (f_1(U_j), ..., f_s(U_j)) with U_j = (u_1(j), ..., u_S(j))
            f_j_vec = calcflux(get_node_vars(u, dg, j, element), dg.equation)

            for node in 1:polydeg(dg)+1
                # To prevent multiple computation of f_vec, this loop is moved inside the j-loop
                for s in 1:nvariables(dg)
                    # sum of D[node, j] * f_j_vec[s] over j yields Df_s
                    du[s, node, element] -= dg.D[node, j] * f_j_vec[s]
                end
            end
        end
    end

    return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,3}, dg)
    # f*_s(u_1,..,u_N+1) mit s={1,...,S} for not-boundary interfaces
    for element in 2:nelements(dg)
        f_star_left = dg.riemann_solver(
            dg.node_pos[1, element],
            get_node_vars(u, dg, polydeg(dg)+1, element-1), # u_L
            get_node_vars(u, dg, 1, element), # u_R
            dg.equation)

        for s in 1:nvariables(dg)
            dg.flux_buffer[s, element] = f_star_left[s]
        end
    end

    return nothing
end


function calc_boundary_flux!(u::AbstractArray{<:Any,3}, dg)
    boundary_condition = dg.boundary_condition

    flux_left = dg.riemann_solver(
        dg.node_pos[1, 1],
        boundary_condition(u, dg.xspan[1], dg), # u_L
        get_node_vars(u, dg, 1, 1), # u_R
        dg.equation)

    flux_right = dg.riemann_solver(
        dg.node_pos[polydeg(dg)+1, nelements(dg)],
        get_node_vars(u, dg, polydeg(dg)+1, nelements(dg)), # u_L
        boundary_condition(u, dg.xspan[2], dg), # u_R
        dg.equation)

    for s in 1:nvariables(dg)
        dg.flux_buffer[s, 1] = flux_left[s]
        dg.flux_buffer[s, nelements(dg)+1] = flux_right[s]
    end

    return nothing
end


function calc_surface_integral!(du::AbstractArray{<:Any,3}, u, dg)
    # -M_(-1)B*(f*_s(u_1,..,u_N+1)-f_s(u_1,..,u_N+1))
    #
    #      | 1/w_1   0   .    .     .      0   | | -1   0   .   .   0 |
    #      |  0    1/w_2                   .   | |  0   0           . |
    #  = - |  .      0   .                 .   | |  .       .       . | (f*_s(u_1,..,u_N+1)-f_s(u_1,..,u_N+1)), mit s={1,...,S}
    #      |  .               .            .   | |  .               . |
    #      |  .                   1/w_N    0   | |  .           0   0 |
    #      |  0      .   .    .     0   1/w_N+1| |  0   .   .   0   1 |
    for element in 1:nelements(dg)
        f_vec_left = calcflux(get_node_vars(u, dg, 1, element), dg.equation)
        f_vec_right = calcflux(get_node_vars(u, dg, polydeg(dg)+1, element), dg.equation)

        for s in 1:nvariables(dg)
            du[s, 1, element] += (dg.flux_buffer[s, element] - f_vec_left[s]) / dg.weights[1]
            du[s, end, element] -= (dg.flux_buffer[s, element+1] - f_vec_right[s]) / dg.weights[end]
        end
    end

    return nothing
end


function calc_correction_term!(du, u, t, dg::DG_1D{S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B, :ecDG}) where {S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B}
    for element in 1:nelements(dg)
        h = get_var_nodes(u, dg, 1, element)
        hv = get_var_nodes(u, dg, 2, element)
        v = hv ./ h
        hv2 = hv .* v

        s1 = 0.5 * (-dg.D * hv2 + hv .* dg.D * v + v .* dg.D * hv)

        s2 = 0.5 * dg.equation.g * (-dg.D * h.^2 + 2 * h .* dg.D * h)

        for node in 1:polydeg(dg)+1
            du[2, node, element] -= s1[node] + s2[node]
        end
    end

    return nothing
end

function calc_correction_term!(du, u, t, dg::DG_1D{S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B, :esDG}) where {S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B}
    for element in 1:nelements(dg)
        h = get_var_nodes(u, dg, 1, element)
        hv = get_var_nodes(u, dg, 2, element)
        v = hv ./ h
        hv2 = hv .* v

        s1 = 0.5 * (-dg.D * hv2 + hv .* dg.D * v + v .* dg.D * hv)

        s2 = 0.5 * dg.equation.g * (-dg.D * h.^2 + 2 * h .* dg.D * h)

        for node in 1:polydeg(dg)+1
            du[2, node, element] -= s1[node] + s2[node]
        end
    end

    return nothing
end


function calc_correction_term!(du, u, t, ::DG_1D{S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B, :DG}) where {S, N, NQ, EQ, SRC, R, X, T, NP, W, FB, B}
    return nothing
end


function calc_shallow_water_source_term!(du, u, t, dg)
    return nothing
end


function calc_shallow_water_source_term!(du, u, t, dg::DG_1D{S, N, NQ, EquationShallowWater{E1, E2}, SRC, R, X, T, NP, W, FB, B, METHOD}) where {S, N, NQ, SRC, R, X, T, NP, W, FB, B, METHOD, E1, E2}
    for element in 1:nelements(dg)
        b_vec = dg.equation.b.(get_node_pos_for_element(dg, element))
        Db = dg.D * b_vec
        ghDb = dg.equation.g * get_var_nodes(u, dg, 1, element) .* Db

        for node in 1:polydeg(dg)+1
            du[2, node, element] -= ghDb[node]
        end
    end

    return nothing
end


function calc_source_term!(du, u, t, dg)
    for element in 1:nelements(dg)
        for node in 1:polydeg(dg)+1
            source = dg.source(get_node_vars(u, dg, node, element), dg.node_pos[node, element], t)

            for s in 1:nvariables(dg)
                du[s, node, element] += source[s]
            end
        end
    end

    return nothing
end


function calc_source_term!(du, u, t, dg::DG_1D{S, N, NQ, EQ, Nothing, R, X, T, NP, W, FB, METHOD}) where {S, N, NQ, EQ, R, X, T, NP, W, FB, METHOD}
    return nothing
end


upwind_a(x, uL, uR, equation) = equation.a > 0 ? calcflux(uL, equation) : calcflux(uR, equation)
# Only works for one variable and the flux function must have a minimum. f*(uL, uR) = max(f(max(uL, minimum)), f(min(uR, minimum)))
godunov_scalar_minimum(x, uL, uR, equation, minimum) = max(calcflux(max(uL[1], minimum), equation), calcflux(min(uR[1], minimum), equation))
rusanov(x, uL, uR, equation) = 0.5 * (calcflux(uL, equation) + calcflux(uR, equation) -
        max(max_abs_eigenvalue(uL, equation), max_abs_eigenvalue(uR, equation)) * (uR - uL))

function entropy_flux(x, uL, uR, equation)
    mean(aL, aR) = 0.5 * (aL + aR)
    v_mean = mean(uL[2] / uL[1], uR[2] / uR[1])
    h_mean = mean(uL[1], uR[1])
    h2_mean = mean(uL[1]^2, uR[1]^2)

    return @SVector [v_mean * h_mean, v_mean^2 * h_mean + 0.5 * equation.g * h2_mean]
end


function entropy_shock_flux(x, uL, uR, equation)
    entropy_deriv(x, u, equation) = @SVector[-0.5 * u[2]^2 / u[1]^2 + equation.g * u[1] + equation.g * equation.b(x), u[2] / u[1]]

    u_avg = 0.5 * (uL + uR)
    λ = @SVector[u_avg[2] / u_avg[1] - sqrt(u_avg[1] * equation.g), u_avg[2] / u_avg[1] + sqrt(u_avg[1] * equation.g)]
    R = @SMatrix[1 1; λ[1] λ[2]]
    v = abs.(λ)
    q_vec = entropy_deriv(x, uR, equation) - entropy_deriv(x, uL, equation)

    return entropy_flux(x, uL, uR, equation) - 0.25 / equation.g * (R * (v .* R') * q_vec)
end


function boundary_condition_periodic(u::AbstractArray{<:Any,3}, x, dg::DG)
    if x == dg.xspan[1]
        return get_node_vars(u, dg, polydeg(dg)+1, nelements(dg))
    elseif x == dg.xspan[2]
        return get_node_vars(u, dg, 1, 1)
    end

    error("This should not happen!")
end


# Interpolate dg nodes to equidistant nodes for plotting
function dg_to_equidistant(u, n_out_per_cell, dg::DG_1D)
    nodes_in, _ = legendre_gauss_lobatto_nodes_weights(polydeg(dg))
    vandermonde = vandermonde_matrix(nodes_in, n_out_per_cell)

    result = zeros(nvariables(dg), n_out_per_cell * dg.Nq)

    for element in 1:dg.Nq
        for i in 1:n_out_per_cell
            index = (element-1) * n_out_per_cell + i

            for ii in 1:polydeg(dg)+1
                for s in 1:nvariables(dg)
                    result[s, index] += vandermonde[i, ii] * u[s, ii, element]
                end
            end
        end
    end

    return result
end


function dg_to_plot_data(u, dg::DG_1D; n_out=1000)
    n_nodes_in = polydeg(dg) + 1
    n_out_per_cell = convert(Integer, ceil(n_out / nelements(dg)))
    n_out = n_out_per_cell * nelements(dg)

    data = dg_to_equidistant(u, n_out_per_cell, dg)
    dx = (dg.xspan[2] - dg.xspan[1]) / n_out
    x = range(dg.xspan[1] + dx/2, dg.xspan[2] - dx/2, step=dx)

    return x, data
end


# Plot dg solution (optionally against exact solution)
function plot_dg(u, dg::DG_1D; n_out=1000, f_exact=nothing, latex=false)
    x, data = dg_to_plot_data(u, dg; n_out=n_out)

    kwargs = Dict{Symbol, Any}(
        :xlabel => "x",
        :ylabel => "u(x)"
    )
    if latex
        pgfplotsx()
        kwargs[:tex_output_standalone] = true
        kwargs[:xlabel] = "\$x\$"
        kwargs[:ylabel] = "\$u(x)\$"
    end

    if f_exact !== nothing
        exact = similar(data)
        for i in eachindex(x)
            exact[:, i] = f_exact(x[i])
        end

        for s in 1:nvariables(dg)
            display(plot(x, hcat(exact[s, :], data[s, :]), label=["Exakte Lösung" "Approximation"],
            title="u$s für N=$(polydeg(dg)) und Nq=$(nelements(dg))"; kwargs...))
        end
    else
        for s in 1:nvariables(dg)
            display(plot(x, data[s, :], label="u$s"; kwargs...))
        end
    end

    return nothing
end
