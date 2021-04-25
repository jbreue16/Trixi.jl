struct DG_2D{S, N, NQ, EQ, SRC, R, X, T, W, E, B, I} <: DG{S, N}
    Nq::NQ # Number of elements
    equation::EQ # equation
    source::SRC
    riemann_solver::R # Riemann solver
    xspan::X
    D::T # Derivation matrix for the corresponding nodes
    weights::W # Quadrature weights
    elements::E
    boundary_condition::B
    is_solid::I

    function DG_2D(N, Nq, S, equation, riemann_solver; xspan=(0, 1), 
                   boundary_condition=boundary_condition_periodic, source=nothing, is_solid=(x,y,Nq)->false)
        if N == 0
            nodes, weights = legendre_gauss_nodes_weights(N)
        else
            nodes, weights = legendre_gauss_lobatto_nodes_weights(N)
        end
        elements = init_elements(N, Nq, S, xspan, nodes, is_solid)

        D_temp = derivation_matrix(InterpolationPolynomial(nodes))
        D = SMatrix{size(D_temp, 1), size(D_temp, 2)}(D_temp)

        new{S, N, typeof(Nq), typeof(equation), typeof(source), typeof(riemann_solver), typeof(xspan),
            typeof(D), typeof(weights), typeof(elements), typeof(boundary_condition), typeof(is_solid)}(
            Nq, equation, source, riemann_solver, xspan, D, weights, elements, boundary_condition, is_solid)
    end
end


struct ElementContainer{N, I}
    node_pos::N # node_pos[element_x, element_y, node_x, node_y]
    interfaces::I # interface[element_x, element_y, orientation], 1 = left, 2 = right, 3 = bottom, 4 = top

    ElementContainer(node_pos, interfaces) = new{typeof(node_pos), typeof(interfaces)}(node_pos, interfaces)
end


struct Interface{S}
    is_boundary::Bool
    flux_buffer::Vector{SVector{S, Float64}}

    function Interface(is_boundary, S, N)
        flux_buffer = Vector{SVector{S, Float64}}(undef, N+1)

        new{S}(is_boundary, flux_buffer)
    end
end


function init_elements(N, Nq, S, xspan, nodes, is_solid)
    node_pos = Array{SVector{2, Float64}}(undef, Nq, Nq, N+1, N+1)
    Δx = (xspan[2] - xspan[1]) / Nq # TODO yspan, Nq x/y

    interfaces = Array{Interface{S}, 3}(undef, Nq, Nq, 4)

    for element_x in 1:Nq, element_y in 1:Nq
        # Calculate node positions
        for node_x in 1:N+1
            # Transform to [0, 1]
            pos_x = nodes[node_x] * 0.5 + 0.5

            for node_y in 1:N+1
                # Transform to [0, 1]
                pos_y = nodes[node_y] * 0.5 + 0.5
                # Global position of node
                node_pos_x = xspan[1] + (element_x - 1) * Δx + pos_x * Δx
                node_pos_y = xspan[1] + (element_y - 1) * Δx + pos_y * Δx

                node_pos[element_x, element_y, node_x, node_y] = @SVector [node_pos_x, node_pos_y]
            end
        end

        # Calculate interface data
        # Look to the left
        is_boundary = element_x == 1 || is_solid(element_x, element_y, Nq) || is_solid(element_x-1, element_y, Nq)
        interface = Interface(is_boundary, S, N)
        interfaces[element_x, element_y, 1] = interface
        if element_x != 1
            interfaces[element_x-1, element_y, 2] = interface
        end

        # Look down
        is_boundary = element_y == 1 || is_solid(element_x, element_y, Nq) || is_solid(element_x, element_y-1, Nq)
        interface = Interface(is_boundary, S, N)
        interfaces[element_x, element_y, 3] = interface
        if element_y != 1
            interfaces[element_x, element_y-1, 4] = interface
        end
    end

    for element in 1:Nq
        # Right boundary
        is_boundary = true
        interface = Interface(is_boundary, S, N)
        interfaces[Nq, element, 2] = interface

        # Upper boundary
        is_boundary = true
        interface = Interface(is_boundary, S, N)
        interfaces[element, Nq, 4] = interface
    end

    return ElementContainer(node_pos, interfaces)
end


# Compute timestep using Δt = CFL * Δx / λ_max for LGL nodes
function timestep(u, CFL, dg::DG_2D)
    # Compute Δx_eff as the smallest distance between interpolation nodes.
    if polydeg(dg) == 0
        # Δx_eff = Δx
        Δx_eff = (dg.xspan[2] - dg.xspan[1]) / dg.Nq
    else
        # Smallest distance is between the first two nodes for LGL nodes
        Δx_eff = dg.elements.node_pos[1, 1, 2, 1][1] - dg.elements.node_pos[1, 1, 1, 1][1] # TODO different Δx and Δy
    end

    # Compute maximum eigenvalue over all values of u
    λ_max = 0.0
    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        if !dg.is_solid(element_x, element_y, dg.Nq)
            for node_x in 1:polydeg(dg)+1, node_y in 1:polydeg(dg)+1
                λ = max_abs_eigenvalue(get_node_vars(u, dg, element_x, element_y, node_x, node_y), dg.equation)
                if λ > λ_max
                    λ_max = λ
                end
            end
        end
    end

    if !isfinite(λ_max) || λ_max ≈ 0
        @info "" λ_max
        error("Something went horribly wrong!")
    end

    return CFL * Δx_eff / λ_max
end


function discretize_initial_condition(dg::DG_2D, initial_condition)
    u = Array{Float64}(undef, nvariables(dg), dg.Nq, dg.Nq, polydeg(dg)+1, polydeg(dg)+1)

    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        if dg.is_solid(element_x, element_y, dg.Nq)
            u[:, element_x, element_y, :, :] .= 0.0
        else
            for node_x in 1:polydeg(dg)+1, node_y in 1:polydeg(dg)+1
                node_vars = initial_condition(dg.elements.node_pos[element_x, element_y, node_x, node_y])

                for s in 1:nvariables(dg)
                    u[s, element_x, element_y, node_x, node_y] = node_vars[s]
                end
            end
        end
    end

    return u
end


function calc_inverse_jacobian!(du::AbstractArray{<:Any,5}, dg)
    Δx = (dg.xspan[2] - dg.xspan[1]) / dg.Nq
    J_inv = 2 / Δx
    @. du *= J_inv
end


# Naive implementation for better readability and comparison.
# This implementation is about 20 times slower.
function calc_volume_integral_naive!(du::AbstractArray{<:Any,5}, u, dg)
    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        if !dg.is_solid(element_x, element_y, dg.Nq)
            for s in 1:nvariables(dg)
                flux1 = [calcflux(u[:, element_x, element_y, node_x, node_y], 1, dg.equation)[s] for node_x in 1:polydeg(dg)+1, node_y in 1:polydeg(dg)+1]
                flux2 = [calcflux(u[:, element_x, element_y, node_x, node_y], 2, dg.equation)[s] for node_x in 1:polydeg(dg)+1, node_y in 1:polydeg(dg)+1]

                du[s, element_x, element_y, :, :] -= dg.D * flux1
                du[s, element_x, element_y, :, :] -= flux2 * dg.D'
            end
        end
    end

    return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,5}, u, dg)
    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        if !dg.is_solid(element_x, element_y, dg.Nq)
            for node_y in 1:polydeg(dg)+1, node_x in 1:polydeg(dg)+1
                for node in 1:polydeg(dg)+1
                    flux1 = calcflux(get_node_vars(u, dg, element_x, element_y, node, node_y), 1, dg.equation)
                    flux2 = calcflux(get_node_vars(u, dg, element_x, element_y, node_x, node), 2, dg.equation)

                    for s in 1:nvariables(dg)
                        du[s, element_x, element_y, node_x, node_y] -= dg.D[node_x, node] * flux1[s]
                        du[s, element_x, element_y, node_x, node_y] -= dg.D[node_y, node] * flux2[s]
                    end
                end
            end
        end
    end

    return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,5}, dg)
    # Right and upper boundary are always boundary interfaces
    # Those are computed in calc_boundary_flux!
    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        if !dg.is_solid(element_x, element_y, dg.Nq)
            # Look to the left
            interface = dg.elements.interfaces[element_x, element_y, 1]
            if !interface.is_boundary
                for node in 1:polydeg(dg)+1
                    u_L = get_node_vars(u, dg, element_x-1, element_y, polydeg(dg)+1, node)
                    u_R = get_node_vars(u, dg, element_x, element_y, 1, node)

                    interface.flux_buffer[node] = dg.riemann_solver(u_L, u_R, 1, dg.equation)
                end
            end

            # Look down
            interface = dg.elements.interfaces[element_x, element_y, 3]
            if !interface.is_boundary
                for node in 1:polydeg(dg)+1
                    u_L = get_node_vars(u, dg, element_x, element_y-1, node, polydeg(dg)+1)
                    u_R = get_node_vars(u, dg, element_x, element_y, node, 1)

                    interface.flux_buffer[node] = dg.riemann_solver(u_L, u_R, 2, dg.equation)
                end
            end
        end
    end

    return nothing
end


function calc_boundary_flux!(u::AbstractArray{<:Any,5}, dg)
    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        # Look to the left
        interface = dg.elements.interfaces[element_x, element_y, 1]
        
        if interface.is_boundary
            for node in 1:polydeg(dg)+1
                if element_x == 1
                    u_L = dg.boundary_condition(u, dg.elements.node_pos[element_x, element_y, 1, node], 
                                                element_x, element_y, 1, node, 1, dg)
                    u_R = get_node_vars(u, dg, element_x, element_y, 1, node)
                elseif dg.is_solid(element_x, element_y, dg.Nq)
                    u_L = get_node_vars(u, dg, element_x-1, element_y, polydeg(dg)+1, node)
                    u_R = @SVector [-u_L[1], u_L[2], u_L[3]] # TODO hardcoded boundary condition
                elseif dg.is_solid(element_x-1, element_y, dg.Nq)
                    u_R = get_node_vars(u, dg, element_x, element_y, 1, node)
                    u_L = @SVector [-u_R[1], u_R[2], u_R[3]] # TODO hardcoded boundary condition
                end
                
                interface.flux_buffer[node] = dg.riemann_solver(u_L, u_R, 1, dg.equation)
            end
        end

        # Look down
        interface = dg.elements.interfaces[element_x, element_y, 3]
        if interface.is_boundary
            for node in 1:polydeg(dg)+1
                if element_y == 1
                    u_L = dg.boundary_condition(u, dg.elements.node_pos[element_x, element_y, node, 1], 
                                                element_x, element_y, node, 1, 3, dg)
                    u_R = get_node_vars(u, dg, element_x, element_y, node, 1)
                elseif dg.is_solid(element_x, element_y, dg.Nq)
                    u_L = get_node_vars(u, dg, element_x, element_y-1, node, polydeg(dg)+1)
                    u_R = @SVector [u_L[1], -u_L[2], u_L[3]] # TODO hardcoded boundary condition
                elseif dg.is_solid(element_x, element_y-1, dg.Nq)
                    u_R = get_node_vars(u, dg, element_x, element_y, node, 1)
                    u_L = @SVector [u_R[1], -u_R[2], u_R[3]] # TODO hardcoded boundary condition
                else
                    error("Something went horribly wrong!")
                end

                interface.flux_buffer[node] = dg.riemann_solver(u_L, u_R, 2, dg.equation)
            end
        end
    end

    for element in 1:dg.Nq
        # Right boundary
        for node in 1:polydeg(dg)+1
            interface = dg.elements.interfaces[dg.Nq, element, 2]
            u_L = get_node_vars(u, dg, dg.Nq, element, polydeg(dg)+1, node)
            u_R = dg.boundary_condition(u, dg.elements.node_pos[dg.Nq, element, polydeg(dg)+1, node], 
                                        dg.Nq, element, polydeg(dg)+1, node, 2, dg)

            interface.flux_buffer[node] = dg.riemann_solver(u_L, u_R, 1, dg.equation)
        end

        # Upper boundary
        for node in 1:polydeg(dg)+1
            interface = dg.elements.interfaces[element, dg.Nq, 4]
            u_L = get_node_vars(u, dg, element, dg.Nq, node, polydeg(dg)+1)
            u_R = dg.boundary_condition(u, dg.elements.node_pos[element, dg.Nq, node, polydeg(dg)+1], 
                                        element, dg.Nq, node, polydeg(dg)+1, 4, dg)

            interface.flux_buffer[node] = dg.riemann_solver(u_L, u_R, 2, dg.equation)
        end
    end

    return nothing
end


function calc_surface_integral!(du::AbstractArray{<:Any,5}, u, dg)
    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        if !dg.is_solid(element_x, element_y, dg.Nq)
            # x-direction
            for node in 1:polydeg(dg)+1
                f_interface_left = dg.elements.interfaces[element_x, element_y, 1].flux_buffer[node]
                f_vec_left = calcflux(get_node_vars(u, dg, element_x, element_y, 1, node), 1, dg.equation)

                f_interface_right = dg.elements.interfaces[element_x, element_y, 2].flux_buffer[node]
                f_vec_right = calcflux(get_node_vars(u, dg, element_x, element_y, polydeg(dg)+1, node), 1, dg.equation)

                for s in 1:nvariables(dg)
                    du[s, element_x, element_y, 1, node] += (f_interface_left[s] - f_vec_left[s]) / dg.weights[1]
                    du[s, element_x, element_y, end, node] -= (f_interface_right[s] - f_vec_right[s]) / dg.weights[end]
                end
            end

            # y-direction
            for node in 1:polydeg(dg)+1
                f_interface_left = dg.elements.interfaces[element_x, element_y, 3].flux_buffer[node]
                f_vec_left = calcflux(get_node_vars(u, dg, element_x, element_y, node, 1), 2, dg.equation)

                f_interface_right = dg.elements.interfaces[element_x, element_y, 4].flux_buffer[node]
                f_vec_right = calcflux(get_node_vars(u, dg, element_x, element_y, node, polydeg(dg)+1), 2, dg.equation)

                for s in 1:nvariables(dg)
                    du[s, element_x, element_y, node, 1] += (f_interface_left[s] - f_vec_left[s]) / dg.weights[1]
                    du[s, element_x, element_y, node, end] -= (f_interface_right[s] - f_vec_right[s]) / dg.weights[end]
                end
            end
        end
    end

    return nothing
end


function calc_source_term!(du, u, t, dg::DG_2D{S, N, NQ, EQ, Nothing, R, X, T, W, E, B, I}) where {S, N, NQ, EQ, R, X, T, W, E, B, I}
    return nothing
end


rusanov(uL, uR, orientation::Real, equation) = 0.5 * (calcflux(uL, orientation, equation) + calcflux(uR, orientation, equation) -
        max(max_abs_eigenvalue(uL, equation), max_abs_eigenvalue(uR, equation)) * (uR - uL))


function boundary_condition_periodic(u::AbstractArray{<:Any,5}, pos, element_x, element_y, node_x, node_y, orientation, dg::DG)
    if orientation == 1 # Left
        return get_node_vars(u, dg, dg.Nq, element_y, polydeg(dg)+1, node_y)
    elseif orientation == 2 # Right
        return get_node_vars(u, dg, 1, element_y, 1, node_y)
    elseif orientation == 3 # Down
        return get_node_vars(u, dg, element_x, dg.Nq, node_x, polydeg(dg)+1)
    elseif orientation == 4 # Up
        return get_node_vars(u, dg, element_x, 1, node_x, 1)
    end

    error("Invalid orientation!")
end


# Interpolate dg nodes to equidistant nodes for plotting
function dg_to_equidistant(u, n_out_per_cell, dg::DG_2D)
    nodes_in, _ = legendre_gauss_lobatto_nodes_weights(polydeg(dg))
    vandermonde = vandermonde_matrix(nodes_in, n_out_per_cell)

    result = zeros(nvariables(dg), n_out_per_cell * dg.Nq, n_out_per_cell * dg.Nq)

    for element_x in 1:dg.Nq, element_y in 1:dg.Nq
        for i in 1:n_out_per_cell, j in 1:n_out_per_cell
            x_index = (element_x-1) * n_out_per_cell + i
            y_index = (element_y-1) * n_out_per_cell + j
            
            for ii in 1:polydeg(dg)+1, jj in 1:polydeg(dg)+1
                for s in 1:nvariables(dg)
                    result[s, x_index, y_index] += vandermonde[i, ii] * vandermonde[j, jj] * u[s, element_x, element_y, ii, jj]
                end
            end
        end
    end

    return result
end


# Returns x, y and interpolated data for plotting
function dg_to_plot_data(u, dg::DG_2D; n_out=1000)
    n_nodes_in = polydeg(dg) + 1
    n_out_per_cell = convert(Integer, ceil(n_out / dg.Nq))
    n_out = n_out_per_cell * dg.Nq

    data = dg_to_equidistant(u, n_out_per_cell, dg)
    dx = (dg.xspan[2] - dg.xspan[1]) / n_out
    x = range(dg.xspan[1] + dx/2, dg.xspan[2] - dx/2, step=dx)
    y = range(dg.xspan[1] + dx/2, dg.xspan[2] - dx/2, step=dx)

    return x, y, data
end