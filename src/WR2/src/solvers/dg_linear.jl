function semidiscretization_matrix(a, Δx, N, Nq; method=:lobatto)
    B, nodes, weights = B_matrix_nodes_weights(N, Val(method))

    M = Diagonal(weights)
    p = InterpolationPolynomial(nodes)
    D = derivation_matrix(p)
    Minv_D_M = M \ (D' * M)
    min_Minv_B = -M \ B

    # Block matrizes
    # A contains blocks of M^-1 D' M (volume integral)
    A = spzeros(Nq * (N+1), Nq * (N+1))
    # MB contains blocks of -M^-1 B (surface integral)
    MB = spzeros(Nq * (N+1), Nq * (N+1))
    # u_matrix * u will give us a block vector where each block is u* for the
    # corresponding cell
    u_matrix = spzeros(Nq * (N+1), Nq * (N+1))

    if method == :gauss
        # l_i(1) with gauss nodes to compute u_matrix with gauss,
        # because we need to interpolate to get the boundary values
        l = B[:, end]
    end

    for i in 1:Nq
        # (pos+1, pos+1) will be the index of the element in the upper left corner
        # of the current block
        pos = (i-1) * (N+1)

        A[pos .+ (1:N+1), pos .+ (1:N+1)] .= Minv_D_M # D

        if method == :lobatto
            #            A 0 0 0 B           0 . . . 0          0 . . 0 1
            #            B A 0 0 0           . .     .          . .     0
            # u_matrix = 0 B A 0 0  with A = .   .   .  and B = .   .   .
            #            0 0 B A 0           .     . 0          .     . .
            #            0 0 0 B A           0 . . 0 1          0 . . . 0
            #
            u_matrix[pos+N+1, pos+N+1] = 1
            if i == Nq
                u_matrix[1, end] = 1
            else
                u_matrix[pos+N+2, pos+N+1] = 1
            end
        elseif method == :gauss
            #            A 0 0 0 B           0 . . . 0          - - l - -
            #            B A 0 0 0           .       .          0 . . . 0
            # u_matrix = 0 B A 0 0  with A = .       .  and B = .       .
            #            0 0 B A 0           0 . . . 0          .       .
            #            0 0 0 B A           - - l - -          0 . . . 0
            #
            u_matrix[pos+N+1, pos .+ (1:N+1)] .= l
            if i == Nq
                u_matrix[1, pos .+ (1:N+1)] .= l
            else
                u_matrix[pos+N+2, pos .+ (1:N+1)] .= l
            end
        end

        MB[pos .+ (1:N+1), pos .+ (1:N+1)] .= min_Minv_B
    end

    return 2*a/Δx * (MB * u_matrix + A) # A
end


function B_matrix_nodes_weights(N, ::Val{:lobatto})
    nodes, weights = legendre_gauss_lobatto_nodes_weights(N)
    B = spzeros(N+1, N+1)
    B[1,1] = -1
    B[N+1,N+1] = 1

    return B, nodes, weights
end


function B_matrix_nodes_weights(N, ::Val{:gauss})
    nodes, weights = legendre_gauss_nodes_weights(N)
    bar_weights = barycentric_weights(nodes)
    B = spzeros(N+1, N+1)
    p1 = 1.0
    p2 = 1.0
    for i in 1:N+1
        p1 *= -1 - nodes[i]
        p2 *= 1 - nodes[i]
    end

    for i in 1:N+1
        B[i, 1] = -bar_weights[i] * p1 / (-1 - nodes[i])
        B[i, end] = bar_weights[i] * p2 / (1 - nodes[i])
    end

    return B, nodes, weights
end


function semidiscretize(a, N, Nq, u0::T; CFL=1.4, xspan=(0, 1), tspan=(0, 1), method=:lobatto) where T
    Δx = (xspan[2] - xspan[1]) / Nq
    A = semidiscretization_matrix(a, Δx, N, Nq; method=method)

    dg_nodes = compute_dg_nodes(Δx, N, Nq, method)
    dt = timestep(CFL, dg_nodes, a, xspan[1])

    return ODEProblemLinear(A, u0.(dg_nodes), tspan; dt=dt)
end


function compute_dg_nodes(Δx, N, Nq, method)
    nodes = quadrature_nodes(N, Val(method))

    n = N+1
    dg_nodes = Vector{eltype(nodes)}(undef, Nq * n)
    for i in 0:Nq-1
        dg_nodes[i*n+1:i*n+n] = (nodes .+ 1) * 0.5 * Δx .+ i*Δx
    end

    return dg_nodes
end


# Compute timestep using Δt = CFL * Δx / a
function timestep(CFL, dg_nodes, a, left_boundary)
    # Compute Δx_eff as the smallest distance between interpolation nodes.
    if length(dg_nodes) > 1
        # Smallest distance is between the first two nodes
        Δx_eff = dg_nodes[2] - dg_nodes[1]

        if !isapprox(dg_nodes[1], left_boundary)
            # For gauss, the smallest distance can be between the first node
            # and the last node of the preceding cell
            Δx_eff = min(Δx_eff, 2 * (dg_nodes[1] - left_boundary))
        end
    else
        Δx_eff = (dg_nodes[1] - left_boundary) * 2
    end

    return CFL * Δx_eff / a
end


# Interpolate dg nodes to equidistant nodes for plotting
function dg_to_equidistant(dg_data, N_in, n_out_per_cell; method=:lobatto)
    n_nodes_in = N_in + 1
    Nq = div(length(dg_data), n_nodes_in)

    nodes_in = quadrature_nodes(N_in, Val(method))
    vandermonde = vandermonde_matrix(nodes_in, n_out_per_cell)

    result = Vector{eltype(dg_data)}(undef, n_out_per_cell * Nq)

    for i in 1:Nq
        pos_in = (i-1) * n_nodes_in
        pos_out = (i-1) * n_out_per_cell
        result[pos_out .+ (1:n_out_per_cell)] = vandermonde * dg_data[pos_in .+ (1:n_nodes_in)]
    end

    return result
end


# Plot dg solution (optionally against exact solution)
function plot_dg(dg_data, N_in; n_out=100, xspan=(-1.0, 1.0), f_exact=nothing, method=:lobatto, latex=false)
    n_nodes_in = N_in + 1
    Nq = div(length(dg_data), n_nodes_in)
    n_out_per_cell = convert(Integer, ceil(n_out / Nq))
    n_out = n_out_per_cell * Nq

    data = dg_to_equidistant(dg_data, N_in, n_out_per_cell; method=method)
    dx = (xspan[2] - xspan[1]) / n_out
    x = range(xspan[1] + dx/2, xspan[2] - dx/2, step=dx)

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
        p = plot(x, [f_exact.(x), data], label=["Exakte Lösung" "Approximation"]; kwargs...)
    else
        p = plot(x, data, label="Approximation"; kwargs...)
    end

    return p
end
