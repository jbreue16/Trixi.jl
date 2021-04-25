struct InterpolationPolynomial{T}
    nodes::Vector{T}
    weights::Vector{T}

    InterpolationPolynomial(nodes) = (weights = barycentric_weights(nodes);
            new{eltype(nodes)}(nodes, weights))
end


function (p::InterpolationPolynomial)(x, f)
    n = length(p.nodes)

    @assert length(f) == n "length of f needs to be equal to the number of nodes"

    result = findfirst(y -> y == x, p.nodes)
    # If x is in p.nodes
    if result !== nothing
        return f[result]
    end

    quotient = @. p.weights / (x - p.nodes)

    numerator = 0
    denominator = 0

    for j in 1:n
        numerator += f[j] * quotient[j]
        denominator += quotient[j]
    end

    return numerator / denominator
end


function barycentric_weights(nodes) 
    n = length(nodes)

    weights = ones(n)
    for j in 1:n, i in 1:n
        if i != j
            weights[j] *= nodes[j] - nodes[i]
        end
    end

    return 1 ./ weights
end


function vandermonde_matrix(nodes_in, n_out)
    n_nodes_in = length(nodes_in)
    dx = 2 / n_out
    nodes_out = range(-1 + dx/2, 1 - dx/2, step=dx)
    weights = barycentric_weights(nodes_in)

    V = zeros(n_out, n_nodes_in)

    for i in 1:n_out
        match = false
        for j in 1:n_nodes_in
            if isapprox(nodes_in[j], nodes_out[i], rtol=eps())
                match = true
                V[i, j] = 1.0
            end
        end

        if !match
            p = 1.0
            for j in 1:n_nodes_in
                p *= nodes_out[i] - nodes_in[j]
            end

            for j in 1:n_nodes_in
                V[i, j] = weights[j] * p / (nodes_out[i] - nodes_in[j])
            end
        end
    end

    return V
end


function derivation_matrix(p::InterpolationPolynomial{T}) where T
    N = length(p.nodes);
    D = zeros(N,N);

    @inbounds for i in 1:N, j in 1:N
        if i==j
            sum = 0.0
            for k in 1:N
                if k != i
                    sum += p.weights[k] / (p.nodes[i] - p.nodes[k])
                end
            end
            D[i,j] = -sum / p.weights[i]
        else
            D[i,j] = p.weights[j] / (p.weights[i] * (p.nodes[i] - p.nodes[j]))
        end
    end

    return D
end