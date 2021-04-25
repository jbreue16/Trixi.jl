function legendre_polynomial_derivative(N, x)
    if N == 0
        LN = 1.0
        LNabl = 0.0
    elseif N == 1
        LN = x
        LNabl = 1.0
    else
        LN2 = 1.0
        LN1 = x
        LN2abl = 0.0
        LN1abl = 1.0

        for k in 2:N
            LN = (2*k-1)/k * x * LN1 - (k-1)/k * LN2
            LNabl = LN2abl + (2*k-1) * LN1
            LN2 = LN1
            LN1 = LN
            LN2abl = LN1abl
            LN1abl = LNabl
        end
    end

    return LN, LNabl
end


function legendre_gauss_nodes_weights(N)
    TOL = 4 * eps()
    x = zeros(N+1)
    w = zeros(N+1)
    n_iter = 10
    if N == 0
        x[1] = 0
        w[1] = 2
    elseif N == 1
        x[1] = -sqrt(1/3)
        w[1] = 1
        x[2] = -x[1]
        w[2] = w[1]
    else
        l = div(N + 1, 2) - 1
        for j in 0:l
            x[j+1] = -cos((2 * j + 1)/(2 * N + 2) * π)
            for k in 0:n_iter
                (LN1, LN1abl) = legendre_polynomial_derivative(N+1, x[j+1])
                Δ = -LN1/LN1abl
                x[j+1] = x[j+1] + Δ
                if abs(Δ) <= TOL * abs(x[j+1])
                    break
                end
            end
            (LN1, LN1abl) = legendre_polynomial_derivative(N+1, x[j+1])
            x[N-j+1] = -x[j+1]
            w[j+1] = 2/((1 - x[j+1]^2) * LN1abl^2)
            w[N-j+1] = w[j+1]
        end
    end
    if mod(N, 2) == 0
        (LN1, LN1abl) = legendre_polynomial_derivative(N+1, 0.0)
        x[div(N,2) + 1] = 0.0
        w[div(N,2) + 1] = 2/LN1abl^2
    end

    return x, w
end


function q_and_L_evaluation(N, x)
    LN2 = 1.0
    LN1 = convert(Float64, x)
    LN2abl = 0.0
    LN1abl = 1.0
    LN = 0.0
    for k in 2:N
        LN = (2*k-1)/k * x * LN1 - (k-1)/k * LN2
        LNabl = LN2abl + (2*k-1) * LN1
        LN2 = LN1
        LN1 = LN
        LN2abl = LN1abl
        LN1abl = LNabl
    end
    k = N + 1
    L_k = (2*k-1)/k * x * LN - (k-1)/k * LN2
    L_k_abl = LN2abl + (2*k-1) * LN1
    q = L_k - LN2
    qabl = L_k_abl - LN2abl

    return q, qabl, LN
end

function legendre_gauss_lobatto_nodes_weights(N)
    x = zeros(N+1)
    w = zeros(N+1)
    n_iter = 10
    TOL = 4 * eps()
    if N == 1
        x[1] = -1.0
        w[1] = 1.0
        x[2] = 1.0
        w[2] = w[1]
    else
        x[1] = -1.0
        w[1] = 2/(N*(N+1))
        x[N+1] = 1.0
        w[N+1] = w[1]
        l = div(N+1, 2) - 1
        for j in 1:l
            x[j+1] = -cos(((j + 1/4) * π)/N - 3/(8 * N * π * (j + 1/4)))
            for k in 0:n_iter
                (q, qabl, LN) = q_and_L_evaluation(N, x[j+1])
                Δ = -q/qabl
                x[j+1] = x[j+1] + Δ
                if abs(Δ) <= TOL * abs(x[j+1])
                    break
                end
            end
            (q, qabl, LN) = q_and_L_evaluation(N, x[j+1])
            x[N-j+1] = -x[j+1]
            w[j+1] = 2/(N * (N+1) * LN^2)
            w[N-j+1] = w[j+1]
        end
    end
    if mod(N, 2) == 0
        (q, qabl, LN) = q_and_L_evaluation(N, 0.0)
        x[div(N,2) + 1] = 0.0
        w[div(N,2) + 1] = 2/(N * (N+1) * LN^2)
    end
    return x, w
end


function quadrature_nodes(N, ::Val{:lobatto})
    nodes, _ = legendre_gauss_lobatto_nodes_weights(N)
    
    return nodes
end


function quadrature_nodes(N, ::Val{:gauss})
    nodes, _ = legendre_gauss_nodes_weights(N)
    
    return nodes
end