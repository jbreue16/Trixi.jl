function rhs!(du, u, t,
              mesh::CurvedMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
    @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # solve the auxilary system for viscous flows
    if typeof(equations) == CompressibleEulerEquations2D{Float64} && equations.viscous
        #  q is an array with one dimension more than u: q = [∂u/∂x] [∂u/∂y]
        @unpack q1 = cache
        @unpack q2 = cache
        q1 .= zero(eltype(q1))
        q2 .= zero(eltype(q2))
        @timeit_debug timer() "auxiliary system" calc_nabla_u!(q1,
        q2, u, t, mesh, Trixi.AuxiliaryEquation(),
        boundary_conditions, # hier noch unstetige boundaries schwierig
        dg, cache)
    end

  # Calculate volume integral
    @timeit_debug timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Calculate interface fluxes
    @timeit_debug timer() "interface flux" calc_interface_flux!(
    cache, u, mesh, equations, dg)

  # Calculate boundary fluxes
    @timeit_debug timer() "boundary flux" calc_boundary_flux!(
    cache, u, t, boundary_conditions, mesh, equations, dg)

  # Calculate surface integrals
    @timeit_debug timer() "surface integral" calc_surface_integral!(
    du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
    @timeit_debug timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
    @timeit_debug timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

    return nothing
end

# calculation of the auxilary system q = ∇u for viscous flows
function calc_nabla_u!(q1, q2, u, t,
    mesh::CurvedMesh{2}, equations,
    boundary_conditions,
    dg::DG, cache)
    # use the standard weak form DGSEM to calculate ∇u
    # Calculate volume integral
    calc_volume_integral_auxiliary!(q1, q2, u,
        mesh::CurvedMesh{2}, equations::AuxiliaryEquation,
        dg::DG, cache)

    # calculate surface integral
    calc_surface_integral_auxiliary!(q1, q2, u, t, 
        mesh::CurvedMesh{2}, equations::AuxiliaryEquation,
        dg::DG, cache,
        boundary_conditions)

  # Apply Jacobian from mapping to reference element
    apply_jacobian!(
        q1, mesh, equations, dg, cache)
    apply_jacobian!(
        q2, mesh, equations, dg, cache)

    return nothing
end

function calc_surface_integral_auxiliary!(q1, q2, u, t,
                                          mesh::CurvedMesh{2}, equations::AuxiliaryEquation,
                                          dg::DG, cache,
                                          boundary_conditions)
# values are stored in surface_flux_values and calculated with u; then copied to q
    @unpack elements = cache
# calc surface integral for q1
    @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2)"

    # Interfaces for u_x (1 dispatch) and in x-direction (`orientation` = 1)
        calc_interface_flux_auxiliary!(1, elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, mesh, equations, dg, cache)

    # Interfaces Interfaces for u_x (1 dispatch) and in y-direction (`orientation` = 2)
        calc_interface_flux_auxiliary!(1, elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, mesh, equations, dg, cache)
    end
# boundary flux with nabla = 1 dispatch for u_x
    calc_boundary_flux_auxiliary!(cache, u, t, boundary_conditions, mesh, equations, dg, 1)
# add the surface integral to q1
    calc_surface_integral!(q1, mesh, equations, dg, cache)

# calc surface integral for q2. The previously calculated surface fluxes are overwritten.
    @threaded for element in eachelement(dg, cache)
        # Interfaces in negative directions
        # Faster version of "for orientation in (1, 2)"
    
        # Interfaces Interfaces for u_y (2 dispatch) and in x-direction (`orientation` = 1)
        calc_interface_flux_auxiliary!(2, elements.surface_flux_values,
                             elements.left_neighbors[1, element],
                             element, 1, u, mesh, equations, dg, cache)
    
        # Interfaces Interfaces for u_y (2 dispatch) and in y-direction (`orientation` = 2)
        calc_interface_flux_auxiliary!(2, elements.surface_flux_values,
                             elements.left_neighbors[2, element],
                             element, 2, u, mesh, equations, dg, cache)
    end
    # boundary flux with nabla = 2 dispatch for u_y
    calc_boundary_flux_auxiliary!(cache, u, t, boundary_conditions, mesh, equations, dg, 2)
    # add the surface integral to q2
    calc_surface_integral!(q2, mesh, equations, dg, cache)

    return nothing
end

function calc_volume_integral_auxiliary!(q1, q2, u,
                                        mesh::CurvedMesh{2}, equations::AuxiliaryEquation,
                                        dg::DGSEM, cache)
    @unpack derivative_dhat = dg.basis
    @unpack contravariant_vectors = cache.elements

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # flux(u) = - u !
            u_node = - get_node_vars(u, equations, dg, i, j, element)
            
      # Compute the contravariant by taking the scalar product of the
      # first contravariant vector Ja^1 and u
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
            contravariant_u_x1 = Ja11 * u_node
            contravariant_u_y1 = Ja12 * u_node
            for ii in eachnode(dg)
                integral_contribution1 = derivative_dhat[ii, i] * contravariant_u_x1
                integral_contribution2 = derivative_dhat[ii, i] * contravariant_u_y1
                add_to_node_vars!(q1, integral_contribution1, equations, dg, ii, j, element)
                add_to_node_vars!(q2, integral_contribution2, equations, dg, ii, j, element)
            end

      # Compute the contravariant by taking the scalar product of the
      # second contravariant vector Ja^2 and u
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
            contravariant_u_x2 = Ja21 * u_node
            contravariant_u_y2 = Ja22 * u_node

            for jj in eachnode(dg)
                integral_contribution1 = derivative_dhat[jj, j] * contravariant_u_x2
                integral_contribution2 = derivative_dhat[jj, j] * contravariant_u_y2
                add_to_node_vars!(q1, integral_contribution1, equations, dg, i, jj, element)
                add_to_node_vars!(q2, integral_contribution2, equations, dg, i, jj, element)
            end
        end
    end
    return nothing
end

function calc_volume_integral!(du, u,
                               mesh::CurvedMesh{2},
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
    @unpack derivative_dhat = dg.basis
    @unpack contravariant_vectors = cache.elements

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            flux1 = flux(u_node, 1, equations)
            flux2 = flux(u_node, 2, equations)

      # Compute the contravariant flux by taking the scalar product of the
      # first contravariant vector Ja^1 and the flux vector
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
            contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2

            for ii in eachnode(dg)
                integral_contribution = derivative_dhat[ii, i] * contravariant_flux1
                add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
            end

      # Compute the contravariant flux by taking the scalar product of the
      # second contravariant vector Ja^2 and the flux vector
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
            contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2

            for jj in eachnode(dg)
                integral_contribution = derivative_dhat[jj, j] * contravariant_flux2
                add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
            end
        end
    end
    return nothing
end

##################BLZ: Volume Integral Pseudo Strong Form ##############
function calc_volume_integral!(du, u,
    mesh::CurvedMesh{2},
    nonconservative_terms::Val{false}, equations,
    volume_integral::VolumeIntegralPseudoStrongForm,
    dg::DGSEM, cache)
    @unpack derivative_matrix = dg.basis
    @unpack contravariant_vectors = cache.elements 
    @unpack weights = dg.basis
# Estimate the Pseudo strong Form D_pseudo = - M^-1 B + D
    S = zeros(nnodes(dg), nnodes(dg)) # actually becomes -S
    S[1, 1] += 1 / weights[1]
    S[end,end] -= 1 / weights[end] 
    D_pseudo = derivative_matrix + S

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            flux1 = flux(u_node, 1, equations)
            flux2 = flux(u_node, 2, equations)

# Compute the contravariant flux by taking the scalar product of the
# first contravariant vector Ja^1 and the flux vector
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
            contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2

            for ii in eachnode(dg)
                integral_contribution = D_pseudo[ii, i] * contravariant_flux1
                add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
            end

# Compute the contravariant flux by taking the scalar product of the
# second contravariant vector Ja^2 and the flux vector
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
            contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2

            for jj in eachnode(dg)
                integral_contribution = D_pseudo[jj, j] * contravariant_flux2
                add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
            end
        end
    end
    return nothing
end

##################BLZ: Volume Integral FLuxDifferencing based on PseudoStrong ############## 
function calc_volume_integral!(du, u,
    mesh::CurvedMesh{2},
    nonconservative_terms::Val{false}, equations,
    volume_integral::VolumeIntegralFluxDifferencing,
    dg::DGSEM, cache)

    if typeof(equations) == CompressibleEulerEquations2D{Float64} && equations.viscous
        calc_volume_integral_viscous!(du, u,
        mesh::CurvedMesh{2},
        nonconservative_terms::Val{false}, equations,
        volume_integral::VolumeIntegralFluxDifferencing,
        dg::DGSEM, cache)
    else
# separate and parallel computation for each element 
        @threaded for element in eachelement(dg, cache)
            @unpack contravariant_vectors = cache.elements
            @unpack inverse_weights = dg.basis
            @unpack derivative_matrix = dg.basis
        # actually -S from Pseudo Strong Form!!
            S = zeros(nnodes(dg), nnodes(dg))
            S[  1,   1] += inverse_weights[1]
            S[end, end] -= inverse_weights[end]

# Volume Integral for one Element, loop over all nodes
            for j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, j, element)
                flux1 = flux(u_node, 1, equations)
                flux2 = flux(u_node, 2, equations)
                
# x direction
                Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)

# two point flux of the same node outside the loop. And Pseudo Strong Part for boundary nodes
                if i == 1 || i == nnodes(dg)
                # Here we can also use the phys. flux.
                    contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
                    integral_contribution = (2 * derivative_matrix[i, i] + S[i, i]) *
                                        contravariant_flux1            
                    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
                else
                    contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
                    integral_contribution = 2 * derivative_matrix[i, i] * contravariant_flux1
                    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
                end

# two point flux symmetry: estimate the volume flux only one time
                for ii in (i + 1):nnodes(dg)
                # arihtmetic mean for two point volume flux metrics
                    Ja11_, Ja12_ = get_contravariant_vector(1, contravariant_vectors, ii, j, element)
                    u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
                    flux1 = volume_integral.volume_flux(u_node, u_node_ii, 1, equations)
                    flux2 = volume_integral.volume_flux(u_node, u_node_ii, 2, equations)
                    contravariant_flux1 = 0.5 * (Ja11 + Ja11_) * flux1 + 0.5 * (Ja12 + Ja12_) * flux2
                    integral_contribution = 2 * derivative_matrix[i, ii] * contravariant_flux1
                    add_to_node_vars!(du, integral_contribution, equations, dg, i,  j, element)
                    integral_contribution = 2 * derivative_matrix[ii, i] * contravariant_flux1
                    add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
                end
# y direction                
                flux1 = flux(u_node, 1, equations)
                flux2 = flux(u_node, 2, equations) 

                Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)

# two point flux of the same node outside the loop. And Pseudo Strong Part for boundary nodes
                if j == 1 || j == nnodes(dg)
                # Here we can also use the phys. flux.
                    contra_flux = Ja22 * flux2 + Ja21 * flux1    
                    integral_contribution = (2 * derivative_matrix[j, j] + S[j, j]) *
                                         contra_flux
                    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
                else
                    contravariant_flux2 = Ja22 * flux2 + Ja21 * flux1    
                    integral_contribution = 2 * derivative_matrix[j, j] * contravariant_flux2
                    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
                end
# two point flux symmetry: estimate the volume flux only one time
                for jj in (j + 1):nnodes(dg)
                # arithmetic mean for two point volume flux metrics
                    Ja21_, Ja22_ = get_contravariant_vector(2, contravariant_vectors, i, jj, element)
                    u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
                    flux1 = volume_integral.volume_flux(u_node, u_node_jj, 1, equations)
                    flux2 = volume_integral.volume_flux(u_node, u_node_jj, 2, equations)
                    contravariant_flux2 = 0.5 * (Ja22 + Ja22_) * flux2 + 0.5 * (Ja21 + Ja21_) * flux1
                    integral_contribution =  2 * derivative_matrix[j, jj] * contravariant_flux2
                    add_to_node_vars!(du, integral_contribution, equations, dg, i, j,  element)
                    integral_contribution =  2 * derivative_matrix[jj, j] * contravariant_flux2
                    add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
                end
            end
        end
    end
end

function calc_volume_integral_viscous!(du, u,
    mesh::CurvedMesh{2},
    nonconservative_terms::Val{false}, equations,
    volume_integral::VolumeIntegralFluxDifferencing,
    dg::DGSEM, cache)

    @unpack q1 = cache
    @unpack q2 = cache

# separate and parallel computation for each element 
    @threaded for element in eachelement(dg, cache)
        @unpack contravariant_vectors = cache.elements
        @unpack inverse_weights = dg.basis
        @unpack derivative_matrix = dg.basis
        # actually -S from Pseudo Strong Form!!
        S = zeros(nnodes(dg), nnodes(dg))
        S[  1,   1] += inverse_weights[1]
        S[end, end] -= inverse_weights[end]

# Volume Integral for one Element, loop over all nodes
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            q1_node = get_node_vars(q1, equations, dg, i, j, element)
            q2_node = get_node_vars(q2, equations, dg, i, j, element)

            flux1 = flux(u_node, 1, equations) + viscous_flux(u_node, q1_node, q2_node, 1, equations)
            flux2 = flux(u_node, 2, equations) + viscous_flux(u_node, q1_node, q2_node, 2, equations)

# x direction
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)

# two point flux of the same node outside the loop. And Pseudo Strong Part for boundary nodes
            if i == 1 || i == nnodes(dg)
                # Here we can also use the phys. flux.
                contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
                integral_contribution = (2 * derivative_matrix[i, i] + S[i, i]) *
                                        contravariant_flux1            
                add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
            else
                contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
                integral_contribution = 2 * derivative_matrix[i, i] * contravariant_flux1
                add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
            end

# two point flux symmetry: estimate the volume flux only one time
            for ii in (i + 1):nnodes(dg)
                # arihtmetic mean for two point volume flux metrics
                Ja11_, Ja12_ = get_contravariant_vector(1, contravariant_vectors, ii, j, element)
                u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
                q1_node_ii = get_node_vars(q1, equations, dg, ii, j, element)
                q2_node_ii = get_node_vars(q2, equations, dg, ii, j, element)

                flux1 = euler_and_viscous_flux(u_node, q1_node, q2_node, u_node_ii, q1_node_ii, q2_node_ii,
                                                1, equations, volume_integral.volume_flux)
                flux2 = euler_and_viscous_flux(u_node, q1_node, q2_node, u_node_ii, q1_node_ii, q2_node_ii,
                                                2, equations, volume_integral.volume_flux)
                contravariant_flux1 = 0.5 * (Ja11 + Ja11_) * flux1 + 0.5 * (Ja12 + Ja12_) * flux2
                integral_contribution = 2 * derivative_matrix[i, ii] * contravariant_flux1
                add_to_node_vars!(du, integral_contribution, equations, dg, i,  j, element)
                integral_contribution = 2 * derivative_matrix[ii, i] * contravariant_flux1
                add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
            end
# y direction                
            flux1 = flux(u_node, 1, equations) + viscous_flux(u_node, q1_node, q2_node, 1, equations)
            flux2 = flux(u_node, 2, equations) + viscous_flux(u_node, q1_node, q2_node, 2, equations)

            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)

# two point flux of the same node outside the loop. And Pseudo Strong Part for boundary nodes
            if j == 1 || j == nnodes(dg)
                # Here we can also use the phys. flux.
                contra_flux = Ja22 * flux2 + Ja21 * flux1    
                integral_contribution = (2 * derivative_matrix[j, j] + S[j, j]) *
                                         contra_flux
                add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
            else
                contravariant_flux2 = Ja22 * flux2 + Ja21 * flux1    
                integral_contribution = 2 * derivative_matrix[j, j] * contravariant_flux2
                add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
            end
# two point flux symmetry: estimate the volume flux only one time
            for jj in (j + 1):nnodes(dg)
                # arithmetic mean for two point volume flux metrics
                Ja21_, Ja22_ = get_contravariant_vector(2, contravariant_vectors, i, jj, element)
                u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
                q1_node_jj = get_node_vars(q1, equations, dg, i, jj, element)
                q2_node_jj = get_node_vars(q2, equations, dg, i, jj, element)

                flux1 = euler_and_viscous_flux(u_node, q1_node, q2_node, u_node_jj, q1_node_jj, q2_node_jj,
                                                    1, equations, volume_integral.volume_flux)
                flux2 = euler_and_viscous_flux(u_node, q1_node, q2_node, u_node_jj, q1_node_jj, q2_node_jj,
                                                    2, equations, volume_integral.volume_flux)
                contravariant_flux2 = 0.5 * (Ja22 + Ja22_) * flux2 + 0.5 * (Ja21 + Ja21_) * flux1
                integral_contribution =  2 * derivative_matrix[j, jj] * contravariant_flux2
                add_to_node_vars!(du, integral_contribution, equations, dg, i, j,  element)
                integral_contribution =  2 * derivative_matrix[jj, j] * contravariant_flux2
                add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
            end
        end
    end
end

function calc_interface_flux!(cache, u,
                              mesh::CurvedMesh{2},
                              equations, dg::DG)
    @unpack elements = cache

    @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2)"

    # Interfaces in x-direction (`orientation` = 1)
        calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, mesh, equations, dg, cache)

    # Interfaces in y-direction (`orientation` = 2)
        calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, mesh, equations, dg, cache)
    end

    return nothing
end

function calc_interface_flux_auxiliary!(nabla, cache, u,
    mesh::CurvedMesh{2},
    equations, dg::DG)
    @unpack elements = cache

    @threaded for element in eachelement(dg, cache)
# Interfaces in negative directions
# Faster version of "for orientation in (1, 2)"

# Interfaces in x-direction (`orientation` = 1)
        calc_interface_flux_auxiliary!(nabla, elements.surface_flux_values,
elements.left_neighbors[1, element],
element, 1, u, mesh, equations, dg, cache)

# Interfaces in y-direction (`orientation` = 2)
        calc_interface_flux_auxiliary!(nabla, elements.surface_flux_values,
elements.left_neighbors[2, element],
element, 2, u, mesh, equations, dg, cache)
    end

    return nothing
end

@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::CurvedMesh{2}, equations,
                                      dg::DG, cache)
  # This is slow for LSA, but for some reason faster for Euler (see #519)
    if left_element <= 0 # left_element = 0 at boundaries
        return nothing
    end

    @unpack surface_flux = dg
    @unpack contravariant_vectors = cache.elements
    if typeof(equations) == AuxiliaryEquation # BLZ: to solve the auxiliary system
        surface_flux = flux_central_auxiliary 
    elseif typeof(equations) == CompressibleEulerEquations2D{Float64} && equations.viscous # BLZ: to add the central flux for viscous terms 
        calc_interface_flux_viscous!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh, equations,
                                      dg, cache)
    end 
    right_direction = 2 * orientation
    left_direction = right_direction - 1

    for i in eachnode(dg)
        if orientation == 1
            u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
            u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

      # First contravariant vector Ja^1 as SVector
            normal_vector = get_contravariant_vector(1, contravariant_vectors, 1, i, right_element)
        else # orientation == 2
            u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
            u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

      # Second contravariant vector Ja^2 as SVector
            normal_vector = get_contravariant_vector(2, contravariant_vectors, i, 1, right_element)
        end

        flux = surface_flux(u_ll, u_rr, normal_vector, equations)

        for v in eachvariable(equations)
            surface_flux_values[v, i, right_direction, left_element] = flux[v]
            surface_flux_values[v, i, left_direction, right_element] = flux[v]
        end
    end

    return nothing
end

@inline function calc_interface_flux_viscous!(surface_flux_values, left_element, right_element,
    orientation, u,
    mesh::CurvedMesh{2}, equations,
    dg::DG, cache)
# This is slow for LSA, but for some reason faster for Euler (see #519)
if left_element <= 0 # left_element = 0 at boundaries
return nothing
end

@unpack surface_flux = dg
@unpack contravariant_vectors = cache.elements

right_direction = 2 * orientation
left_direction = right_direction - 1

for i in eachnode(dg)
if orientation == 1
u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

# First contravariant vector Ja^1 as SVector
normal_vector = get_contravariant_vector(1, contravariant_vectors, 1, i, right_element)
else # orientation == 2
u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

# Second contravariant vector Ja^2 as SVector
normal_vector = get_contravariant_vector(2, contravariant_vectors, i, 1, right_element)
end

flux = surface_flux_viscous(u_ll, u_rr, normal_vector, equations)

for v in eachvariable(equations)
surface_flux_values[v, i, right_direction, left_element] = flux[v]
surface_flux_values[v, i, left_direction, right_element] = flux[v]
end
end

return nothing
end

@inline function calc_interface_flux_auxiliary!(nabla, surface_flux_values, left_element, right_element,
    orientation, u,
    mesh::CurvedMesh{2}, equations,
    dg::DG, cache)
# This is slow for LSA, but for some reason faster for Euler (see #519)
    if left_element <= 0 # left_element = 0 at boundaries
        return nothing
    end

    @unpack contravariant_vectors = cache.elements
    surface_flux = flux_central_auxiliary 

    right_direction = 2 * orientation
    left_direction = right_direction - 1

    for i in eachnode(dg)
        if orientation == 1
            u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
            u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

# First contravariant vector Ja^1 as SVector
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, 1, i, right_element)
# derivative in only one direction; y_η for u_x, -x_η for u_y, the other one is 0
            if nabla == 1 normal_vector = SVector(Ja11, 0) else normal_vector = SVector(0, Ja12) end
        else # orientation == 2
            u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
            u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

# Second contravariant vector Ja^2 as SVector
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, 1, right_element)
# derivative in only one direction; -y_ξ for u_x, x_ξ for u_y, the other one is 0
            if nabla == 2 normal_vector = SVector(Ja22, 0) else normal_vector = normal_vector = SVector(0, Ja21) end
        end

        flux = surface_flux(u_ll, u_rr, normal_vector, equations)

        for v in eachvariable(equations)
            surface_flux_values[v, i, right_direction, left_element] = flux[v]
            surface_flux_values[v, i, left_direction, right_element] = flux[v]
        end
    end

    return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::CurvedMesh{2}, equations, dg::DG)
    @assert isperiodic(mesh)
end
function calc_boundary_flux_auxiliary!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
    mesh::CurvedMesh{2}, equations, dg::DG, nabla)
    @assert isperiodic(mesh)
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             mesh::CurvedMesh{2}, equations, dg::DG)
    calc_boundary_flux!(cache, u, t,
                      (boundary_condition, boundary_condition,
                       boundary_condition, boundary_condition),
                      mesh, equations, dg)
end
function calc_boundary_flux_auxiliary!(cache, u, t, boundary_condition,
    mesh::CurvedMesh{2}, equations, dg::DG, nabla)
    calc_boundary_flux_auxiliary!(cache, u, t,
(boundary_condition, boundary_condition,
boundary_condition, boundary_condition),
mesh, equations, dg, nabla)
end


function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             mesh::CurvedMesh{2}, equations, dg::DG)
    @unpack surface_flux = dg
    if typeof(equations) == AuxiliaryEquation surface_flux = flux_central end
    @unpack surface_flux_values = cache.elements
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2)
    # Negative x-direction
        direction = 1
        element = linear_indices[begin, cell_y]

        for j in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (1, j), (j,), element)
        end

    # Positive x-direction
        direction = 2
        element = linear_indices[end, cell_y]

        for j in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (nnodes(dg), j), (j,), element)
        end
    end

    for cell_x in axes(mesh, 1)
    # Negative y-direction
        direction = 3
        element = linear_indices[cell_x, begin]

        for i in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (i, 1), (i,), element)
        end

    # Positive y-direction
        direction = 4
        element = linear_indices[cell_x, end]

        for i in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (i, nnodes(dg)), (i,), element)
        end
    end
end
function calc_boundary_flux_auxiliary!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
    mesh::CurvedMesh{2}, equations, dg::DG, nabla)
    @unpack surface_flux = dg
    if typeof(equations) == AuxiliaryEquation surface_flux = flux_central end
    @unpack surface_flux_values = cache.elements
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2)
# Negative x-direction
        direction = 1
        element = linear_indices[begin, cell_y]

        for j in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
              boundary_conditions[direction],
              mesh, equations, dg, cache,
              direction, (1, j), (j,), element, nabla)
        end

# Positive x-direction
        direction = 2
        element = linear_indices[end, cell_y]

        for j in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
              boundary_conditions[direction],
              mesh, equations, dg, cache,
              direction, (nnodes(dg), j), (j,), element, nabla)
        end
    end

    for cell_x in axes(mesh, 1)
# Negative y-direction
        direction = 3
        element = linear_indices[cell_x, begin]

        for i in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
              boundary_conditions[direction],
              mesh, equations, dg, cache,
              direction, (i, 1), (i,), element, nabla)
        end

# Positive y-direction
        direction = 4
        element = linear_indices[cell_x, end]

        for i in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
              boundary_conditions[direction],
              mesh, equations, dg, cache,
              direction, (i, nnodes(dg)), (i,), element, nabla)
        end
    end
end

# BLZ modification neccessary for viscous equation, original function is in dg.jl
@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t, orientation,
    boundary_condition,
    mesh::CurvedMesh, equations, dg::DG, cache,
    direction, node_indices, surface_node_indices, element)
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack surface_flux = dg
    if typeof(equations) == AuxiliaryEquation  surface_flux = flux_central end # BLZ

    u_inner = get_node_vars(u, equations, dg, node_indices..., element)
    x = get_node_coords(node_coordinates, equations, dg, node_indices..., element)
  
  # Contravariant vector Ja^i is the normal vector
    normal = get_contravariant_vector(orientation, contravariant_vectors, node_indices..., element)

    flux = boundary_condition(u_inner, normal, direction, x, t, surface_flux, equations)
  
    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
    end
end
@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t, orientation,
    boundary_condition,
    mesh::CurvedMesh, equations, dg::DG, cache,
    direction, node_indices, surface_node_indices, element, nabla)
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack surface_flux = dg
    if typeof(equations) == AuxiliaryEquation  surface_flux = flux_central end # BLZ

    u_inner = get_node_vars(u, equations, dg, node_indices..., element)
    x = get_node_coords(node_coordinates, equations, dg, node_indices..., element)
  
  # Contravariant vector Ja^i is the normal vector
    normal = get_contravariant_vector(orientation, contravariant_vectors, node_indices..., element)

  # derivative in only one direction, depending on orientation; 
    # orientation = 2 : -y_ξ for u_x, x_ξ for u_y, the other one is 0
    # orientation = 1 : y_η for u_x, -x_η for u_y, the other one is 0
    if orientation == 2 
        if nabla == 2 
            normal_vector = SVector(normal[2], 0)
        else normal_vector = normal_vector = SVector(0, normal[1])
        end
    else # orientation == 1
        if nabla == 1 
            normal_vector = SVector(normal[1], 0) 
        else normal_vector = SVector(0, normal[2])
        end
    end
    flux = boundary_condition(u_inner, normal_vector, direction, x, t, surface_flux, equations)
  
    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
    end
end

  
function apply_jacobian!(du,
                         mesh::Union{CurvedMesh{2},UnstructuredQuadMesh},
                         equations, dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            factor = -inverse_jacobian[i, j, element]

            for v in eachvariable(equations)
                du[v, i, j, element] *= factor
            end
        end
    end

    return nothing
end
