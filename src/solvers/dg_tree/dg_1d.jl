
# everything related to a DG semidiscretization in 1D,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::TreeMesh{1}, equations,
                      dg::DG, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  cache = (; elements, interfaces, boundaries)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
  cache = (;cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

  return cache
end


# The methods below are specialized on the volume integral type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::TreeMesh{1}, equations,
                      volume_integral::VolumeIntegralFluxDifferencing, dg::DG, uEltype)
  create_cache(mesh, have_nonconservative_terms(equations), equations, volume_integral, dg, uEltype)
end

function create_cache(mesh::TreeMesh{1}, nonconservative_terms::Val{false}, equations,
                      ::VolumeIntegralFluxDifferencing, dg, uEltype)
  NamedTuple()
end

# TODO: MHD in 1D
# function create_cache(mesh::TreeMesh{1}, nonconservative_terms::Val{true}, equations,
#                       ::VolumeIntegralFluxDifferencing, dg, uEltype)
# end


function create_cache(mesh::TreeMesh{1}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG, dg::DG, uEltype)
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  cache = create_cache(mesh, equations,
                       VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                       dg, uEltype)

  A2dp1_x = Array{uEltype, 2}
  fstar1_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; cache..., element_ids_dg, element_ids_dgfv, fstar1_threaded)
end


function create_cache(mesh::TreeMesh{1}, equations,
                      volume_integral::VolumeIntegralPureLGLFiniteVolume, dg::DG, uEltype)

  A2dp1_x = Array{uEltype, 2}
  fstar1_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; fstar1_threaded)
end



# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::TreeMesh{1}, equations, mortar_l2::LobattoLegendreMortarL2, uEltype)
  NamedTuple()
end


# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?

function rhs!(du, u, t,
              mesh::TreeMesh{1}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg, cache)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(
    cache, u, mesh, equations, dg)

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations, dg)

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


function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, CurvedMesh{1}},
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis

  @threaded for element in eachelement(dg, cache)
    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)

      flux1 = flux(u_node, 1, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, element)
      end
    end
  end

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::TreeMesh{1},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
  @threaded for element in eachelement(dg, cache)
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_integral.volume_flux, dg, cache, element)
  end
end

@inline function split_form_kernel!(du::AbstractArray{<:Any,3}, u,
                                    nonconservative_terms::Val{false}, equations,
                                    volume_flux, dg::DGSEM, cache,
                                    element, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis

  # Calculate volume integral in one element
  for i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, element)

    # x direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux1 = flux(u_node, 1, equations)
    integral_contribution = alpha * derivative_split[i, i] * flux1
    add_to_node_vars!(du, integral_contribution, equations, dg, i, element)
    # use symmetry of the volume flux for the remaining terms
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      integral_contribution = alpha * derivative_split[i, ii] * flux1
      add_to_node_vars!(du, integral_contribution, equations, dg, i,  element)
      integral_contribution = alpha * derivative_split[ii, i] * flux1
      add_to_node_vars!(du, integral_contribution, equations, dg, ii, element)
    end
  end
end


# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{1},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
  @unpack element_ids_dg, element_ids_dgfv = cache
  @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  alpha = @timeit_debug timer() "blending factors" indicator(u, equations, dg, cache)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

  # Loop over pure DG elements
  @timeit_debug timer() "pure DG" @threaded for element in element_ids_dg
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_flux_dg, dg, cache, element)
  end

  # Loop over blended DG-FV elements
  @timeit_debug timer() "blended DG-FV" @threaded for element in element_ids_dgfv
    alpha_element = alpha[element]

    # Calculate DG volume integral contribution
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_flux_dg, dg, cache, element, 1 - alpha_element)

    # Calculate FV volume integral contribution
    fv_kernel!(du, u, equations, volume_flux_fv, dg, cache, element, alpha_element)
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{1},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolume,
                               dg::DGSEM, cache)
  @unpack volume_flux_fv = volume_integral

  # Calculate LGL FV volume integral
  @threaded for element in eachelement(dg, cache)
    fv_kernel!(du, u, equations, volume_flux_fv, dg, cache, element, true)
  end

  return nothing
end



@inline function fv_kernel!(du::AbstractArray{<:Any,3}, u::AbstractArray{<:Any,3},
                            equations, volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
  @unpack fstar1_threaded = cache
  @unpack inverse_weights = dg.basis

  # Calculate FV two-point fluxes
  fstar1 = fstar1_threaded[Threads.threadid()]
  calcflux_fv!(fstar1, u, equations, volume_flux_fv, dg, element)

  # Calculate FV volume integral contribution
  for i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, element] += ( alpha *
                             (inverse_weights[i] * (fstar1[v, i+1] - fstar1[v, i])) )

    end
  end

  return nothing
end

@inline function calcflux_fv!(fstar1, u::AbstractArray{<:Any,3},
                              equations, volume_flux_fv, dg::DGSEM, element)

  fstar1[:, 1,           ] .= zero(eltype(fstar1))
  fstar1[:, nnodes(dg)+1,] .= zero(eltype(fstar1))

  for i in 2:nnodes(dg)
    u_ll = get_node_vars(u, equations, dg, i-1, element)
    u_rr = get_node_vars(u, equations, dg, i,   element)
    flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
    set_node_vars!(fstar1, flux, equations, dg, i)
  end

  return nothing
end


function prolong2interfaces!(cache, u,
                             mesh::TreeMesh{1}, equations, dg::DG)
  @unpack interfaces = cache

  @threaded for interface in eachinterface(dg, cache)
    left_element  = interfaces.neighbor_ids[1, interface]
    right_element = interfaces.neighbor_ids[2, interface]

    # interface in x-direction
    for v in eachvariable(equations)
      interfaces.u[1, v, interface] = u[v, nnodes(dg), left_element]
      interfaces.u[2, v, interface] = u[v,          1, right_element]
    end
  end

  return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{1},
                              nonconservative_terms::Val{false}, equations,
                              dg::DG, cache)
  @unpack surface_flux = dg
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    # Call pointwise Riemann solver
    u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface)
    flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

    # Copy flux to left and right element storage
    for v in eachvariable(equations)
      surface_flux_values[v, left_direction,  left_id]  = flux[v]
      surface_flux_values[v, right_direction, right_id] = flux[v]
    end
  end
end

# TODO: MHD in 1D
# function calc_interface_flux!(surface_flux_values, mesh::TreeMesh{1},
#                               nonconservative_terms::Val{true}, equations,
#                               dg::DG, cache)
# end


function prolong2boundaries!(cache, u,
                             mesh::TreeMesh{1}, equations, dg::DG)
  @unpack boundaries = cache
  @unpack orientations, neighbor_sides = boundaries

  @threaded for boundary in eachboundary(dg, cache)
    element = boundaries.neighbor_ids[boundary]

    # boundary in x-direction
    if neighbor_sides[boundary] == 1
      # element in -x direction of boundary
      for v in eachvariable(equations)
        boundaries.u[1, v, boundary] = u[v, nnodes(dg), element]
      end
    else # Element in +x direction of boundary
      for v in eachvariable(equations)
        boundaries.u[2, v, boundary] = u[v, 1,          element]
      end
    end
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::TreeMesh{1}, equations, dg::DG)
  @assert isempty(eachboundary(dg, cache))
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition,
                             mesh::TreeMesh{1}, equations, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  for direction in eachindex(firsts)
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_condition,
                                     equations, dg, cache,
                                     direction, firsts[direction], lasts[direction])
  end
end

function calc_boundary_flux!(cache, t, boundary_conditions::Union{NamedTuple,Tuple},
                             mesh::TreeMesh{1}, equations, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                   equations, dg, cache, 1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                   equations, dg, cache, 2, firsts[2], lasts[2])
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,3}, t,
                                          boundary_condition, equations, dg::DG, cache,
                                          direction, first_boundary, last_boundary)
  @unpack surface_flux = dg
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    # Get boundary flux
    u_ll, u_rr = get_surface_node_vars(u, equations, dg, boundary)
    if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
      u_inner = u_ll
    else # Element is on the right, boundary on the left
      u_inner = u_rr
    end
    x = get_node_coords(node_coordinates, equations, dg, boundary)
    flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
                              equations)

    # Copy flux to left and right element storage
    for v in eachvariable(equations)
      surface_flux_values[v, direction, neighbor] = flux[v]
    end
  end

  return nothing
end


function calc_surface_integral!(du, mesh::Union{TreeMesh{1}, CurvedMesh{1}},
                                equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  @threaded for element in eachelement(dg, cache)
    for v in eachvariable(equations)
      # surface at -x
      du[v, 1,          element] -= surface_flux_values[v, 1, element] * boundary_interpolation[1,          1]
      # surface at +x
      du[v, nnodes(dg), element] += surface_flux_values[v, 2, element] * boundary_interpolation[nnodes(dg), 2]
    end
  end

  return nothing
end


function apply_jacobian!(du, mesh::Union{TreeMesh{1}, CurvedMesh{1}},
                         equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    factor = -cache.elements.inverse_jacobian[element]

    for i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, element] *= factor
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_sources!(du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{1}, dg::DG, cache)
  return nothing
end

function calc_sources!(du, u, t, source_terms,
                       equations::AbstractEquations{1}, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element)
      x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, element)
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, element)
    end
  end

  return nothing
end
