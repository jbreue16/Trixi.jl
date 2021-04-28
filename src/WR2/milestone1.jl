

function initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    rho_v1 = 0.1
    rho_v2 = -0.2
    rho_e = 10.0
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end
  
  
  """
      initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)
  
  A smooth initial condition used for convergence tests in combination with
  [`source_terms_convergence_test`](@ref)
  (and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
  """
  function initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)
    c = 2
    A = 0.1
    L = 2
    f = 1/L
    ω = 2 * pi * f
    ini = c + A * sin(ω * (x[1] + x[2] - t))
  
    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2
  
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end
  
  """
      source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations2D)
  
  Source terms used for convergence tests in combination with
  [`initial_condition_convergence_test`](@ref)
  (and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
  """
  @inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations2D)
    # Same settings as in `initial_condition`
    c = 2
    A = 0.1
    L = 2
    f = 1/L
    ω = 2 * pi * f
    γ = equations.gamma
  
    x1, x2 = x
    si, co = sincos((x1 + x2 - t)*ω)
    tmp1 = co * A * ω
    tmp2 = si * A
    tmp3 = γ - 1
    tmp4 = (2*c - 1)*tmp3
    tmp5 = (2*tmp2*γ - 2*tmp2 + tmp4 + 1)*tmp1
    tmp6 = tmp2 + c
  
    du1 = tmp1
    du2 = tmp5
    du3 = tmp5
    du4 = 2*((tmp6 - 1)*tmp3 + tmp6*γ)*tmp1
  
    # Original terms (without performanc enhancements)
    # du1 = cos((x1 + x2 - t)*ω)*A*ω
    # du2 = (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
    #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
    # du3 = (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
    #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
    # du3 = 2*((c - 1 + sin((x1 + x2 - t)*ω)*A)*(γ - 1) +
    #                             (sin((x1 + x2 - t)*ω)*A + c)*γ)*cos((x1 + x2 - t)*ω)*A*ω
  
    return SVector(du1, du2, du3, du4)
  end
  
  """
      boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
                                          surface_flux_function,
                                          equations::CompressibleEulerEquations2D)
  
  Boundary conditions used for convergence tests in combination with
  [`initial_condition_convergence_test`](@ref) and [`source_terms_convergence_test`](@ref).
  """
  function boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_convergence_test(x, t, equations)
  
    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
      flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
  
    return flux
  end












