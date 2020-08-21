
include("containers.jl")
include("math.jl")


# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer

# Initialize top-level parameters structure for program-wide parameters
const parameters = Dict{Symbol,Any}()


# Parse parameters file into global dict
function parse_parameters_file(filename)
  parameters[:default] = parsefile(filename)
  parameters[:default]["parameters_file"] = filename
end


# Return parameter by name, optionally taking a default value and a range of valid values.
#
# If no default value is specified, the parameter is required and the program
# stops if the parameter was not found. The range of valid parameters is used
# to restrict parameters to sane values.
function parameter(name, default=nothing; valid=nothing)
  if haskey(parameters[:default], name)
    # If parameter exists, use its value
    value = parameters[:default][name]
  else
    # Otherwise check whether a default is given and abort if not
    if default === nothing
      error("requested paramter '$name' does not exist and no default value was provided")
    else
      value = default
    end
  end

  # If a range of valid values has been specified, check parameter value against it
  if valid !== nothing
    if !(value in valid)
      error("'$value' is not a valid value for parameter '$name' (valid: $valid)")
    end
  end

  return value
end


"""
    setparameter(name::String, value)

Set parameter with the specified `name` to the specified `value`.
"""
function setparameter(name::String, value)
  parameters[:default][name] = value
end

# Return true if parameter exists.
parameter_exists(name::String) = haskey(parameters[:default], name)


# Parse command line arguments and return as dict
function parse_commandline_arguments(args=ARGS; testing=false)
  # Copy arguments such that we can modify them without changing the function argument
  myargs = copy(args)

  # Initialize dictionary with parsed arguments
  parsed = Dict{String, Any}()

  # Verbose if disabled by default
  parsed["verbose"] = false

  # The output was bravely copied and pasted by using the ArgParse settings below
  while !isempty(myargs)
    current = popfirst!(myargs)
    if current in ("-h", "--help")
      println("""
              usage: trixi [-v] [-h] parameters_file

              positional arguments:
                parameters_file  Name of file with runtime parameters.

              optional arguments:
                -v, --verbose    Enable verbose output, which might help with
                                debugging.
                -h, --help       show this help message and exit
                """)
      if testing
        return 1
      else
        exit(0)
      end
    elseif current in ("-v", "--verbose")
      # Enable verbose output
      parsed["verbose"] = true
    elseif startswith(current, "-")
      # Unknown option
      println(stderr, """
              unrecognized option $current
              usage: trixi [-v] parameters_file
              """)
      if testing
        return 2
      else
        exit(1)
      end
    else
      # Must be non-option argument -> parameters file
      # If a parameters file was already given, throw error
      if haskey(parsed, "parameters_file")
        println(stderr, """
                too many arguments
                usage: trixi [-v] parameters_file
                """)
        if testing
          return 3
        else
          exit(1)
        end
      end

      # Otherwise store parameters file
      parsed["parameters_file"] = current
    end
  end

  # Error if no parameters file was given
  if !haskey(parsed, "parameters_file")
    println(stderr, """
            required argument parameters_file was not provided
            usage: trixi [-v] parameters_file
            """)
    if testing
      return 4
    else
      exit(1)
    end
  end

  return parsed
end


# Allow an expression to be terminated gracefully by Ctrl-c.
#
# On Unix-like operating systems, gracefully handle user interrupts (SIGINT), also known as
# Ctrl-c, while evaluation expression `ex`.
macro interruptable(ex)
  @static Sys.isunix() && quote
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

    try
      # Try to run code
      $(esc(ex))
    catch e
      # Only catch interrupt exceptions and end with a nice message
      isa(e, InterruptException) || rethrow(e)
      println(stderr, "\nExecution interrupted by user (Ctrl-c)")
    end

    # Disable interrupt handling again
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 1)
  end
end


# Print informative message at startup
function print_startup_message()
  s = """

    ████████╗██████╗ ██╗██╗  ██╗██╗
    ╚══██╔══╝██╔══██╗██║╚██╗██╔╝██║
       ██║   ██████╔╝██║ ╚███╔╝ ██║
       ██║   ██╔══██╗██║ ██╔██╗ ██║
       ██║   ██║  ██║██║██╔╝ ██╗██║
       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
    """
  println(s)
end


"""
    get_name(x)

Returns a name of `x` ready for pretty printing.
By default, return `string(y)` if `x isa Val{y}` and return `string(x)` otherwise.

# Examples

```jldoctest
julia> Trixi.get_name("test")
"test"

julia> Trixi.get_name(Val(:test))
"test"
```
"""
get_name(x) = string(x)
get_name(::Val{x}) where x = string(x)
