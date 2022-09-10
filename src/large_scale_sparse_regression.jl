# Code for large scale sparse regression model

## function that computes the tolerance for proximal evaluation for a given inner iteration number i

struct SettingsLSR

    # first comes description of different data type

    μ_max::Float64 # starting μ
    μ_min::Float64 # ending μ
    μ_mult_fact::Float64 # multiplicative factor to reduce μ
    n_iter_min::Int64 # minimum number of iterations per fixed μ
    n_iter_max::Int64 # maximum number of iterations per fixed μ
    big_M::Float64 # a big number, it suffices to hard code it, but for now let me keep it
    β::Float64 # β is the strong convexity parameter
    tol::Float64 # tolerance for the fixed point gap
    verbose::Bool # whether to print iteration information
    freq::Int64 # how often to print the iteration solution
    γ_updt_rule::Symbol # for now two rule: :safe and :adaptive, deafult will be :safe

    # constructor function
    function SettingsLSR(;
        μ_max = 1.0,
        μ_min = 1e-12,
        μ_mult_fact = 0.85,
        n_iter_min = 1000,
        n_iter_max = 1000,
        big_M = 1e99,
        β = 1e-10,
        tol = 1e-8,
        verbose = true,
        freq = 10,
        γ_updt_rule = :safe
        )
        new(μ_max, μ_min, μ_mult_fact, n_iter_min, n_iter_max, big_M, β, tol, verbose, freq, γ_updt_rule)
    end

end

##

function γ_from_μ_LSR(μ::R, settings::SettingsLSR) where {R <: Real}
    if settings.γ_updt_rule == :safe
        return abs(cbrt(settings.μ_min)) # cbrt also works quite well
    else
        @error "either select :adaptive or :safe or :adaptiveplus as your γ selection rule"
        return nothing
    end
end

methods(γ_from_μ_LSR)

##

function ϵProxGenerate(i::Int64; tolerance_type = :default)
    # @info "invoking ϵProxGenerate function"
    if tolerance_type == :default 
        return 1/(i+1)^1.5
    elseif tolerance_type == :constant
            return 1e-4       
    else
        @error "correct tolerance_type is not given"
    end
end

## Define the one shot function to describe the model
# ==================================================

# The function is f(x) = (λ/2) || A x - b || ^2, we set λ = 2
# its proximal operator
# y = prox_{γ f} (x) can be computed by solving the system
# G y = r, 
# where G = (λ*A'*A) + (1/γ)*I
# and    r = (1/γ)*x + (λ*A'*b), and we set h = (λ*A'*b)
# So, we can define the entire data structure of the Large Scale Sparse Regression Model by the matrices G and h

struct ProblemLargeSparseRegModel{A <: AbstractMatrix{ <: Real }, I <: Integer, R <: Real, V <: AbstractVector{ <:Real }}
    GMat::A # corresponds to A in ||Ax-b||^2
    hVec::V # corresponds to b in ||Ax-b||^2
    k::I # represents cardinality of x, ie card(x) <= k
    Γ::R # corresponds to ||x||_∞ <= Γ
    z0::V # one initial state provided by the user 
    PMat # Preconditioner matrix of GMat
end

## This structure contains all the necessary information to describe a state

mutable struct StateLargeSparseRegModel{R <: Real, V <: AbstractVector{ <: Real}, I <: Integer} # Done
    xPrev::V # the first iterate for the previous iteration (i-1)
    yPrev::V # the second iterate for the previous iteration (i-1)
    zPrev::V # the third iterate for the previous iteration (i-1)
    x::V # the first iterate for the current iteration (i)
    y::V # the second iterate for the current iteration (i)
    z::V # the third iterate for the current iteration (i)
    i::I # iteration number
    fxd_pnt_gap::R # fixed point gap for state
    fsblt_gap::R # how feasible the current point is, this is measured by || x- Π_C(x) ||
    μ::R # current value of μ
    γ::R # current value of γ
    ϵProx::R # tolerance for the proximal evaluation
end

    ## this function constructs the very first state for the problem and settings for the factor analysis model
    # ========================================================= 

function StateLargeSparseRegModel(problem::ProblemLargeSparseRegModel, settings::SettingsLSR)

    # this function constructs the very first state for the problem and settings
    z0 = copy(problem.z0)
    x0 = zero(z0)
    y0 = zero(z0)
    z0Prev = zero(z0)
    x0Prev = zero(z0)
    y0Prev = zero(z0)
    i = copy(settings.n_iter_max) # iteration number where best fixed point gap is reached
    fxd_pnt_gap = copy(settings.big_M)
    fsblt_gap = copy(settings.big_M)
    μ = copy(settings.μ_max)
    γ = abs(cbrt(settings.μ_min)) # γ_from_μ_LSR(μ, settings)
    ϵProx = ϵProxGenerate(i; tolerance_type = :constant)

    return StateLargeSparseRegModel(x0Prev, y0Prev, z0Prev, x0, y0, z0, i, fxd_pnt_gap, fsblt_gap, μ, γ, ϵProx)

end

## structure that describes the information to intialize our algorithm
# ====================================================================

mutable struct InitInfoLargeSparseRegModel{V <: AbstractVector{ <: Real}, R <: Real}

    # this will have z0, γ, μ
    z0::V # initial condition (vector part) to start an inner iteration
    μ::R # μ required start an inner iteration
    γ::R # γ required to start an inner iteration
    ϵProx::R # tolerance for the proximal evaluation

end

## Of course, we will need a constructor for the InitInfoLargeSparseRegModeltype, given the user input. The user inputs are problem::problem, settings::settings.
# =========================================================

function InitInfoLargeSparseRegModel(problem::ProblemLargeSparseRegModel, settings::SettingsLSR)

    ## constructs the initial NExOS information
    z0 = copy(problem.z0)
    μ = copy(settings.μ_max)
    γ = abs(cbrt(settings.μ_min)) # γ_from_μ_LSR(μ, settings)
    ϵProx = ϵProxGenerate(1; tolerance_type = :constant)# copy(settings.ϵ_prox)
    return InitInfoLargeSparseRegModel(z0, μ, γ, ϵProx)

end

## update the state for large scale sparse regression model
# =========================================================

function update_state_lsr!(state::StateLargeSparseRegModel, init_info::InitInfoLargeSparseRegModel, problem::ProblemLargeSparseRegModel, settings::SettingsLSR)

    # extract information from the state and init info
    xPrev = state.xPrev
    yPrev = state.yPrev
    zPrev = init_info.z0 
    x = state.x
    y = state.y
    z = init_info.z0 
    β = settings.β
    γ = init_info.γ
    μ = init_info.μ
    ϵProx = init_info.ϵProx
    k = problem.k
    Γ = problem.Γ

    # create best point variables
    best_point_xPrev = xPrev
    best_point_yPrev = yPrev
    best_point_zPrev = zPrev
    best_point_x = x
    best_point_y = y
    best_point_z = z
    best_fxd_pnt_gap = settings.big_M
    best_fsblt_gap = settings.big_M

    i = 1
    while i <= settings.n_iter_max
        xPrev, yPrev, zPrev, x, y, z = inner_iteration_lsr(xPrev, yPrev, zPrev, x, y, z, β, γ, μ, problem, ϵProx, i) 
        crnt_fxd_pnt_gap = norm(x-y, Inf)
        y_fsbl = Π_exact_lsr(k, Γ, y)
        crnt_fsblt_gap = norm(y-y_fsbl, Inf)

        # update the best points so far if we have lower objective value
        if crnt_fxd_pnt_gap <= best_fxd_pnt_gap
            best_point_xPrev = xPrev
            best_point_yPrev = yPrev
            best_point_zPrev = zPrev
            best_point_x = x
            best_point_y = y
            best_point_z = z
            best_fxd_pnt_gap = crnt_fxd_pnt_gap
            best_fsblt_gap = crnt_fsblt_gap
        end # if

        # display important information
        if settings.verbose == true
            if mod(i, settings.freq) == 0
                @info "μ = $μ | iteration = $i | log fixed point gap = $(log10(crnt_fxd_pnt_gap)) | log feasibility gap = $(log10(crnt_fsblt_gap))"
            end # if
        end # if

        # termination condition check
        if crnt_fxd_pnt_gap <= settings.tol
            if settings.verbose == true
                @info "μ = $μ | log fixed point gap reached $(log10(settings.tol))"
            end
            break
        end #if

        # increment the count by 1
        i = i+1
        # update ϵProx 
        ϵProx = ϵProxGenerate(i; tolerance_type = :constant) 

    end # while

    ## update the state
    state.xPrev = best_point_xPrev
    state.yPrev = best_point_yPrev 
    state.zPrev = best_point_zPrev
    state.x = best_point_x
    state.y = best_point_y
    state.z = best_point_z
    state.i = i
    state.fxd_pnt_gap = best_fxd_pnt_gap
    state.fsblt_gap = best_fsblt_gap
    state.μ = μ
    state.γ = γ
    state.ϵProx = ϵProxGenerate(i; tolerance_type = :constant) 

    # Print information about the best point
    if settings.verbose == true
        @info "information about the best state found for μ = $(state.μ)"
        @info "μ = $(state.μ) | log fixed point gap = $(log10(state.fxd_pnt_gap)) | log feasibility gap = $(log10(state.fsblt_gap)) | inner iterations = $(state.i)"
    end

    return state
end

## Inner iteration function LargeSparseRegModel
# =============================================


function inner_iteration_lsr(xPrev::V, yPrev::V, zPrev::V, x::V, y::V, z::V, β::R, γ::R, μ::R, problem::ProblemLargeSparseRegModel, ϵProx::R, i::I) where {V <: AbstractVector{ <: Real}, R <: Real, I <: Integer}
    # old iteration: x = prox_NExOS(problem.f, β, γ, z)
    xPrev = x
    yPrev = y
    zPrev = z
    x = prox_NExOS_lsr(problem.GMat, problem.hVec, γ, zPrev, xPrev, ϵProx, problem.PMat) # xPrev is used as the warm-starting point for the prox evaluation and ϵProx is the termination tolerance for cg!
    y = Π_NExOS_lsr(problem.k, problem.Γ, β, γ, μ, 2*x - z)
    z = z + y - x
    # println("size of the new matrices = ", size(Z), size(z))
    return xPrev, yPrev, zPrev, x, y, z
end

## The component functions for the innter iteration
# =================================================

## Exact projection onto the cardinality set
# ==========================================

function Π_exact_lsr(k::I, Γ::R, x::AbstractVector{R}) where {I <: Integer, R <: Real}
    ## this function takes a vector x and returns a projection of x on the set
    ## card(⋅) <= k, ||⋅||_∞ <= Γ

    n = length(x)
    y = zero(x)

    perm = sortperm(abs.(x), rev=true)
    ## sortperm(abs.(x), rev=true) returns a sorted array index set from larger to smaller components of absolute values of x
    for i in 1:n
        if i in perm[1:k]
            if x[i]>Γ
                y[i] = Γ
            elseif x[i] < -Γ
                y[i] = -Γ
            else
                y[i] = x[i]
            end
        end
    end

    return y
end

## NExOS inner iteration associated with projection
# =================================================

function Π_NExOS_lsr(k::I, Γ::R, β::R, γ::R, μ::R, x::V) where {I <: Integer, R <: Real, V <: AbstractVector{ <: Real}}
    dnm = γ + (μ*((β*γ)+1))
    scaled_input = (1/((β*γ)+1)).*x
    pi_x_scaled_input = Π_exact_lsr(k, Γ, scaled_input)
    t1 = (μ/dnm).*x
    t2 = (γ/dnm)*pi_x_scaled_input
    return t1+t2
end

## prox_NExOS_lsr function to compute the proximal operator
# =========================================================

function prox_NExOS_lsr(GMat::A, hVec::V,  γ::R, zPrev::V, xPrev::V, ϵProx::R, PMat) where {R <: Real, A <: AbstractMatrix{R}, V <:  AbstractVector{R}}
    sol = copy(xPrev)
    # cg!(sol, GMat, hVec + (1/γ).*zPrev; Pl = PMat, abstol = 1e-4)
    # recall that: GMat ≡ (λ*A'*A) + (1/γ)*I
    #          and hVec ≡ λ*A'*b
    Anew = [0.0026751207469390887 -1.7874010558441433 1.9570015234024314 0.48297821744005304 0.5499471860787609; -1.1926192117844148 -0.2962711608418759 0.34270253049651844 -0.9366799738457191 2.6135543879953094; -2.3956100254067567 1.5957363630260428 -0.5408329087542932 -1.595958552841769 -1.6414598361443185; 0.6791192541307551 2.895535427621422 -0.6676549207144326 -0.49424052539586016 -0.4336822064078515; 2.201382399088591 -0.578227952607833 -0.17602060199064307 -0.46261612338452723 0.24784285430931077; -0.49522601422578916 0.44636764792131967 1.2906418721153354 0.8301596930345838 0.5837176074011505];

    bnew = [-0.7765194270735608, 0.28815810383363427, -0.07926006593887291, 0.5659412043036721, 1.3718921913877151, 0.8834692513807884];

    sol1 = inv((2*Anew'*Anew) + (I/γ))*( (2*Anew'*bnew) + (zPrev ./ γ))

    f = LeastSquares(Anew, bnew, iterative = true)
    prox_point = similar(xPrev) # allocate prox_NExOS output
    prox!(prox_point, f, zPrev, γ) # this is the prox! function from the ProximalOperators package
    sol = prox_point
    @show norm(sol1-sol)
    return sol
end

## Function to update the initial information
# ===========================================

function update_init_info!(state::StateLargeSparseRegModel, init_info::InitInfoLargeSparseRegModel, problem::ProblemLargeSparseRegModel, settings::SettingsLSR)

    init_info.μ = init_info.μ*settings.μ_mult_fact
    init_info.γ = abs(cbrt(settings.μ_min)) # γ_from_μ(init_info.μ, settings)
    init_info.z0 = state.z
    init_info.ϵProx = ϵProxGenerate(1; tolerance_type = :constant)

    return init_info

end

## Final solve funciton
# =====================

# Dedicated solver for the large sparse regression problem
function solve!(problem::ProblemLargeSparseRegModel, settings::SettingsLSR)

    # create the initial state
    state = StateLargeSparseRegModel(problem, settings) # create the initial state, keep in mind actually we can run a proximal evaluation now that we can use to warm start later
    init_info = InitInfoLargeSparseRegModel(problem, settings) # create intial information

    #create the optimization problem to compute the proximal operator

    # now this first state goes into the iteration_outer!(state, problem, settings) and we keep running it until our termination condtion has been met
    
    while state.μ >= μ_min # settings.μ_min
        # run the inner iteration algorithm update procedure
        state = update_state_lsr!(state, init_info, problem, settings)
        init_info = update_init_info!(state, init_info, problem, settings)
    end

    if settings.verbose == true
        @info "information about the best state found for smallest μ = $(state.μ)"
        @info "μ = $(state.μ) | log fixed point gap = $(log10(state.fxd_pnt_gap)) | log feasibility gap = $(log10(state.fsblt_gap)) | inner iterations = $(state.i)"
    end

    return state
end






