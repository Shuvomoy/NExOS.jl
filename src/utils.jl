## Our rule for chosing γ from μ

function γ_from_μ(μ::R, setting::Setting) where {R <: Real}
    if setting.γ_updt_rule == :safe
        return abs(cbrt(setting.μ_min)) # cbrt also works quite well
    elseif setting.γ_updt_rule == :adaptive
        γ_adaptive = (10^-3)*abs(sqrt(μ))
        return γ_adaptive # adaptive actually does not work very well
    elseif setting.γ_updt_rule == :adaptiveplus
        log_γ_adaptiveplus = ((3/(log10(setting.μ_max)+10))*(log10(μ)+10))-6
        γ_adaptiveplus = 10^log_γ_adaptiveplus
        return γ_adaptiveplus
    else
        @error "either select :adaptive or :safe or :adaptiveplus as your γ selection rule"
        return nothing
    end
end

# # Projection onto the sparse set
#  First thing to do is, projecting onto the sparse set. Projection on the sparse set can be computed as follows.

function Π_exact(C::SparseSet, x::AbstractVector{R}) where {R <: Real}
    ## this function takes a vector x and returns a projection of x on the set
    ## card(⋅) <= k, ||⋅||_∞ <= M
    ## Extract the information from teh SparseSet
    M = C.M
    k = C.r

    n = length(x)
    y = zero(x)

    perm = sortperm(abs.(x), rev=true)
    ## sortperm(abs.(x), rev=true) returns a sorted array index set from larger to smaller components of absolute values of x
    for i in 1:n
        if i in perm[1:k]
            if x[i]>M
                y[i] = M
            elseif x[i] < -M
                y[i] = -M
            else
                y[i] = x[i]
            end
        end
    end

    return y
end

# Finally, the projection onto the low rank set.
# # Pseudocode for `Π_exact`
# ---

# ![Algorithm](/literate_assets/low_rank_projection.png)

function Π_exact(C::RankSet, x::AbstractMatrix{R}; iterative::Bool = true) where {R <: Real}

    ## extract the values
    y = similar(x)
    M = C.M # bound on the matrix elements
    r = C.r # rank constraint rank(x) ≦ r
    if iterative == true
        U, S, V = tsvd(x, r)
        for i in 1:r
            if S[i] > M
                S[i] = M
            end
        end
        SVt = S.* V' # this line essentially does the same as Diagonal(X)*V' but more efficiently
        mul!(y, U, SVt)
        return y
    else
        F = svd(x)
        σ_x = F.S
        for i in 1:r
            if σ_x[i] > M
                σ_x[i] = M
            end
        end
        y = F.U[:,1:r]*Diagonal(σ_x[1:r])*V[:,1:r]'
        return y
    end # if else
end

# Projection onto the unit hypercube.

function Π_exact(C::UnitHyperCube, x::AbstractVector{R}) where {R <: Real}
    ## this function takes a vector x and returns a projection of x on the set {0,1}^n
    n = C.n
    y = zero(x) # creates a 0 vector with the same dimension as x
    for i=1:n
        if x[i] > 0.5
            y[i]=1
        end
    end
    return y
end




# update the state

function update_state!(state::State, init_info::InitInfo, problem::Problem, setting::Setting)

    # extract information from the state and init info
    x = state.x
    y = state.y
    z = init_info.z0
    β = setting.β
    γ = init_info.γ
    μ = init_info.μ

    # create best point variables
    best_point_x = x
    best_point_y = y
    best_point_z = z
    best_fxd_pnt_gap = setting.big_M
    best_fsblt_gap = setting.big_M

    i = 1
    while i <= setting.n_iter_max
        x, y, z = inner_iteration(x, y, z, β, γ, μ, problem)
        crnt_fxd_pnt_gap = norm(x-y, Inf)
        y_fsbl = Π_exact(problem.C, y)
        crnt_fsblt_gap = norm(y-y_fsbl, Inf)

        # update the best points so far if we have lower objective value
        if crnt_fxd_pnt_gap <= best_fxd_pnt_gap
            best_point_x = x
            best_point_y = y
            best_point_z = z
            best_fxd_pnt_gap = crnt_fxd_pnt_gap
            best_fsblt_gap = crnt_fsblt_gap
        end # if

        # display important information
        if setting.verbose == true
            if mod(i, setting.freq) == 0
                @info "μ = $μ | iteration = $i | log fixed point gap = $(log10(crnt_fxd_pnt_gap)) | log feasibility gap = $(log10(crnt_fsblt_gap))"
            end # if
        end # if

        # termination condition check
        if crnt_fxd_pnt_gap <= setting.tol
            if setting.verbose == true
                @info "μ = $μ | log fixed point gap reached $(log10(setting.tol))"
            end
            break
        end #if

        # increment the count by 1
        i = i+1

    end # while

    ## update the state
    state.x = best_point_x
    state.y = best_point_y
    state.z = best_point_z
    state.i = i
    state.fxd_pnt_gap = best_fxd_pnt_gap
    state.fsblt_gap = best_fsblt_gap
    state.μ = μ
    state.γ = γ

    # Print information about the best point
    if setting.verbose == true
        @info "information about the best state found for μ = $(state.μ)"
        @info "μ = $(state.μ) | log fixed point gap = $(log10(state.fxd_pnt_gap)) | log feasibility gap = $(log10(state.fsblt_gap)) | inner iterations = $(state.i)"
    end

    return state
end

## the inner iteration function
# The inner iteration function has the following pseudocde
# # Pseudocode for `Π_exact`
# ---

# ![Algorithm](/literate_assets/inner_iteration.png)

function inner_iteration(x::A, y::A, z::A, β::R, γ::R, μ::R, problem::Problem) where {A <: AbstractVecOrMat{<:Real}, R <: Real}
    # old iteration: x = prox_NExOS(problem.f, β, γ, z)
    x = prox_NExOS(problem.f, γ, z)
    y = Π_NExOS(problem.C, β, γ, μ, 2*x - z)
    z = z + y - x
    return x, y, z
end

## proximal function on f: this is the old function to be removed later
# function prox_NExOS(f::ProximableFunction, β::R, γ::R, x::A) where {R <: Real, A <: AbstractVecOrMat{<:Real}}
#     prox_param = γ/((β*γ)+1)
#     scaled_input = (1/((β*γ)+1)).*x
#     prox_point = similar(x) # allocate prox_NExOS output
#     prox!(prox_point, f, scaled_input, prox_param) # this is the prox! function from the ProximalOperators package
#     return prox_point
# end

## New proximal function, no scaling required
function prox_NExOS(f::ProximableFunction, γ::R, x::A) where {R <: Real, A <: AbstractVecOrMat{<:Real}}
    prox_point = similar(x) # allocate prox_NExOS output
    prox!(prox_point, f, x, γ) # this is the prox! function from the ProximalOperators package
    return prox_point
end



## NExOS projection function
# This implements the following pseudocode
# ![relaxed_projection_algorithm](/literate_assets/relaxed_projection.png)

function Π_NExOS(C::ProxRegularSet, β::R, γ::R, μ::R, x::A) where {R <: Real, A <: AbstractVecOrMat{<:Real}}
    dnm = γ + (μ*((β*γ)+1))
    scaled_input = (1/((β*γ)+1)).*x
    pi_x_scaled_input = Π_exact(C, scaled_input)
    t1 = (μ/dnm).*x
    t2 = (γ/dnm)*pi_x_scaled_input
    return t1+t2
end


# Now we write the update init information function.

function update_init_info!(state::State, init_info::InitInfo, problem::Problem, setting::Setting)

    init_info.μ = init_info.μ*setting.μ_mult_fact
    init_info.γ = γ_from_μ(init_info.μ, setting)
    x_μ = Π_NExOS(problem.C, setting.β, init_info.γ, init_info.μ, state.x)
    u1 = similar(x_μ) # allocate memory
    gradient!(u1, problem.f, x_μ)
    u = u1 + setting.β.*x_μ # this is a gradient of the function f + (β/2)*||⋅||^2 at x_μ
    # we are using the function gradient from proximaloperators.jl
    init_info.z0 = x_μ + init_info.γ*u

    return init_info

end

# Experiment with the initial information

function update_init_info_experimental!(state::State, init_info::InitInfo, problem::Problem, setting::Setting)

    init_info.μ = init_info.μ*setting.μ_mult_fact
    init_info.γ = γ_from_μ(init_info.μ, setting)
    init_info.z0 = state.z

    return init_info

end

## Extending the proximal opertor for least sqaures penalty over matrices



function ProximalOperators.prox!(Y::Array{R,2}, f::ProximalOperators.LeastSquaresIterative, X::Array{R,2}, gamma::R=R(1)) where {R <: Real}
    m, n = size(Y)
    y = similar(vec(Y)) # create storaget to store the proximal operator
    x = vec(X)
    prox!(y, f, x, gamma)
    # need to convert y back to Y
    Y[:,:] = reshape(y,m,n) # Y[:,:] is used so that the contents are changed
    return f(y)
end

function ProximalOperators.prox!(Y::Array{R,2}, f::ProximalOperators.LeastSquaresDirect, X::Array{R,2}, gamma::R=R(1)) where {R <: Real}
    m, n = size(Y)
    y = similar(vec(Y)) # create storaget to store the proximal operator
    x = vec(X)
    prox!(y, f, x, gamma)
    # need to convert y back to Y
    Y[:,:] = reshape(y,m,n) # Y[:,:] is used so that the contents are changed
    return f(y)
end

# Computing gradient for the leastsquares over matrix function direct case
function ProximalOperators.gradient!(Y::Array{R,2}, f::ProximalOperators.LeastSquaresDirect, X::Array{R,2}) where {R<: Real}
    m, n = size(Y)
    y = similar(vec(Y)) # create storaget to store the proximal operator
    x = vec(X)
    gradient!(y, f, x)
    Y[:,:] = reshape(y,m,n) # Y[:,:] is used so that the contents are changed
    return f(y)
end


# Computing gradient for the leastsquares over matrix function direct case
function ProximalOperators.gradient!(Y::Array{R,2}, f::ProximalOperators.LeastSquaresIterative, X::Array{R,2}) where {R<: Real}
    m, n = size(Y)
    y = similar(vec(Y)) # create storaget to store the proximal operator
    x = vec(X)
    gradient!(y, f, x)
    Y[:,:] = reshape(y,m,n) # Y[:,:] is used so that the contents are changed
    return f(y)
end
