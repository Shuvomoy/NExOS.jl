# Code for factor analysis model. We write the code for it seperately because it has a very special structure.

## Define the one shot function FactorAnalysisModel

struct ProblemFactorAnalysisModel{A <: AbstractMatrix{ <: Real }, I <: Integer, R <: Real, V <: AbstractVector{ <:Real }}
    Σ::A # This is part of the data, which is a positive semidefinite matrix
    r::I # represents rank of the decision variable matrix X
    M::R # upper bound on the operator-2 norm of decision variable matrix X
    Z0::A # one initial state provided by the user (for the matrix part)
    z0::V # one initial state provided by the user (the vector part)
end

## This structure contains all the necessary information to describe a state

mutable struct StateFactorAnalysisModel{R <: Real, A <: AbstractMatrix{ <: Real }, V <: AbstractVector{ <: Real}, I <: Integer} # Done
    X::A # the first iterate matrix part
    x::V # the first iterate vector part
    Y::A # the second iterate matrix part
    y::V # the second iterate vector part
    Z::A # the third iterate matrix part
    z::V # the third iterate vector part
    i::I # iteration number
    fxd_pnt_gap::R # fixed point gap for state
    fsblt_gap::R # how feasible the current point is, this is measured by || x- Π_C(x) ||
    μ::R # current value of μ
    γ::R # current value of γ
end

## this function constructs the very first state for the problem and settings for the factor analysis model
function StateFactorAnalysisModel(problem::ProblemFactorAnalysisModel, settings::Settings)

    # this function constructs the very first state for the problem and settings
    Z0 = copy(problem.Z0)
    z0 = copy(problem.z0)
    X0 = zero(Z0)
    x0 = zero(z0)
    Y0 = zero(Z0)
    y0 = zero(z0)
    i = copy(settings.n_iter_max) # iteration number where best fixed point gap is reached
    fxd_pnt_gap = copy(settings.big_M)
    fsblt_gap = copy(settings.big_M)
    μ = copy(settings.μ_max)
    γ = γ_from_μ(μ, settings)

    return StateFactorAnalysisModel(X0, x0, Y0, y0, Z0, z0, i, fxd_pnt_gap, fsblt_gap, μ, γ)

end

## structure that describes the information to intialize our algorithm:

mutable struct InitInfoFactorAnalysisModel{ A <: AbstractMatrix{ <: Real }, V <: AbstractVector{ <: Real}, R <: Real}

    # this will have Z0, z0, γ, μ
    Z0::A # initial condition (matrix part) to start an inner iteration
    z0::V # initial condition (vector part) to start an inner iteration
    μ::R # μ required start an inner iteration
    γ::R # γ required to start an inner iteration

end

## Of course, we will need a constructor for the InitInfoFactorAnalysisModel type, given the user input. The user inputs are problem::problem, settings::settings.

function InitInfoFactorAnalysisModel(problem::ProblemFactorAnalysisModel, settings::Settings)

    ## constructs the initial NExOS information
    Z0 = copy(problem.Z0)
    z0 = copy(problem.z0) # need to understand if copy is necessary
    μ = copy(settings.μ_max)
    γ = γ_from_μ(μ, settings)

    return InitInfoFactorAnalysisModel(Z0, z0, μ, γ)

end



## update the state for factor analysis model

function update_state_fam!(state::StateFactorAnalysisModel, init_info::InitInfoFactorAnalysisModel, problem::ProblemFactorAnalysisModel, settings::Settings)

    # extract information from the state and init info
    X = state.X
    x = state.x
    Y = state.Y
    y = state.y
    Z = init_info.Z0
    z = init_info.z0 # TODO: Need to write the InitInfoFactorAnalysisModel
    β = settings.β
    γ = init_info.γ
    μ = init_info.μ

    # create best point variables
    best_point_X = X
    best_point_x = x
    best_point_Y = Y
    best_point_y = y
    best_point_Z = Z
    best_point_z = z
    best_fxd_pnt_gap = settings.big_M
    best_fsblt_gap = settings.big_M

    i = 1
    while i <= settings.n_iter_max
        X, x, Y, y, Z, z = inner_iteration_fam(X, x, Y, y, Z, z, β, γ, μ, problem) # Done: Need to write inner iteration factor analysis model: inner_iteration_fam
        crnt_fxd_pnt_gap = norm(x-y, Inf)+norm(X-Y, Inf)
        Y_fsbl = Π_exact(RankSet(problem.M, problem.r), Y)
        crnt_fsblt_gap = norm(Y-Y_fsbl, Inf) # the vector part y is not changed in the second step of the factor analysis inner iteration, so we do not need norm(y-y_fsbl, Inf) part

        # update the best points so far if we have lower objective value
        if crnt_fxd_pnt_gap <= best_fxd_pnt_gap
            best_point_X = X
            best_point_x = x
            best_point_Y = Y
            best_point_y = y
            best_point_Z = Z
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

    end # while

    ## update the state
    state.X = best_point_X
    state.x = best_point_x
    state.Y = best_point_Y
    state.y = best_point_y
    state.Z = best_point_Z
    state.z = best_point_z
    state.i = i
    state.fxd_pnt_gap = best_fxd_pnt_gap
    state.fsblt_gap = best_fsblt_gap
    state.μ = μ
    state.γ = γ

    # Print information about the best point
    if settings.verbose == true
        @info "information about the best state found for μ = $(state.μ)"
        @info "μ = $(state.μ) | log fixed point gap = $(log10(state.fxd_pnt_gap)) | log feasibility gap = $(log10(state.fsblt_gap)) | inner iterations = $(state.i)"
    end

    return state
end

# inner iteration function for FactorAnalysisModel
function inner_iteration_fam(X::A, x::V, Y::A, y::V, Z::A, z::V, β::R, γ::R, μ::R, problem::ProblemFactorAnalysisModel) where { A <: AbstractMatrix{ <: Real }, V <: AbstractVector{ <: Real}, R <: Real}
    # old iteration: x = prox_NExOS(problem.f, β, γ, z)
    X, x = prox_NExOS_fam(problem.Σ, problem.M, γ, Z, z)
    Y = Π_NExOS(RankSet(problem.M, problem.r), β, γ, μ, 2*X - Z)
    y = max.(2*x - z,0)
    Z = Z + Y - X
    z = z + y - x
    # println("size of the new matrices = ", size(Z), size(z))
    return X, vec(x), Y, vec(y), Z, vec(z)
end

# Proximal operator funciton evaluation using Convex, this is much slower than using JuMP, so I am commenting it for now.

# function prox_NExOS_fam(Σ, M, γ, X, d) #(Σ::A, M::R, γ::R, X::A, d::V) where {R <: Real, A <: AbstractMatrix{R}, V <:  AbstractVector{R}} # For now M is not used, may use it in a future version
#
#   # This functions takes the input data Σ, γ, X, d and evaluates
#   # the proximal operator of the function f at the point (X,D)
#
#   # Data extraction
#   # ---------------
#   n = length(d) # dimension of the problem
#   # println("*****************************")
#   # println(size(d))
#   # println("the value of d is = ", d)
#   # println("the type of d is", typeof(d))
#   D = LinearAlgebra.diagm(d) # creates the diagonal matrix D that embed
#
#   # Create the variables
#   #  --------------------
#   X_tl = Convex.Semidefinite(n) # Here Semidefinite(n) encodes that
#   # X_tl ≡ ̃X is a positive semidefinite matrix
#   d_tl = Convex.Variable(n) # d_tl ≡ ̃d
#   D_tl = diagm(d_tl) # Create the diagonal matrix ̃D from ̃d
#
#   # Create terms of the objective function, which we write down
#   #  in three parts
#   #  ----------------------------------------------------------
#   t1 = square(norm2(Σ - X_tl - D_tl)) # norm2 computes Frobenius norm in Convex.jl
#   t2 = square(norm2(X-X_tl))
#   t3 = square(norm2(D-D_tl))
#
#   # Create objective
#   # ----------------
#   objective = t1 + (1/(2*γ))*(t2 + t3) # the objective to be minimized
#
#   # create the problem instance
#   # ---------------------------
#   convex_problem = Convex.minimize(objective, [d_tl >= 0, Σ - D_tl  in :SDP])
#
#   # set the solver
#   # --------------
#   convex_solver = () -> SCS.Optimizer(verbose=false)#COSMO.Optimizer(verbose=false)
#
#   # solve the problem
#   # -----------------
#   Convex.solve!(convex_problem, convex_solver)
#
#   # get the optimal solution
#   # ------------------------
#   X_sol = X_tl.value
#   d_sol = d_tl.value
#   # println("d_sol = ", d_sol)
#
#   # return the output
#   return X_sol, vec(d_sol)
#
# end

## put everything in a function
# uses JuMP

function prox_NExOS_fam(Σ::A, M::R, γ::R, X::A, d::V; X_tl_sv = nothing, d_tl_sv = nothing, warm_start = false) where {R <: Real, A <: AbstractMatrix{R}, V <:  AbstractVector{R}}

	# This functions takes the input data Σ, γ, X, d and evaluates the proximal operator of the function f at the point (X,d)

	n = length(d)

	# prox_model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => false))

	prox_model = JuMP.Model(optimizer_with_attributes(Mosek.Optimizer))


	@variables( prox_model,
	begin
		d_tl[1:n] >= 0
		X_tl[1:n, 1:n], PSD
	end
	)

	if warm_start == true
		println("warm start enabled")
		# Set warm-start
		set_start_value.(X_tl, X_tl_sv) # Warm start
		set_start_value.(d_tl, d_tl_sv) # Warm start
	    println("norm difference is = ", norm(start_value.(X_tl) - X_tl_sv))
	end

    t_1 = vec(Σ - X_tl - diagm(d_tl))
	t_2 = vec(X_tl-X)
	t_3 = vec(diagm(d_tl)-diagm(d))
	obj = t_1'*t_1 + ((1/(2*γ))*(t_2'*t_2 + t_3'*t_3))

	@objective(prox_model, Min, obj)

	@constraints(prox_model, begin
		Symmetric(Σ - diagm(d_tl)) in PSDCone()
	end)

	set_silent(prox_model)

	JuMP.optimize!(prox_model)

	# obj_val = JuMP.objective_value(prox_model)
	X_sol = JuMP.value.(X_tl)
	d_sol = JuMP.value.(d_tl)

	return X_sol, d_sol

end

function update_init_info_experimental_fam!(state::StateFactorAnalysisModel, init_info::InitInfoFactorAnalysisModel, problem::ProblemFactorAnalysisModel, settings::Settings)

    init_info.μ = init_info.μ*settings.μ_mult_fact
    init_info.γ = γ_from_μ(init_info.μ, settings)
    init_info.z0 = state.z
    init_info.Z0 = state.Z

    return init_info

end
