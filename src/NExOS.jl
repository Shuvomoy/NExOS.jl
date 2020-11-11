module NExOS

# First, we include the type file

include("./types.jl")

# Export all the types, and functions for usage

export ProxRegularSet, Problem, Settings, State, InitInfo, SparseSet, RankSet,  LeastSquaresOverMatrix, SquaredLossMatrixCompletion, affine_operator_to_matrix

# Next, we include the utils file

include("./utils.jl")

export Π_exact, update_state!, inner_iteration, prox_NExOS,  Π_NExOS, update_init_info!, update_init_info_experimental!, prox!

# Next, we include the file that solves factor analysis problem, this a special file, as it is using somewhat specialized implementation

include("./factor_analysis.jl")

export ProblemFactorAnalysisModel, StateFactorAnalysisModel, InitInfoFactorAnalysisModel, update_state_fam!, inner_iteration_fam, prox_NExOS_fam



# Export all the types, and functions for usage from the utils

# the main solver function

# export the solver function

# Final solver that does everything

function solve!(problem::Problem, settings::Settings)

    # create the initial state
    state = State(problem, settings) # create the initial information
    init_info = InitInfo(problem, settings) # create intial information

    # now this first state goes into the iteration_outer!(state, problem, settings) and we keep running it until our termination condtion has been met
    while state.μ >= settings.μ_min
        # run the outer iteration update procedure
        state = update_state!(state, init_info, problem, settings)
        # init_info = update_init_info!(state, init_info, problem, settings )
        # experimental version: uncomment the previous line after you are done experimenting
        init_info = update_init_info_experimental!(state, init_info, problem, settings )
    end

    if settings.verbose == true
        @info "information about the best state found for smallest μ = $(state.μ)"
        @info "μ = $(state.μ) | log fixed point gap = $(log10(state.fxd_pnt_gap)) | log feasibility gap = $(log10(state.fsblt_gap)) | inner iterations = $(state.i)"
    end


    return state
end

# Dedicated solver for factor analysis problem
function solve!(problem::ProblemFactorAnalysisModel, settings::Settings)

    # create the initial state
    state = StateFactorAnalysisModel(problem, settings) # create the initial state, keep in mind actually we can run a proximal evaluation now that we can use to warm start later
    init_info = InitInfoFactorAnalysisModel(problem, settings) # create intial information

    #create the optimization problem to compute the proximal operator

    # now this first state goes into the iteration_outer!(state, problem, settings) and we keep running it until our termination condtion has been met
    while state.μ >= settings.μ_min
        # run the outer iteration update procedure
        state = update_state_fam!(state, init_info, problem, settings)
        # init_info = update_init_info!(state, init_info, problem, settings )
        # experimental version: uncomment the previous line after you are done experimenting
        init_info = update_init_info_experimental_fam!(state, init_info, problem, settings )
    end

    if settings.verbose == true
        @info "information about the best state found for smallest μ = $(state.μ)"
        @info "μ = $(state.μ) | log fixed point gap = $(log10(state.fxd_pnt_gap)) | log feasibility gap = $(log10(state.fsblt_gap)) | inner iterations = $(state.i)"
    end


    return state
end


export solve!


end # module
