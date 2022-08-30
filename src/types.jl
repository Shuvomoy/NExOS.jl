##%% Load the packages

# constant value

μ_min = 10^-8

# First, we load the packages.

using ProximalOperators, LinearAlgebra, TSVD, SparseArrays, OSQP, JuMP, MosekTools, IterativeSolvers

##%% All the necessary types

# # ProxRegularSet
#  This is an abstract type that describes the nature of the set.

abstract type ProxRegularSet end


# # problem
# This structure contains the function, the set, value of parameter β, and an intial point.

struct Problem{F, S <: ProxRegularSet, R <: Real, A <: AbstractVecOrMat{ <:Real}}

    f::F # the objective function
    C::S # the constraint set
    β::R # what value of β to pick
    z0::A # one intial state provided by the user

    ## consider having a constructor that will create an intial staate if the user does not give one

end

##%% Settings function

# # settings
#  This struct represents the user settings. It contains information regarding user specified value of different parameters

struct Settings

    # first comes description of different data type

    μ_max::Float64 # starting μ
    # μ_min::Float64 # ending μ
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
    function Settings(;
        μ_max = 1.0,
        # μ_min = 1e-12,
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
        # new(μ_max, μ_min, μ_mult_fact, n_iter_min, n_iter_max, big_M, β, tol, verbose, freq, γ_updt_rule)
		new(μ_max, μ_mult_fact, n_iter_min, n_iter_max, big_M, β, tol, verbose, freq, γ_updt_rule)
    end

end

## structure that describes the state (for the outer loop)

# # state
# This structure contains all the necessary information to describe a state

mutable struct State{T <: AbstractVecOrMat{<: Real}, I <: Integer, R <: Real} # Done
    x::T # the first iterate
    y::T # the second iterate
    z::T # the third iterate
    i::I # iteration number
    fxd_pnt_gap::R # fixed point gap for state
    fsblt_gap::R # how feasible the current point is, this is measured by || x- Π_C(x) ||
    μ::R # current value of μ
    γ::R # current value of γ
end

# Let us initialize the state data structure.

function State(problem::Problem, settings::Settings)

    # this function constructs the very first state for the problem and settings
    z0 = copy(problem.z0)
    x0 = zero(z0)
    y0 = zero(z0)
    i = copy(settings.n_iter_max) # iteration number where best fixed point gap is reached
    fxd_pnt_gap = copy(settings.big_M)
    fsblt_gap = copy(settings.big_M)
    μ = copy(settings.μ_max)
    γ = γ_from_μ(μ, settings) #

    return State(x0, y0, z0, i, fxd_pnt_gap, fsblt_gap, μ, γ)

end

# Time to describe the data structure that we require to initialize our algorithm.

## structure that describes the information to intialize our algorithm:

mutable struct InitInfo{T <: AbstractVecOrMat{<: Real}, R <: Real}

    # this will have z0, γ, μ
    z0::T # initial condition to start an inner iteration
    μ::R # μ required start an inner iteration
    γ::R # γ required to start an inner iteration

end

# Of course, we will need a constructor for the InitInfo type, given the user input. The user inputs are problem::problem, settings::settings.

function InitInfo(problem::Problem, settings::Settings)

    ## constructs the initial NExOS information
    z0 = copy(problem.z0) # need to understand if copy is necessary
    μ = copy(settings.μ_max)
    γ = γ_from_μ(μ, settings)

    return InitInfo(z0, μ, γ)

end

#%%
# Time to define our sets. We have two prox regular sets for now:
# 1. SparseSet: It is of the form: card(x) ≦ r, ||x||_∞ ≦ M
# 2. RankSet: It is of the form: rank(x) ≦ r, ||x||_2 ≦ M

# We also consider the unit hypercube, which is not prox-regular however is defined here to apply our algorithm as a heuristics to the 3-SAT problem.

# define sparse set first, via the `SparseSet` structure.

struct SparseSet{R <: Real, I <: Integer} <: ProxRegularSet
    M::R # M is the upper bound on the elements of x
    r::I # r is the maximum cardinality
    function SparseSet{R, I}(M::R, r::I) where {R <: Real, I <: Integer}
        if r <= 0 || M <= 0
            error("parameter M and r must be a positive integer")
        else
            new(M, r)
        end
    end
end

# Constructor for the sparse set
SparseSet(M::R, r::I) where {R <: Real, I <: Integer} = SparseSet{R,I}(M, r)


# Okay, we have done everything for the sparse set. Now, we do the same for the low rank set. First, we define the structure.

struct RankSet{R<: Real, I <: Integer} <: ProxRegularSet
    M::R
    r::I
    function RankSet{R, I}(M::R, r::I) where {R <: Real, I <: Integer}
        if r <= 0 || M <= 0
            error("parameter M and r must be a positive integer")
        else
            new(M, r)
        end
    end
end


# Let us define the constructor for the low rank set.

RankSet(M::R, r::I) where {R <: Real, I <: Integer} = RankSet{R, I}(M, r)

# Time to define the unit hypercube: {0,1}^n

struct UnitHyperCube{I <: Integer} <: ProxRegularSet
    n::I
    function  UnitHyperCube{I}(n::I) where {I <: Integer}
        if n <= 0
            error("parameter n must be a positive integer")
        else
            new(n)
        end
    end
end

# Constructor for the unit hypercube

UnitHyperCube(n::I) where {I <: Integer} = UnitHyperCube{I}(n)

## Define the Least squarers function for affine rank minimization:


## Start with some helper functions

function Afn_Op_ith_Mat(𝓐, k, i)
    ## gives the i-th matrix of the affine oerator of 𝓐, such that
    ## [𝓐(X)]_i = trace(A_i^T X)
    m, kn = size(𝓐)
    n = convert(Int64, kn/k)
    m, kn = size(𝓐)
    n = convert(Int64, kn/k)
    return 𝓐[:, (i-1)*n+1:i*n]
end

# TODO: Make this function faster
function Affn_Op(𝓐, k, X)
    # Computes 𝓐(X), k is the dimension of the affine operator
    # 𝓐 = [A_1 A_2 ... A_k] ∈ R^ {m × kn}
    m, n = size(X)
    m1, kn = size(𝓐)
    n1 = convert(Int64, kn/k)
    if m1 ≠ m || n1 ≠ n
        @error "sizes of the input matrix and the affine operator matrix do not match"
    end
    z = zeros(k) # output will be stored in this matrix
    for i in 1:k
        z[i] = tr(Afn_Op_ith_Mat(𝓐, k, i)'*X)
    end
    return z
end

function Affn_Op_Mat(𝓐, k, X)
    # Computes 𝓐(X), k is the dimension of the affine operator
    # 𝓐 = [A_1 A_2 ... A_k] ∈ R^ {m × kn}
   A = affine_operator_to_matrix(𝓐, k)
   # output will be stored in this matrix
   return A*vec(X)
end

# TODO: need to make this function faster
function affine_operator_to_matrix(𝓐::AbstractMatrix{R}, k::I) where {R <: Real, I <: Integer}
    ## convert 𝓐 to a matrix A of size k × mn
    ## 𝓐 = [A_1 A_2 ... A_k] ∈ R^ {m × kn}, where each
    ## A_i is an m × n matrix, and our
    ## AffineOp(X) = [ tr(A_1^T X) ; tr(A_2^T X); ... ; tr(A_k^T X) ], where X is an m × n matrix
    ## our goal is coming up with a matrix A such that
    ## AffineOp(X) = A*vec(X)
    m, kn = size(𝓐)
    if mod(kn, k) ≠ 0
        @error "dimension of the matrix [A_1 ... A_k] is not correct"
    end
    n = convert(Int64, kn/k)
    A = zeros(k,m*n)
    for i in 1:k
        A[i,:] = transpose(vec(𝓐[:, (i-1)*n+1:i*n]))
    end
    return A
end




## Let us define the function least squares over matrices


abstract type LeastSquaresOverMatrix end

is_smooth(f::LeastSquaresOverMatrix) = true

is_quadratic(f::LeastSquaresOverMatrix) = true

fun_name(f::LeastSquaresOverMatrix) = "Least squares penalty over matrices"

function LeastSquaresOverMatrix(𝓐::M, b::V, k::I, lam::R=R(1); iterative=false) where {R <: Real, V <: AbstractArray{R}, I <: Integer, M} # BUG: seems that this version of the objective value has a bug, for now use the other one that directly takes a k×mn matrix
    # FIXME: Fix this function later on, for now work with the other version
    ## In this case, 𝓐 is as follows:
    ## 𝓐 = [A_1 A_2 ... A_k] ∈ R^ {m × kn}, where each
    ## A_i is an m × n matrix, and
    # b is the output observation with k entries
    # this function will convert 𝓐 into a matrix Amat of size k × mn that can operate on vec(X) of size mn, which is the vectorized version of the input matrix X of size m × n
    Amat = affine_operator_to_matrix(𝓐, k)
    if iterative == false
        ProximalOperators.LeastSquaresDirect(Amat, b, lam)
    else
        ProximalOperators.LeastSquaresIterative(Amat, b, lam)
    end
end

function LeastSquaresOverMatrix(𝐀::M, b::V, lam::R=R(1); iterative=false) where {R <: Real, V <: AbstractArray{R}, I <: Integer, M}
    # in this case 𝐀 is given as a k × mn matrix, so that it acts directly on the vectorized operator
    if size(𝐀)[1] != length(b)
        @error "number of rows of A must be equal to number of elements of b "
    end
    if iterative == false
        ProximalOperators.LeastSquaresDirect(𝐀, b, lam)
    else
        ProximalOperators.LeastSquaresIterative(𝐀, b, lam)
    end
end

# evaluating the LeastSquaresOverMatrix
# for direct
function (f::ProximalOperators.LeastSquaresDirect)(X::Array{R,2}) where {R <: Real}
    return f(vec(X))
end

# for iterative
function (f::ProximalOperators.LeastSquaresIterative)(X::Array{R,2}) where {R <: Real}
    return f(vec(X))
end

## Function for squared loss for matrix completion problem

# data conversion for the matrix completion problem

# older function: takes much longer to compute
# function matrix_completion_A_b(Zobs::AbstractMatrix{R}) where {R <: Real}
#     # The function will take the matrix Zobs of size mxn such that has it has many missing (NaN) values and few observed value. Let the index of available values of Zobs be Ω. Then the function will create a matrix A and vector b, such that for any matrix X we have
#     #  \[
#     # f(X)=\sum_{(i,j)\in\Omega}(X_{ij}-Z_{ij})^{2}=\|Ax-b\|_{2}^{2}.
#     # \]
#     m, n = size(Zobs)
#     zobs = vec(Zobs)
#     # find the corresponding indices of the missing data again:
#     index_available_vec = findall(x -> isnan(x)== false, zobs)
#     b = zobs[index_available_vec]
#     p = length(index_available_vec)
#     A = spzeros(p, m*n) # m is size of b,
#     for i in 1:p
#         A[i,index_available_vec[i]] = 1.0
#     end
#     return A, b
# end

function matrix_completion_A_b(Zobs::AbstractMatrix{R}) where {R <: Real}
    # The function will take the matrix Zobs of size mxn such that has it has many missing (NaN) values and few observed value. Let the index of available values of Zobs be Ω. Then the function will create a matrix A and vector b, such that for any matrix X we have
    #  \[
    # f(X)=\sum_{(i,j)\in\Omega}(X_{ij}-Z_{ij})^{2}=\|Ax-b\|_{2}^{2}.
    # \]
    m, n = size(Zobs)
    zobs = vec(Zobs)
    # find the corresponding indices of the missing data again:
    index_available_vec = findall(x -> isnan(x)== false, zobs)
    b = zobs[index_available_vec]
    p = length(index_available_vec)
    I_A = 1:p
    J_A = index_available_vec
    V_A = ones(p)
    A = sparse(I_A,J_A,V_A,p,m*n)
    return A, b
end

# time to define the squared loss function

abstract type SquaredLossMatrixCompletion end

is_smooth(f::SquaredLossMatrixCompletion) = true

is_quadratic(f::SquaredLossMatrixCompletion) = true

fun_name(f::SquaredLossMatrixCompletion) = "Squared loss function for matrix completion problem"

function SquaredLossMatrixCompletion(Zobs::AbstractMatrix{R}, lam::R=R(1); iterative = false) where {R <: Real}
    # construct the data from Zobs
    A, b = matrix_completion_A_b(Zobs)

    if iterative == false
        ProximalOperators.LeastSquaresDirect(A, b, lam)
    else
        ProximalOperators.LeastSquaresIterative(A, b, lam)
    end

end

## least squares penalty with variable tolerance via iterative cg solver

export LeastSquaresNExOS

### ABSTRACT TYPE

abstract type LeastSquaresNExOS end

is_smooth(::Type{<:LeastSquaresNExOS}) = true
is_generalized_quadratic(::Type{<:LeastSquaresNExOS}) = true

### CONSTRUCTORS


function LeastSquaresNExOS(A, b; lam=1, tol = 1e-4)
        LeastSquaresIterativeCstmTol(A, b, lam, tol)
end

infer_shape_of_x(A, ::AbstractVector) = (size(A, 2), )
infer_shape_of_x(A, b::AbstractMatrix) = (size(A, 2), size(b, 2))

using LinearAlgebra

struct LeastSquaresIterativeCstmTol{N, R, RC, M, V, O, IsConvex} <: LeastSquaresNExOS
    A::M # m-by-n operator
    b::V # m (by-p)
    lambda::R
    lambdaAtb::V
    shape::Symbol
    S::O
    res::Array{RC, N} # m (by-p)
    res2::Array{RC, N} # m (by-p)
    q::Array{RC, N} # n (by-p)
    tol::R
end

is_prox_accurate(f::Type{<:LeastSquaresIterativeCstmTol}) = false
is_convex(::Type{LeastSquaresIterativeCstmTol{N, R, RC, M, V, O, IsConvex}}) where {N, R, RC, M, V, O, IsConvex} = IsConvex

function LeastSquaresIterativeCstmTol(A::M, b, lambda, tol) where M
    if size(A, 1) != size(b, 1)
        error("A and b have incompatible dimensions")
    end
    m, n = size(A)
    x_shape = infer_shape_of_x(A, b)
    shape, S, res2 = if m >= n
        :Tall, AcA(A, x_shape), []
    else
        :Fat, AAc(A, size(b)), zero(b)
    end
    RC = eltype(A)
    R = real(RC)
    LeastSquaresIterativeCstmTol{ndims(b), R, RC, M, typeof(b), typeof(S), lambda >= 0}(A, b, R(lambda), lambda*(A'*b), shape, S, zero(b), res2, zeros(RC, x_shape), R(tol))
end

function (f::LeastSquaresIterativeCstmTol)(x)
    mul!(f.res, f.A, x)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function prox!(y, f::LeastSquaresIterativeCstmTol, x, gamma)
    @info "[🎴 ] calling Shuvo's custom prox on LeastSquaresIterativeCstmTol"
    f.q .= f.lambdaAtb .+ x./gamma
    RC = eltype(f.S)
    # two cases: (1) tall A, (2) fat A
    if f.shape == :Tall
        y .= x
        op = ScaleShift(RC(f.lambda), f.S, RC(1)/gamma)
        IterativeSolvers.cg!(y, op, f.q; abstol = f.tol)
    else # f.shape == :Fat
        # y .= gamma*(f.q - lambda*(f.A'*(f.fact\(f.A*f.q))))
        mul!(f.res, f.A, f.q)
        op = ScaleShift(RC(f.lambda), f.S, RC(1)/gamma)
        IterativeSolvers.cg!(f.res2, op, f.res; abstol = f.tol)
        mul!(y, adjoint(f.A), f.res2)
        y .*= -f.lambda
        y .+= f.q
        y .*= gamma
    end
    mul!(f.res, f.A, y)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function gradient!(y, f::LeastSquaresIterativeCstmTol, x)
    mul!(f.res, f.A, x)
    f.res .-= f.b
    mul!(y, adjoint(f.A), f.res)
    y .*= f.lambda
    return (f.lambda / 2) * real(dot(f.res, f.res))
end

function prox_naive(f::LeastSquaresIterativeCstmTol, x, gamma)
    y = IterativeSolvers.cg(f.lambda*f.A'*f.A + I/gamma, f.lambda*f.A'*f.b + x/gamma)
    fy = (f.lambda/2)*norm(f.A*y-f.b)^2
    return y, fy
end
