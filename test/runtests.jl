using NExOS
using ProximalOperators
using Test
using Random, LinearAlgebra

@testset "sparse_regression" begin

    M = 1.0

    r = 3

    C = SparseSet(M, r)

    A = [0.0026751207469390887 -1.7874010558441433 1.9570015234024314 0.48297821744005304 0.5499471860787609; -1.1926192117844148 -0.2962711608418759 0.34270253049651844 -0.9366799738457191 2.6135543879953094; -2.3956100254067567 1.5957363630260428 -0.5408329087542932 -1.595958552841769 -1.6414598361443185; 0.6791192541307551 2.895535427621422 -0.6676549207144326 -0.49424052539586016 -0.4336822064078515; 2.201382399088591 -0.578227952607833 -0.17602060199064307 -0.46261612338452723 0.24784285430931077; -0.49522601422578916 0.44636764792131967 1.2906418721153354 0.8301596930345838 0.5837176074011505]

    b = [-0.7765194270735608, 0.28815810383363427, -0.07926006593887291, 0.5659412043036721, 1.3718921913877151, 0.8834692513807884]

    m, n = size(A)

    z0 = [0.7363657041121843, 1.7271074489333478, -1.7929456862340958, 0.9359690211495514, 1.771070535933835]

    f = LeastSquares(A, b, iterative = true)

    settings = NExOS.Settings(μ_max = 2, μ_min = 1e-8, μ_mult_fact = 0.85, verbose = true, freq = 250, γ_updt_rule = :adaptive)

    problem = NExOS.Problem(f, C, settings.β, z0)

    state_final = NExOS.solve!(problem, settings)

    @test log10(state_final.fxd_pnt_gap) <= -4

    @test log10(state_final.fsblt_gap) <= -4

    @test abs(f(state_final.x) - 0.934) <= 1e-3

end

@testset "affine_rank_minimization" begin

    m = 10

    n = 2*m

    M = 1.0

    k = convert(Int64, m*n/20) # k is the number of components in the affine operator A: R^m×n → R^k

    r = convert(Int64,round(m*.35))  # r corresponds to the rank of the matrix rank(X) <= r

    mcA = randn(k, m*n)

    b = randn(k)

    Z0 = zeros(m,n)

    f = LeastSquaresOverMatrix(mcA, b, 1.0, iterative = true);

    D = NExOS.RankSet(M, r)

    settings = NExOS.Settings(μ_max = 2, μ_min = 1e-8, μ_mult_fact = 0.5, verbose = true, freq = 250, γ_updt_rule = :safe)

    problem = NExOS.Problem(f, D, settings.β, Z0)

    state_final = NExOS.solve!(problem, settings)

    f(state_final.x)

    @test log10(state_final.fxd_pnt_gap) <= -4

    @test log10(state_final.fsblt_gap) <= -4

end

@testset "matrix_completion" begin

    Random.seed!(1234)

    # Construct a random m-by-n matrix matrix of rank k
    m,n,k = 5,5,2

    Zfull = randn(m,k)*randn(k,n) # ground truth data

    M = opnorm(Zfull,2)

    n_obs = 13

    Zobs = fill(NaN,(m,n))

    obs = randperm(m*n)[1:n_obs]

    Zobs[obs] = Zfull[obs] # partially observed matrix

    # let us find the indices of all the elements that are available

    f = SquaredLossMatrixCompletion(Zobs, iterative = true)

    r = k

    Z0 = zeros(size(Zobs))

    C = RankSet(M, r)

    settings = NExOS.Settings(μ_max = 5, μ_min = 1e-8, μ_mult_fact = 0.5, n_iter_min = 1000, n_iter_max = 1000, verbose = true, freq = 250, tol = 1e-4, γ_updt_rule = :safe)

    problem = NExOS.Problem(f, C, settings.β, Z0)

    state_final = NExOS.solve!(problem, settings)

    @test log10(state_final.fxd_pnt_gap) <= -4

    @test log10(state_final.fsblt_gap) <= -4

end

# Low rank factor analysis testing, requires MOSEK

# using NExOS
#
# using LinearAlgebra, Convex, JuMP, MosekTools
#
# Σ = [1.0 -0.34019653769952096 -0.263030887801514 -0.14349389289304187 -0.18605860686109255; -0.34019653769952096 1.0 0.4848473200092671 0.3421745595621214 0.38218138592185846; -0.263030887801514 0.4848473200092671 1.0 0.3768343949936584 0.5028863662242727; -0.14349389289304187 0.3421745595621214 0.3768343949936584 1.0 0.3150998750134158; -0.18605860686109255 0.38218138592185846 0.5028863662242727 0.3150998750134158 1.0]
#
# n, _ = size(Σ)
#
# r = convert(Int64, round(rank(Σ)/2))
#
# M = 2*opnorm(Σ ,2)
#
# Z0 = Σ # zeros(n,n) Initial condition
#
# z0 = zeros(n) # Initial condition
#
# problem =  NExOS.ProblemFactorAnalysisModel(Σ, r, M, Z0, z0)
#
# settings = NExOS.Settings(μ_max = 1, μ_min = 1e-4, μ_mult_fact = 0.5, n_iter_min = 10, n_iter_max = 10, verbose = true, freq = 50, tol = 1e-2, γ_updt_rule = :adaptive)
#
# state_final = NExOS.solve!(problem, settings)
