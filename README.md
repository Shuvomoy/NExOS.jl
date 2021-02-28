# ```NExOS.jl```
[![Build Status](https://travis-ci.com/Shuvomoy/NExOS.jl.svg?branch=master)](https://travis-ci.com/Shuvomoy/NExOS.jl)

<p align="center">
  <a href="#Installation">Installation</a> •
  <a href="#Usage">Usage</a> •
  <a href="#Tutorials">Tutorials</a> •
  <a href="#Citing">Citing</a> •
  <a href="#Contact">Contact</a> 
</p>

``NExOS.jl`` is a `Julia` package that implements [**N**onconvex **Ex**terior-point **O**perator **S**plitting algorithm](https://arxiv.org/abs/2011.04552) (**NExOS**). The package is tailored for minimizing a convex cost function over a nonconvex constraint set, where projection onto the constraint set is single-valued around points of interest. These types of sets are called *prox-regular* sets, *e.g.*, sets containing low-rank and sparsity constraints. 

``NExOS.jl`` considers nonconvex optimization problems of the following form:

```
minimize    f(x)+(β/2)‖x‖^2
subject to  x ∈ X,
```

where the decision variable `x` can be a vector in `ℜ^d` or a matrix in `ℜ^(m×d)` or a combination of both. The cost function `f` is convex, `β` is a positive parameter (can be arbitrarily small), and the constraint set `X` is a nonconvex prox-regular set (please see [Acceptable functions and sets](#Acceptable-functions-and-sets) for more details). 

## Installation

In `Julia REPL`, type

```] add NExOS```

## Usage

Below is a short usage example on using `NExOS.jl` for [sparse regression problem](https://nbviewer.jupyter.org/github/Shuvomoy/NExOS.jl/blob/master/tutorials/sparse_regression_using_NExOS.ipynb) (for other examples, please see [Tutorials](#Tutorials)). 

```julia
# Load the packages
using Random, NExOS, ProximalOperators

# Random data generation 
m = 25
n = 50
A = randn(m,n)
b = randn(m)
M = 100
k = convert(Int64, round(m/3))
beta = 10^-10

# Create the problem instance in NExOS
C = SparseSet(M, k) # Create the set
f = LeastSquares(A, b, iterative = true) # Create the function
settings = Settings(μ_max = 2, μ_min = 1e-8, μ_mult_fact = 0.85, verbose = false, freq = 250, γ_updt_rule = :adaptive, β = beta) # settings
z0 = zeros(n) # create an initial point
problem = Problem(f, C, settings.β, z0) # problem instance

# Solve the problem
state_final = solve!(problem, settings)

# Extract solution info
x_NExOS = state_final.x # solution found by NExOS
p_star = f(x_NExOS) # objective value
```

### Acceptable functions and sets

##### Objective function `f`

`NExOS.jl` allows for any `f` that is convex. We can classify them into two types:

1. The function `f` is an unconstrained convex function with an easy-to-compute proximal operator. To compute the proximal operators for this category of functions, `NExOS.jl` uses the package [`ProximalOperators.jl`](https://github.com/kul-forbes/ProximalOperators.jl). The list of available functions for this type is available at this [link](https://kul-forbes.github.io/ProximalOperators.jl/stable/functions/). 

2. The function `f` is a constrained convex function (*i.e.*, a convex function over some convex constraint set). For such a function, no closed form solution usually exists, and in this situation `NExOS` computes the proximal operator of `f` by solving a convex optimization problem using [`JuMP`](https://github.com/jump-dev/JuMP.jl)  and any of the solvers supported by it (listed [here](https://jump.dev/JuMP.jl/stable/installation/#Getting-Solvers-1)). To know more about this proximal operator computation process, please see [this blog post](https://shuvomoy.github.io/blog/programming/2020/09/08/proximal_operator_over_matrix.html) I wrote.

##### Constraint set `X`

The constraint set `X` is nonconvex, but it can be decomposed into a convex compact set `C` and a nonconvex prox-regular set `N`, *i.e.*, `X = C ⋂ N`. For example:

1. **Sparse set.**  `N = {x ∈ ℜ^d ∣ card(x) ≦ k}`, where `card(x)` denotes the number of nonzero components in `x`,
2. **Low-rank set.**  `N = { X ∈ ℜ^(m×d) ∣ rank(X) ≦ r}`, where `rank(X)` denotes the rank of the matrix `X`,
3. **Combination of low-rank and sparse set.**  `N = {X ∈ ℜ^(m×d), x ∈ ℜ^d ∣ card(x) ≦ k, rank(X) ≦ r}`,  
4. **Any other prox-regular set.**  `N` can be any other prox-regular sets, *e.g.,* weakly convex sets, proximally smooth sets, *etc.* 

## Tutorials

Please see the following `jupyter notebook` tutorials that describe in more detail how to use `NExOS.jl`. 

1. [Affine rank minimization](https://nbviewer.jupyter.org/github/Shuvomoy/NExOS.jl/blob/master/tutorials/Affine%20rank%20minimization%20using%20NExOS.jl.ipynb).
2. [Matrix completion](https://nbviewer.jupyter.org/github/Shuvomoy/NExOS.jl/blob/master/tutorials/Matrix_completion_problem_NEXOS.ipynb).
3. [Sparse regression](https://nbviewer.jupyter.org/github/Shuvomoy/NExOS.jl/blob/master/tutorials/sparse_regression_using_NExOS.ipynb).
4. [Low-rank factor analysis](https://nbviewer.jupyter.org/github/Shuvomoy/NExOS.jl/blob/master/tutorials/Low-rank_factor_analysis_using_NExOS.ipynb).


## Citing
If you find `NExOS.jl` useful in your project, we kindly request that you cite the following paper:
```
@article{NExOS,
  title={Exterior-point Operator Splitting for Nonconvex Learning},
  author={Das Gupta, Shuvomoy and Stellato, Bartolomeo and Van Parys, Bart P.G.},
  journal={arXiv preprint arXiv:2011.04552},
  note={\url{https://arxiv.org/pdf/2011.04552.pdf}},
  year={2020}
}
```
A preprint can be downloaded [here](https://arxiv.org/pdf/2011.04552.pdf).

## Reporting issues
Please report any issues via the [Github issue tracker](https://github.com/Shuvomoy/NExOS.jl/issues). All types of issues are welcome including bug reports, feature requests, implementation for a specific research problem and so on.

## Contact
Send an email :email: to [sdgupta@mit.edu](mailto:sdgupta@mit.edu) :rocket:!	


