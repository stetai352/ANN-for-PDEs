from pymor.basic import *

# scale arguments of cos and sin by π to get more oscillations and thus more interesting patterns
# offset waves by only 1.1 (instead of 2) to make patterns more visible (still strictly positive because e₀ ≥ 1 and μ₀ ≥ 0.1)
def defineSinusoidProblem():
    
    p0 = ProjectionParameterFunctional('diffusion', size = 4, index = 0)
    p1 = ProjectionParameterFunctional('diffusion', size = 4, index = 1)
    p2 = ProjectionParameterFunctional('diffusion', size = 4, index = 2)
    p3 = ProjectionParameterFunctional('diffusion', size = 4, index = 3)


    e0 = ExpressionFunction('1.1 + cos(x[0]*(2**0)*pi) * sin(x[1]*(2**0)*pi)', 2)
    e1 = ExpressionFunction('1.1 + cos(x[0]*(2**1)*pi) * sin(x[1]*(2**1)*pi)', 2)
    e2 = ExpressionFunction('1.1 + cos(x[0]*(2**2)*pi) * sin(x[1]*(2**2)*pi)', 2)
    e3 = ExpressionFunction('1.1 + cos(x[0]*(2**3)*pi) * sin(x[1]*(2**3)*pi)', 2)

    problem = StationaryProblem(
        domain = RectDomain(),
        rhs = ConstantFunction(dim_domain = 2, value = 1.),
        diffusion = LincombFunction([e0, e1, e2, e3],
                                    [p0, p1, p2, p3]),
        parameter_ranges = (0.1, 1.),
    )

    fom, data = discretize_stationary_cg(problem, diameter = 1/200)

    """mu = [0.1,1,0.1,1]
    U = fom.solve(mu)
    fom.visualize(U)"""

    return fom, data