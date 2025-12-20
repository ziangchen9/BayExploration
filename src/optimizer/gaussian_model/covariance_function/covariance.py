from gpytorch.kernels import MaternKernel, RBFKernel

COVARIANCE_MODULE_MAP = {
    "rbf": RBFKernel,
    "matern1_5": lambda nu: MaternKernel(nu=1.5),
    "matern2_5": lambda nu: MaternKernel(nu=2.5),
}
