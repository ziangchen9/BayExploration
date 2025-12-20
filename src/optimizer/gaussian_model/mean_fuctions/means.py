import gpytorch
import torch
from gpytorch.means import ConstantMean, LinearMean


class QuadraticMean(gpytorch.means.Mean):
    def __init__(
        self, batch_shape: torch.Size = torch.Size(), bias: bool = True, d: int = 2
    ):
        super().__init__()
        self.register_parameter(
            name="second", parameter=torch.nn.Parameter(torch.randn(*batch_shape, d, 1))
        )
        self.register_parameter(
            name="first", parameter=torch.nn.Parameter(torch.randn(*batch_shape, d, 1))
        )
        if bias:
            self.register_parameter(
                name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
            )
        else:
            self.bias = None

    def forward(self, x):
        result = x.pow(2).matmul(self.second).squeeze(-1) + x.matmul(
            self.first
        ).squeeze(-1)
        if self.bias is not None:
            result = result + self.bias
        return result


MEAN_MODULE_MAP = {
    "constant": ConstantMean,
    "linear": LinearMean,
    "quadratic": QuadraticMean,
}
