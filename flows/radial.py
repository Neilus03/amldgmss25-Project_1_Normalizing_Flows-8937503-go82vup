import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nf_utils import Flow


class Radial(Flow):
    """Radial transformation.

    Args:
        dim: dimension of input/output data, int
    """

    def __init__(self, dim: int = 2):
        """Create and initialize an affine transformation."""
        super().__init__()

        self.dim = dim

        self.x0 = nn.Parameter(
            torch.Tensor(
                self.dim,
            )
        )  # Vector used to parametrize z_0
        self.pre_alpha = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scalar used to indirectly parametrized \alpha
        self.pre_beta = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scaler used to indireclty parametrized \beta

        stdv = 1.0 / math.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation for the given input x.

        Args:
            x: input sample, shape [batch_size, dim]

        Returns:
            y: sample after forward transformation, shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE

        # Task 3: Radial forward transformation. f(z) = z + beta * H(alpha, r) * (z - z0)

        # Softplus to get positive values for alpha and beta
        alpha = F.softplus(self.pre_alpha).expand(B, -1)  # alpha = softplus(pre_alpha) -> alpha > 0, shape [B, 1] -> [B, 1]
        beta = -alpha + F.softplus(self.pre_beta).expand(B, -1)  # beta = -alpha + softplus(pre_beta), shape [B, 1] -> [B, 1]

        # Calculate r = ||x - x0||_2 (Euclidean distance between each x and z0)
        x0 = self.x0.expand(B, -1)  # Broadcast x0 to match batch size (shape [B, D])
        r = torch.norm(x - x0, dim=-1, keepdim=True)  # r is the distance for each sample, shape [B, 1]

        # Calculate the vector difference z = x - x0
        z = x - x0  # shape [B, D]

        # Normalize z to avoid scaling issues (z / r)
        z = z / (r + 1e-8)  # Avoid division by zero, shape [B, D]

        # Apply the transformation: y = x0 + (r ** alpha) * z + beta
        y = x0 + (r ** alpha) * z + beta * torch.ones_like(x)  # Broadcasting over batch, shape [B, D]

        # The log determinant of the Jacobian for this transformation is calculated as:
        # log_det_jac = sum of log(1 + alpha / r), since it's a radial transformation
        log_det_jac = torch.sum(torch.log(1 + alpha / (r + 1e-8)), dim=-1)  # shape [B]

        ##########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> None:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        raise ValueError("The inverse transformation is not known in closed form.")

if __name__ == "__main__":
    # Example usage
    radial_flow = Radial(dim=2)
    x = torch.randn(5, 2)  # Batch of 5 samples in 2D
    y, log_det_jac = radial_flow.forward(x)
    print("Forward transformation:")
    print("Input:")
    print(x)
    print("Output:")
    print(y)
    print("Log determinant Jacobian:")
    print(log_det_jac)