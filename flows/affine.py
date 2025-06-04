from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .nf_utils import Flow


class Affine(Flow):
    """Affine transformation y = e^a * x + b.

    Args:
        dim (int): dimension of input/output data. int
    """

    def __init__(self, dim: int = 2):
        """Create and init an affine transformation."""
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.ones(self.dim) * 0.5) # a
        self.shift = nn.Parameter(torch.zeros(self.dim))  # b

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward transformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################

        scale = torch.exp(self.log_scale) # e^a (shape [D])
        y = scale * x + self.shift # y = e^a * x + b (broadcasting over batch) 
        log_det_jac = torch.sum(self.log_scale, dim=0)  # sum of log scales
        log_det_jac = log_det_jac * torch.ones(B, device=x.device)  # shape [B]

        ##########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        B, D = y.shape

        ##########################################################

        scale = torch.exp(self.log_scale) # shape [D]
        x = (y - self.shift) / scale # same as y * exp(-log_scale)
        inv_log_det_jac = -torch.sum(self.log_scale) # sum of log scales
        inv_log_det_jac = inv_log_det_jac * torch.ones(B, device=y.device)
        

        ##########################################################

        assert x.shape == (B, D)
        assert inv_log_det_jac.shape == (B,)

        return x, inv_log_det_jac

if __name__ == "__main__":
    # Example usage
    affine = Affine(dim=2)
    x = torch.randn(5, 2)  # Batch of 5 samples, each of dimension 2
    y, log_det_jac = affine.forward(x)
    print("Forward transformation:")
    print("Input:")
    print(x)
    print("Output:")
    print(y)
    print("Log determinant Jacobian:")
    print(log_det_jac)