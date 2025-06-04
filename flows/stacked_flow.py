from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .nf_utils import Flow


class StackedFlows(nn.Module):
    """Stack a list of transformations with a given based distribtuion.

    Args:
        transforms: list fo stacked transformations. list of Flows
        dim: dimension of input/output data. int
        base_dist: name of the base distribution. options: ['Normal']
    """

    def __init__(
        self,
        transforms: List[Flow],
        dim: int = 2,
        base_dist: str = "Normal",
        device="cpu",
    ):
        super().__init__()

        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList(
                [
                    transforms,
                ]
            )
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(
                f"transforms must a Flow or a list, but was {type(transforms)}"
            )

        self.dim = dim
        if base_dist == "Normal":
            self.base_dist = MultivariateNormal(
                torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device)
            )
        else:
            raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of a batch of data (slide 27).

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            log_prob: Log probability of the data, shape [batch_size]
        """

        B, D = x.shape

        ##########################################################
       
        z = x #Startting from the data point
        log_prob = torch.zeros(B, device=x.device) # this shape will be enforced by the assert below so we might as well set it already

        #Go through stacked transforms in reverse, each time applying the inverse pass and adding the log-determinant.
        for transform in reversed(self.transforms):
            z, inv_log_det = transform.inverse(z) # shape: z = [B,D], inv_log_det = [B,]
            log_prob += inv_log_det #Accumulate

        #Now z is in base space; and we need to add the base log-probabiliy
        log_prob += self.base_dist.log_prob(z) # shape: log_prob = [B], as set in the beginnign and enforced below

        ##########################################################

        assert log_prob.shape == (B,)

        return log_prob

    def rsample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from the transformed distribution (slide 31).

        Returns:
            x: sample after forward transformation, shape [batch_size, dim]
            log_prob: Log probability of x, shape [batch_size]
        """
        ##########################################################
        
        # Draw reparameterised samples from the base distibution
        z = self.base_dist.rsample((batch_size,))        # shape (B, D)
        log_prob = self.base_dist.log_prob(z)            # shape (B,)

        # Push samples forward through every transformation in order
        for transform in self.transforms:
            z, log_det = transform.forward(z)            # (B,D), (B,)
            log_prob = log_prob - log_det                # change-of-variables, get the log prob

        # After the loop z is in data space
        x = z

        ##########################################################

        assert x.shape == (batch_size, self.dim)
        assert log_prob.shape == (batch_size,)

        return x, log_prob


