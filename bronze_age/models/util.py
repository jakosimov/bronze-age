
import torch
from torch import nn
from torch.nn import functional as F

from bronze_age.config import BronzeConfig as Config


def differentiable_argmax(x):
    y_soft = x.softmax(dim=-1)
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(
        -1, index, 1.0
    )

    # Return y_hard detached to prevent mixing correct gradients

    # Use the STE trick:
    # We add the softmax and subtract the detached softmax, letting the gradient pass through
    return y_hard.detach() + y_soft - y_soft.detach()


class GumbelSoftmax(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tau = config.temperature

    def forward(self, x):
        if self.training:
            return F.gumbel_softmax(x, hard=True, tau=self.tau)
        else:
            return F.one_hot(x.argmax(dim=-1), x.shape[-1]).to(
                dtype=x.dtype, device=x.device
            )  # exact ones are needed for Decision Trees for some reason


class DifferentiableArgmax(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x):
        if self.training:
            return differentiable_argmax(x)
        else:
            return F.one_hot(x.argmax(dim=-1), x.shape[-1]).to(
                dtype=x.dtype, device=x.device
            )  # exact ones are needed for Decision Trees for some reason
