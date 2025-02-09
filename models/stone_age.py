import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum

class ActivationType(Enum):
    GUMBEL = "gumbel"
    ARGMAX = "argmax"

def to_float_tensor(x):
    if x is not torch.FloatTensor:
        x = x.float()
    return x

def gumbel_softmax(logits, tau=1.0, beta=1.0, hard=False, dim=-1):
    noise = -torch.empty_like(
        logits, memory_format=torch.legacy_contiguous_format)
    gumbels = noise.exponential_().log()
    gumbels = logits + gumbels*beta
    gumbels = gumbels / tau
    m = torch.nn.Softmax(dim)
    y_soft = m(gumbels)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        zeroes = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format)
        y_hard = zeroes.scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class ArgMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        y_soft = input.softmax(dim=-1)
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(input, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)

        ctx.save_for_backward(y_soft, y_hard)
        return y_hard, y_soft

    @staticmethod
    def backward(ctx, grad_output, grad_out_y_soft):
        y_soft, y_hard = ctx.saved_tensors
        grad = grad_output * y_hard
        grad += grad_out_y_soft * y_soft
        return grad


def argmax(x):
    # Create a wrapper that only returns the first output
    return ArgMax.apply(x)[0]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class LinearSoftmax(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=ActivationType.GUMBEL, temperature=1.0, use_batch_norm=True):
        super(LinearSoftmax, self).__init__()
        self.__name__ = 'LinearSoftmax'
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.argmax = False
        self.activation = activation
        self.softmax_temp = temperature
        self.beta = 0.0
        self.alpha = 1.0
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.lin1(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation == ActivationType.ARGMAX:
            x_d = argmax(x)
        else:
            x_d = gumbel_softmax(x, hard=True, tau=self.softmax_temp, beta=self.beta)

        if np.random.random() > self.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x


class MLPSoftmax(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=ActivationType.GUMBEL, temperature=1.0, hidden_units=16, dropout=0.0):
        super(MLPSoftmax, self).__init__()
        self.__name__ = 'LinearSoftmax'
        self.mlp = MLP(in_channels, hidden_units, out_channels, 2, dropout)
        self.activation = activation
        self.argmax = False
        self.beta = 0.0
        self.alpha = 1.0
        self.softmax_temp = temperature

    def forward(self, x):
        x = self.mlp(x)
        if self.acivation == ActivationType.ARGMAX:
            x_d = argmax(x)
        else:
            x_d = gumbel_softmax(x, hard=True, tau=self.softmax_temp, beta=self.beta)

        if np.random.random() > self.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x

# class InputLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, softmax_temp, activation=ActivationType.GUMBEL):
#         super(InputLayer, self).__init__()
#         self.lin1 = torch.nn.Linear(in_channels, out_channels)
#         self.activation = activation
#         self.softmax_temp = softmax_temp
#         self.alpha = 1.0
#         self.beta = 0.0

#     def forward(self, x):
#         x = to_float_tensor(x)
#         x = self.lin1(x)
#         if self.:
#             x_d = torch.nn.functional.argmax(x)

