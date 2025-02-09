
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.functional import log_softmax
from torch_geometric.nn import MessagePassing, global_add_pool

from bronze_age.config import Config, NetworkType


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

class SoftmaxLayer(torch.nn.Module):
    """For usage in StoneAgeGNNLayer"""

    def __init__(self, out_channels, config: Config, linear_layer, use_batch_norm=None):
        super(SoftmaxLayer, self).__init__()
        self.__name__ = 'LinearSoftmax'
        self.lin1 = linear_layer
        self.config = config
        if use_batch_norm is None:
            self.use_batch_norm = self.config.use_batch_norm
        else:
            self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels)

    def use_argmax(self):
        # return self.config.activation == ActivationType.ARGMAX
        return not self.training

    def forward(self, x):
        x = self.lin1(x)

        if self.use_batch_norm:
            x = self.bn(x)

        if self.use_argmax():
            x_d = argmax(x)
        else:
            x_d = gumbel_softmax(x, hard=True, tau=self.config.temperature, beta=self.config.beta)

        if torch.rand(1).item() > self.config.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x

class LinearSoftmax(SoftmaxLayer):
    """For usage in StoneAgeGNNLayer"""

    def __init__(self, in_channels, out_channels, config: Config, use_batch_norm=None):
        linear_layer = torch.nn.Linear(in_channels, out_channels)
        super(LinearSoftmax, self).__init__(out_channels=out_channels, config=config, linear_layer=linear_layer, use_batch_norm=use_batch_norm)
        self.__name__ = 'LinearSoftmax'

class MLPSoftmax(SoftmaxLayer):
    """For usage in StoneAgeGNNLayer"""

    def __init__(self, in_channels, out_channels, config: Config):
        linear_layer = MLP(in_channels, config.hidden_units, out_channels, 2, config.dropout)
        super(MLPSoftmax, self).__init__(out_channels=out_channels, config=config, linear_layer=linear_layer, use_batch_norm=False)
        self.__name__ = 'MLPSoftmax'


class InputLayer(LinearSoftmax):
    def __init__(self, in_channels, out_channels, config: Config):
        super(InputLayer, self).__init__(in_channels, out_channels, config, use_batch_norm=False)
        self.__name__ = 'InputLayer'


class PoolingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, config: Config):
        super(PoolingLayer, self).__init__()
        self.__name__ = 'PoolingLayer'
        if config.network == NetworkType.MLP:
            self.lin2 = MLP(in_channels, config.hidden_units, out_channels, 2, config.dropout)
        else:
            self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.lin2(x)
        return log_softmax(x, dim=-1)


class StoneAgeGNNLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, bounding_parameter, config: Config, index=0):
        super().__init__(aggr='add')
        self.__name__ = 'stone-age-' + str(index)
        self.bounding_parameter = bounding_parameter

        if config.network == NetworkType.MLP:
            self.linear_softmax = MLPSoftmax(in_channels, out_channels, config)
        else:
            self.linear_softmax = LinearSoftmax(in_channels, out_channels, config)


    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        return torch.clamp(message_sums, min=0, max=self.bounding_parameter)

    def update(self, inputs, x):
        combined = torch.cat((inputs, x), 1)
        x = self.linear_softmax(combined)
        return x


class StoneAgeGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, config: Config):
        super().__init__()

        self.use_pooling = config.use_pooling
        self.skip_connection = config.skip_connection
        state_size = config.state_size
        num_layers = config.num_layers
        bounding_parameter = config.bounding_parameter
        
        # self.num_layers = num_layers

        self.input = InputLayer(in_channels, state_size, config)

        if self.skip_connection:
            self.output = PoolingLayer((num_layers + 1) * state_size, out_channels, config=config)
        else:
            self.output = PoolingLayer(state_size, out_channels, config)

        self.stone_age = ModuleList()
        for i in range(num_layers):
            self.stone_age.append(
                StoneAgeGNNLayer(state_size * 2,
                                 state_size,
                                 bounding_parameter=bounding_parameter,
                                 config=config,
                                 index=i))

    def forward(self, x, edge_index, batch=None):

        x = self.input(x)
        xs = [x]
        for layer in self.stone_age:
            x = layer(x, edge_index)
            xs.append(x)

        if self.use_pooling:
            x = global_add_pool(x, batch)
            xs = [global_add_pool(xi, batch) for xi in xs]
        if self.skip_connection:
            x = torch.cat(xs, dim=1)
        x = self.output(x)
        return x

