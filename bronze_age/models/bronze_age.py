import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

from bronze_age.config import AggregationMode
from bronze_age.config import BronzeConfig as Config
from bronze_age.config import LayerTypeBronze as LayerType
from bronze_age.models.stone_age import ConceptReasonerModule, InputLayer


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
            return differentiable_argmax(x)

class DifferentiableArgmax(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x):
        return differentiable_argmax(x)        


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
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

    def forward(self, x, explain=False):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    


class BronzeAgeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, config: Config, name=None, non_linearity=None):
        super().__init__()
        self.config = config
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        if name is not None:
            self.__name__ = name
        
        if config.layer_type == LayerType.LINEAR:
            self.f = torch.nn.Linear(in_channels, out_channels)
            self.non_linearity = GumbelSoftmax(config)
        elif config.layer_type == LayerType.MLP:
            self.f = MLP(in_channels, config.hidden_units, out_channels, 2, config.dropout)
            self.non_linearity = GumbelSoftmax(config)
        elif config.layer_type == LayerType.DEEP_CONCEPT_REASONER:
            self.f = ConceptReasonerModule(in_channels, out_channels, config)
            self.non_linearity = nn.Identity() if not self.config.use_one_hot_output else DifferentiableArgmax(config)
        elif config.layer_type == LayerType.GLOBAL_DEEP_CONCEPT_REASONER:
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.non_linearity = non_linearity or self.non_linearity

    def forward(self, x):
        x1 = self.f(x)
        x2 = self.non_linearity(x1)

        entropy_loss = nn.functional.mse_loss(x2, x1, reduction="mean")
        return x2, entropy_loss
    

class BronzeAgeGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, config: Config):
        super(BronzeAgeGNNLayer, self).__init__(aggr="add")
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels

        if config.aggregation_mode == AggregationMode.STONE_AGE:
            self.layer = BronzeAgeLayer(in_channels, 2*out_channels, config)
        elif config.aggregation_mode == AggregationMode.BRONZE_AGE:
            bounding_parameter = config.bounding_parameter
            out_channels = out_channels * (bounding_parameter + 1)
            self.bounding_parameter = bounding_parameter
            self.register_buffer("_Y_range", torch.arange(bounding_parameter).float())
        
            self.layer = BronzeAgeLayer(in_channels, out_channels, config)
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        clamped_sum = torch.clamp(message_sums, min=0, max=self.bounding_parameter)
        if self.config.aggregation_mode == AggregationMode.STONE_AGE:
            return clamped_sum
        elif self.config.aggregation_mode == AggregationMode.BRONZE_AGE:
            states = F.elu(clamped_sum[..., None] - self._Y_range) - 0.5
            states = F.sigmoid(self.a * states)
            states = states.view(*states.shape[:-2], -1)
            if self.config.use_one_hot_output:
                states = states + states.detach().round().float() - states.detach()
            return states
        else:
            raise NotImplementedError

    def update(self, inputs, x):
        combined = torch.cat((inputs, x), 1)
        return self.layer(combined)
     
    
LOGIC_LAYERS = {
    LayerType.DEEP_CONCEPT_REASONER: ConceptReasonerModule,
    LayerType.GLOBAL_DEEP_CONCEPT_REASONER: None, # TODO
    LayerType.LINEAR: InputLayer,
    LayerType.MLP: InputLayer,
}
class BronzeAgeGNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, config: Config):
        super(BronzeAgeGNN, self).__init__()
        self.config = config
        
        self.use_pooling = config.dataset.uses_pooling
        self.skip_connection = config.skip_connection
        self.config = config
        
        state_size = config.state_size
        num_layers = config.num_layers
        bounding_parameter = config.bounding_parameter

        input_emb_size = 16
        output_emb_size = 32

        self.input = BronzeAgeLayer(in_channels, input_emb_size, config, name="InputLayer")

        final_layer_inputs = (num_layers + 1) * state_size if self.skip_connection else state_size
        final_non_linearity = nn.LogSoftmax(dim=-1) if config.layer_type in [LayerType.LINEAR, LayerType.MLP] else None
        self.output = BronzeAgeLayer(final_layer_inputs, out_channels, config, name="PoolingLayer", non_linearity=final_non_linearity)

        self.stone_age = nn.ModuleList()
        for i in range(num_layers):
            self.stone_age.append(BronzeAgeGNNLayer(state_size, state_size, config, name=f"StoneAgeLayer-{i}"))
    
    def forward(self, x, edge_index, batch=None):
        x, loss_term = self.input(x.float())
        entropy = {"input": loss_term}

        for layer in self.stone_age:
            x, loss_term = layer(x, edge_index)
            entropy[layer.__name__] = loss_term
            xs.append(x)

        if self.use_pooling:
            x = global_add_pool(x, batch)
            xs = [global_add_pool(xi, batch) for xi in xs]
        if self.skip_connection:
            x = torch.cat(xs, dim=1)

        x, _ = self.output(x)

        return x, entropy