import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

from bronze_age.config import AggregationMode
from bronze_age.config import BronzeConfig as Config
from bronze_age.config import LayerTypeBronze as LayerType
from bronze_age.config import NonLinearity
from bronze_age.models.concept_reasoner import (
    ConceptReasonerModule,
    GlobalConceptReasonerModule,
)


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

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    


class BronzeAgeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, config: Config, layer_type: LayerType = None, non_linearity=None, name=None):
        super().__init__()
        self.config = config
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        if name is not None:
            self.__name__ = name
        
        layer_type = layer_type or config.layer_type
        if layer_type == LayerType.LINEAR:
            self.f = torch.nn.Linear(in_channels, out_channels)
        elif layer_type == LayerType.MLP:
            self.f = MLP(in_channels, config.hidden_units, out_channels, 2, config.dropout)
        elif layer_type == LayerType.DEEP_CONCEPT_REASONER:
            self.f = ConceptReasonerModule(in_channels, out_channels, config.concept_embedding_size, config)
        elif layer_type == LayerType.GLOBAL_DEEP_CONCEPT_REASONER:
            self.f = GlobalConceptReasonerModule(in_channels, out_channels, config)
        else:
            raise NotImplementedError

        if config.nonlinearity is None:
            self.non_linearity = nn.Identity()
        elif config.nonlinearity == NonLinearity.GUMBEL_SOFTMAX:
            self.non_linearity = GumbelSoftmax(config)
        elif config.nonlinearity == NonLinearity.DIFFERENTIABLE_ARGMAX:
            self.non_linearity = DifferentiableArgmax(config)
        else:
            raise NotImplementedError
        
        if config.evaluation_nonlinearity is None:
            self.eval_non_linearity = self.non_linearity
        elif config.evaluation_nonlinearity == NonLinearity.GUMBEL_SOFTMAX:
            self.eval_non_linearity = GumbelSoftmax(config)
        elif config.evaluation_nonlinearity == NonLinearity.DIFFERENTIABLE_ARGMAX:
            self.eval_non_linearity = DifferentiableArgmax(config)
        else:
            raise NotImplementedError
        
        self.non_linearity = non_linearity or self.non_linearity
        self.eval_non_linearity = non_linearity or self.eval_non_linearity

    def forward(self, x, return_explanation=False):
        if return_explanation:
            x1, explanation = self.f(x, return_explanation=True)
        else:
            x1, explanation = self.f(x), None
            
        x2 = self.non_linearity(x1) if self.training else self.eval_non_linearity(x1)

        entropy_loss = nn.functional.mse_loss(x2, x1, reduction="mean")

        
            
        return x2, entropy_loss, explanation
    

class BronzeAgeGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, config: Config, name=None):
        super(BronzeAgeGNNLayer, self).__init__(aggr="add")
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.a = config.a
        bounding_parameter = config.bounding_parameter
        self.bounding_parameter = bounding_parameter
        if config.aggregation_mode == AggregationMode.STONE_AGE:
            self.layer = BronzeAgeLayer(2*in_channels, out_channels, config)
        elif config.aggregation_mode == AggregationMode.BRONZE_AGE:
            bounding_parameter = config.bounding_parameter
            self.register_buffer("_Y_range", torch.arange(bounding_parameter).float())
        
            self.layer = BronzeAgeLayer(in_channels * (bounding_parameter + 1), out_channels, config)
        else:
            raise NotImplementedError
        
        if name is not None:
            self.__name__ = name

    def forward(self, x, edge_index, return_explanation=False):
        return self.propagate(edge_index, x=x, return_explanation=return_explanation)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        clamped_sum = torch.clamp(message_sums, min=0, max=self.bounding_parameter)
        if self.config.aggregation_mode == AggregationMode.STONE_AGE:
            return clamped_sum
        elif self.config.aggregation_mode in [AggregationMode.BRONZE_AGE, AggregationMode.BRONZE_AGE_ROUNDED]:
            states = F.elu(clamped_sum[..., None] - self._Y_range) - 0.5
            states = F.sigmoid(self.a * states)
            states = states.view(*states.shape[:-2], -1)
            if self.config.aggregation_mode == AggregationMode.BRONZE_AGE_ROUNDED:
                states = states + states.detach().round().float() - states.detach()
            return states
        else:
            raise NotImplementedError

    def update(self, inputs, x, return_explanation=False):
        combined = torch.cat((x, inputs), 1)
        return self.layer(combined, return_explanation=return_explanation)
     
    
class BronzeAgeGNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, config: Config):
        super(BronzeAgeGNN, self).__init__()
        self.config = config
        
        self.use_pooling = config.dataset.uses_pooling
        self.skip_connection = config.skip_connection
        self.config = config
        
        state_size = config.state_size
        num_layers = config.num_layers

        self.input = BronzeAgeLayer(in_channels, state_size, config, layer_type=LayerType.LINEAR if config.layer_type == LayerType.MLP else config.layer_type, name="InputLayer")

        final_layer_inputs = (num_layers + 1) * state_size if self.skip_connection else state_size
        final_non_linearity = nn.LogSoftmax(dim=-1) if config.layer_type in [LayerType.LINEAR, LayerType.MLP] else None
        self.output = BronzeAgeLayer(final_layer_inputs, out_channels, config, non_linearity=final_non_linearity, name="PoolingLayer",)

        self.stone_age = nn.ModuleList()
        for i in range(num_layers):
            self.stone_age.append(BronzeAgeGNNLayer(state_size, state_size, config, name=f"StoneAgeLayer-{i}"))
    
    def forward(self, x, edge_index, batch=None, return_explanation=False):
        x, loss_term, explanation = self.input(x.float())
        entropy = {"input": loss_term}
        explanations = {"input": explanation}
        xs = [x]
        for layer in self.stone_age:
            x, loss_term, explanation = layer(x, edge_index)
            entropy[layer.__name__] = loss_term
            explanations[layer.__name__] = explanation
            xs.append(x)

        if self.use_pooling:
            x = global_add_pool(x, batch)
            xs = [global_add_pool(xi, batch) for xi in xs]
        if self.skip_connection:
            x = torch.cat(xs, dim=1)

        x, _, explanation = self.output(x)
        explanations["output"] = explanation

        if not return_explanation:
            explanations = None
    
        return x, entropy, explanations