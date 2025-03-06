import random

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.functional import log_softmax
from torch_geometric.nn import MessagePassing, global_add_pool

from bronze_age.config import Config, LayerType, NetworkType
from bronze_age.models.concept_reasoner import (
    ConceptReasoningLayer,
    GlobalConceptReasoningLayer,
)


def to_float_tensor(x):
    if x is not torch.FloatTensor:
        x = x.float()
    return x


def gumbel_softmax(logits, tau=1.0, beta=1.0, hard=False, dim=-1):
    noise = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
    gumbels = noise.exponential_().log()
    gumbels = logits + gumbels * beta
    gumbels = gumbels / tau
    m = torch.nn.Softmax(dim)
    y_soft = m(gumbels)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        zeroes = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
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
        y_hard = torch.zeros_like(
            input, memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, index, 1.0)

        ctx.save_for_backward(y_soft, y_hard)
        return y_hard, y_soft

    @staticmethod
    def backward(ctx, grad_output, grad_out_y_soft):
        y_soft, y_hard = ctx.saved_tensors
        grad = grad_output * y_hard
        grad += grad_out_y_soft * y_soft
        return grad


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


def argmax(x):
    # Create a wrapper that only returns the first output
    return ArgMax.apply(x)[0]


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


class SoftmaxLayer(torch.nn.Module):
    """For usage in StoneAgeGNNLayer"""

    def __init__(self, out_channels, config: Config, linear_layer, use_batch_norm=None):
        super(SoftmaxLayer, self).__init__()
        self.__name__ = "LinearSoftmax"
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

    def forward(self, x, explain=False):
        x = self.lin1(x)

        if self.use_batch_norm:
            x = self.bn(x)

        if self.use_argmax():
            x_d = argmax(x)
        else:
            x_d = gumbel_softmax(
                x, hard=True, tau=self.config.temperature, beta=self.config.beta
            )

        if torch.rand(1).item() > self.config.alpha and self.training:
            x = (x + x_d) / 2
        else:
            x = x_d
        return x


class LinearSoftmax(SoftmaxLayer):
    """For usage in StoneAgeGNNLayer"""

    def __init__(self, in_channels, out_channels, config: Config, use_batch_norm=None):
        linear_layer = torch.nn.Linear(in_channels, out_channels)
        super(LinearSoftmax, self).__init__(
            out_channels=out_channels,
            config=config,
            linear_layer=linear_layer,
            use_batch_norm=use_batch_norm,
        )
        self.__name__ = "LinearSoftmax"


class MLPSoftmax(SoftmaxLayer):
    """For usage in StoneAgeGNNLayer"""

    def __init__(self, in_channels, out_channels, config: Config):
        linear_layer = MLP(
            in_channels, config.hidden_units, out_channels, 2, config.dropout
        )
        super(MLPSoftmax, self).__init__(
            out_channels=out_channels,
            config=config,
            linear_layer=linear_layer,
            use_batch_norm=False,
        )
        self.__name__ = "MLPSoftmax"


class InputLayer(LinearSoftmax):
    def __init__(self, in_channels, out_channels, config: Config):
        super(InputLayer, self).__init__(
            in_channels, out_channels, config, use_batch_norm=False
        )
        self.__name__ = "InputLayer"


class PoolingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, config: Config):
        super(PoolingLayer, self).__init__()
        self.__name__ = "PoolingLayer"
        if config.network == NetworkType.MLP:
            self.lin2 = MLP(
                in_channels, config.hidden_units, out_channels, 2, config.dropout
            )
        else:
            self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.lin2(x)
        return log_softmax(x, dim=-1)


class StoneAgeGNNLayer(MessagePassing):
    def __init__(
        self, in_channels, out_channels, bounding_parameter, config: Config, index=0
    ):
        super().__init__(aggr="add")
        self.__name__ = "stone-age-" + str(index)
        self.bounding_parameter = bounding_parameter

        if config.network == NetworkType.MLP:
            self.linear_softmax = MLPSoftmax(2 * in_channels, out_channels, config)
        else:
            self.linear_softmax = LinearSoftmax(in_channels, out_channels, config)

    def forward(self, x, edge_index, return_explanation=False):
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


def generate_names(n_concepts, in_channels, bounding_parameter):
    names = [f"s_{i}" for i in range(n_concepts)] + [
        f"s_{i}_count>{j}"
        for i in range(in_channels)
        for j in range(bounding_parameter)
    ]
    return names


class ConceptReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, emb_size, config: Config):
        super(ConceptReasonerModule, self).__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        self.concept_reasoner = ConceptReasoningLayer(
            emb_size=emb_size,
            n_classes=n_classes,
            temperature=config.concept_temperature,
        )
        self.concept_context_generator = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts, 2 * emb_size * self.n_concepts),
            torch.nn.LeakyReLU(),
        )

    def forward(self, combined, return_explanation=False, concept_names=None):
        concept_embs = self.concept_context_generator(combined)
        concept_embs_shape = combined.shape[:-1] + (self.n_concepts, 2 * self.emb_size)
        concept_embs = concept_embs.view(*concept_embs_shape)
        concept_pos = concept_embs[..., : self.emb_size]
        concept_neg = concept_embs[..., self.emb_size :]
        embedding = concept_pos * combined[..., None] + concept_neg * (
            1 - combined[..., None]
        )

        x = self.concept_reasoner(embedding, combined)

        if return_explanation:
            explanation = self.concept_reasoner.explain(
                embedding, combined, mode="global", concept_names=concept_names
            )
            return x, explanation

        return x


class GlobalConceptReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, config: Config):
        super(GlobalConceptReasonerModule, self).__init__()
        self.n_classes = n_classes
        self.concept_reasoner = GlobalConceptReasoningLayer(
            n_concepts, n_classes, temperature=config.concept_temperature
        )

    def forward(self, combined, return_explanation=False, concept_names=None):
        x = self.concept_reasoner(combined)

        if return_explanation:
            explanation = self.concept_reasoner.explain(
                combined, mode="global", concept_names=concept_names
            )
            return x, explanation

        return x


class BronzeAgeGNNLayerConceptReasoner(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        bounding_parameter,
        config: Config,
        index=0,
        a=10,
    ):
        super().__init__(aggr="add")
        self.__name__ = "stone-age-" + str(index)
        self.bounding_parameter = bounding_parameter
        self.register_buffer("_Y_range", torch.arange(bounding_parameter).float())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_size = config.concept_embedding_size
        self.n_concepts = in_channels + in_channels * bounding_parameter
        self.a = a
        self.index = index
        self.config = config
        if config.layer_type == LayerType.BronzeAgeConcept:
            self.reasoning_module = ConceptReasonerModule(
                n_concepts=in_channels + in_channels * bounding_parameter,
                emb_size=config.concept_embedding_size,
                n_classes=out_channels,
                config=config,
            )
        elif config.layer_type == LayerType.BronzeAgeGeneralConcept:
            self.reasoning_module = GlobalConceptReasonerModule(
                n_concepts=in_channels + in_channels * bounding_parameter,
                n_classes=out_channels,
                config=config,
            )
        else:
            raise ValueError(f"Invalid layer type {config.layer_type}")

    def forward(self, x, edge_index, return_explanation=False, return_entropy=False):
        return self.propagate(
            edge_index, x=x, return_explanation=return_explanation, return_entropy=return_entropy
        )

    def message(self, x_j, return_explanation=False):
        return x_j

    def aggregate(
        self, inputs, index, ptr, dim_size, return_explanation=False, return_entropy=False
    ):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        clamped_sum = torch.clamp(message_sums, min=0, max=self.bounding_parameter)
        states = F.elu(clamped_sum[..., None] - self._Y_range) - 0.5
        states = F.sigmoid(self.a * states)
        states = states.view(*states.shape[:-2], -1)
        if self.config.use_one_hot_output:
            states = states + states.detach().round().float() - states.detach()
        return states

    def update(self, inputs, x, return_explanation=False, return_entropy=False):
        # inputs is one hot encoding of current state
        # x has shape (num_states * bounding_parameter)
        # where x.reshape(num_states, bounding_parameter) a per state one-hot encoding
        # of the number of states in neighborhood
        # x[i*bounding_parameter + j] = 1 if there are more than j nodes in state i in the neighborhood of our node
        combined = torch.cat((x, inputs), 1)

        if return_explanation:
            concept_names = generate_names(
                self.in_channels, self.in_channels, self.bounding_parameter
            )
            x, explanation = self.reasoning_module(
                combined,
                return_explanation=True,
                concept_names=concept_names,
            )
        else:
            x = self.reasoning_module(combined)

        one_hot = differentiable_argmax(x)

        outputs = []
        if self.config.use_one_hot_output:
            outputs.append(one_hot)
        else:
            outputs.append(x)
        
        if return_entropy:
            entropy = torch.sum((x - one_hot) ** 2)
            outputs.append(entropy)
        
        if return_explanation:
            outputs.append(explanation)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


class BronzeAgeGNNLayer(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        bounding_parameter,
        config: Config,
        index=0,
        a=10,
    ):
        super().__init__(aggr="add")
        self.__name__ = "stone-age-" + str(index)
        self.bounding_parameter = bounding_parameter
        self.register_buffer("_Y_range", torch.arange(bounding_parameter).float())
        self.a = a
        if config.network == NetworkType.MLP:
            self.linear_softmax = MLPSoftmax(
                in_channels + bounding_parameter * in_channels, out_channels, config
            )
        else:
            self.linear_softmax = LinearSoftmax(
                in_channels + bounding_parameter * in_channels, out_channels, config
            )

    def forward(self, x, edge_index, return_explanation=False):
        return self.propagate(edge_index, x=x, return_explanation=return_explanation)

    def message(self, x_j, return_explanation=False):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size, return_explanation=False):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        clamped_sum = torch.clamp(message_sums, min=0, max=self.bounding_parameter)
        states = F.elu(clamped_sum[..., None] - self._Y_range) - 0.5
        states = F.sigmoid(self.a * states)
        return states.view(*states.shape[:-2], -1)

    def update(self, inputs, x, return_explanation=False):
        # inputs is one hot encoding of current state
        # x has shape (num_states * bounding_parameter)
        # where x.reshape(num_states, bounding_parameter) a per state one-hot encoding
        # of the number of states in neighborhood
        # x[i*bounding_parameter + j] = 1 if there are more than j nodes in state i in the neighborhood of our node
        combined = torch.cat((inputs, x), 1)
        x = self.linear_softmax(combined)
        return x


class StoneAgeGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, config: Config):
        super().__init__()

        self.use_pooling = config.dataset.uses_pooling
        self.skip_connection = config.skip_connection
        self.config = config
        state_size = config.state_size
        num_layers = config.num_layers
        bounding_parameter = config.bounding_parameter

        input_emb_size = 16
        output_emb_size = 32

        self.is_interpretable = (
            config.layer_type == LayerType.BronzeAgeConcept
            or config.layer_type == LayerType.BronzeAgeGeneralConcept
        )

        if self.is_interpretable:
            self.input = ConceptReasonerModule(
                n_concepts=in_channels,
                emb_size=input_emb_size,
                n_classes=state_size,
                config=config,
            )
        else:
            self.input = InputLayer(in_channels, state_size, config)

        if self.skip_connection:
            final_layer_inputs = (num_layers + 1) * state_size
        else:
            final_layer_inputs = state_size

        if self.is_interpretable:  # This gives awful results
            self.output = ConceptReasonerModule(
                n_concepts=final_layer_inputs,
                n_classes=out_channels,
                emb_size=output_emb_size,
                config=config,
            )
        else:
            self.output = PoolingLayer(final_layer_inputs, out_channels, config=config)

        self.stone_age = ModuleList()
        for i in range(num_layers):
            if self.is_interpretable:
                layer_constructor = BronzeAgeGNNLayerConceptReasoner
            elif config.layer_type == LayerType.BronzeAge:
                layer_constructor = BronzeAgeGNNLayer
            else:
                layer_constructor = StoneAgeGNNLayer
            self.stone_age.append(
                layer_constructor(
                    state_size,
                    state_size,
                    bounding_parameter=bounding_parameter,
                    config=config,
                    index=i,
                )
            )

    def forward(self, x, edge_index, batch=None, return_explanation=False, return_entropy=False):
        explanations = {}
        if return_explanation:
            x, explanation = self.input(x.float(), return_explanation=True)
            explanations["input"] = explanation
        else:
            x = self.input(x.float())

        x_one_hot = differentiable_argmax(x)
        if self.config.use_one_hot_output:
            x = x_one_hot

        xs = [x]
        entropy = {"input": torch.sum((x - x_one_hot) ** 2)}

        for layer in self.stone_age:
            if not return_explanation:
                x, ent = layer(x, edge_index, return_explanation=return_explanation, return_entropy=True)
            else:
                x, ent, explanation = layer(x, edge_index, return_explanation=True, return_entropy=True)
                explanations[layer.__name__] = explanation
            entropy[layer.__name__] = ent
            xs.append(x)

        if self.use_pooling:
            x = global_add_pool(x, batch)
            xs = [global_add_pool(xi, batch) for xi in xs]
        if self.skip_connection:
            x = torch.cat(xs, dim=1)

        if return_explanation:
            x, explanation = self.output(x, return_explanation=True)
            explanations["output"] = explanation
        else:
            x = self.output(x)

        if self.config.use_one_hot_output:
            x = differentiable_argmax(x)
        
        outputs = [x]
        if return_entropy:
            outputs.append(entropy)
        if return_explanation:
            outputs.append(explanations)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def explain(self, x, edge_index, batch=None):
        result = self.forward(x, edge_index, batch=batch, explain=True)
        # TODO: Implement explanation
