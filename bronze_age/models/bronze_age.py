from collections import defaultdict
from copy import deepcopy
from functools import partial

import lightning
import numpy as np
import sklearn.tree as sktree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from lightning.pytorch import loggers as pl_loggers
from sklearn.tree import DecisionTreeClassifier
from torch_geometric.nn import MessagePassing, global_add_pool

from bronze_age.config import AggregationMode
from bronze_age.config import BronzeConfig as Config
from bronze_age.config import LayerTypeBronze as LayerType
from bronze_age.config import LossMode, NonLinearity
from bronze_age.models.concept_reasoner import (
    ConceptReasonerModule,
    GlobalConceptReasonerModule,
    MemoryBasedReasonerModule,
)
from bronze_age.models.prune_tree import (
    get_max_depth,
    get_traversal_order,
    prune_node_from_tree,
)
from bronze_age.models.util import DifferentiableArgmax, GumbelSoftmax


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

    def forward(self, x, return_explanation=False, concept_names=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x, torch.tensor(0.0), None


class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, return_explanation=False, concept_names=None):
        x = self.lin(x)
        return x, torch.tensor(0.0), None


class BronzeAgeLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        config: Config,
        layer_type: LayerType | None = None,
        non_linearity=None,
        name=None,
    ):
        super().__init__()
        self.config = config
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        if name is not None:
            self.__name__ = name

        layer_type = layer_type or config.layer_type
        if layer_type == LayerType.LINEAR:
            self.f = Linear(in_channels, out_channels)
        elif layer_type == LayerType.MLP:
            self.f = MLP(
                in_channels, config.hidden_units, out_channels, 2, config.dropout
            )
        elif layer_type == LayerType.DEEP_CONCEPT_REASONER:
            self.f = ConceptReasonerModule(
                in_channels, out_channels, config.concept_embedding_size, config
            )
        elif layer_type == LayerType.GLOBAL_DEEP_CONCEPT_REASONER:
            self.f = GlobalConceptReasonerModule(in_channels, out_channels, config)
        elif layer_type == LayerType.MEMORY_BASED_CONCEPT_REASONER:
            self.f = MemoryBasedReasonerModule(in_channels, out_channels, config)
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

        self.inputs_list = None
        self.outputs_list = None
        self.mask_ref = None

    def set_inputs_outputs(self, inputs_list, outputs_list, mask_ref):
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list
        self.mask_ref = mask_ref

    def clear_inputs_outputs(self):
        self.inputs_list = None
        self.outputs_list = None
        self.mask_ref = None

    def forward(self, x, return_explanation=False, concept_names=None):
        if self.inputs_list is not None:
            self.inputs_list.append(
                x[self.mask_ref.current_mask].detach().cpu().numpy()
            )

        x1, aux_loss, explanation = self.f(
            x, return_explanation=return_explanation, concept_names=concept_names
        )

        x2 = self.non_linearity(x1) if self.training else self.eval_non_linearity(x1)

        if self.outputs_list is not None:
            self.outputs_list.append(
                x2[self.mask_ref.current_mask].detach().cpu().numpy()
            )

        return x2, aux_loss, explanation


class BronzeAgeDecisionTree(nn.Module):
    def __init__(
        self,
        tree,
        out_channels: int,
        state_size: int,
        config: Config,
        use_linear_feature_combinations=False,
    ):
        super().__init__()
        self.tree = tree
        self.config = config
        self.state_size = state_size
        self.use_linear_feature_combinations = use_linear_feature_combinations
        self.out_channels = out_channels

    @staticmethod
    def _preprocess_features(x, num_states, use_linear_feature_combinations=False):
        if use_linear_feature_combinations:
            current_state = x[..., :num_states]
            neighbors = x[..., num_states:]
            neighbors_difference = neighbors[..., :, None] > neighbors[..., None, :]
            # remove diagonal and flatten rest
            neighbors_difference = neighbors_difference[
                :, ~np.eye(neighbors_difference.shape[-1], dtype=bool)
            ].reshape(neighbors_difference.shape[0], -1)
            x = np.concatenate(
                (current_state, neighbors, neighbors_difference), axis=-1
            )
        return x

    @staticmethod
    def from_data(
        x,
        y,
        out_channels: int,
        state_size: int,
        config: Config,
        use_linear_feature_combinations=False,
        layer_name=None,
    ):
        tree = DecisionTreeClassifier(
            random_state=0, max_leaf_nodes=config.max_leaf_nodes
        )
        x = BronzeAgeDecisionTree._preprocess_features(
            x, state_size, use_linear_feature_combinations
        )
        tree.fit(x, y)
        return BronzeAgeDecisionTree(
            tree,
            out_channels,
            state_size,
            config,
            use_linear_feature_combinations=use_linear_feature_combinations,
        )

    def forward(self, x, return_explanation=False, concept_names=None):
        x1 = x.cpu().detach().numpy()
        x1 = self._preprocess_features(
            x1, self.state_size, self.use_linear_feature_combinations
        )
        y = torch.tensor(self.tree.predict(x1)).to(device=x.device, dtype=torch.long)
        return (
            F.one_hot(y, self.out_channels).to(dtype=x.dtype, device=x.device),
            torch.tensor(0.0).to(device=x.device),
            None,
        )


def _binary_cross_entropy_loss(y_hat, y, class_weights):
    y_one_hot = F.one_hot(y.long(), num_classes=y_hat.shape[-1]).float()
    return F.binary_cross_entropy(y_hat, y_one_hot, weight=class_weights)


def _cross_entropy_loss(y_hat, y, class_weights):
    return F.cross_entropy(y_hat, y, weight=class_weights)


class ConceptReasonerTrainerModule(lightning.LightningModule):
    def __init__(self, layer_dict, config):
        super().__init__()
        self.layer_dict = torch.nn.ModuleDict(
            {fmt(key): layer for key, layer in layer_dict.items()}
        )
        self.config = config

    def forward(self, x_dict):
        return {
            key: self.layer_dict[fmt(key)](
                x,
            )[0]
            for key, x in x_dict.items()
        }

    def training_step(self, batch):
        x = {key: x[0] for key, x in batch.items()}
        y = {key: x[1] for key, x in batch.items()}

        y_hat = self(x)
        loss = 0
        for key in y.keys():
            loss += _binary_cross_entropy_loss(y_hat[key], y[key], class_weights=None)
        loss = loss / len(y)
        self.log(
            "train_loss_trainer_teacher",
            loss,
            on_epoch=True,
            on_step=False,
            batch_size=y["input"].size(0),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


class BronzeAgeGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, config: Config, name=None):
        super(BronzeAgeGNNLayer, self).__init__(aggr="add")
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.a = config.a

        bounding_parameter = config.bounding_parameter
        self.bounding_parameter = bounding_parameter

        bounding_parameter = config.bounding_parameter
        self.register_buffer("_Y_range", torch.arange(bounding_parameter).float())

        if config.aggregation_mode == AggregationMode.STONE_AGE:
            self.layer = BronzeAgeLayer(2 * in_channels, out_channels, config)
        elif config.aggregation_mode in [
            AggregationMode.BRONZE_AGE,
            AggregationMode.BRONZE_AGE_ROUNDED,
        ]:
            self.layer = BronzeAgeLayer(
                in_channels * (bounding_parameter + 1), out_channels, config
            )
        else:
            raise NotImplementedError

        if name is not None:
            self.__name__ = name

        self.inputs_list = None
        self.outputs_list = None
        self.mask_ref = None

    def set_inputs_outputs(self, inputs_list, outputs_list, mask_ref):
        self.inputs_list = inputs_list
        self.outputs_list = outputs_list
        self.mask_ref = mask_ref

    def clear_inputs_outputs(self):
        self.inputs_list = None
        self.outputs_list = None
        self.mask_ref = None

    def forward(self, x, edge_index, return_explanation=False):
        return self.propagate(edge_index, x=x, return_explanation=return_explanation)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        clamped_sum = torch.clamp(message_sums, min=0, max=self.bounding_parameter)

        states = F.elu(clamped_sum[..., None] - self._Y_range) - 0.5
        states = F.sigmoid(self.a * states)
        states = states.view(*states.shape[:-2], -1)

        rounded_states = states + states.detach().round().float() - states.detach()
        if self.inputs_list is not None:
            return clamped_sum, states, rounded_states

        if self.config.aggregation_mode == AggregationMode.STONE_AGE:
            return clamped_sum
        elif self.config.aggregation_mode == AggregationMode.BRONZE_AGE:
            return states
        elif self.config.aggregation_mode == AggregationMode.BRONZE_AGE_ROUNDED:
            return rounded_states
        else:
            raise NotImplementedError

    def _get_concept_names(self):
        if self.config.aggregation_mode == AggregationMode.STONE_AGE:
            return [f"s_{i}" for i in range(self.in_channels)] + [
                f"s_{i}_count" for i in range(self.in_channels)
            ]
        elif self.config.aggregation_mode in [
            AggregationMode.BRONZE_AGE,
            AggregationMode.BRONZE_AGE_ROUNDED,
        ]:
            return [f"s_{i}" for i in range(self.in_channels)] + [
                f"s_{i}_count>{j}"
                for i in range(self.in_channels)
                for j in range(self.bounding_parameter)
            ]

    def update(self, inputs, x, return_explanation=False):
        if self.inputs_list is not None:
            clamped_sum, states, rounded_states = inputs
            combined_clamped = torch.cat((x, clamped_sum), 1)
            combined_states = torch.cat((x, states), 1)
            combined_rounded = torch.cat((x, rounded_states), 1)
            self.inputs_list.append(
                (
                    combined_clamped[self.mask_ref.current_mask].detach().cpu().numpy(),
                    combined_states[self.mask_ref.current_mask].detach().cpu().numpy(),
                    combined_rounded[self.mask_ref.current_mask].detach().cpu().numpy(),
                )
            )
            if self.config.aggregation_mode == AggregationMode.STONE_AGE:
                combined = combined_clamped
            elif self.config.aggregation_mode == AggregationMode.BRONZE_AGE:
                combined = combined_states
            elif self.config.aggregation_mode == AggregationMode.BRONZE_AGE_ROUNDED:
                combined = combined_rounded
        else:
            combined = torch.cat((x, inputs), 1)

        output, aux_loss, explanation = self.layer(
            combined,
            return_explanation=return_explanation,
            concept_names=self._get_concept_names(),
        )

        if self.outputs_list is not None:
            self.outputs_list.append(
                output[self.mask_ref.current_mask].detach().cpu().numpy()
            )
        return output, aux_loss, explanation


def fmt(key):
    return key.replace(".", "_")


class BronzeAgeGNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, config: Config):
        super(BronzeAgeGNN, self).__init__()
        self.config = config

        self.use_pooling = config.dataset.uses_pooling
        self.skip_connection = config.skip_connection
        self.config = config

        state_size = config.state_size
        num_layers = config.num_layers

        self.input = BronzeAgeLayer(
            in_channels,
            state_size,
            config,
            layer_type=(
                LayerType.LINEAR
                if config.layer_type == LayerType.MLP
                else config.layer_type
            ),
            name="InputLayer",
        )

        final_layer_inputs = (
            (self.config.num_recurrent_iterations * num_layers + 1) * state_size
            if self.skip_connection
            else state_size
        )
        final_non_linearity = (
            nn.LogSoftmax(dim=-1)
            if config.layer_type in [LayerType.LINEAR, LayerType.MLP]
            else None
        )
        if (
            config.layer_type in [LayerType.LINEAR, LayerType.MLP]
            and config.loss_mode == LossMode.BINARY_CROSS_ENTROPY
        ):
            final_non_linearity = nn.Sigmoid()
        self.output = BronzeAgeLayer(
            final_layer_inputs,
            out_channels,
            config,
            non_linearity=final_non_linearity,
            name="PoolingLayer",
        )

        self.stone_age = nn.ModuleList()
        for i in range(num_layers):
            self.stone_age.append(
                BronzeAgeGNNLayer(
                    state_size, state_size, config, name=f"StoneAgeLayer-{i}"
                )
            )

    def forward(self, x, edge_index, batch=None, return_explanation=False):
        x, loss_term, explanation = self.input(
            x.float(), return_explanation=return_explanation
        )
        entropy = {"input": loss_term}
        explanations = {"input": explanation}
        xs = [x]

        for iteration in range(self.config.num_recurrent_iterations):
            for layer in self.stone_age:
                x, loss_term, explanation = layer(
                    x, edge_index, return_explanation=return_explanation
                )
                entropy[f"{layer.__name__}_{iteration}"] = loss_term
                if iteration == 0:
                    explanations[layer.__name__] = explanation
                xs.append(x)

        if self.use_pooling:
            x = global_add_pool(x, batch)
            xs = [global_add_pool(xi, batch) for xi in xs]
        if self.skip_connection:
            x = torch.cat(xs, dim=1)

        x, _, explanation = self.output(x, return_explanation=return_explanation)
        explanations["output"] = explanation

        if not return_explanation:
            explanations = None

        return x, entropy, explanations

    def get_inputs_outputs_fancy(self, train_loader, aggregation_mode):
        inputs_train = defaultdict(list)
        outputs_train = defaultdict(list)

        class MaskRef:
            def __init__(self):
                self.current_mask = None

        mask_ref = MaskRef()

        self.get_submodule("input").set_inputs_outputs(
            inputs_train["input"], outputs_train["input"], mask_ref
        )
        self.get_submodule("output").set_inputs_outputs(
            inputs_train["output"], outputs_train["output"], mask_ref
        )

        for name, module in self.named_modules():
            if isinstance(module, BronzeAgeGNNLayer):
                module.set_inputs_outputs(
                    inputs_train[name], outputs_train[name], mask_ref
                )

        self.eval()
        for data in train_loader:
            if hasattr(data, "train_mask"):
                mask_ref.current_mask = data.train_mask
            else:
                mask_ref.current_mask = torch.ones(data.x.size(0), dtype=torch.bool)
            self(data.x, data.edge_index, batch=data.batch)
            mask_ref.current_mask = None

        for name in inputs_train.keys():
            module = self.get_submodule(name)
            module.clear_inputs_outputs()

        for key in inputs_train.keys():
            inputs = inputs_train[key]
            if key not in ["input", "output"]:
                stone_age_inputs = [stone_age_input for stone_age_input, _, _ in inputs]
                bronze_age_inputs = [
                    bronze_age_input for _, bronze_age_input, _ in inputs
                ]
                bronze_age_rounded_inputs = [
                    bronze_age_rounded_input
                    for _, _, bronze_age_rounded_input in inputs
                ]
                if aggregation_mode == AggregationMode.STONE_AGE:
                    inputs = stone_age_inputs
                elif aggregation_mode == AggregationMode.BRONZE_AGE:
                    inputs = bronze_age_inputs
                elif aggregation_mode == AggregationMode.BRONZE_AGE_ROUNDED:
                    inputs = bronze_age_rounded_inputs
            outputs = outputs_train[key]
            inputs_train[key] = np.concatenate(inputs)
            outputs_train[key] = np.concatenate(
                [np.argmax(output, axis=-1) for output in outputs]
            )

        new_inputs_train = {}
        new_outputs_train = {}
        for key in inputs_train.keys():
            if key not in ["input", "output"]:
                new_key = key + ".layer"
            else:
                new_key = key
            new_inputs_train[new_key] = inputs_train[key]
            new_outputs_train[new_key] = outputs_train[key]

        return new_inputs_train, new_outputs_train

    def get_inputs_outputs(self, train_loader):
        inputs_train = defaultdict(list)
        outputs_train = defaultdict(list)
        current_mask = None

        def _hook(module, input, output, key=None):
            if key is None:
                raise ValueError("Key must be provided")
            x = input[0]
            y = output[0]
            if current_mask is not None:
                x = x[current_mask]
                y = y[current_mask]
            y = torch.argmax(y, dim=-1, keepdim=False)
            inputs_train[key].append(x.detach().cpu().numpy())
            outputs_train[key].append(y.detach().cpu().numpy())

        hooks = []
        for name, module in self.named_modules():
            if isinstance(module, BronzeAgeLayer):
                hooks.append(module.register_forward_hook(partial(_hook, key=name)))

        self.eval()
        for data in train_loader:
            if hasattr(data, "train_mask"):
                current_mask = data.train_mask
            self(data.x, data.edge_index, batch=data.batch)
            current_mask = None

        for hook in hooks:
            hook.remove()
        for key in inputs_train.keys():
            inputs_train[key] = np.concatenate(inputs_train[key])
            outputs_train[key] = np.concatenate(outputs_train[key])

        return inputs_train, outputs_train

    def train_concept_model(self, train_loader, experiment_title=""):
        final_model = deepcopy(self)
        config = deepcopy(self.config)
        student_aggregation_mode = (
            config.student_aggregation_mode or config.aggregation_mode
        )

        inputs_train, outputs_train = final_model.get_inputs_outputs_fancy(
            train_loader, student_aggregation_mode
        )

        per_layer_datasets = {
            key: torch.utils.data.TensorDataset(
                torch.tensor(inputs_train[key]), torch.tensor(outputs_train[key])
            )
            for key in inputs_train.keys()
        }

        stack_dataset = torch.utils.data.StackDataset(**per_layer_datasets)
        train_data_loader = torch.utils.data.DataLoader(
            stack_dataset, batch_size=final_model.config.batch_size, shuffle=True
        )
        layer_dict = {}
        config.nonlinearity = None
        config.evaluation_nonlinearity = None
        final_model.config.aggregation_mode = student_aggregation_mode
        config.aggregation_mode = student_aggregation_mode
        for key in inputs_train.keys():
            out_channels = final_model.get_submodule(key).out_channels
            in_channels = inputs_train[key].shape[-1]

            new_module = BronzeAgeLayer(
                in_channels,
                out_channels,
                config,
                layer_type=self.config.student_layer_type,
                name=key,
            )
            layer_dict[key] = new_module

        trainer_model = ConceptReasonerTrainerModule(layer_dict, config)

        logger = pl_loggers.TensorBoardLogger(
            save_dir="lightning_logs", name=experiment_title + " concept_trainer"
        )
        trainer = lightning.Trainer(
            max_epochs=config.teacher_max_epochs,
            log_every_n_steps=1,
            enable_progress_bar=False,
            logger=logger,
        )

        trainer.fit(trainer_model, train_data_loader)

        for key, layer in layer_dict.items():
            final_model.set_submodule(key, layer)

        return final_model

    def to_decision_tree(self, train_loader):
        decision_tree = deepcopy(self)
        inputs_train, outputs_train = decision_tree.get_inputs_outputs(train_loader)

        for key in inputs_train.keys():
            out_channels = decision_tree.get_submodule(key).out_channels
            use_linear_feature_combinations = not (
                key == "input"
                or (key == "output" and not self.config.dataset.uses_pooling)
            )
            num_states = (
                0
                if key == "output" and self.config.dataset.uses_pooling
                else self.config.state_size
            )
            decision_tree_module = BronzeAgeDecisionTree.from_data(
                inputs_train[key],
                outputs_train[key],
                out_channels,
                num_states,
                self.config,
                use_linear_feature_combinations=use_linear_feature_combinations,
                layer_name=key,
            )
            decision_tree.set_submodule(key, decision_tree_module)

        return decision_tree

    def update_trees(self, trees):
        for name, tree in trees.items():
            self.get_submodule(name).tree = tree

    def prune_decision_trees(self, train_loader, validation_loader, score_model):
        """
        Prunes the decision trees in the model by removing nodes that do not improve the validation score
        :param train_loader: The training data loader
        :param validation_loader: The validation data loader
        :param score_model: The function to score the model
        :return: A new model with pruned trees and the number of nodes pruned
        """
        decision_tree_model = deepcopy(self)

        # score_before_train = score_model(decision_tree_model, train_loader)
        score_before_val = score_model(decision_tree_model, validation_loader)

        trees = {}
        for name, module in decision_tree_model.named_modules():
            if isinstance(module, BronzeAgeDecisionTree):
                trees[name] = module.tree

        for tree_name, tree in trees.items():
            print(f"Tree {tree_name}")
            print(sktree.export_text(tree))
        print()

        traversal_order = get_traversal_order(trees)

        to_prune = []
        for traversal_element in traversal_order:
            node_id = traversal_element.node_id
            tree_name = traversal_element.tree

            original_tree = deepcopy(trees[tree_name])
            tree_copy = deepcopy(trees[tree_name])

            prune_node_from_tree(tree_copy.tree_, node_id)
            trees[tree_name] = tree_copy
            decision_tree_model.update_trees(trees)

            score_after_train = score_model(decision_tree_model, train_loader)
            score_after_val = score_model(decision_tree_model, validation_loader)
            # What this condition should be may need to be adjusted
            if (
                score_after_val >= score_before_val
                and score_after_train >= score_before_val
            ):
                to_prune.append((tree_name, node_id))
                print(f"Pruned node {node_id} in {tree_name}")
            else:
                trees[tree_name] = original_tree
                decision_tree_model.update_trees(trees)
        print(f"Pruned {len(to_prune)} out of {len(traversal_order)} nodes")

        for tree_name, tree in trees.items():
            tree.tree_.max_depth = get_max_depth(tree, 0)

        print("Pruned trees:")
        for tree_name, tree in trees.items():
            print(f"Tree {tree_name}")
            print(sktree.export_text(tree))
        print()

        return decision_tree_model, len(to_prune)
