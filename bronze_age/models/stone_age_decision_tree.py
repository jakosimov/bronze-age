import torch
import numpy as np
import shap

from bronze_age.config import Config
from torch_geometric.nn import MessagePassing, global_add_pool


def linear_combo_features(input_data, state_size):
    difference_features = (
        input_data[np.newaxis, :, :state_size, None]
        > input_data[:, np.newaxis, :state_size]
    )
    difference_features = difference_features.reshape(len(input_data), -1).astype(int)
    return np.concatenate((input_data, difference_features), axis=1)


def get_decision_path_features(
    estimator, data, num_classes=0, message=False, num_features=0, pooling_gc=False
):
    feature = estimator.tree_.feature
    node_indicator = estimator.decision_path(data)
    leave_id = estimator.apply(data)
    threshold = estimator.tree_.threshold
    features_used = np.zeros((np.shape(data)[0], np.shape(data)[1]))
    # features_used = np.ones((np.shape(data)[0], np.shape(data)[1]))
    for sample_id in range(len(data)):
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        for node_id in node_index:
            # If Current Node is a Leaf -> Skip
            if leave_id[sample_id] == node_id:
                continue

            features_used[sample_id][feature[node_id]] = 1.0

    return features_used


def get_features_used(
    estimator, data, num_classes=0, message=False, num_features=0, pooling_gc=False
):
    features_used = np.zeros((np.shape(data)[0], num_classes, np.shape(data)[1]))
    predictions = estimator.predict(data)
    explainer = shap.TreeExplainer(
        estimator, feature_perturbation="tree_path_dependent"
    )

    shap_values = explainer.shap_values(data)
    if shap_values.ndim != 3:
        shap_values = shap_values[..., None]

    decision_path_features = get_decision_path_features(
        estimator, data, num_classes, message, num_features, pooling_gc
    )
    for sample_id in range(len(data)):
        for class_index in range(shap_values.shape[-1]):
            current_class = estimator.classes_[class_index]
            multiplies = 1.0
            if current_class != predictions[sample_id]:
                multiplies = -1.0

            for feature_number in range(np.shape(data)[1]):
                features_used[sample_id][current_class][feature_number] = (
                    multiplies * shap_values[sample_id][feature_number][class_index]
                )
    return features_used


def tree_predict(
    x, tree, num_classes, linear_feature_combinations=False, pooling=False
):
    data = x.cpu().detach().numpy()
    if linear_feature_combinations:
        num_features = len(data[0]) // 2
        if pooling:
            num_features = len(data[0])
        data = linear_combo_features(data, num_features)
    x = tree.predict(data)
    one_hot = np.zeros((x.size, num_classes))
    one_hot[np.arange(x.size), x] = 1
    return torch.from_numpy(one_hot)


class DTModule(torch.nn.Module):
    def __init__(
        self,
        tree,
        name,
        state_size,
        linear_feature_combinations=False,
        use_pooling=False,
    ):
        super(DTModule, self).__init__()
        self.__name__ = name
        self.tree = tree
        self.state_size = state_size
        self.linear_feature_combinations = linear_feature_combinations
        self.pooling = use_pooling

    def forward(self, x):
        x = tree_predict(
            x,
            self.tree,
            self.state_size,
            linear_feature_combinations=self.linear_feature_combinations,
            pooling=self.pooling,
        )
        return x


class StoneAgeGNNLayerDT(MessagePassing):

    def __init__(
        self,
        tree,
        config: Config,
        index=0,
        linear_feature_combinations=False,
    ):
        super().__init__(aggr="add")
        self.__name__ = "stone-age-" + str(index)
        self.tree = DTModule(
            tree=tree,
            name="linear_softmax",
            state_size=config.state_size,
            linear_feature_combinations=linear_feature_combinations,
        )
        self.bounding_parameter = config.bounding_parameter
        self.state_size = config.state_size
        self.messages = None
        self.edge_index = None
        self.importance = None
        self.inputs = None
        self.tree_input = None
        self.features_used = None
        self.linear_feature_combinations = linear_feature_combinations

    def forward(self, x, edge_index, explain=False):
        if explain:
            self.edge_index = edge_index.cpu().detach().numpy()
        return self.propagate(edge_index, x=x, explain=explain)

    def message(self, x_j, explain=False):
        if explain:
            self.messages = x_j.cpu().detach().numpy()
        return x_j

    def aggregate(self, inputs, index, ptr, dim_size, explain=False):
        message_sums = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        if explain:
            self.inputs = message_sums.cpu().detach().numpy()
        return torch.clamp(message_sums, min=0, max=self.bounding_parameter)

    def update(self, inputs, x, explain=False):
        combined = torch.cat((inputs, x), 1)
        self.tree_input = combined.cpu().detach().numpy()

        if self.linear_feature_combinations:
            num_features = len(self.tree_input[0]) // 2
            self.tree_input = linear_combo_features(self.tree_input, num_features)

        if explain:
            feature_importance = self.tree.tree.feature_importances_
            features_used = get_features_used(
                self.tree.tree,
                self.tree_input,
                num_classes=self.state_size,
                message=True,
                num_features=len(feature_importance),
            )
            self.features_used = features_used

        x = self.tree(combined)
        return x


class StoneAgeDecisionTree(torch.nn.Module):
    def __init__(
        self, config: Config, trees, out_channels, linear_feature_combinations=False
    ):
        super().__init__()
        self.input = DTModule(
            trees["input"],
            "input",
            state_size=config.state_size,
        )
        self.pooling = DTModule(
            trees["output"],
            "output",
            state_size=out_channels,
            use_pooling=config.use_pooling,
            linear_feature_combinations=config.use_pooling,
        )
        self.tree_depths = [self.input.tree.get_depth()]
        self.stone_age = torch.nn.ModuleList()

        for i in range(config.num_layers):
            self.tree_depths.append(trees[f"stone_age.{i}.linear_softmax"].get_depth())
            self.stone_age.append(
                StoneAgeGNNLayerDT(
                    tree=trees[f"stone_age.{i}.linear_softmax"],
                    index=i,
                    config=config,
                    linear_feature_combinations=linear_feature_combinations,
                )
            )
        self.tree_depths.append(self.pooling.tree.get_depth())

    def forward(self, x, edge_index, batch=None, **kwargs):
        x = self.input(x)
        xs = [x]
        for layer in self.stone_age:
            x = layer(x, edge_index)
            xs.append(x)

        if self.use_pooling:
            if batch is None:
                batch = torch.from_numpy(np.array([0 for _ in range(len(x))]))
            x = global_add_pool(x, batch)
            xs = [global_add_pool(xi, batch) for xi in xs]
            # x = global_max_pool(x, batch)
            # xs = [global_max_pool(xi, batch) for xi in xs]
        if self.skip_connection:
            x = torch.cat(xs, dim=1)

        x = self.pooling(x)
        return x
