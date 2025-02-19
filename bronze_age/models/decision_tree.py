from bronze_age.config import Config
from bronze_age.datasets import get_dataset
from bronze_age.models.feature_extractor import extract_features
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import torch

from bronze_age.models.stone_age_decision_tree import (
    StoneAgeDecisionTree,
    linear_combo_features,
)


def scale_pooling_num_node_samples(estimator, tree_data, data):
    node_indicator = estimator.decision_path(tree_data)
    num_nodes = estimator.tree_.node_count
    new_sample_count = np.zeros(num_nodes)
    for sample_id in range(len(tree_data)):
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        for node_id in node_index:
            new_sample_count[node_id] += len(data[sample_id].x)

    for node_id in range(num_nodes):
        estimator.tree_.n_node_samples[node_id] = new_sample_count[node_id]

    # print(f'Root Samples After: {estimator.tree_.n_node_samples[0]}')
    return estimator


def get_layer_names(config):
    """Returns the name of all (relevant) layers in the stone age gnn model"""
    number_of_layers = config.num_layers
    layer_names = ["input"]
    for i in range(number_of_layers):
        layer_names.append(f"stone_age.{i}.linear_softmax")
    layer_names.append("output")
    return layer_names


def extract_input_output(
    model,
    config: Config,
    layer_names,
    train_dataset,
    test_dataset,
    train_mask=None,
    test_mask=None,
):
    """Extracts the input and output of the specified layers for the train and test dataset"""
    if config.dataset.uses_mask:
        data = train_dataset[0]
        train_mask = data.train_mask
        test_mask = data.test_mask
        input_outputs_train = extract_features(
            model, layer_names, data, "cpu", mask=train_mask
        )
        input_outputs_test = extract_features(
            model, layer_names, data, "cpu", mask=test_mask
        )
    else:
        input_outputs_train = extract_features(
            model,
            layer_names,
            train_dataset,
            "cpu",
            batch_size=config.batch_size,
        )
        input_outputs_test = extract_features(
            model,
            layer_names,
            test_dataset,
            "cpu",
            batch_size=config.batch_size,
        )
    return input_outputs_train, input_outputs_test


def train_decision_tree(layer_name, input_outputs_train, config: Config, train_dataset):
    """Trains a decision tree model for a given layer"""
    data_input_train = input_outputs_train[layer_name]["inputs"]
    data_output_train = np.argmax(input_outputs_train[layer_name]["outputs"], axis=1)

    if layer_name != "input" and layer_name != "output":
        num_features = len(data_input_train[0]) // 2
        data_input_train = linear_combo_features(data_input_train, num_features)

    if layer_name == "output" and config.dataset.uses_pooling:
        num_features = len(data_input_train[0])
        data_input_train = linear_combo_features(data_input_train, num_features)
    clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=config.max_leaf_nodes)
    clf.fit(data_input_train, data_output_train)

    if layer_name == "output" and config.dataset.uses_pooling:
        clf = scale_pooling_num_node_samples(clf, data_input_train, train_dataset)
    return clf


def train_decision_tree_model(
    gnn_model, config: Config, out_channels, train_dataset, test_dataset
):
    """Trains a decision tree model for the stone age gnn model"""
    layer_names = get_layer_names(config)

    (
        input_outputs_train,
        _,
    ) = extract_input_output(
        gnn_model, config, layer_names, train_dataset, test_dataset
    )

    trees = {}
    for layer_name in layer_names:
        clf = train_decision_tree(
            layer_name, input_outputs_train, config, train_dataset
        )
        trees[layer_name] = clf

    model_dt = StoneAgeDecisionTree(
        config, trees, out_channels, linear_feature_combinations=True
    )

    return model_dt
