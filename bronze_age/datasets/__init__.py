from enum import StrEnum

import numpy as np
import torch
import torch_geometric.transforms as T
from dig.xgraph.dataset import MoleculeDataset, SynGraphDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import Compose

from bronze_age.datasets.distance import Distance
from bronze_age.datasets.game_of_life import GameOfLifeGraph, HexagonalGameOfLifeGraph
from bronze_age.datasets.infection import Infection
from bronze_age.datasets.pathfinding import PathFinding
from bronze_age.datasets.prefixsum import PrefixSum
from bronze_age.datasets.rootvalue import RootValue
from bronze_age.datasets.saturation import Saturation
from bronze_age.datasets.simple_saturation import SimpleSaturation


class CustomDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        data = dataset[0]
        self.num_node_features = data.num_node_features
        self.num_classes = data.num_classes

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(
        self,
        idx,
    ):
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            return self.dataset[idx]

        else:
            return CustomDataset([self.dataset[i] for i in idx])


def _x_to_float(data):
    data.x = data.x.to(torch.float32)
    return data


def _flatten_y(data):
    y = torch.flatten(data.y)
    data.y = y
    return data


def _y_to_long(data):
    data.y = data.y.long()
    return data


def _add_features(data):
    x = torch.full((data.num_nodes, 1), 1.0, dtype=torch.float)
    data.x = x
    return data


def _syn_graph_transform(data):
    data.x = data.x[:, :1]
    return data


class DatasetEnum(StrEnum):
    MUTAG = "MUTAG"
    PROTEINS = "PROTEINS"
    IMDB_BINARY = "IMDB-BINARY"
    REDDIT_BINARY = "REDDIT-BINARY"
    MUTAGENICITY = "Mutagenicity"
    BBBP = "BBBP"
    COLLAB = "COLLAB"
    BA_2MOTIFS = "BA_2Motifs"
    INFECTION = "Infection"
    SATURATION = "Saturation"
    BA_SHAPES = "BA_shapes"
    TREE_CYCLE = "Tree_Cycle"
    TREE_GRID = "Tree_Grid"
    CORA = "Cora"
    CITESEER = "CiteSeer"
    PUBMED = "PubMed"
    OGBA = "OGBA"
    OGB_MOLHIV = "OGB-molhiv"
    OGB_PPA = "OGB-ppa"
    OGB_CODE2 = "OGB-code2"
    SIMPLE_SATURATION = "Simple_Saturation"
    DISTANCE = "Distance"
    PATH_FINDING = "PathFinding"
    PREFIX_SUM = "PrefixSum"
    ROOT_VALUE = "RootValue"
    GAME_OF_LIFE = "GameOfLife"
    HEXAGONAL_GAME_OF_LIFE = "HexagonalGameOfLife"

    @property
    def uses_pooling(self):
        return _USES_POOLING[self]

    @property
    def uses_mask(self):
        return _USES_MASK[self]


_DATASETS = {
    DatasetEnum.MUTAG: lambda c: MoleculeDataset(c.data_dir, "MUTAG"),
    DatasetEnum.BBBP: lambda c: MoleculeDataset(
        c.data_dir, "BBBP", pre_transform=Compose([_flatten_y, _y_to_long])
    ),
    DatasetEnum.PROTEINS: lambda c: TUDataset(root=c.data_dir, name="PROTEINS"),
    DatasetEnum.IMDB_BINARY: lambda c: TUDataset(
        root=c.data_dir, name="IMDB-BINARY", pre_transform=_add_features
    ),
    DatasetEnum.REDDIT_BINARY: lambda c: TUDataset(
        root=c.data_dir, name="REDDIT-BINARY", pre_transform=_add_features
    ),
    DatasetEnum.MUTAGENICITY: lambda c: TUDataset(root=c.data_dir, name="Mutagenicity"),
    DatasetEnum.COLLAB: lambda c: TUDataset(
        root=c.data_dir, name="COLLAB", pre_transform=_add_features
    ),
    DatasetEnum.INFECTION: lambda c: CustomDataset(
        [
            Infection(num_layers=c.num_layers).create_dataset(
                num_nodes=1000, edge_probability=0.004
            )
            for _ in range(10)
        ]
    ),
    DatasetEnum.BA_2MOTIFS: lambda c: SynGraphDataset(
        c.data_dir, "BA_2Motifs", transform=_syn_graph_transform
    ),
    DatasetEnum.SATURATION: lambda c: CustomDataset(
        [
            Saturation(
                sample_count=1,
                num_layers=c.num_layers,
                concat_features=False,
                conv_type=None,
            ).create_dataset()
            for _ in range(10)
        ]
    ),
    DatasetEnum.SIMPLE_SATURATION: lambda c: CustomDataset(
        [
            SimpleSaturation(
                sample_count=1,
                num_layers=c.num_layers,
                concat_features=False,
                conv_type=None,
            ).create_dataset()
            for _ in range(10)
        ]
    ),
    DatasetEnum.BA_SHAPES: lambda c: SynGraphDataset(
        c.data_dir, "BA_shapes", transform=Compose([_syn_graph_transform, _x_to_float])
    ),
    DatasetEnum.TREE_CYCLE: lambda c: SynGraphDataset(
        c.data_dir, "Tree_Cycle", transform=Compose([_syn_graph_transform, _x_to_float])
    ),
    DatasetEnum.TREE_GRID: lambda c: SynGraphDataset(
        c.data_dir, "Tree_Grid", transform=Compose([_syn_graph_transform, _x_to_float])
    ),
    DatasetEnum.CORA: lambda c: Planetoid(
        c.data_dir, name="Cora", pre_transform=_x_to_float
    ),
    DatasetEnum.CITESEER: lambda c: Planetoid(
        c.data_dir, name="CiteSeer", pre_transform=_x_to_float
    ),
    DatasetEnum.PUBMED: lambda c: Planetoid(
        c.data_dir, name="PubMed", pre_transform=_x_to_float
    ),
    DatasetEnum.OGBA: lambda c: PygNodePropPredDataset(
        "ogbn-arxiv",
        root=c.data_dir,
        transform=Compose([T.ToUndirected(), _x_to_float]),
    ),
    DatasetEnum.OGB_MOLHIV: lambda c: PygGraphPropPredDataset(
        "ogbg-molhiv", root=c.data_dir
    ),
    DatasetEnum.OGB_PPA: lambda c: PygGraphPropPredDataset("ogbg-ppa", root=c.data_dir),
    DatasetEnum.OGB_CODE2: lambda c: PygGraphPropPredDataset(
        "ogbg-code2", root=c.data_dir
    ),
    DatasetEnum.DISTANCE: lambda c: CustomDataset(Distance().data),
    DatasetEnum.PATH_FINDING: lambda c: CustomDataset(
        PathFinding(num_nodes=15, num_graphs=200, rangeValue=0).data
    ),
    DatasetEnum.PREFIX_SUM: lambda c: CustomDataset(
        PrefixSum(num_nodes=8, num_graphs=100, rangeValue=0).data
    ),
    DatasetEnum.ROOT_VALUE: lambda c: CustomDataset(
        RootValue(num_nodes=8, num_graphs=100, rangeValue=0).data
    ),
    DatasetEnum.GAME_OF_LIFE: lambda c: CustomDataset(
        GameOfLifeGraph(num_graphs=10, grid_size=8, steps=1).data
    ),
    DatasetEnum.HEXAGONAL_GAME_OF_LIFE: lambda c: CustomDataset(
        HexagonalGameOfLifeGraph(num_graphs=10, grid_size=8, steps=1).data
    ),
}

_USES_POOLING = {
    DatasetEnum.MUTAG: True,
    DatasetEnum.PROTEINS: True,
    DatasetEnum.IMDB_BINARY: True,
    DatasetEnum.REDDIT_BINARY: True,
    DatasetEnum.MUTAGENICITY: True,
    DatasetEnum.BBBP: True,
    DatasetEnum.COLLAB: True,
    DatasetEnum.BA_2MOTIFS: True,
    DatasetEnum.INFECTION: False,
    DatasetEnum.SATURATION: False,
    DatasetEnum.BA_SHAPES: False,
    DatasetEnum.TREE_CYCLE: False,
    DatasetEnum.TREE_GRID: False,
    DatasetEnum.CORA: False,
    DatasetEnum.CITESEER: False,
    DatasetEnum.PUBMED: False,
    DatasetEnum.OGBA: False,
    DatasetEnum.OGB_MOLHIV: True,
    DatasetEnum.OGB_PPA: True,
    DatasetEnum.OGB_CODE2: True,
    DatasetEnum.SIMPLE_SATURATION: False,
    DatasetEnum.DISTANCE: False,
    DatasetEnum.PATH_FINDING: False,
    DatasetEnum.PREFIX_SUM: False,
    DatasetEnum.ROOT_VALUE: False,
    DatasetEnum.GAME_OF_LIFE: False,
    DatasetEnum.HEXAGONAL_GAME_OF_LIFE: False,
}

_USES_MASK = {
    DatasetEnum.MUTAG: False,
    DatasetEnum.PROTEINS: False,
    DatasetEnum.IMDB_BINARY: False,
    DatasetEnum.REDDIT_BINARY: False,
    DatasetEnum.MUTAGENICITY: False,
    DatasetEnum.BBBP: False,
    DatasetEnum.COLLAB: False,
    DatasetEnum.BA_2MOTIFS: False,
    DatasetEnum.INFECTION: False,
    DatasetEnum.SATURATION: False,
    DatasetEnum.BA_SHAPES: True,
    DatasetEnum.TREE_CYCLE: True,
    DatasetEnum.TREE_GRID: True,
    DatasetEnum.CORA: True,
    DatasetEnum.CITESEER: True,
    DatasetEnum.PUBMED: True,
    DatasetEnum.OGBA: True,
    DatasetEnum.OGB_MOLHIV: False,
    DatasetEnum.OGB_PPA: False,
    DatasetEnum.OGB_CODE2: False,
    DatasetEnum.SIMPLE_SATURATION: False,
    DatasetEnum.DISTANCE: False,
    DatasetEnum.PATH_FINDING: False,
    DatasetEnum.PREFIX_SUM: False,
    DatasetEnum.ROOT_VALUE: False,
    DatasetEnum.GAME_OF_LIFE: False,
    DatasetEnum.HEXAGONAL_GAME_OF_LIFE: False,
}


def get_dataset(config):
    return _DATASETS[config.dataset](config)


if __name__ == "__main__":
    from bronze_age.config import Config

    for dataset in DatasetEnum:
        print(dataset)
        config = Config(dataset=dataset, data_dir="downloads", number_of_layers=1)
        dataset = get_dataset(config)
        print(len(dataset))
        print(dataset[0])
        print(dataset.num_node_features)
        print(dataset.num_classes)
        print()
