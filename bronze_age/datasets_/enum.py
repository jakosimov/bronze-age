from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional

import torch
from dig.xgraph.dataset import MoleculeDataset, SynGraphDataset
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset


class DatasetEnum(StrEnum):
    MUTAG = 'MUTAG'
    PROTEINS = 'PROTEINS'
    IMDB_BINARY = 'IMDB-BINARY'
    REDDIT_BINARY = 'REDDIT-BINARY'
    MUTAGENICITY = 'Mutagenicity'
    BBBP = 'BBBP'
    COLLAB = 'COLLAB'
    BA_2MOTIFS = 'BA_2Motifs'
    BA_SHAPES = 'BA_Shapes'
    TREE_CYCLE = 'Tree_Cycle'
    TREE_GRID = 'Tree_Grid'



def _add_node_feature(data):
    data.x = torch.full((data.num_nodes, 1), 1.0, dtype=torch.float)
    return data

def _bbbp_transform(data):
    data.y = data.y.flatten()
    return data

def _ba_transform(data):
    data.x = data.x.to(torch.float32)
    data.x = data.x[:, :1]
    return data

@dataclass
class _DatasetConfig:
    name: str
    init: Callable[["Config"], Dataset]
    use_pooling: bool
    use_mask : bool = False


"""
    if dataset_name == 'MUTAG':
        dataset = MoleculeDataset(args.data_dir + '/datasets', dataset_name)
        use_pooling = True

    if dataset_name == 'PROTEINS':
        dataset = TUDataset(root=args.data_dir + '/datasets', name=dataset_name)
        use_pooling = True

    if dataset_name == 'IMDB-BINARY':
        dataset = TUDataset(root=args.data_dir + '/datasets', name=dataset_name, pre_transform=AddFeatures())
        use_pooling = True

    if dataset_name == 'REDDIT-BINARY':
        dataset = TUDataset(root=args.data_dir + '/datasets', name=dataset_name, pre_transform=AddFeatures())
        use_pooling = True

    if dataset_name == 'Mutagenicity':
        dataset = TUDataset(root=args.data_dir + '/datasets', name=dataset_name)
        use_pooling = True

    if dataset_name == 'BBBP':
        dataset = MoleculeDataset(args.data_dir + '/datasets', dataset_name, pre_transform=FlattenY())
        use_pooling = True

    if dataset_name == 'COLLAB':
        dataset = TUDataset(args.data_dir + '/datasets', name=dataset_name, pre_transform=AddFeatures())
        use_pooling = True
"""

_DATASETS = {
    DatasetEnum.MUTAG: _DatasetConfig(
        name='MUTAG',
        init=lambda c: MoleculeDataset(c.data_dir, 'MUTAG'),
        use_pooling=True,
    ),
    DatasetEnum.PROTEINS: _DatasetConfig(
        name='PROTEINS',
        init=lambda c: TUDataset(root=c.data_dir, name='PROTEINS'),
        use_pooling=True,
    ),
    DatasetEnum.IMDB_BINARY: _DatasetConfig(
        name='IMDB-BINARY',
        init=lambda c: TUDataset(root=c.data_dir, name='IMDB-BINARY', pre_transform=_add_node_feature),
        use_pooling=True,
    ),
    DatasetEnum.REDDIT_BINARY: _DatasetConfig(
        name='REDDIT-BINARY',
        init=lambda c: TUDataset(root=c.data_dir, name='REDDIT-BINARY', pre_transform=_add_node_feature),
        use_pooling=True,
    ),
    DatasetEnum.MUTAGENICITY: _DatasetConfig(
        name='Mutagenicity',
        init=lambda c: TUDataset(root=c.data_dir, name='Mutagenicity'),
        use_pooling=True,
    ),
    DatasetEnum.BBBP: _DatasetConfig(
        name='BBBP',
        init=lambda c: MoleculeDataset(c.data_dir, 'BBBP', pre_transform=_bbbp_transform),
        use_pooling=True,
    ),
    DatasetEnum.COLLAB: _DatasetConfig(
        name='COLLAB',
        init=lambda c: TUDataset(root=c.data_dir, name='COLLAB', pre_transform=_add_node_feature),
        use_pooling=True,
    ),
    DatasetEnum.BA_2MOTIFS: _DatasetConfig(
        name='BA_2Motifs',
        init=lambda c: SynGraphDataset(c.data_dir, 'BA_2Motifs', pre_transform=_ba_transform),
        use_pooling=True,
    ),
    DatasetEnum.BA_SHAPES: _DatasetConfig(
        name='BA_Shapes',
        init=lambda c: SynGraphDataset(c.data_dir, 'BA_Shapes', pre_transform=_ba_transform),
        use_pooling=True,
        use_mask=True,
    ),
    DatasetEnum.TREE_CYCLE: _DatasetConfig(
        name='Tree_Cycle',
        init=lambda c: SynGraphDataset(c.data_dir, 'Tree_Cycle', pre_transform=_ba_transform),
        use_pooling=True,
        use_mask=True,
    ),
}