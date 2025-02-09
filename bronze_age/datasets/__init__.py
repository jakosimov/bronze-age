import torch_geometric.data as geom_data

from bronze_age.config import Config

_DATASETS = {
    'MUTAG': lambda c: geom_data.TUDataset(root=c.data_root, name='MUTAG'),
}
def get_dataset(config):
    if config.dataset not in _DATASETS:
        raise ValueError(f'Unknown dataset: {config.dataset}')
    return _DATASETS[config.dataset](config)  