from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bronze_age.datasets import DatasetEnum


class NetworkType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"

@dataclass
class Config:
    dataset: DatasetEnum
    data_dir: Path
    temperature : float = 1.0
    alpha : float  = 1.0
    beta : float  = 1.0
    dropout : float  = 0.0
    use_batch_norm : float  = True
    network : NetworkType = NetworkType.MLP
    hidden_units : int = 16
    state_size : int = 10
    num_layers : int = 5
    skip_connection : bool = True
    use_pooling : bool = True
    bounding_parameter : int = 1000
    batch_size : int = 128
    device: str = "mps"
    max_leaf_nodes: int = 100
