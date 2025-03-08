from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bronze_age.datasets import DatasetEnum


class NetworkType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"


class LayerType(StrEnum):
    StoneAge = "stone-age"
    BronzeAge = "bronze-age"
    BronzeAgeConcept = "bronze-age-concept"
    BronzeAgeGeneralConcept = "bronze-age-general-concept"


class LossMode(StrEnum):
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    CROSS_ENTROPY = "cross_entropy"


@dataclass
class Config:
    dataset: DatasetEnum
    data_dir: Path
    temperature: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    dropout: float = 0.0
    use_batch_norm: float = True
    network: NetworkType = NetworkType.MLP
    hidden_units: int = 16
    state_size: int = 10
    num_layers: int = 5
    skip_connection: bool = True
    bounding_parameter: int = 10
    batch_size: int = 128
    device: str = "mps"
    max_leaf_nodes: int = 100
    learning_rate: float = 0.01
    max_epochs: int = 1500
    num_cv: int = 10
    train_decision_tree: bool = False
    layer_type: LayerType = LayerType.BronzeAgeConcept
    use_one_hot_output: bool = False
    concept_embedding_size: int = 16
    concept_temperature: float = 0.5
    entropy_loss_scaling: float = 0.2
    early_stopping: bool = True
    loss_mode: LossMode = LossMode.BINARY_CROSS_ENTROPY
    one_hot_evaluation: bool = True
