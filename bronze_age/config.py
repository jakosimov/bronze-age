from dataclasses import dataclass
from pathlib import Path

from bronze_age.datasets import DatasetEnum


@dataclass
class Config:
    dataset: DatasetEnum
    data_dir: Path