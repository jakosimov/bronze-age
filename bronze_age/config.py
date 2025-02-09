from dataclasses import dataclass
from enum import StrEnum


class Dataset(StrEnum):
    MUTAG = 'MUTAG'

@dataclass
class Config:
    dataset: Dataset