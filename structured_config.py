from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING

@dataclass
class OptimizerConfig:
    name: str = MISSING
    lr: float = 0.001
    weight_decay: float = 0.0

@dataclass
class TrainingConfig:
    epochs: int = 100
    optimizer: OptimizerConfig = MISSING 