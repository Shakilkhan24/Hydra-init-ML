from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    num_classes: int

@dataclass
class MainConfig:
    model: ModelConfig
    seed: int

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig) 