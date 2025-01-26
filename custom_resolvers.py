from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import random

OmegaConf.register_new_resolver(
    "random_int",
    lambda min_val, max_val: random.randint(min_val, max_val)
) 