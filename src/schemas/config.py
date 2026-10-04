from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    lr: float = MISSING
    type: str = MISSING


@dataclass
class TailConfig:
    type: str = MISSING
    accuracy: float = MISSING


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tail: TailConfig = field(default_factory=TailConfig)


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
