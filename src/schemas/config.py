from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    lr: float = 0.1
    type: str = "GARCH"


@dataclass
class TailConfig:
    type: str = "TAIL"
    accuracy: float = 0.5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tail: TailConfig = field(default_factory=TailConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
