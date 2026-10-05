from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass(frozen=True)
class ModelConfig:
    lr: float = 0.1
    type: str = "GARCH"


@dataclass(frozen=True)
class TailConfig:
    type: str = "TAIL"
    accuracy: float = 0.5


@dataclass(frozen=True)
class DataConfig:
    file_path: Path = Path("data/price_data.csv")


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tail: TailConfig = field(default_factory=TailConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
