from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass()
class ModelConfig:
    lr: float = 0.1
    type: str = "GARCH"


@dataclass()
class TailConfig:
    type: str = "TAIL"
    accuracy: float = 0.5


@dataclass()
class DataConfig:
    file_path: Path = Path("data/price_data.csv")
    crypto_tickers: tuple[str, ...] = ("btcusd",)
    oth_tickers: tuple[str, ...] = ("GLD", "SPY", "QQQ", "TLT", "BIL", "XLP", "XLV")


@dataclass()
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tail: TailConfig = field(default_factory=TailConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
