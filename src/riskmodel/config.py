from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass()
class ARIMAConfig:
    order: tuple[int, ...] = (1, 0, 1)


@dataclass
class GARCHConfig:
    order: tuple[int, ...] = (1, 0, 1)
    dist: str = "normal"


@dataclass
class TestConfig:
    lb_max_lag: int = 3


@dataclass()
class TailConfig:
    type: str = "TAIL"
    accuracy: float = 0.5


@dataclass()
class DataConfig:
    start_date: str = "1900-01-01"
    file_path: Path = Path("data/price_data.csv")
    crypto_tickers: tuple[str, ...] = ("btcusd",)
    cash_tickers: tuple[str, ...] = ("BIL",)
    oth_tickers: tuple[str, ...] = ("GLD", "SPY", "QQQ", "TLT", "BIL", "XLP", "XLV")


@dataclass()
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    tail: TailConfig = field(default_factory=TailConfig)
    test: TestConfig = field(default_factory=TestConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
