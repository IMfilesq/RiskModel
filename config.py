from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass
class GARCHConfig:
    order: tuple[int, ...] = (1, 0, 1)
    dist: str = "normal"


@dataclass
class TestConfig:
    lb_max_lag: int = 3


@dataclass()
class DataConfig:
    start_date: str = "1900-01-01"
    file_path: Path = Path("data/price_data.csv")
    crypto_tickers: tuple[str, ...] = ("btcusd",)
    oth_tickers: tuple[str, ...] = (
        "GLD",
        "SPY",
        "QQQ",
        "TLT",
        "XLP",
        "XLV",
    )  # temporarily removing BIL, has scaling factor 10000
    scaling_factors: tuple[float, ...] = (100, 100, 10000, 100, 10000, 100, 10)
    weights: tuple[float, ...] = tuple(1 / 7 for _ in range(7))


@dataclass()
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    test: TestConfig = field(default_factory=TestConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
