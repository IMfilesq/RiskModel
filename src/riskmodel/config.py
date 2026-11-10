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
    use_stooq: bool = True
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
    )  # temporarily removing BIL, has scaling factor 10000 (100, 100, 10000, 100, 10000, 100, 10)
    scaling_factors: dict[str, float] = field(
        default_factory=lambda: {
            "GLD": 100,
            "SPY": 100,
            "QQQ": 10000,
            "TLT": 100,
            "XLP": 10000,
            "XLV": 100,
            "BTCUSD": 10,
        }
    )
    weights: tuple[float, ...] = tuple(1 / 7 for _ in range(7))


@dataclass()
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    test: TestConfig = field(default_factory=TestConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
