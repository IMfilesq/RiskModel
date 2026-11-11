from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class GARCHConfig:
    # order: tuple[int, ...] = (1, 0, 1)
    # Wyniki na podstawie AIC
    order: Any = (
        (6, 0, 9),
        (2, 0, 2),
        (4, 0, 2),
        (2, 0, 2),
        (1, 0, 3),
        (2, 0, 5),
        (1, 0, 1),
    )
    aic_garch: bool = False
    dist: str = "normal"
    p_max: int = 10
    q_max: int = 10


@dataclass
class TestConfig:
    lb_max_lag: int = 3


@dataclass()
class DataConfig:
    use_stooq: bool = True
    start_date: str = "2010-07-19"
    end_date: str = "2026-11-09"
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
            "QQQ": 1000,
            "TLT": 100,
            "XLP": 10000,
            "XLV": 100,
            "BTCUSD": 10,
        }
    )
    weights: tuple[float, ...] = tuple(1 / 7 for _ in range(7))


@dataclass()
class SimulationConfig:
    horizon: int = 10
    n_sims: int = 10000


@dataclass()
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    test: TestConfig = field(default_factory=TestConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
