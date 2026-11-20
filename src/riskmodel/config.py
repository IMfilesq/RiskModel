from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class GARCHConfig:
    order: Any = (
        (1, 0, 1),
        (1, 0, 1),
        (1, 0, 1),
        (1, 0, 1),
        (1, 0, 1),
        (1, 0, 1),
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
    window_size: int = 1000
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
    )
    weights: tuple[float, ...] = tuple(1 / 7 for _ in range(7))


@dataclass()
class SimulationConfig:
    horizon: int = 10
    n_sims: int = 10000


@dataclass()
class PlotConfig:
    create_plots: bool = True


@dataclass()
class BacktestConfig:
    run: bool = False


@dataclass()
class RisMeasureConfig:
    es_level: float = 0.975
    var_level: float = 0.975


@dataclass()
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    test: TestConfig = field(default_factory=TestConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk_measure: RisMeasureConfig = field(default_factory=RisMeasureConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
