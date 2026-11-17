from collections.abc import Generator

import numpy as np
import pandas as pd

from riskmodel.config import Config
from riskmodel.data.market_data import MarketData
from riskmodel.data.transforms import weighting
from riskmodel.main_model import main_model
from riskmodel.schemas import TrainTestSplit


def _rolling_windows(
    df: pd.DataFrame, window_size: int, step: int
) -> Generator[TrainTestSplit, None, None]:
    """Creates rolling windows of the data."""
    for i in range(0, len(df) - window_size + 1, step):
        train = df.iloc[i : i + window_size]
        test = df.iloc[i + window_size : i + window_size + step]
        yield TrainTestSplit(train=train, test=test)


def back_metrics(
    data: MarketData, window_size: int, step: int, cfg: Config
) -> pd.DataFrame:
    """Calculates backtest metrics for rolling windows of the data."""
    results = {"ES": [], "VaR": [], "actual": []}
    for split in _rolling_windows(data.df, window_size, step):
        train_data = MarketData(split.train)
        test_data = MarketData(split.test)
        measures = main_model(market_data=train_data, cfg=cfg)
        actual_returns = (
            np.array(test_data.df.iloc[-1] / test_data.df.iloc[0]).reshape(1, -1) - 1
        )
        weighted_returns = weighting(actual_returns, cfg.data.weights)
        portfolio_return = weighted_returns.sum()
        results["ES"].append(measures.ES)
        results["VaR"].append(measures.VaR)
        results["actual"].append(portfolio_return)
    return pd.DataFrame(results)


def empirical_metrics(back_metricks: pd.DataFrame):
    """Calculates empirical metrics based on backtest results."""
    var_violations = (back_metricks["actual"] < back_metricks["VaR"]).sum()
    emp_es = back_metricks[back_metricks["actual"] < back_metricks["VaR"]][
        "actual"
    ].mean()
    return var_violations, emp_es
