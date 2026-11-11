import numpy as np
import pandas as pd

from riskmodel.data.market_data import MarketData


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provided all the data frame data is positive and numerical, this function calculates the log returns of the data frame.

    """
    return df.apply(lambda x: np.log(x), axis=0).diff()[1:]


def scale(data: MarketData, scaling_factors: dict[str, float]) -> MarketData:
    assert len(data.df.columns) == len(scaling_factors)
    for ticker, factor in scaling_factors.items():
        print(f"scaling {ticker} by {factor}")
        data.df[ticker] = data.df[ticker] * factor
    return data


def unscale(data: np.ndarray, scaling_factors: tuple[float, ...]) -> np.ndarray:
    assert data.shape[1] == len(scaling_factors)
    for i, factor in enumerate(scaling_factors):
        print(f"unscaling column {i} by {factor}")
        data[:, i] = data[:, i] / factor
    return data


def weighting(data: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    assert data.shape[1] == len(weights)
    assert sum(weights) == 1
    # assert np.isclose(sum(weights), 1.0)
    for index in range(len(weights)):
        data[:, index] = data[:, index] * weights[index]
    return data
