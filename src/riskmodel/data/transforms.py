import numpy as np
import pandas as pd

from riskmodel.data.market_data import MarketData


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provided all the data frame data is positive and numerical, this function calculates the log returns of the data frame.

    """
    return df.apply(lambda x: np.log(x), axis=0).diff()[1:]


def scale(data: MarketData, scaling_factors: tuple[float, ...]) -> MarketData:
    print("Remember to check the order of scaling and tickers")
    assert len(data.df.columns) == len(scaling_factors)
    for index, ticker in enumerate(data.df.columns):
        print(f"scaling {ticker} by {scaling_factors[index]}")
        data.df[ticker] = data.df[ticker] * scaling_factors[index]
    return data


def unscale(data: np.ndarray, scaling_factors: tuple[float, ...]) -> np.ndarray:
    print("Remember to check the order of scaling and columns")
    assert data.shape[1] == len(scaling_factors)
    for index in range(len(scaling_factors)):
        data[:, index] = data[:, index] / scaling_factors[index]
    return data


def weighting(data: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    print("Remember to check the order of scaling and columns")
    assert data.shape[1] == len(weights)
    assert sum(weights) == 1
    for index in range(len(weights)):
        data[:, index] = data[:, index] * weights[index]
    return data
