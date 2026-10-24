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


def unscale(data: MarketData, scaling_factors: tuple[float, ...]) -> MarketData:
    print("Remember to check the order of scaling and tickers")
    assert len(data.df.columns) == len(scaling_factors)
    for index, ticker in enumerate(data.df.columns):
        data.df[ticker] = data.df[ticker] / scaling_factors[index]
    return data
