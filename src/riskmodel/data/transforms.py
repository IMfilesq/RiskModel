import numpy as np
import pandas as pd


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provided all the data frame data is positive and numerical, this function calculates the log returns of the data frame.

    """
    return df.apply(lambda x: np.log(x), axis=0).diff()[1:]


def unscale(data: np.ndarray, scaling_factors: tuple[float, ...]) -> np.ndarray:
    """Unscales the data by dividing each column by its corresponding scaling factor."""
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
