import numpy as np
import pandas as pd


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provided all the data frame data is positive and numerical, this function calculates the log returns of the data frame.

    """
    return df.apply(lambda x: np.log(x), axis=0).diff()[1:]
