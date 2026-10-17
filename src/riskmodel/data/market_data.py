import pandas as pd


class MarketData:
    def __init__(self, data: pd.DataFrame):
        self.df = data

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, data: pd.DataFrame) -> None:
        self._validate(data)
        self._df = data.copy()

    def _validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be a pandas DatetimeIndex.")

        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted in ascending order.")

        if data.shape[0] < 3:
            raise ValueError("Data must contain at least 3 rows.")

        if data.shape[1] == 1:
            raise ValueError("Data must contain at least 1 column.")

        if data.isna().any().any():
            raise ValueError("Data cannot contain NaN values.")

        if not all(pd.api.types.is_float_dtype(dtype) for dtype in data.dtypes):
            raise TypeError(
                "All columns in data must be of float type (e.g., float64)."
            )
