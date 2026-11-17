from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StatioResult:
    """Stores pvalues of the stationatity tests"""

    kpss: float
    adf: float
    pp: float


@dataclass(frozen=True)
class ModelResult:
    """Stores the results of the risk model"""

    ES: float
    VaR: float


@dataclass(frozen=True)
class TrainTestSplit:
    """Stores the train-test split of the data"""

    train: pd.DataFrame
    test: pd.DataFrame
