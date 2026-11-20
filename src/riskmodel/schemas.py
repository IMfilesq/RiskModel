from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StatioResult:
    """Stores pvalues of the stationatity tests"""

    kpss: float
    adf: float
    pp: float


@dataclass(frozen=True)
class TrainTestSplit:
    """Stores the train-test split of the data"""

    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class EmpiricalMetrics:
    """Stores the results of the empirical_metrics"""

    var_violations: int
    emp_es: float


@dataclass(frozen=True)
class RiskMeasures:
    """Stores the results of risk measure calculations"""

    var: float
    es: float
