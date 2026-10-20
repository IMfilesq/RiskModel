import warnings

import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from riskmodel.schemas import StatioResult

warnings.simplefilter("ignore", category=InterpolationWarning)


def _statio_tests(data: pd.Series) -> StatioResult:
    warnings.simplefilter("ignore", category=InterpolationWarning)
    adf = adfuller(data, result_object=False)
    kp = kpss(data, result_object=False)
    pp = PhillipsPerron(data)

    return StatioResult(kp[1], adf[1], pp.pvalue)


def statio_tests(data: pd.DataFrame):
    store = {}
    for ticker in data.columns:
        pvals = _statio_tests(data[ticker])
        store[ticker] = [pvals.adf, pvals.kpss, pvals.pp]
    index = ["ADF", "KPSS", "PP"]

    return pd.DataFrame(store, index=index)


def multi_ac_tests(data: pd.DataFrame, lags: int) -> pd.DataFrame:
    store = {}
    for ticker in data.columns:
        store[ticker] = list(acorr_ljungbox(data[ticker], lags=lags)["lb_pvalue"])
    index = [f"LB lag = {i + 1}" for i in range(lags)]
    return pd.DataFrame(store, index=index)
