import logging
from typing import Literal, cast

import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from riskmodel.data.market_data import MarketData

logger = logging.getLogger(__name__)


class MissingParams(Exception):
    "Model does not have params that are acessed"


def garch(
    data: MarketData, order: tuple[int, ...], dist: str
) -> dict[str, ARCHModelResult]:
    """Fits univariate GARCH models to each column of the MarketData.df"""

    dist = cast(
        Literal[
            "normal",
            "gaussian",
            "t",
            "studentst",
            "skewstudent",
            "skewt",
            "ged",
            "generalized error",
        ],
        dist,
    )

    result = {}
    for ticker in data.df.columns:
        p, o, q = order
        logger.info(f"Fitting GARCH model to {ticker}")
        result[ticker] = arch_model(data.df[ticker], p=p, o=o, q=q, dist=dist).fit(
            disp="off"
        )
    return result


def get_pvals(garch_models: dict[str, ARCHModelResult]) -> pd.DataFrame:
    """Gets pvalues for GARCH coefficients"""
    pvals = {ticker: model.pvalues for ticker, model in garch_models.items()}
    return pd.DataFrame(pvals)


def get_params(garch_models: dict[str, ARCHModelResult]) -> pd.DataFrame:
    """Gets coefficients of fitted GARCH models"""
    params_dct = {ticker: model.params for ticker, model in garch_models.items()}
    df = pd.DataFrame(params_dct)
    if "alpha[1]" in df.index and "beta[1]" in df.index:
        df.loc["alpha[1] + beta[1]"] = df.loc["alpha[1]"] + df.loc["beta[1]"]
        return df
    else:
        raise MissingParams()


def get_std_residuals(garch_models: dict[str, ARCHModelResult]) -> MarketData:
    """Gets coefficients of fitted GARCH models"""
    resid_dct = {ticker: model.std_resid for ticker, model in garch_models.items()}
    return MarketData(pd.DataFrame(resid_dct))
