from typing import Literal, cast

from arch import arch_model
from arch.univariate.base import ARCHModel  # , ARCHModelResult

from riskmodel.data.market_data import MarketData


def garch(data: MarketData, order: tuple[int, ...], dist: str) -> dict[str, ARCHModel]:

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
        result[ticker] = arch_model(data.df[ticker], p=p, o=o, q=q, dist=dist)
    return result
