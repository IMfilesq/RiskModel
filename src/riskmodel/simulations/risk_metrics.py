import numpy as np

from riskmodel.data.transforms import unscale, weighting
from riskmodel.schemas import RiskMeasures


def metrics(
    data: np.ndarray,
    scaling_factors: tuple,
    weights: tuple,
    ES_level: float,
    VaR_level: float,
) -> RiskMeasures:

    data = unscale(data, scaling_factors)
    data = np.exp(data) - 1
    data = weighting(data, weights)
    data = np.sum(data, 1)

    var = np.percentile(data, 100 * (1 - VaR_level))

    es_quantile = np.percentile(data, 100 * (1 - ES_level))
    es = np.mean(data[data <= es_quantile])

    return RiskMeasures(var=var, es=es, es_quantile=es_quantile)
