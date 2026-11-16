import numpy as np

from riskmodel.data.transforms import unscale, weighting


def metrics(data: np.ndarray, scaling_factors: tuple, weights: tuple) -> None:

    data = unscale(data, scaling_factors)
    data = np.exp(data) - 1
    data = weighting(data, weights)
    data = np.sum(data, 1)

    var = np.percentile(data, 2.5)
    print(f"VaR: {var}")

    es = np.mean(data[data <= var])
    print(f"ES: {es}")
