import numpy as np
import pyvinecopulib as pv

from riskmodel.data.market_data import MarketData
from riskmodel.models.returns import MixedPareto


def _copula(data: MarketData, dist: dict[str, MixedPareto]) -> pv.Vinecop:

    residuals_df = data.df
    u_data = np.zeros_like(residuals_df, dtype=float)

    for i, ticker in enumerate(residuals_df.columns):
        x = residuals_df[ticker].to_numpy()
        dist_single = dist[ticker]
        u_data[:, i] = dist_single._cdf(x)  # type: ignore[attr-defined]

    # Zabezpieczenie przed wartościami dokładnie 0.0 lub 1.0
    u_data = np.clip(u_data, 1e-10, 1.0 - 1e-10)

    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.student])
    copula = pv.Vinecop.from_data(u_data, controls=controls)

    return copula
