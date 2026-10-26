from scipy.stats import jf_skew_t

from riskmodel.data.market_data import MarketData


def skew_t(data: MarketData) -> dict[str, tuple[float, ...]]:
    result = {}
    for ticker in data.df.columns:
        result[ticker] = jf_skew_t.fit(data.df[ticker])
    return result
