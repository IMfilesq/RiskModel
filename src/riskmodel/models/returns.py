import numpy as np
from scipy.stats import genpareto, jf_skew_t

from riskmodel.data.market_data import MarketData


class MixedPareto:
    data: np.ndarray
    q_l: float
    q_u: float
    v_l: float
    v_u: float
    data_c: np.ndarray
    bins: np.ndarray
    pdf_c: np.ndarray
    cdf_c: np.ndarray
    c_l: float
    c_u: float
    scale_l: float
    scale_u: float

    def __init__(
        self, data: np.typing.ArrayLike, q_l: float = 0.05, q_u: float = 0.95
    ) -> None:

        # l - lower
        # u - upper
        # c - center
        # q - percentile (oznaczenie z dokumentacji numpy)
        if not (0 < q_l < 1) or not (0 < q_u < 1) or (q_l >= q_u):
            raise (ValueError("Incorrectly selected percentiles"))
        data = np.asarray(data, dtype=np.float64).flatten()
        self.data = data
        self.q_l = q_l
        self.q_u = q_u
        self.q_c = q_u - q_l
        self._fit(data, q_l, q_u)

    def _fit(self, data: np.ndarray, q_l: float, q_u: float) -> None:

        # Wartości percentyli
        v_l = np.percentile(data, q_l * 100)
        v_u = np.percentile(data, q_u * 100)
        self.v_l, self.v_u = v_l, v_u

        # Dane w środku
        data_c = data[(self.v_l <= data) & (data <= self.v_u)]
        self.data_c = data_c

        # Rozkład empiryczny
        hist, bins = np.histogram(data_c, bins=20, density=True)
        self.bins = bins
        self.pdf_c = hist
        self.cdf_c = np.concatenate([[0], np.cumsum(hist * np.diff(bins))])

        # Parametry ogonów
        self.c_l, _, self.scale_l = genpareto.fit(
            self.v_l - data[data < self.v_l], floc=0
        )
        self.c_u, _, self.scale_u = genpareto.fit(
            data[data > self.v_u] - self.v_u, floc=0
        )

    def _pdf_single(self, x: float) -> np.ndarray:

        # PDF lewy ogon
        if x < self.v_l:
            return self.q_l * genpareto.pdf(self.v_l - x, self.c_l, 0, self.scale_l)

        # PDF prawy ogon
        if x > self.v_u:
            return (1 - self.q_u) * genpareto.pdf(
                x - self.v_u, self.c_u, 0, self.scale_u
            )

        # PDF środek
        if x == self.bins[-1]:
            return self.q_c * self.pdf_c[-1]

        idx = np.searchsorted(self.bins, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.pdf_c) - 1)
        return self.q_c * self.pdf_c[idx]

    def _cdf_single(self, t: float) -> np.ndarray:

        # CDF lewy ogon
        if t < self.v_l:
            return self.q_l * (
                1 - genpareto.cdf(self.v_l - t, self.c_l, 0, self.scale_l)
            )

        # CDF prawy ogon
        if t >= self.v_u:
            return self.q_u + (1 - self.q_u) * genpareto.cdf(
                t - self.v_u, self.c_u, 0, self.scale_u
            )

        # CDF środek
        # idx = np.searchsorted(self.bins, t, side="right") - 1
        # idx = np.clip(idx, 0, len(self.cdf_c)-1)
        return np.squeeze(self.q_l + self.q_c * np.interp(t, self.bins, self.cdf_c))

    def _ppf_single(self, p: float) -> np.ndarray:

        if not (0 < p < 1):
            raise (ValueError("Probability must be in the (0;1) interval"))

        # PPF lewy ogon
        if p <= self.q_l:
            return self.v_l - genpareto.ppf(1 - p / self.q_l, self.c_l, 0, self.scale_l)

        # PPF prawy ogon
        if p > self.q_u:
            return self.v_u + genpareto.ppf(
                (p - self.q_u) / (1 - self.q_u), self.c_u, 0, self.scale_u
            )

        # PPF środek
        p_c = (p - self.q_l) / self.q_c
        return np.asarray(np.interp(p_c, self.cdf_c, self.bins))

    def _pdf(self, x: np.typing.ArrayLike) -> np.ndarray:
        return np.vectorize(self._pdf_single)(np.asarray(x))

    def _cdf(self, t: np.typing.ArrayLike) -> np.ndarray:
        return np.vectorize(self._cdf_single)(np.asarray(t))

    def _ppf(self, p: np.typing.ArrayLike) -> np.ndarray:
        return np.vectorize(self._ppf_single)(np.asarray(p))


class SkewT:
    a: float
    b: float
    loc: float
    scale: float

    def __init__(self, data: np.typing.ArrayLike) -> None:

        self._fit(data)

    def _fit(self, data: np.typing.ArrayLike) -> None:

        self.a, self.b, self.loc, self.scale = jf_skew_t.fit(data)

    def _pdf_single(self, x: float) -> np.ndarray:

        return jf_skew_t.pdf(x, self.a, self.b, self.loc, self.scale)

    def _cdf_single(self, t: float) -> np.ndarray:

        return jf_skew_t.cdf(t, self.a, self.b, self.loc, self.scale)

    def _ppf_single(self, p: float) -> np.ndarray:

        if not (0 < p < 1):
            raise (ValueError("Probability must be in the (0;1) interval"))

        return jf_skew_t.ppf(p, self.a, self.b, self.loc, self.scale)

    def _pdf(self, x: np.typing.ArrayLike) -> np.ndarray:
        return np.vectorize(self._pdf_single)(np.asarray(x))

    def _cdf(self, t: np.typing.ArrayLike) -> np.ndarray:
        return np.vectorize(self._cdf_single)(np.asarray(t))

    def _ppf(self, p: np.typing.ArrayLike) -> np.ndarray:
        return np.vectorize(self._ppf_single)(np.asarray(p))


def skew_t(data: MarketData) -> dict[str, tuple[float, ...]]:
    result = {}
    for ticker in data.df.columns:
        result[ticker] = jf_skew_t.fit(data.df[ticker])
    return result


def mixed(data: MarketData) -> dict[str, MixedPareto]:
    result = {}
    for ticker in data.df.columns:
        result[ticker] = MixedPareto(data.df[ticker])
    return result
