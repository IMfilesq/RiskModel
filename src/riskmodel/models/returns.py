import numpy as np
from scipy.stats import genpareto, jf_skew_t

from riskmodel.data.market_data import MarketData


class MixedPareto:
    def __init__(self, data, q_l=0.05, q_u=0.95):
        
        # l - lower
        # u - upper
        # c - center
        # q - percentile (oznaczenie z dokumentacji numpy)
        if not(0<q_l<1) or not(0<q_u<1) or (q_l>=q_u): raise(ValueError("Incorrectly selected percentiles"))
        self.data = data
        self.q_l = q_l
        self.q_u = q_u
        self.q_c = q_u - q_l
        self._fit(data, q_l, q_u)

    def _fit(self, data, q_l, q_u):
        
        # Wartości percentyli
        v_l = np.percentile(data, q_l*100)
        v_u = np.percentile(data, q_u*100)
        self.v_l, self.v_u = v_l, v_u
        
        # Dane w środku
        data_c = data[(self.v_l<=data) & (data<=self.v_u)]
        self.data_c = data_c
        
        # Rozkład empiryczny
        hist, bins = np.histogram(data_c, bins=20, density=True)
        self.bins = bins
        self.pdf_c = hist
        self.cdf_c = np.concatenate([[0], np.cumsum(hist * np.diff(bins))])

        # Parametry ogonów
        self.c_l, _, self.scale_l = genpareto.fit(self.v_l-data[data<self.v_l], floc=0)
        self.c_u, _, self.scale_u = genpareto.fit(data[data>self.v_u]-self.v_u, floc=0)

    def _pdf_single(self, x):

        # PDF lewy ogon
        if x < self.v_l:
            return self.q_l * genpareto.pdf(self.v_l-x, self.c_l, 0, self.scale_l)

        # PDF prawy ogon
        if x > self.v_u:
            return (1 - self.q_u) * genpareto.pdf(x-self.v_u, self.c_u, 0, self.scale_u)

        # PDF środek
        if x == self.bins[-1]:
            return self.q_c * self.pdf_c[-1]

        idx = np.searchsorted(self.bins, x, side="right")-1
        idx = np.clip(idx, 0, len(self.pdf_c)-1)
        return self.q_c * self.pdf_c[idx]

    def _cdf_single(self, t):
        
        # CDF lewy ogon
        if t < self.v_l:
            return self.q_l * (1 - genpareto.cdf(self.v_l-t, self.c_l, 0, self.scale_l))
        
        # CDF prawy ogon
        if t >= self.v_u:
            return self.q_u + (1 - self.q_u) * genpareto.cdf(t-self.v_u, self.c_u, 0, self.scale_u)

        # CDF środek
        p_c = np.interp(t, self.bins, self.cdf_c)
        return self.q_l + self.q_c * p_c

    def _ppf_single(self, p):
        
        if not(0<p<1): raise(ValueError("Probability must be in the (0;1) interval"))
        
        # PPF lewy ogon
        if p < self.q_l:
            return self.v_l - genpareto.ppf(1 - p/self.q_l, self.c_l, 0, self.scale_l)
        
        # PPF prawy ogon
        if p >= self.q_u:
            return self.v_u + genpareto.ppf((p-self.q_u)/(1-self.q_u), self.c_u, 0, self.scale_u)

        # PPF środek
        p_c = (p-self.q_l)/self.q_u
        return np.interp(p_c, self.cdf_c, self.bins)

    def _pdf(self, x):
        return np.vectorize(self._pdf_single)(np.asarray(x))

    def _cdf(self, t):
        return np.vectorize(self._cdf_single)(np.asarray(t))

    def _ppf(self, p):
        return np.vectorize(self._ppf_single)(np.asarray(p))


def skew_t(data: MarketData) -> dict[str, tuple[float, ...]]:
    result = {}
    for ticker in data.df.columns:
        result[ticker] = jf_skew_t.fit(data.df[ticker])
    return result


def pareto(data: MarketData) -> dict[str, tuple[float, ...]]:
    result = {}
    for ticker in data.df.columns:
        result[ticker] = genpareto.fit(data.df[ticker])
    return result


def mixed(data):
    result = {}
    for ticker in data.df.columns:
        result[ticker] = MixedPareto(data.df[ticker])
    return result