import numpy as np
from arch.univariate.base import ARCHModelResult
from numba import njit
from pandas import Index
from pyvinecopulib import Vinecop

from riskmodel.models.returns import MixedPareto


def simulation_garch(
    copula: Vinecop,
    garch_models: dict[str, ARCHModelResult],
    dist_models: dict[str, MixedPareto],
    tickers: Index,
    horizon: int,
    M: int,
) -> np.ndarray:

    sample_list = np.stack(
        [copula.sample(n=M) for _ in range(horizon)], axis=2
    )  # otrzymujemy tensor M x len(tickers) x horizon

    final_returns = np.zeros((M, len(tickers)))

    for i, ticker in enumerate(tickers):
        # Zczytywanie danych na temat modelu
        model = garch_models[ticker]

        volatility = model.model.volatility
        p = int(getattr(volatility, "p", 1))
        q = int(getattr(volatility, "q", 1))
        k = int(max(p, q))
        eps_init = np.asarray(model.resid[-k:])
        sig2_init = np.asarray(model.conditional_volatility[-k:] ** 2)
        params = model.params
        mu = float(params["mu"])
        omega = float(params["omega"])
        alpha = np.array([params[f"alpha[{idx}]"] for idx in range(1, p + 1)])[::-1]
        beta = np.array([params[f"beta[{idx}]"] for idx in range(1, q + 1)])[::-1]

        # Przepuszczenie wygenerowanych danych przez funkcję kwantylową
        dist = dist_models[ticker]
        resid = dist._ppf(sample_list[:, i, :])

        final_returns[:, i] = simulation_garch_numba(
            resid, eps_init, sig2_init, alpha, beta, mu, omega
        )

    return final_returns


@njit(cache=True)
def simulation_garch_numba(
    resid: np.ndarray,
    eps_init: np.ndarray,
    sig2_init: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    mu: float,
    omega: float,
):

    M, horizon = resid.shape
    p = len(alpha)
    q = len(beta)
    k = max(p, q)

    eps_buf = np.empty(k + horizon)
    sig2_buf = np.empty(k + horizon)

    simulated_returns = np.zeros(M)

    for x in range(M):
        eps_buf[:k] = eps_init
        sig2_buf[:k] = sig2_init
        for y in range(horizon):
            idx = k + y
            alpha_term = 0
            for i in range(p):
                alpha_term += alpha[i] * eps_buf[idx - p + i] ** 2
            beta_term = 0
            for i in range(q):
                beta_term += beta[i] * sig2_buf[idx - q + i]
            sig2 = omega + alpha_term + beta_term
            eps_t = np.sqrt(sig2) * resid[x, y]
            sig2_buf[idx] = sig2
            eps_buf[idx] = eps_t
            simulated_returns[x] += eps_t + mu

    return simulated_returns
