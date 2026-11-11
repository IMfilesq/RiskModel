import numpy as np
from arch.univariate.base import ARCHModelResult
from pyvinecopulib import Vinecop

from riskmodel.models.returns import MixedPareto


def simulation_garch(
    copula: Vinecop,
    garch_models: dict[str, ARCHModelResult],
    dist_models: dict[str, MixedPareto],
    tickers: tuple[str, ...],
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
        p = getattr(volatility, "p", 1)
        q = getattr(volatility, "q", 1)
        k = max(p, q)
        eps_init = model.resid[-k:]
        sig2_init = model.conditional_volatility[-k:] ** 2
        params = model.params
        mu = params["mu"]
        omega = params["omega"]
        alpha = np.array([params[f"alpha[{idx}]"] for idx in range(1, p + 1)])[::-1]
        beta = np.array([params[f"beta[{idx}]"] for idx in range(1, q + 1)])[::-1]

        # Przepuszczenie wygenerowanych danych przez funkcję kwantylową
        dist = dist_models[ticker]
        resid = dist._ppf(sample_list[:, i, :])

        total_len = k + horizon
        eps_buf = np.empty(total_len)
        sig2_buf = np.empty(total_len)
        simulated_returns = np.zeros((M, horizon))

        for x in range(M):
            eps_buf[:k] = eps_init
            sig2_buf[:k] = sig2_init
            for y in range(horizon):
                idx = k + y
                alpha_term = np.dot(alpha, eps_buf[idx - p : idx] ** 2)
                beta_term = np.dot(beta, sig2_buf[idx - q : idx])
                sig2 = omega + alpha_term + beta_term
                eps_t = np.sqrt(sig2) * resid[x, y]
                sig2_buf[idx] = sig2
                eps_buf[idx] = eps_t
                simulated_returns[x, y] = eps_t

        daily_returns = simulated_returns + mu
        final_returns[:, i] = np.sum(daily_returns, axis=1)

    return final_returns
