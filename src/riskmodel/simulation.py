import numpy as np

h_0 = 10


def simulation_garch(copula, garch_models, dist_models, tickers, horizon=h_0, M=10000):
    
    sample_list = np.stack([copula.sample(n=M) for _ in range(horizon)], axis=2) # otrzymujemy tensor M x len(tickers) x horizon

    final_returns = np.zeros((M, len(tickers)))
    
    for i, ticker in enumerate(tickers):

        # Zczytywanie danych na temat modelu
        model = garch_models[ticker]
        p = model.model.volatility.p
        q = model.model.volatility.q
        k = max(p, q)
        eps_init = model.resid[-k:].values
        sig2_init = model.conditional_volatility[-k:] ** 2
        params = model.params
        mu = params["mu"]
        omega = params["omega"]
        alpha = np.array([params[f"alpha[{idx}]"] for idx in range(1, p+1)])[::-1]
        beta = np.array([params[f"beta[{idx}]"] for idx in range(1, q+1)])[::-1]

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
                alpha_term = np.dot(alpha, eps_buf[idx-p:idx]**2)
                beta_term = np.dot(beta, sig2_buf[idx-q:idx])
                sig2 = omega + alpha_term + beta_term
                eps_t = np.sqrt(sig2) * resid[x, y]
                sig2_buf[idx] = sig2
                eps_buf[idx] = eps_t
                simulated_returns[x, y] = eps_t
        
        daily_returns = simulated_returns + mu
        final_returns[:, i] = np.sum(daily_returns, axis=1)
    
    return(final_returns)