from logging import getLogger

import hydra
import numpy as np
import pyvinecopulib as pv
from omegaconf import OmegaConf

from riskmodel.config import Config
from riskmodel.data.download import write_tiingo_data
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.stooq import write_stooq_data
from riskmodel.data.transforms import log_returns, scale, unscale, weighting
from riskmodel.models.garch import garch, get_params, get_pvals, get_std_residuals
from riskmodel.models.returns import mixed
from riskmodel.plot import hist
from riskmodel.simulation import simulation_garch
from riskmodel.stat_tests import het_test, multi_ac_tests, normality_tests, statio_tests

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(omega_cfg: Config):
    cfg: Config = OmegaConf.to_object(omega_cfg)  # type: ignore[assignment]

    logger.info("Downloading the price data in case the file is missing...")
    if cfg.data.use_stooq:
        write_stooq_data(cfg.data)
    else:
        write_tiingo_data(cfg.data)

    logger.info("Loading price data...")
    data = load_price_data(cfg.data.file_path)
    market_data = MarketData(data)

    logger.info("Transforming to log returns")
    market_data.df = log_returns(market_data.df)

    logger.info("Performing stationarity tests")
    logger.warning(
        "KPSS pvalues might be actually higher due to silenced warning in the test module"
    )
    statio_pvals = statio_tests(market_data.df)
    print(statio_pvals)

    logger.info("Performing LB autocorrelation tests")
    ac_pvals = multi_ac_tests(market_data.df, cfg.test.lb_max_lag)
    print(ac_pvals)

    logger.info("Performing heteroscedasticy tests")
    het_pvals = het_test(market_data.df)
    print(het_pvals)

    logger.info("Scaling data according to arch_model warnings")
    market_data = scale(market_data, cfg.data.scaling_factors)

    logger.info(f"Fitting GARCH {cfg.garch.order} to data")
    garch_models = garch(market_data, cfg.garch.order, cfg.garch.dist)

    logger.info("GARCH model param pvalues:")
    print(get_pvals(garch_models))

    logger.info("GARCH model params")
    print(get_params(garch_models))

    logger.info("Acessing GARCH models residuals")
    garch_resid = get_std_residuals(garch_models)

    logger.info("Testing heteroscedasticy on standarized GARCH residuals")
    print(het_test(garch_resid.df))

    logger.info("Testing normality on standarized GARCH residuals")
    print(normality_tests(market_data))

    logger.info("Creating histograms with GARCH standarized residuals")
    hist(garch_resid)

    logger.info(
        "Fitting semi-parametric distribution with Pareto tails and a histogram-based body to data"
    )
    dist_models = mixed(garch_resid)

    # Kod z czata
    logger.info(
        "Calculating theoretical CDF of semi-parametric distribution with Pareto tails and a histogram-based body residuals for copula input"
    )
    residuals_df = garch_resid.df
    u_data = np.zeros_like(residuals_df, dtype=float)

    for i, ticker in enumerate(residuals_df.columns):
        print(ticker)
        x = residuals_df[ticker].values
        # Rozpakowujemy dopasowany model dla danego aktywa
        dist = dist_models[ticker]
        # Liczymy teoretyczną dystrybuantę (CDF) dla każdej reszty GARCH
        u_data[:, i] = dist._cdf(x)

    # Zabezpieczenie przed wartościami dokładnie 0.0 lub 1.0 (ważne numerycznie dla kopul)
    u_data = np.clip(u_data, 1e-10, 1.0 - 1e-10)

    logger.info("Setting up and fitting the Vine Copula (Student-t)")
    # Settings of copula - use t-student copula
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.student])

    # Fitting the copula using parametric pseudo-observations (CDF values)
    copula = pv.Vinecop.from_data(u_data, controls=controls)
    print(copula)

    logger.info("Starting Monte Carlo simulations for 10-day VaR and ES")
    simulated_data = simulation_garch(
        copula, garch_models, dist_models, residuals_df.columns
    )
    print(simulated_data)
    logger.info("Calculating estimated VaR and ES values")

    # tymczasowe rozwiązanie z unscaling_tuple, to jest do poprawy bo jest podatne na błędy w kolejności tickerów
    # na razie trudno to zmienić bo kopuła i symulacja nie mają nazwanych kolumn
    unscaling_tuple = tuple(
        cfg.data.scaling_factors[ticker] for ticker in residuals_df.columns
    )
    data = unscale(simulated_data, unscaling_tuple)
    data = np.exp(data) - 1
    data = weighting(data, cfg.data.weights)
    data = np.sum(data, 1)

    var = np.percentile(data, 1)
    logger.info("Var")
    print(abs(var * 100))

    es = np.mean(data[data <= np.percentile(data, 2.5)])
    logger.info("ES")
    print(abs(es * 100))


if __name__ == "__main__":
    main()
