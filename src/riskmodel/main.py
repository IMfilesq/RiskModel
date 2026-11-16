from logging import getLogger

import hydra
from omegaconf import OmegaConf

from riskmodel.config import Config
from riskmodel.data.copula import _copula
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.stooq import write_stooq_data
from riskmodel.data.tiingo import write_tiingo_data
from riskmodel.data.transforms import log_returns, scale
from riskmodel.models.garch import (
    garch,
    garch_selection,
    get_params,
    get_pvals,
    get_std_residuals,
)
from riskmodel.models.returns import mixed
from riskmodel.plot import graphs, graphs_single
from riskmodel.simulations.risk_metrics import metrics
from riskmodel.simulations.simulation import simulation_garch
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

    if cfg.garch.aic_garch:
        logger.info(
            f"Fitting GARCH models based on AIC (p_max={cfg.garch.p_max}, q_max={cfg.garch.q_max})"
        )
        garch_models = garch_selection(
            market_data, cfg.garch.dist, cfg.garch.p_max, cfg.garch.q_max
        )
    else:
        logger.info("Fitting user-specified GARCH models to data")
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
    print(normality_tests(garch_resid))

    logger.info(
        "Fitting semi-parametric distribution with Pareto tails and a histogram-based body to data"
    )
    dist_models = mixed(garch_resid)

    logger.info("Creating stacked graphs for GARCH standarized residuals")
    graphs(garch_resid, dist_models)

    logger.info("Creating single graphs for GARCH standarized residuals")
    graphs_single(garch_resid, dist_models)

    logger.info("Setting up and fitting the Vine Copula (Student-t)")
    copula = _copula(garch_resid, dist_models)
    print(copula)
    print(copula.order)

    logger.info("Starting Monte Carlo simulations for 10-day VaR and ES")
    simulated_data = simulation_garch(
        copula,
        garch_models,
        dist_models,
        garch_resid.df.columns,
        cfg.simulation.horizon,
        cfg.simulation.n_sims,
    )

    logger.info("Calculating estimated VaR and ES values")
    unscaling_factors = tuple(
        cfg.data.scaling_factors[ticker] for ticker in garch_resid.df.columns
    )
    metrics(simulated_data, unscaling_factors, cfg.data.weights)


if __name__ == "__main__":
    main()
