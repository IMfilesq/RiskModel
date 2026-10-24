from logging import getLogger

import hydra
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from riskmodel.config import Config
from riskmodel.data.download import write
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.transforms import log_returns, scale
from riskmodel.models.garch import garch, get_params, get_pvals, get_std_residuals
from riskmodel.stat_tests import het_test, multi_ac_tests, normality_tests, statio_tests

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(omega_cfg: Config):
    cfg: Config = OmegaConf.to_object(omega_cfg)  # type: ignore[assignment]

    logger.info("Downloading the price data in case the file is missing...")
    write(cfg.data)

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

    logger.warning(f"{cfg.data.cash_tickers} probably requires ARMA modelling")

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

    logger.info("Testing heteroscedasticy on GARCH residuals")
    print(het_test(garch_resid.df))

    logger.info("Testing normality on GARCH residuals")
    print(normality_tests(market_data))

    plt.hist(garch_resid.df["GLD"], bins=50)
    plt.savefig("test_plot")


if __name__ == "__main__":
    main()
