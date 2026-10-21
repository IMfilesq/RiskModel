from logging import getLogger

import hydra
from omegaconf import OmegaConf

from riskmodel.config import Config
from riskmodel.data.download import write
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.transforms import log_returns
from riskmodel.stat_tests import het_test, multi_ac_tests, statio_tests

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

    # arima_mdl = ARIMA(market_data.df["TLT"], order=(1, 0, 1))
    # garch_mdl = arch_model(
    #     market_data.df["BTCUSD"], dist="studentst"
    # )  # dla btc wychodzą reszty z ciężkimi ogonami, dla GLD reszty bardziej normalne
    # # w przyszłości i tak będziemy wyciągać reszty i samodzielnie je modelować
    # print(garch_mdl.fit().summary())


if __name__ == "__main__":
    main()
