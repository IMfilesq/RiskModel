from logging import getLogger

import hydra
from omegaconf import OmegaConf
from scipy.stats import binomtest

from riskmodel.backtest import back_metrics, empirical_metrics
from riskmodel.config import Config
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.stooq import write_stooq_data
from riskmodel.data.tiingo import write_tiingo_data
from riskmodel.main_model import main_model

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

    logger.info("Running the main model on the entire dataset...")
    main_model(cfg, MarketData(data))

    if cfg.backtest.run:
        logger.info("Running backtests...")
        bm = back_metrics(
            MarketData(data), cfg.data.window_size, cfg.simulation.horizon, cfg
        )
        print(bm)
        logger.info("Evaluating backtest results...")
        em = empirical_metrics(bm)
        print(em)
        t = binomtest(
            em.var_violations,
            n=len(bm),
            p=1 - cfg.risk_measure.var_level,
            alternative="two-sided",
        )
        logger.info(
            f"P-value with H0 that VaR is {cfg.risk_measure.var_level}: {t.pvalue}"
        )
        print(t.pvalue)


if __name__ == "__main__":
    main()
