from logging import getLogger

import hydra
from arch import arch_model
from omegaconf import OmegaConf

# from statsmodels.tsa.arima.model import ARIMA
from riskmodel.config import Config
from riskmodel.data.download import write
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.transforms import log_returns

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(omega_cfg: Config):
    cfg: Config = OmegaConf.to_object(omega_cfg)  # type: ignore[assignment]
    logger.info("Downloading the price data in case the file is missing...")
    write(cfg.data)
    logger.info("Loading price data...")
    data = load_price_data(cfg.data.file_path)
    market_data = MarketData(data)
    market_data.df = log_returns(market_data.df)
    # arima_mdl = ARIMA(market_data.df["GLD"], order=(1, 0, 1)) #dla złota współczynniki są zerowe
    # print(arima_mdl.fit().summary())
    garch_mdl = arch_model(
        market_data.df["BTCUSD"], dist="studentst"
    )  # dla btc wychodzą reszty z ciężkimi ogonami, dla GLD reszty bardziej normalne
    # w przyszłości i tak będziemy wyciągać reszty i samodzielnie je modelować
    print(garch_mdl.fit().summary())
    print(market_data.df)
    print(cfg)


if __name__ == "__main__":
    main()
