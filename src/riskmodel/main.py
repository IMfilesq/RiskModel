from logging import getLogger

import hydra
from omegaconf import OmegaConf

from riskmodel.config import Config
from riskmodel.data.download import write
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(omega_cfg: Config):
    cfg: Config = OmegaConf.to_object(omega_cfg)  # type: ignore[assignment]
    logger.info("Downloading the price data in case the file is missing...")
    write(cfg.data)
    logger.info("Loading price data...")
    data = load_price_data(cfg.data.file_path)
    market_data = MarketData(data)
    print(market_data.df)
    print(cfg)


if __name__ == "__main__":
    main()
