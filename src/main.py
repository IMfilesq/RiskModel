from logging import getLogger

import hydra

from src.data.download import save
from src.data.load import load_price_data
from src.schemas.config import Config

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    logger.info("Downloading the price data in case the file is missing...")
    save(cfg.data.file_path)
    logger.info("Loading price data...")
    data = load_price_data(cfg.data.file_path)
    print(data.head())
    print(cfg)


if __name__ == "__main__":
    main()
