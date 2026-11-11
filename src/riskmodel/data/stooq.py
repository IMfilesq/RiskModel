from logging import getLogger

import pandas as pd

from riskmodel.config import DataConfig

logger = getLogger(__name__)


def _merge(
    crypto_tickers: tuple[str, ...],
    oth_tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    prices_dict = {}
    for ticker in crypto_tickers:
        df_crypto_raw = pd.read_csv(
            f"data/stooq_data/{ticker.lower()}_d.csv",
            parse_dates=["Data"],
            date_format="%Y-%m-%d",
        )
        prices_dict[ticker.upper()] = df_crypto_raw.set_index("Data")["Zamkniecie"]
    for ticker in oth_tickers:
        df_oth_raw = pd.read_csv(
            f"data/stooq_data/{ticker.lower()}_us_d.csv",
            parse_dates=["Data"],
            date_format="%Y-%m-%d",
        )
        prices_dict[ticker.upper()] = df_oth_raw.set_index("Data")["Zamkniecie"]

    df = (
        pd.DataFrame(prices_dict)
        .rename_axis("date")
        .sort_index()  # Sortowanie indeksu dat (wymagane dla poprawnego slicing'u)
        .dropna()
        .loc[start_date:end_date]  # Ucięcie do zadanego zakresu
    )
    return df


def write_stooq_data(
    cfg: DataConfig,
) -> None:
    """Saves the DataFrame to a CSV file."""
    if cfg.file_path.exists():
        logger.debug(f"File {cfg.file_path} already exists, skipping download.")
    else:
        df = _merge(cfg.crypto_tickers, cfg.oth_tickers, cfg.start_date, cfg.end_date)
        df.to_csv(cfg.file_path)
