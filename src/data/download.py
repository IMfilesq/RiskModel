import os
from logging import getLogger
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = getLogger(__name__)


class DownloadDataError(Exception):
    """Raised when market data cannot be downloaded or is empty."""


def _get_eod_stock_data(ticker: str, start_date: str, api_key: str) -> pd.Series:
    """Downloads historical EOD data for stocks / ETFs starting from the specified date."""
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    params = {"startDate": start_date, "token": api_key}
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if response.status_code != 200 or not data:
        raise DownloadDataError(f"Error downloading {ticker}: {data}")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.set_index("date", inplace=True)
    return df["adjClose"]


def _get_crypto_data(ticker: str, start_date: str, api_key: str) -> pd.Series:
    """Downloads historical crypto data from Tiingo starting from the specified date."""
    url = "https://api.tiingo.com/tiingo/crypto/prices"
    params = {
        "tickers": ticker,
        "startDate": start_date,
        "resampleFreq": "1day",
        "token": api_key,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if response.status_code != 200 or not data:
        raise DownloadDataError(f"Error downloading crypto {ticker}: {data}")

    price_data = data[0]["priceData"]
    df = pd.DataFrame(price_data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.set_index("date", inplace=True)
    return df["close"]


def _download(
    api_key: str,
    start_date: str = "2012-01-01",
    crypto_tickers: tuple[str, ...] = ("btcusd",),
    oth_tickers: tuple[str, ...] = ("GLD", "SPY"),
) -> pd.DataFrame:
    """Downoloads specified market data and returns a DataFrame with the adjusted close prices for the given tickers starting from the specified date."""
    prices_dict = {}

    logger.info(f"Downloading data from Tiingo starting {start_date}...")

    for ticker in oth_tickers:
        prices_dict[ticker.upper()] = _get_eod_stock_data(ticker, start_date, api_key)

    for ticker in crypto_tickers:
        prices_dict[ticker.upper()] = _get_crypto_data(ticker, start_date, api_key)

    df_prices = pd.DataFrame(prices_dict).dropna()

    return df_prices


def save(file_path: Path = Path("data/price_data.csv")) -> None:
    """Saves the DataFrame to a CSV file."""
    if file_path.exists():
        logger.debug(f"File {file_path} already exists, skipping download.")
    else:
        logger.debug("Creating the missing data directory...")
        os.makedirs(file_path.parent, exist_ok=True)
        logger.debug("Acessing the Ting api key from .env file...")
        API_KEY = os.getenv("TIINGO_API_KEY")
        if not API_KEY:
            raise ValueError(
                "Nie znaleziono klucza API! Upewnij się, że plik .env istnieje i zawiera"
                " wpis 'TIINGO_API_KEY'."
            )
        logger.debug("Downloading data from Tiingo...")
        df = _download(API_KEY)
        logger.debug("Saving data locally...")
        df.to_csv(file_path)
        logger.info(f"Data saved to {file_path}.")


if __name__ == "__main__":
    save()
