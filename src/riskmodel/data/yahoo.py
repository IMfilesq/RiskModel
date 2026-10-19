import pandas as pd
import yfinance as yf

crypto_tickers: tuple[str, ...] = ("BTC-USD",)
oth_tickers: tuple[str, ...] = ("GLD", "SPY", "QQQ", "TLT", "BIL", "XLP", "XLV")


def load_data(start: str = "1900-01-01") -> pd.DataFrame:

    tickers = list(crypto_tickers) + list(oth_tickers)

    raw = yf.download(
        tickers,
        start=start,
        interval="1d",
        group_by="column",
        progress=False,
    )

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise TypeError(
            "No data downloaded. Check your tickers and internet connection."
        )

    raw = raw.dropna(how="any").sort_index()

    raw = raw[[("Close", t) for t in tickers]]

    raw.columns = [t for _, t in raw.columns]

    return raw
