import pandas as pd


def _merge(crypto_tickers: tuple[str, ...], oth_tickers: tuple[str, ...]):
    prices_dict = {}
    for ticker in crypto_tickers:
        df_crypto_raw = pd.read_csv(
            f"data/{ticker}_us_d.csv", parse_dates=["Data"], date_format="%Y-%m-%d"
        )
        prices_dict[ticker.upper()] = df_crypto_raw.set_index("Data")["Zamkniecie"]
    for ticker in oth_tickers:
        df_oth_raw = pd.read_csv(
            f"data/{ticker}_us_d.csv", parse_dates=["Data"], date_format="%Y-%m-%d"
        )
        prices_dict[ticker.upper()] = df_oth_raw.set_index("Data")["Zamkniecie"]
    df = pd.DataFrame(prices_dict).rename_axis("Date").dropna()
    return df
