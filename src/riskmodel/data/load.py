from pathlib import Path

import pandas as pd


def load_price_data(file_path: Path = Path("data/price_data.csv")) -> pd.DataFrame:
    """Loads price data from a CSV file."""
    try:
        df = pd.read_csv(file_path, index_col="date", parse_dates=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File {file_path} not found. Please download the data first. Try: uv run python -m src.data.download in the bash terminal"
        )
