from pathlib import Path

import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

from riskmodel.data.market_data import MarketData


def hist(data: MarketData) -> None:
    "Cretes and saves histograms in outputs foler in each run"
    base_path = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = base_path / "plots" / "resid_hist"

    plots_dir.mkdir(parents=True, exist_ok=True)

    for ticker in data.df.columns:
        plt.hist(data.df[ticker], bins=20)
        plt.title(f"{ticker} garch standardized residuals")
        file_path = plots_dir / f"{ticker}.png"
        print(file_path)

        plt.savefig(file_path)
        plt.clf()
