from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig

from riskmodel.data.market_data import MarketData
from riskmodel.models.returns import MixedPareto


def hist(data: np.ndarray, ticker: str) -> None:
    "Creates and saves histograms in outputs folder in each run"
    base_path = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = base_path / "plots" / "resid_hist"

    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.hist(data, bins="fd")
    plt.title(f"{ticker} garch standardized residuals")
    plt.grid(True, alpha=0.3)

    file_path = plots_dir / f"{ticker}.png"
    plt.savefig(file_path)
    plt.close()


def density(dist: MixedPareto, ticker: str, _min: float, _max: float) -> None:

    base_path = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = base_path / "plots" / "resid_pdf"

    plots_dir.mkdir(parents=True, exist_ok=True)

    v_l = dist.v_l
    v_u = dist.v_u

    x = np.linspace(_min, _max, min(1000, int(np.ceil((_max - _min) * 1000))))
    mask_l = x < v_l
    mask_c = (v_l <= x) & (x <= v_u)
    mask_u = x > v_u

    p = dist._pdf(x)

    plt.plot(x[mask_l], p[mask_l], color="red")
    plt.axvline(v_l, color="gray", linestyle="--", alpha=0.75)
    plt.plot(x[mask_c], p[mask_c], color="blue")
    plt.axvline(v_u, color="gray", linestyle="--", alpha=0.75)
    plt.plot(x[mask_u], p[mask_u], color="red")
    plt.title(f"{ticker} probability density function (PDF)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(_min - 0.05, _max + 0.05)
    plt.ylim(min(p) - 0.01, max(p) + 0.01)
    plt.grid(True, alpha=0.3)

    file_path = plots_dir / f"{ticker}.png"
    plt.savefig(file_path)
    plt.close()


def distribution(dist: MixedPareto, ticker: str, _min: float, _max: float) -> None:

    base_path = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = base_path / "plots" / "resid_cdf"

    plots_dir.mkdir(parents=True, exist_ok=True)

    v_l = dist.v_l
    v_u = dist.v_u

    t = np.linspace(_min, _max, int(np.ceil((_max - _min) * 1000)))
    mask_l = t < v_l
    mask_c = (v_l <= t) & (t <= v_u)
    mask_u = t > v_u

    p = dist._cdf(t)

    plt.plot(t[mask_l], p[mask_l], color="red")
    plt.plot(t[mask_c], p[mask_c], color="blue")
    plt.plot(t[mask_u], p[mask_u], color="red")
    plt.title(f"{ticker} cumulative distribution function")
    plt.xlabel("t")
    plt.ylabel("p")
    plt.xlim(_min - 0.1, _max + 0.1)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    file_path = plots_dir / f"{ticker}.png"
    plt.savefig(file_path)
    plt.close()


def quantile(dist: MixedPareto, ticker: str) -> None:

    base_path = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = base_path / "plots" / "resid_ppf"

    plots_dir.mkdir(parents=True, exist_ok=True)

    _min_p = 0.001
    _max_p = 0.999

    q_l = dist.q_l
    q_u = dist.q_u

    p = np.linspace(_min_p, _max_p, int(np.ceil((_max_p - _min_p) * 1000)))
    mask_l = p < q_l
    mask_c = (q_l <= p) & (p <= q_u)
    mask_u = p > q_u

    t = dist._ppf(p)

    plt.plot(p[mask_l], t[mask_l], color="red")
    plt.plot(p[mask_c], t[mask_c], color="blue")
    plt.plot(p[mask_u], t[mask_u], color="red")
    plt.title(f"{ticker} quantile function")
    plt.xlabel("p")
    plt.ylabel("t")
    plt.xlim(-0.05, 1.05)
    plt.ylim(min(t) - 0.1, max(t) + 0.1)
    plt.grid(True, alpha=0.3)

    file_path = plots_dir / f"{ticker}.png"
    plt.savefig(file_path)
    plt.close()


def graphs_single(data: MarketData, dist: dict[str, MixedPareto]) -> None:

    for ticker in data.df.columns:
        data_single = data.df[ticker].to_numpy()
        dist_single = dist[ticker]
        _min = min(data_single)
        _max = max(data_single)

        hist(data_single, ticker)
        density(dist_single, ticker, _min, _max)
        distribution(dist_single, ticker, _min, _max)
        quantile(dist_single, ticker)


def _plot(ticker: str, data: np.ndarray, dist: MixedPareto, plots_dir: Path) -> None:

    _min = min(data)
    _max = max(data)
    v_l = dist.v_l
    v_u = dist.v_u
    q_l = dist.q_l
    q_u = dist.q_u

    # PDF and CDF arguments
    x = np.linspace(_min, _max, min(1000, int(np.ceil((_max - _min) * 1000))))
    mask_l = x < v_l
    mask_c = (v_l <= x) & (x <= v_u)
    mask_u = x > v_u

    pdf_values = dist._pdf(x)
    cdf_values = dist._cdf(x)

    # PPF arguments
    p = np.linspace(0.001, 0.999, 1000)
    p_l = p < q_l
    p_c = (q_l < p) & (p <= q_u)
    p_u = p >= q_u

    ppf_values = dist._ppf(p)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        f"{ticker} GARCH standarized resiudals", fontsize=14, fontweight="bold"
    )

    # Histogram
    ax = axes[0, 0]
    ax.hist(data, bins="fd")
    ax.set_title("Histogram")
    ax.grid(True, alpha=0.3)

    # Density
    ax = axes[0, 1]
    ax.plot(x[mask_l], pdf_values[mask_l], color="red")
    ax.axvline(v_l, color="gray", linestyle="--", alpha=0.75)
    ax.plot(x[mask_c], pdf_values[mask_c], color="blue")
    ax.axvline(v_u, color="gray", linestyle="--", alpha=0.75)
    ax.plot(x[mask_u], pdf_values[mask_u], color="red")
    ax.set_title("Probability density function (PDF)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_xlim(_min - 0.05, _max + 0.05)
    ax.set_ylim(min(pdf_values) - 0.01, max(pdf_values) + 0.01)
    ax.grid(True, alpha=0.3)

    # Distribution
    ax = axes[1, 0]
    ax.plot(x[mask_l], cdf_values[mask_l], color="red")
    ax.plot(x[mask_c], cdf_values[mask_c], color="blue")
    ax.plot(x[mask_u], cdf_values[mask_u], color="red")
    ax.set_title("Cumulative distribution function (CDF)")
    ax.set_xlabel("t")
    ax.set_ylabel("p")
    ax.set_xlim(_min - 0.1, _max + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Quantile function
    ax = axes[1, 1]
    ax.plot(p[p_l], ppf_values[p_l], color="red")
    ax.plot(p[p_c], ppf_values[p_c], color="blue")
    ax.plot(p[p_u], ppf_values[p_u], color="red")
    ax.set_title("Percent point function (PPF)")
    ax.set_xlabel("p")
    ax.set_ylabel("t")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(min(ppf_values) - 0.1, max(ppf_values) + 0.1)
    ax.grid(True, alpha=0.3)

    file_path = plots_dir / f"{ticker}.png"
    fig.savefig(file_path)
    plt.close(fig)


def graphs(data: MarketData, dist: dict[str, MixedPareto]) -> None:

    base_path = Path(HydraConfig.get().runtime.output_dir)
    plots_dir = base_path / "plots" / "stacked_plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    for ticker in data.df.columns:
        _plot(ticker, data.df[ticker].to_numpy(), dist[ticker], plots_dir)
