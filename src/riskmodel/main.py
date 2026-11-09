from logging import getLogger

import hydra
import numpy as np
import pyvinecopulib as pv
from omegaconf import OmegaConf
from scipy.stats import jf_skew_t

from riskmodel.config import Config
from riskmodel.data.download import write
from riskmodel.data.load import load_price_data
from riskmodel.data.market_data import MarketData
from riskmodel.data.transforms import log_returns, scale, unscale, weighting
from riskmodel.models.garch import garch, get_params, get_pvals, get_std_residuals
from riskmodel.models.returns import skew_t
from riskmodel.plot import hist
from riskmodel.stat_tests import het_test, multi_ac_tests, normality_tests, statio_tests

logger = getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(omega_cfg: Config):
    cfg: Config = OmegaConf.to_object(omega_cfg)  # type: ignore[assignment]

    logger.info("Downloading the price data in case the file is missing...")
    write(cfg.data)

    logger.info("Loading price data...")
    data = load_price_data(cfg.data.file_path)
    market_data = MarketData(data)

    logger.info("Transforming to log returns")
    market_data.df = log_returns(market_data.df)

    logger.info("Performing stationarity tests")
    logger.warning(
        "KPSS pvalues might be actually higher due to silenced warning in the test module"
    )
    statio_pvals = statio_tests(market_data.df)
    print(statio_pvals)

    logger.info("Performing LB autocorrelation tests")
    ac_pvals = multi_ac_tests(market_data.df, cfg.test.lb_max_lag)
    print(ac_pvals)

    logger.info("Performing heteroscedasticy tests")
    het_pvals = het_test(market_data.df)
    print(het_pvals)

    logger.info("Scaling data according to arch_model warnings")
    market_data = scale(market_data, cfg.data.scaling_factors)

    logger.info(f"Fitting GARCH {cfg.garch.order} to data")
    garch_models = garch(market_data, cfg.garch.order, cfg.garch.dist)

    logger.info("GARCH model param pvalues:")
    print(get_pvals(garch_models))

    logger.info("GARCH model params")
    print(get_params(garch_models))

    logger.info("Acessing GARCH models residuals")
    garch_resid = get_std_residuals(garch_models)

    logger.info("Testing heteroscedasticy on standarized GARCH residuals")
    print(het_test(garch_resid.df))

    logger.info("Testing normality on standarized GARCH residuals")
    print(normality_tests(market_data))

    logger.info("Creating histograms with GARCH standarized residuals")
    hist(garch_resid)

    logger.info("Fitting skewed-t distributions to data")
    params = skew_t(market_data)
    params = skew_t(garch_resid)
    print(params)

    # Kod z czata
    logger.info("Calculating theoretical CDF of skewed-t residuals for copula input")
    residuals_df = garch_resid.df
    u_data = np.zeros_like(residuals_df, dtype=float)

    for i, ticker in enumerate(residuals_df.columns):
        print(ticker)
        x = residuals_df[ticker].values
        # Rozpakowujemy dopasowane parametry dla danego aktywa
        a, b, loc, sc = params[ticker]

        # Liczymy teoretyczną dystrybuantę (CDF) dla każdej reszty GARCH
        u_data[:, i] = jf_skew_t.cdf(x, a, b, loc=loc, scale=sc)

    # Zabezpieczenie przed wartościami dokładnie 0.0 lub 1.0 (ważne numerycznie dla kopul)
    u_data = np.clip(u_data, 1e-10, 1.0 - 1e-10)

    logger.info("Setting up and fitting the Vine Copula (Student-t)")
    # Settings of copula - use t-student copula
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.student])

    # Fitting the copula using parametric pseudo-observations (CDF values)
    copula = pv.Vinecop.from_data(u_data, controls=controls)
    print(copula)

    # Dla jednego dnia raczej bez sensu liczyć, ale zostawiam w razie czego
    # I nie poprawiam ewentualnych błędów jeżeli są
    logger.info("Starting Monte Carlo simulation for 1-day VaR")
    # logger.info("Generating new data from copula")
    sample = 10000
    new_data = copula.sample(n=sample)
    # print(new_data)
    for i, ticker in enumerate(residuals_df.columns):
        model = garch_models[ticker]
        forecast = model.forecast(horizon=1, reindex=False)
        # print(forecast.variance.iloc[0,0])
        variance = forecast.variance.iloc[0, 0]
        # print(forecast.mean.iloc[0,0])
        mean = forecast.mean.iloc[0, 0]
        a, b, loc, sc = params[ticker]
        new_data[:, i] = mean + variance**0.5 * jf_skew_t.ppf(  # type: ignore
            new_data[:, i], a, b, loc=loc, scale=sc
        )
    new_data = unscale(new_data, cfg.data.scaling_factors) / 7
    var = np.percentile(np.sum(new_data, 1), 1)
    logger.info("One day VaR")
    print(var * 100)

    logger.info("Starting Monte Carlo simulation for 10-day VaR")
    list_samples = [copula.sample(n=sample) for _ in range(10)]
    for i, ticker in enumerate(residuals_df.columns):
        model = garch_models[ticker]
        forecast = model.forecast(horizon=1, reindex=False)
        variance = forecast.variance.iloc[0, 0]
        mean = forecast.mean.iloc[0, 0]
        a, b, loc, sc = params[ticker]
        alpha = model.params["alpha[1]"]
        beta = model.params["beta[1]"]
        omega = model.params["omega"]

        shock = variance**0.5 * jf_skew_t.ppf(  # type: ignore
            list_samples[0][:, i], a, b, loc=loc, scale=sc
        )
        list_samples[0][:, i] = mean + shock  # type: ignore
        for j in range(1, 10):
            variance = omega + alpha * np.square(shock) + beta * variance
            shock = variance**0.5 * jf_skew_t.ppf(
                list_samples[j][:, i], a, b, loc=loc, scale=sc
            )
            list_samples[j][:, i] = mean + shock

    final_data = np.sum(
        list_samples, axis=0
    )  # Logarytmiczne stopy zwrotu dla każdego tickera
    final_data = unscale(final_data, cfg.data.scaling_factors)  # Odskalowanie
    final_data = np.exp(final_data) - 1  # Przeliczenie stóp logarytmicznych na proste
    final_data = weighting(final_data, cfg.data.weights)  # Nakładanie wag
    final_data = np.sum(final_data, 1)  # Łączna stopa dla całego portfela
    var10 = np.percentile(final_data, 1)
    logger.info("Ten day VaR 99")
    print(var10 * 100)
    logger.info("Ten day ES 97.5")
    es10 = np.mean(final_data[final_data <= np.percentile(final_data, 2.5)])
    print(es10 * 100)


if __name__ == "__main__":
    main()
