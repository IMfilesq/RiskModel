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
from riskmodel.data.transforms import log_returns, scale
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
    print(params)

    # od tego miejsca leci kod z czata, na razie tylko prototyp, nie ufać temu za bardzo

    logger.info("Calculating theoretical CDF of skewed-t residuals for copula input")
    residuals_df = garch_resid.df
    u_data = np.zeros_like(residuals_df, dtype=float)

    for i, ticker in enumerate(residuals_df.columns):
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

    logger.info("Generating new data from copula")
    new_data = copula.sample(n=500)
    print(new_data)

    # tutaj należy spowrotem przeskalować dane do formatu reszt z garch (przepuścić przez kwantylowe)
    # potem żeby obliczyć symulowany dzień (kolejny po danych jakie mieliśmy) korzystamy z rówania GARCH
    # można teraz obliczy VaR na dzień w przód
    # żeby obliczyć VaR na 10 dni w przód trzeba 10 razy użyć równania GARCH
    # taka jest głowna idea tego co zostało do roboty ale sam nie do końca wszystko rozumiem


if __name__ == "__main__":
    main()
