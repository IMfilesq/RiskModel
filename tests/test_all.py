import numpy as np
import pandas as pd

from riskmodel.data.transforms import log_returns


def test_log_returns():
    input = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    output = pd.DataFrame(
        {
            "A": [np.log(2) - np.log(1), np.log(3) - np.log(2)],
            "B": [np.log(5) - np.log(4), np.log(6) - np.log(5)],
        },
        index=[1, 2],
    )

    assert log_returns(input).equals(output), (
        "log_returns function did not return expected output"
    )
