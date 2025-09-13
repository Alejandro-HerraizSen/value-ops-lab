import numpy as np
import pandas as pd
import pytest

from value_ops_lab.risk_models import quantile_forecast, cvar_cash_buffer

# Try to detect cvxpy availability to skip CVaR test if missing
try:
    import cvxpy as _cp  # noqa: F401
    _HAS_CVX = True
except Exception:
    _HAS_CVX = False


def test_quantile_forecast_basic_shapes_and_monotonicity():
    # Simple AR(1)-ish synthetic series for net cash flow
    rng = np.random.default_rng(7)
    n = 80
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.normal(0, 5.0)

    df = pd.DataFrame({"net_cf": y})
    df["lag1"] = df["net_cf"].shift(1)
    df["lag2"] = df["net_cf"].shift(2)
    df = df.dropna().reset_index(drop=True)

    X = df[["lag1", "lag2"]]
    y = df["net_cf"]
    # Forecast the last 5 rows as a future horizon example
    X_train, X_future = X.iloc[:-5], X.iloc[-5:]
    y_train = y.iloc[:-5]

    preds = quantile_forecast(X_train, y_train, X_future, quantiles=(0.1, 0.5, 0.9))

    # Shape checks
    assert set(preds.keys()) == {0.1, 0.5, 0.9}
    assert all(len(v) == 5 for v in preds.values())

    # Monotonicity across quantiles at each horizon step
    for i in range(5):
        assert preds[0.1][i] <= preds[0.5][i] <= preds[0.9][i]


@pytest.mark.skipif(not _HAS_CVX, reason="cvxpy not available")
def test_cvar_cash_buffer_returns_reasonable_values():
    # Scenarios represent end-of-period cash without buffer
    # Include some negatives to force a positive buffer
    scenarios = np.array([2000, -1500, 500, -3000, 1000, -500])
    buffer, t = cvar_cash_buffer(scenarios, alpha=0.95)

    assert isinstance(buffer, float) and isinstance(t, float)
    assert buffer >= 0.0

    # If scenarios are all very positive, buffer should be near zero
    good_scenarios = np.array([5000, 4000, 6000, 4500, 5500], dtype=float)
    buffer2, _ = cvar_cash_buffer(good_scenarios, alpha=0.95)
    assert buffer2 >= 0.0
    assert buffer2 <= buffer + 1e-6  # should not need more buffer than the risky set