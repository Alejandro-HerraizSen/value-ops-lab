import pandas as pd
import numpy as np

from value_ops_lab.ccc import dso, dpo, dio, ccc, cash_unlocked


def test_dso_dpo_dio_and_ccc_basic():
    # Build tiny deterministic monthly data
    months = pd.date_range("2025-01-01", periods=2, freq="MS")
    sales_df = pd.DataFrame({"month": months, "sales": [300.0, 300.0]})
    cogs_df = pd.DataFrame({"month": months, "cogs": [200.0, 200.0]})
    ar_df = pd.DataFrame({"month": months, "ar_balance": [100.0, 100.0]})
    ap_df = pd.DataFrame({"month": months, "ap_balance": [50.0, 50.0]})
    inv_df = pd.DataFrame({"month": months, "inventory": [80.0, 80.0]})

    dso_df = dso(ar_df, sales_df)
    dpo_df = dpo(ap_df, cogs_df)
    dio_df = dio(inv_df, cogs_df)
    ccc_df = ccc(dso_df, dpo_df, dio_df)

    # Expected formulas with 365 day year
    expected_dso = 365.0 * (100.0 / 300.0)
    expected_dpo = 365.0 * (50.0 / 200.0)
    expected_dio = 365.0 * (80.0 / 200.0)
    expected_ccc = expected_dso + expected_dio - expected_dpo

    assert np.isclose(dso_df["DSO"].iat[0], expected_dso)
    assert np.isclose(dpo_df["DPO"].iat[0], expected_dpo)
    assert np.isclose(dio_df["DIO"].iat[0], expected_dio)
    assert np.isclose(ccc_df["CCC"].iat[0], expected_ccc)

    # Shape and columns sanity
    assert set(ccc_df.columns) == {"month", "DSO", "DPO", "DIO", "CCC"}
    assert len(ccc_df) == 2


def test_cash_unlocked_nonnegative_and_value():
    # If current CCC reduces from 60 to 50 with 1k daily sales, unlock is 10k
    assert cash_unlocked(60.0, 50.0, 1000.0) == 10000.0