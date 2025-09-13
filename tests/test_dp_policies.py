import numpy as np

from value_ops_lab.dp_policies import dp_pay_terms


def test_dp_policy_structure_and_actions_present():
    # One obligation due now. Large late penalty should push "pay_now".
    cash_today = 0.0
    obligations = [(1000.0, 0)]  # amount, due now
    discount = 0.02
    penalty = 0.50  # very high late penalty
    horizon = 1
    grid_size = 21

    policy = dp_pay_terms(
        cash_today=cash_today,
        obligations=obligations,
        discount=discount,
        penalty=penalty,
        horizon=horizon,
        grid_size=grid_size,
    )

    # Policy should have entries for period 0 and all grid points
    assert len(policy) == grid_size
    # There should be at least one "pay_now" recommendation
    assert any(a == "pay_now" for a in policy.values())


def test_dp_policy_delay_possible_with_low_penalty_and_zero_discount():
    # Same obligation, but no discount and tiny penalty can make "delay" appear
    cash_today = 0.0
    obligations = [(1000.0, 0)]
    discount = 0.0
    penalty = 0.0
    horizon = 1
    grid_size = 21

    policy = dp_pay_terms(
        cash_today=cash_today,
        obligations=obligations,
        discount=discount,
        penalty=penalty,
        horizon=horizon,
        grid_size=grid_size,
    )

    # Expect at least one "delay" action somewhere on the grid
    assert any(a == "delay" for a in policy.values())
    # And at least one "pay_now" for higher cash states
    assert any(a == "pay_now" for a in policy.values())