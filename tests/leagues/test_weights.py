import pandas as pd
from leagues.weights import decay_weights, HALF_LIFE_DAYS


def test_weight_is_one_at_reference_date():
    dates = pd.Series([pd.Timestamp("2026-07-01")])
    w = decay_weights(dates, ref=pd.Timestamp("2026-07-01"))
    assert abs(w.iloc[0] - 1.0) < 1e-9


def test_weight_halves_after_the_half_life():
    ref = pd.Timestamp("2026-07-01")
    old = ref - pd.Timedelta(days=HALF_LIFE_DAYS)
    w = decay_weights(pd.Series([old]), ref=ref)
    assert abs(w.iloc[0] - 0.5) < 1e-6


def test_older_matches_weigh_less():
    ref = pd.Timestamp("2026-07-01")
    dates = pd.Series([ref - pd.Timedelta(days=d) for d in (0, 200, 800)])
    w = decay_weights(dates, ref=ref)
    assert w.iloc[0] > w.iloc[1] > w.iloc[2] > 0
