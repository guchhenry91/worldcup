"""Exponential time-decay: recent matches count more (Dixon-Coles xi).

CAUTION: the famous xi=0.0065 from the 1997 paper is per HALF-WEEK, not per day.
In days the sweet spot is ~0.0018-0.0033; we default to 0.003 (half-life ~231d)
and tune per league by walk-forward RPS in backtest.py.
"""
import numpy as np
import pandas as pd

XI_PER_DAY = 0.003
HALF_LIFE_DAYS = np.log(2) / XI_PER_DAY   # ~231 days


def decay_weights(dates: pd.Series, ref: pd.Timestamp, xi: float = XI_PER_DAY) -> pd.Series:
    """weight = exp(-xi * days_before_ref), clipped at 0 for future matches."""
    dates = pd.to_datetime(dates)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if getattr(ref, "tzinfo", None) is not None:
        ref = ref.tz_localize(None)
    age_days = (ref - dates).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0)
    return np.exp(-xi * age_days)
