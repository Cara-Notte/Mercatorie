from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LagHorizonConfig:
    """Calendar-aware lag matching configuration.

    `max_lookback_days` controls tolerance when exact `date - horizon_days` does not exist:
    - 0 => exact-match-only
    - >0 => use nearest observed date on or before target date within tolerance
    """

    horizon_days: int
    max_lookback_days: int = 0


@dataclass(frozen=True)
class TargetHorizonConfig:
    """Calendar-aware target matching configuration.

    Uses first observed future row on/after `date + horizon_days` within `max_lookahead_days`.
    """

    horizon_days: int
    max_lookahead_days: int = 7


@dataclass(frozen=True)
class RollingWindowConfig:
    """Calendar-aware trailing rolling-window configuration."""

    window_days: int
    min_observations: int = 1


DEFAULT_LAG_CONFIGS: dict[str, LagHorizonConfig] = {
    "price_lag_1d": LagHorizonConfig(horizon_days=1, max_lookback_days=0),
    "price_lag_7d": LagHorizonConfig(horizon_days=7, max_lookback_days=1),
    "price_lag_30d": LagHorizonConfig(horizon_days=30, max_lookback_days=2),
}

DEFAULT_ROLLING_CONFIGS: dict[str, RollingWindowConfig] = {
    "rolling_mean_7d": RollingWindowConfig(window_days=7, min_observations=1),
    "rolling_mean_30d": RollingWindowConfig(window_days=30, min_observations=1),
    "rolling_std_30d": RollingWindowConfig(window_days=30, min_observations=2),
}

DEFAULT_TARGET_CONFIGS: dict[str, TargetHorizonConfig] = {
    "target_7d_inflation_pct": TargetHorizonConfig(horizon_days=7, max_lookahead_days=3),
    "target_30d_inflation_pct": TargetHorizonConfig(horizon_days=30, max_lookahead_days=7),
}
