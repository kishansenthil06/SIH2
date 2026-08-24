"""FROZEN AT PHASE 0 -- agent-side protocols and the feature contract.

`FEATURE_NAMES` is frozen here so that the belief implementation (agent B) and the
learned model (agent D) can be built in parallel without either touching the
other's files.  D tests against a synthetic (N, F) matrix; B guarantees
`Belief.feature_matrix(t).shape == (n_channels, len(FEATURE_NAMES))`.

Nothing in this module may import from `sim` except `sim.contract`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from sim.contract import Action, ChannelGrid, Mission, Obs

# ---------------------------------------------------------------------------
# Rung-2 feature contract.  F = 16.
#
# Deliberately ABSENT: the raw channel index.  Including it would let the model
# memorise emitter positions from `sparse` and destroy generalisation to the
# held-out `agile` scenario.  Only *relative* spectral context is permitted,
# which is what `nbr_recent_hits` provides.
# ---------------------------------------------------------------------------
FEATURE_NAMES: tuple[str, ...] = (
    "p_rung1",              # 0  analytic belief -- model can only REFINE it
    "log_staleness",        # 1  log1p(t - t_last_visit)
    "log_since_detect",     # 2  log1p(t - t_last_detect); log1p(t) if never
    "n_visits",             # 3  visits to this channel this episode
    "emp_rate",             # 4  (n_det + 1) / (n_visits + 2)   Laplace
    "hit_ema_fast",         # 5  alpha = 0.30
    "hit_ema_slow",         # 6  alpha = 0.05
    "misses_since_detect",  # 7  run length
    "mean_dwell_log",       # 8  log1p(dwell_sum / max(n_visits, 1))
    "mean_snr_db",          # 9  mean reported SNR of detections; -30.0 if none
    "idi_mean",             # 10 mean inter-detection interval; -1.0 if < 2
    "idi_std",              # 11 std of same; -1.0 if < 3
    "nbr_recent_hits",      # 12 detections on c+/-1..2 within 1.0 s -- CATCHES THE HOPPER
    "band_activity",        # 13 fraction of channels detected-on in last 1.0 s
    "w_channel",            # 14 mission weight (joules)
    "t_frac",               # 15 t / horizon_s
)
N_FEATURES: int = len(FEATURE_NAMES)

# Two extra columns present only at TRAIN time, appended after FEATURE_NAMES.
# The label is "did the NEXT observation of this channel report a detection",
# which depends on the dwell/bw of that observation -- so the model is given
# them rather than being made to average over them.
TRAIN_EXTRA_NAMES: tuple[str, ...] = ("tau_next_log", "bw_next_log")

SENTINEL_NO_SNR: float = -30.0
SENTINEL_NO_IDI: float = -1.0


@dataclass(slots=True)
class EnergyState:
    """Energy bookkeeping handed to the scheduler."""

    spent_j: float = 0.0
    budget_j: float = float("inf")

    @property
    def remaining_j(self) -> float:
        return self.budget_j - self.spent_j

    def allowed_by(self, t: float, horizon_s: float, slack: float = 0.05) -> float:
        """Pacing target: how much we are permitted to have spent by time t."""
        if not np.isfinite(self.budget_j):
            return float("inf")
        return self.budget_j * min(1.0, t / max(horizon_s, 1e-12) + slack)

    def over_pace(self, t: float, horizon_s: float, slack: float = 0.05) -> bool:
        return self.spent_j > self.allowed_by(t, horizon_s, slack)


# Reasons a scheduler may give for its choice.  Rendered beside the waterfall in
# the dashboard -- this is the "explain any decision to a judge" property.
REASONS: tuple[str, ...] = (
    "index",       # the reward-rate index picked it
    "deadline",    # a hard revisit deadline forced it (suffixed ":ch=k")
    "watchlist",   # a watch-list channel came due (suffixed ":ch=k")
    "budget-pace", # over the energy pacing curve; forced to sleep
    "sleep",       # every scan candidate had negative reward rate
    "fallback",    # no feasible candidate; degenerate sleep
)


@runtime_checkable
class Policy(Protocol):
    """Every baseline and every learned policy satisfies this."""

    name: str

    def reset(
        self,
        grid: ChannelGrid,
        mission: Mission,
        horizon_s: float,
        seed: int,
        cfg: dict,
    ) -> None: ...

    def act(self, obs: Obs) -> Action: ...

    def log_rows(self) -> list[dict]:
        """Per-decision agent-side log used to train rung 2.  May be empty."""
        ...


@runtime_checkable
class BeliefLike(Protocol):
    """The belief layer, as seen by the policy and by the learned model."""

    def update(self, obs: Obs) -> None: ...
    def p_active(self, t: float) -> np.ndarray: ...
    def p_effective(self, t: float) -> np.ndarray: ...
    def staleness(self, t: float) -> np.ndarray: ...
    def feature_matrix(self, t: float) -> np.ndarray: ...
    def attach_model(self, model, beta: float) -> None: ...
