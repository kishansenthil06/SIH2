"""`epsilon_greedy` -- the classic bandit baseline, wired in so it can be measured.

`agent/vendor/ml_scheduler.py` is a standard epsilon-greedy multi-armed bandit:
it tracks a lifetime hit rate per band, exploits the best with probability
`1 - epsilon` and explores uniformly otherwise. This module adapts it to the
project's action space and energy model so it competes on equal terms.

WHY THIS IS A BASELINE AND NOT THE POLICY
-----------------------------------------
The architecture review this project is built on names this exact approach as the
one that underperforms:

    "Bands turn on and off whether or not you are watching. That makes this a
     RESTLESS multi-armed bandit, not a standard one, and it is exactly why plain
     epsilon-greedy or UCB underperforms here -- those assume the arms wait for
     you."

`MLScheduler.get_hit_rate` is `hits / scans` over the whole episode with no decay
and no notion of elapsed time, so a band that was busy at t=2 s keeps that score
at t=50 s regardless of whether the emitter has since gone quiet. That is the
stationarity assumption made concrete. `agent/belief.py` differs in exactly one
respect that matters: it propagates `p(t+dt) = pi + (p - pi)*exp(-Lam*dt)` between
visits, so information *ages*.

Asserting that difference matters is cheap; measuring it is not, so this ships as
a baseline and the ablation reports the answer either way. If it wins, that is a
real result about the scenarios and belongs in the write-up.

WHAT THE WRAPPER HAS TO SUPPLY
------------------------------
The bandit chooses only *which band*. It has no dwell, no bandwidth, no energy and
no clock, which is the other half of the action space (DESIGN.md section 3). To
keep the comparison fair rather than rigged:

* **Bands are the widest legal scan.** One band == one `bw_mhz`-wide window, so a
  band maps to exactly one action and the bandit is never asked to point at
  something it cannot scan in one go.
* **Fixed `(bw, dwell)`, taken from the same tuned values as `round_robin`,** so
  it is not handicapped by a worse action than the sweep gets.
* **It is paced to survive the horizon.** `budget_j = 6.0` over 60 s is 0.1 W --
  roughly a 10% duty cycle -- so an unpaced scanner exhausts the budget at
  t ~ 5 s. A baseline that dies early looks artificially efficient because
  energy-per-detection divides by a truncated denominator, which would flatter
  our own result. Same treatment as the other baselines.
* **Reward is priority-weighted detections**, the same `w_p` in joules the index
  policy optimises, so both are scored against the same objective.

FIREWALL: imports only `sim.contract`; no `sim.env`, no `.truth*`.
"""
from __future__ import annotations

import numpy as np

from agent.vendor.ml_scheduler import MLScheduler
from sim.contract import Action, ChannelGrid, Mission, Obs, Scan, Sleep


class EpsilonGreedyPolicy:
    """`MLScheduler` adapted to `agent.base.Policy`.

    The decision logic is entirely the vendored bandit's; everything here is the
    adaptation described in the module docstring.
    """

    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.2, collect_logs: bool = False):
        self.epsilon = float(epsilon)
        self.collect_logs = collect_logs
        self._rows: list[dict] = []

    # ------------------------------------------------------------------ setup
    def reset(
        self,
        grid: ChannelGrid,
        mission: Mission,
        horizon_s: float,
        seed: int,
        cfg: dict,
    ) -> None:
        self.grid = grid
        self.mission = mission
        self.horizon_s = float(horizon_s)
        self.rng = np.random.default_rng(seed)

        en = cfg["energy"]
        rx = cfg["receiver"]
        self.L_d = float(en["L_d_w"])
        self.L_0 = float(en["L_0_j"])
        self.L_f = float(en["L_f_j_per_hz"])
        self.L_sleep = float(en["L_sleep_w"])
        self.budget_j = float(en["budget_j"])
        self.t_settle = float(rx["t_settle_s"])
        self.f_slew = float(rx["f_slew_hz_per_s"])

        # Same action the tuned sweep gets, so the bandit is not handicapped by a
        # worse dwell than its comparator.
        rr = dict(cfg.get("baselines", {}).get("round_robin", {}))
        self.n_ch = int(rr.get("bw_mhz", 5))
        self.dwell_s = float(rr.get("dwell_ms", 10.0)) * 1e-3

        # One band == one full-width scan, so every band is reachable in a single
        # action.  Any remainder at the top of the band is dropped rather than
        # producing a short final window the bandit would over-select.
        self.n_bands = grid.n_channels // self.n_ch
        bands = list(range(self.n_bands))
        self.sched = MLScheduler(frequency_bands=bands, epsilon=self.epsilon)

        # The vendored module uses the global `random`; seed it so a run is
        # reproducible on `(policy, scenario, seed)` like every other policy.
        import random as _random

        _random.seed(seed)

        # Pace to the energy budget: spread the affordable scans evenly over the
        # horizon rather than spending the budget in the first few seconds.
        #
        # The retune term dominates and must not be left out.  The bandit picks
        # bands with no regard for frequency locality, so consecutive scans hop
        # anywhere in the 2 GHz span: for a uniform random pair the mean |df| is
        # span/3, giving L_f*span/3 = 13.3 mJ against a 12.1 mJ scan.  Costing
        # only the scan (as a first version of this did) under-estimates by ~2x,
        # and the episode died at t = 39 s with the budget gone -- which would
        # have flattered the baseline's energy-per-detection by truncating its
        # denominator, the exact failure this pacing exists to prevent.
        mean_retune_hz = grid.span_hz / 3.0
        step_cost = (
            self.L_0
            + self.L_d * self.dwell_s
            + self.L_f * mean_retune_hz
        )
        affordable = max(1.0, self.budget_j / max(step_cost, 1e-12))
        self.sleep_between_s = max(0.0, self.horizon_s / affordable - self.dwell_s)

        self.f_last_hz = float(grid.center_hz(0))
        self.t = 0.0
        self.energy_spent = 0.0
        self._last_scan_band: int | None = None
        self._next_is_scan: bool = True
        self.last_reason = ""
        self.last_score = float("nan")
        self._rows = []

    # -------------------------------------------------------------------- act
    def act(self, obs: Obs) -> Action:
        self.t = obs.t
        self.energy_spent = obs.energy_total

        # Credit the scan that produced this observation, if it was one of ours.
        if self._last_scan_band is not None and obs.scanned_channels.size:
            hit = len(obs.detections) > 0
            # Reward is priority-weighted detections in joules -- the same `w_p`
            # the index policy optimises -- so both are scored against one
            # objective rather than each against its own.
            reward = float(sum(self.mission.w[d.channel] for d in obs.detections))
            self.sched.update(self._last_scan_band, reward, hit)
        self._last_scan_band = None

        # Pacing belongs to the wrapper, not the bandit: alternate scan and sleep
        # so the episode reaches the horizon instead of burning the 0.1 W budget
        # in the first few seconds.
        if self.sleep_between_s > 0.0 and not self._next_is_scan:
            self._next_is_scan = True
            self.last_reason = "pace"
            dt = min(self.sleep_between_s, max(0.0, self.horizon_s - self.t))
            return Sleep(max(dt, 1e-3))

        self._next_is_scan = False
        band = self.sched.choose_band()          # <- the vendored bandit decides
        self._last_scan_band = band
        self.last_reason = "epsilon-greedy"
        action = self.grid.action_for(band * self.n_ch, self.n_ch, self.dwell_s)
        self.f_last_hz = action.f_center_hz
        return action

    def log_rows(self) -> list[dict]:
        return self._rows

    # ------------------------------------------------------------- diagnostics
    def statistics(self) -> dict:
        """The bandit's own learned table, for inspection and the dashboard."""
        return self.sched.get_statistics()
