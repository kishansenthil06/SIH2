"""Hard-constraint scheduler.  It picks; it never scores.

```python
Scheduler.select(cands, scores, belief, t, energy) -> (action, reason)
```

The type signature *is* the architecture: `scores` arrives as an opaque `(M,)`
array this class did not compute and cannot recompute.  The learner proposes
value; the scheduler picks under hard constraints.  That separation is what makes
every decision explainable live -- each one comes back with a `reason` from
`agent.base.REASONS`, rendered beside the waterfall, and it is exactly why revisit
deadlines live here rather than being folded into the index (a deadline buried in
a score is a suggestion; a deadline in a filter is a guarantee).

Layers, applied in order (DESIGN.md section 7):

1. **Feasibility (hard).**  Drop candidates that overrun the horizon or the
   remaining budget.  Extended, deliberately: a candidate must also leave enough
   energy to *sleep out the rest of the episode* (`L_sleep*(horizon - t_end)`).
   Without that reserve a policy can strand itself -- unable to afford even
   standby -- and terminate early on the budget rather than the horizon, which
   DESIGN.md section 1 calls a failed policy, not a frugal one.
2. **Revisit deadlines (hard).**  `{1: 0.5, 2: 2.0, 3: 10.0}` s on mission
   channels.  If anything is overdue, restrict to candidates covering the *most*
   overdue channel and take the best-scoring one.  This is what makes "max
   staleness is hard-bounded" a provable property rather than an empirical one.
3. **Watch list.**  Own 0.3 s deadline, treated as priority 1.
4. **Budget pacing (soft).**  `allowed(t) = budget*(t/horizon + slack)`; if
   overspent, suppress scans and sleep -- *except* under an active deadline
   override.  Soft because layer 1 is already the hard cap; this only shapes
   *when* the budget is spent, so a deadline can borrow against it.
5. **Sleep clamp.**  `dt = min(dt, next_deadline - t)`, floored at 1 ms, so a long
   sleep can never be the thing that blows a deadline.

`Scheduler(enabled=False)` degrades to layer 1 only.  That is the `greedy`
baseline's configuration: no deadlines, no watch list, no pacing, no clamp.  Layer
1 is kept even there because emitting an action the receiver cannot afford is not
policy behaviour, it is just early termination, and DESIGN.md section 6 requires
baselines to survive the full horizon to be a fair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.base import EnergyState
from sim.contract import Action, ChannelGrid, Mission, Sleep

# Floor on a clamped sleep.  Zero-length sleeps would spin the episode loop
# without advancing the clock.
MIN_SLEEP_S: float = 1.0e-3

# Slop for float comparisons on energy/time budgets.  1 nJ / 1 ns: far below any
# quantity in the model, far above float64 accumulation error over an episode.
_EPS: float = 1.0e-9


@dataclass(slots=True)
class CandidateSet:
    """Columnar candidate list.  No `Scan` object exists until one is chosen.

    Materialising 320+ frozen dataclasses per decision would cost more than the
    entire scoring pass, so candidates live as parallel numpy arrays and exactly
    one action is built -- via `ChannelGrid.action_for`, so it is legal by
    construction.

    Sleep rows carry `k_lo = -1`, `n_ch = 0`, and their `dt` in `dwell_s`.
    """

    grid: ChannelGrid
    k_lo: np.ndarray        # int32 (M,)
    n_ch: np.ndarray        # int32 (M,)
    dwell_s: np.ndarray     # float64 (M,)  -- dt_s for sleep rows
    cost_j: np.ndarray      # float64 (M,)
    duration_s: np.ndarray  # float64 (M,)
    is_sleep: np.ndarray    # bool (M,)

    def __len__(self) -> int:
        return int(self.k_lo.size)

    def covers(self, ch: int) -> np.ndarray:
        """(M,) bool -- which scan candidates include channel `ch`."""
        return (~self.is_sleep) & (self.k_lo <= ch) & (ch < self.k_lo + self.n_ch)

    def action(self, i: int) -> Action:
        """Materialise candidate `i`.  Scans go through `grid.action_for` only."""
        if bool(self.is_sleep[i]):
            return Sleep(float(self.dwell_s[i]))
        return self.grid.action_for(int(self.k_lo[i]), int(self.n_ch[i]),
                                    float(self.dwell_s[i]))


class Scheduler:
    """Hard constraints only.  Stateless between decisions apart from its config."""

    def __init__(
        self,
        grid: ChannelGrid,
        mission: Mission,
        horizon_s: float,
        budget_slack: float = 0.05,
        l_sleep_w: float = 0.01,
        enabled: bool = True,
    ):
        self.grid = grid
        self.mission = mission
        self.horizon_s = float(horizon_s)
        self.budget_slack = float(budget_slack)
        self.l_sleep_w = float(l_sleep_w)
        self.enabled = bool(enabled)

        # Priority deadlines and watch-list deadlines are kept SEPARATE (rather
        # than using Mission.deadline_for(), which mins them together) so the
        # reason string can say which of the two fired.
        prio = np.asarray(mission.priority)
        self.deadline_prio = np.full(prio.shape, np.inf, dtype=np.float64)
        for p, d in mission.deadlines_s.items():
            self.deadline_prio[prio == int(p)] = float(d)

        self.watch = np.asarray(mission.watch_list, dtype=np.int64)
        self.watch_deadline_s = float(mission.watch_deadline_s)

        self.deadline_all = self._combined_deadline()

        self.last_reason: str = "index"

    def _combined_deadline(self) -> np.ndarray:
        out = self.deadline_prio.copy()
        out[self.watch] = np.minimum(out[self.watch], self.watch_deadline_s)
        return out

    # ------------------------------------------------------------------ main
    def select(
        self,
        cands: CandidateSet,
        scores: np.ndarray,
        belief,
        t: float,
        energy: EnergyState,
    ) -> tuple[Action, str]:
        """Choose one candidate under hard constraints.  Never scores anything."""
        m = len(cands)
        if m == 0:
            return self._fallback_sleep(t)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (m,):
            raise ValueError(f"scores must be ({m},), got {scores.shape}")

        t = float(t)

        # ---- layer 1: feasibility (HARD) ---------------------------------
        t_end = t + cands.duration_s
        feas = t_end <= self.horizon_s + _EPS
        # Reserve enough to stand by until the horizon; see the module docstring.
        reserve = self.l_sleep_w * np.maximum(self.horizon_s - t_end, 0.0)
        feas &= (cands.cost_j + reserve) <= (energy.remaining_j + _EPS)
        if not feas.any():
            return self._fallback_sleep(t)

        if not self.enabled:
            # `greedy` baseline: layer 1 only, then pure argmax of a score this
            # class did not compute.
            i = int(np.argmax(np.where(feas, scores, -np.inf)))
            return cands.action(i), ("sleep" if bool(cands.is_sleep[i]) else "index")

        stale = belief.staleness(t)

        # Earliest deadline still in the FUTURE.  Already-blown deadlines are
        # handled by layer 2; this quantity exists to stop a long action from
        # blowing the *next* one.
        due_at = belief.t_last_visit + self.deadline_all
        future = due_at[np.isfinite(due_at) & (due_at > t)]
        next_deadline_t = float(future.min()) if future.size else np.inf

        # ---- layer 1b: do not overshoot the next deadline -----------------
        # A soft narrowing of layer 1: prefer candidates that finish before the
        # next deadline comes due.  If that empties the set we fall back to plain
        # feasibility, so this can never deadlock.
        soft = feas & (t_end <= next_deadline_t + _EPS)
        pool = soft if soft.any() else feas

        # ---- layer 2: revisit deadlines (HARD) ---------------------------
        over = stale - self.deadline_prio      # inf-deadline channels give -inf
        act = self._deadline_override(cands, scores, over, pool, feas, "deadline")
        if act is not None:
            return act

        # ---- layer 3: watch list ------------------------------------------
        if self.watch.size:
            over_w = np.full(stale.shape, -np.inf, dtype=np.float64)
            over_w[self.watch] = stale[self.watch] - self.watch_deadline_s
            act = self._deadline_override(cands, scores, over_w, pool, feas, "watchlist")
            if act is not None:
                return act

        # ---- layer 4: budget pacing (SOFT) -------------------------------
        if energy.over_pace(t, self.horizon_s, self.budget_slack):
            sl = pool & cands.is_sleep
            if sl.any():
                # Sleeping "as long as allowed" is the constraint's own semantics,
                # not a score: layer 5 clamps it to the next deadline anyway.
                i = int(np.argmax(np.where(sl, cands.dwell_s, -np.inf)))
                return self._clamp_sleep(cands, i, t, next_deadline_t), "budget-pace"

        # ---- normal path: argmax of someone else's score ------------------
        i = int(np.argmax(np.where(pool, scores, -np.inf)))
        if bool(cands.is_sleep[i]):
            # Sleep won on merit: every scan candidate had a negative reward rate.
            return self._clamp_sleep(cands, i, t, next_deadline_t), "sleep"
        return cands.action(i), "index"

    # ------------------------------------------------------------- internals
    def _deadline_override(self, cands, scores, over, pool, feas, tag):
        """Restrict to candidates covering the MOST overdue channel, then argmax."""
        if not np.isfinite(over).any():
            return None
        c_star = int(np.argmax(over))
        if not (over[c_star] > 0.0):
            return None
        cov = cands.covers(c_star)
        sel = cov & pool
        if not sel.any():
            sel = cov & feas          # the covering scan may itself overshoot
        if not sel.any():
            return None               # cannot afford it; fall through to pacing
        i = int(np.argmax(np.where(sel, scores, -np.inf)))
        return cands.action(i), f"{tag}:ch={c_star}"

    def _clamp_sleep(self, cands: CandidateSet, i: int, t: float,
                     next_deadline_t: float) -> Action:
        """Layer 5.  A sleep must never be the thing that blows a deadline."""
        dt = float(cands.dwell_s[i])
        if np.isfinite(next_deadline_t):
            dt = min(dt, max(next_deadline_t - t, 0.0))
        dt = min(dt, max(self.horizon_s - t, 0.0))
        return Sleep(max(dt, MIN_SLEEP_S))

    def _fallback_sleep(self, t: float) -> tuple[Action, str]:
        """Nothing is affordable.  Stand by; the env clamps to the horizon."""
        return Sleep(MIN_SLEEP_S), "fallback"
