"""Rung 1 -- the reward-rate index policy.  This is the thing that beats the sweep.

Reads DESIGN.md section 3.  The source architecture document writes the index as

    score(b) = P_hat(active_b) * w_b * (1 + alpha*staleness_b) - cost(a)

which is dimensionally unsound: it subtracts joules from a probability.  Since the
largest cost term is ~4 mJ and belief is O(1), the cost is effectively zero, the
policy thrashes across the band paying retune on every step, and it measurably
LOSES to a round-robin sweep (79.7 J at POI 0.50 versus 61.1 J at POI 0.88).

Two changes fix it, both in DESIGN.md:

  1. `w_p` is expressed in JOULES (see configs/*.yaml `mission.weights`), so
     `gain - cost` is a quantity in joules and the subtraction means something.
  2. The index is a reward RATE, because actions have variable duration.  Without
     dividing by duration a 200 ms dwell gaining 60 mJ beats a 5 ms dwell gaining
     20 mJ, even though the latter earns 4x the reward per second of mission time.

`score_mode: "raw"` keeps the document's literal form available so the difference
can be ablated rather than asserted.

Sleep needs no threshold.  Sleep(dt) has gain=0, cost=L_sleep*dt, duration=dt, so
score_rate(Sleep) = -L_sleep = -0.010 W regardless of dt, while a hopeless scan
scores about -1.1 W.  Sleep is therefore chosen exactly when every scan candidate
has negative reward rate -- the correct answer to "is doing nothing optimal now?"

FIREWALL: this module imports only `sim.contract` and `sim.config`.  It must never
import `sim.env`, `sim.emitters`, `sim.channel` or `sim.receiver`, and must never
touch `.truth*`.  See eval/tests/test_firewall.py.

log_rows() schema
-----------------
One row per channel per decision, consumed by agent/policy_learned.py, which
retroactively labels each row with "did the NEXT observation of this channel
report a detection".  Columns:

    t, step, channel, scanned, detected, dwell_s, bw_hz, <FEATURE_NAMES...>
"""
from __future__ import annotations

import numpy as np

from agent.base import FEATURE_NAMES, EnergyState, Policy
from agent.belief import Belief
from agent.scheduler import CandidateSet, Scheduler
from sim.contract import Action, ChannelGrid, Mission, Obs, Scan, Sleep

# Scratch run of 0x01 for marking a blocked span in the greedy window picker.
# Longer than any (2*bw - 1) we will ever slice out of it, so the picker never
# allocates inside its loop.
_ONES = b"\x01" * 4096


def _top_windows(nwin: np.ndarray, n: int, k_max: int) -> list[int]:
    """Greedy top-`k_max` NON-OVERLAPPING window starts, by descending value.

    `nwin` is the NEGATED window value, so ascending `argsort` is descending
    value and the caller gets the negation for free out of its prefix-sum
    subtraction instead of paying for a separate `-win` array.

    Picking the global best window, blanking everything it overlaps, and
    repeating is the whole selection rule; the only interesting part is doing it
    without walking the full sort order in Python.

    The bound that makes truncation exact: a pick at `k` blocks the `2n-1`
    starts in `[k-n+1, k+n)`, so after `j` picks at most `j*(2n-1)` entries of
    the sort order have been consumed.  Examining the first `k_max*(2n-1)`
    entries therefore yields *exactly* the same picks as scanning all of it --
    this is a faster spelling of the same function, not an approximation.

    `taken` is a bytearray and `order` a list of Python ints on purpose: indexing
    a numpy bool array with a numpy scalar costs ~1 us a time, which at ~325
    iterations per decision was most of the policy's runtime.
    """
    if k_max <= 0:
        return []
    w_n = nwin.size
    span = 2 * n - 1                      # starts blocked by one pick
    k_cap = k_max * span
    if k_cap > w_n:
        k_cap = w_n
    taken = bytearray(w_n)
    picks: list[int] = []
    n_got = 0
    for k in nwin.argsort()[:k_cap].tolist():
        if taken[k]:
            continue
        picks.append(k)
        n_got += 1
        if n_got >= k_max:
            break
        lo = k - n + 1
        if lo < 0:
            lo = 0
        hi = k + n
        if hi > w_n:
            hi = w_n
        taken[lo:hi] = _ONES[: hi - lo] if hi - lo <= len(_ONES) else b"\x01" * (hi - lo)
    return picks


class IndexPolicy:
    """Rung-1 policy: belief -> reward-rate index -> constrained scheduler.

    `greedy=True` produces the ablation baseline that isolates what the belief and
    the scheduler actually contribute: no Markov propagation (belief is a raw
    Laplace hit rate that never decays), no staleness bonus, and the scheduler's
    hard constraints disabled.
    """

    name = "index"

    def __init__(self, greedy: bool = False, collect_logs: bool = False):
        self.greedy = greedy
        self.name = "greedy" if greedy else "index"
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
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        acfg = cfg["agent"]
        en = cfg["energy"]
        rx = cfg["receiver"]

        # Greedy ablation uses a non-decaying empirical hit rate.
        self.belief = Belief(grid, mission, cfg, mode="laplace" if self.greedy else "bayes")

        self.scheduler = Scheduler(
            grid,
            mission,
            horizon_s,
            budget_slack=float(acfg.get("budget_slack", 0.05)),
            l_sleep_w=float(en["L_sleep_w"]),
            enabled=not self.greedy,
        )
        self.energy = EnergyState(spent_j=0.0, budget_j=float(en["budget_j"]))

        self.L_d = float(en["L_d_w"])
        self.L_0 = float(en["L_0_j"])
        self.L_f = float(en["L_f_j_per_hz"])
        self.L_sleep = float(en["L_sleep_w"])
        self.t_settle = float(rx["t_settle_s"])
        self.f_slew = float(rx["f_slew_hz_per_s"])

        self.alpha = 0.0 if self.greedy else float(acfg["alpha_staleness"])
        self.t_ref = float(acfg["staleness_t_ref_s"])
        self.stale_cap = float(acfg["staleness_cap"])
        self.score_mode = str(acfg.get("score_mode", "rate"))
        self.windows_per_bw = int(acfg.get("windows_per_bw", 8))

        self.bw_list = np.asarray(
            [int(b) for b in acfg["bw_candidates_mhz"]], dtype=np.int32
        )
        self.dwell_list = np.asarray(
            [float(d) * 1e-3 for d in acfg["dwell_candidates_ms"]], dtype=np.float64
        )
        self.sleep_list = np.asarray(
            [float(d) * 1e-3 for d in acfg["sleep_candidates_ms"]], dtype=np.float64
        )

        # pd_bar[bw_idx, dwell_idx] -- the MARGINAL detection probability, used both
        # for expected gain here and for miss likelihoods inside the belief.
        self.pd_bar = np.empty((self.bw_list.size, self.dwell_list.size), dtype=np.float64)
        for i, bw in enumerate(self.bw_list):
            for j, dw in enumerate(self.dwell_list):
                self.pd_bar[i, j] = self.belief.pd_bar_for(float(bw) * 1e6, float(dw))

        self._build_candidate_template(grid)

        self.f_last_hz = float(grid.center_hz(0))
        self.t = 0.0
        self.step_index = -1
        self.last_score = float("nan")
        self.last_reason = ""
        self._rows = []
        self._pending: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------- act
    def act(self, obs: Obs) -> Action:
        self.belief.update(obs)
        self.energy.spent_j = obs.energy_total
        self.t = obs.t
        self.step_index = obs.step_index + 1
        if self.collect_logs:
            self._label_pending(obs)

        cands, gain = self._enumerate(self.t)
        scores = self._score(cands, gain)

        action, reason = self.scheduler.select(
            cands, scores, self.belief, self.t, self.energy
        )
        self.last_score = float(np.max(scores)) if scores.size else float("nan")
        self.last_reason = reason

        if self.collect_logs:
            self._emit_rows(action)

        if isinstance(action, Scan):
            self.f_last_hz = action.f_center_hz
        return action

    # ----------------------------------------------------- candidate assembly
    def _build_candidate_template(self, grid: ChannelGrid) -> None:
        """Precompute everything about the candidate set that never changes.

        `_enumerate` runs once per decision (~4600 per episode) and its shape is
        fixed: `n_bw * n_dwell * windows_per_bw` scans plus the sleep rows.  Only
        three things actually vary from decision to decision -- which windows the
        greedy picks, their gains, and `f_last_hz`.  Every other column is a
        function of the config alone, so it is built here, once.

        Layout (must not change): bandwidth-major, then dwell-major, then window,
        with the sleep rows last.  `np.argmax` breaks ties on the first index, so
        re-ordering candidates would silently change which action gets chosen.
        """
        n_grid = int(grid.n_channels)
        # Python ints, so the per-bandwidth loop never touches a numpy scalar.
        self._bw_ints: tuple[tuple[int, int], ...] = tuple(
            (i, int(b)) for i, b in enumerate(self.bw_list) if int(b) <= n_grid
        )
        # cumsum buffer with the leading 0 already in place; only [1:] is written.
        self._cumsum_buf = np.zeros(n_grid + 1, dtype=np.float64)

        n_bw = len(self._bw_ints)
        n_dw = int(self.dwell_list.size)
        n_pk = max(1, int(self.windows_per_bw))
        n_sleep = int(self.sleep_list.size)
        n_scan = n_bw * n_dw * n_pk
        m = n_scan + n_sleep
        self._n_bw, self._n_dw, self._n_pk = n_bw, n_dw, n_pk
        self._n_scan, self._n_sleep = n_scan, m - n_scan

        cbw = float(grid.channel_bw_hz)
        blk = n_dw * n_pk

        n_ch = np.zeros(m, dtype=np.int32)
        dwell = np.empty(m, dtype=np.float64)
        dwell[n_scan:] = self.sleep_list
        dwell_blk = np.repeat(self.dwell_list, n_pk)
        for j, (_i_bw, n) in enumerate(self._bw_ints):
            n_ch[j * blk:(j + 1) * blk] = n
            dwell[j * blk:(j + 1) * blk] = dwell_blk

        # Columns that are identical every decision are SHARED, not copied, and
        # marked read-only so an accidental in-place write is an exception rather
        # than silent cross-decision corruption.
        for arr in (n_ch, dwell):
            arr.flags.writeable = False
        self._n_ch_col = n_ch
        self._dwell_col = dwell
        is_sleep = np.zeros(m, dtype=bool)
        is_sleep[n_scan:] = True
        is_sleep.flags.writeable = False
        self._is_sleep_col = is_sleep

        # Templates: copied per decision, sleep tail already correct, scan head
        # overwritten in place.
        self._k_lo_tpl = np.zeros(m, dtype=np.int32)
        self._k_lo_tpl[n_scan:] = -1
        self._gain_tpl = np.zeros(m, dtype=np.float64)
        self._cost_tpl = np.zeros(m, dtype=np.float64)
        self._cost_tpl[n_scan:] = self.L_sleep * self.sleep_list
        self._dur_tpl = np.zeros(m, dtype=np.float64)
        self._dur_tpl[n_scan:] = self.sleep_list

        # f_center = f_start + k_lo*cbw + n_ch*cbw/2; only the k_lo term varies.
        self._f_base = (
            float(grid.f_start_hz)
            + np.asarray(n_ch[:n_scan], dtype=np.float64) * (cbw * 0.5)
        )
        # cost = L_0 + L_d*dwell + L_f*|df|; only the |df| term varies.
        self._cost_base = self.L_0 + self.L_d * np.asarray(dwell[:n_scan])
        self._dwell_scan = np.asarray(dwell[:n_scan])
        self._cbw = cbw

        # Scratch, never escapes `_enumerate`.
        self._picks = np.empty((n_bw, n_pk), dtype=np.int32)
        self._df = np.empty(n_scan, dtype=np.float64)
        self._nwin = np.empty(n_grid, dtype=np.float64)
        # Negated once here so the negated window sums multiply straight through,
        # and pre-shaped as (n_dwell, 1) columns indexed by loop position.
        self._neg_pd_col = tuple(
            (-self.pd_bar[i_bw])[:, None] for i_bw, _n in self._bw_ints
        )

    def _enumerate(self, t: float) -> tuple[CandidateSet, np.ndarray]:
        """Vectorised candidate generation via prefix sums.

        Per-channel value is summed over each contiguous window with a cumsum, so
        finding the best window of every width costs O(N) rather than O(N*width).
        """
        p = self.belief.p_effective(t)
        w = self.mission.w
        if self.alpha > 0.0:
            stale = self.belief.staleness(t)
            stale_norm = np.minimum(stale / self.t_ref, self.stale_cap)
            value = p * w * (1.0 + self.alpha * stale_norm)
        else:
            value = p * w

        # Prefix sums straight into a preallocated buffer whose leading 0 is
        # permanent; `np.concatenate(([0.0], np.cumsum(value)))` allocated twice.
        cs = self._cumsum_buf
        np.cumsum(value, out=cs[1:])

        n_bw, n_dw, n_pk = self._n_bw, self._n_dw, self._n_pk
        n_scan = self._n_scan
        picks_all = self._picks

        gain = self._gain_tpl.copy()
        gain_view = gain[:n_scan].reshape(n_bw, n_dw, n_pk)

        nwin_buf = self._nwin
        for j, (i_bw, n) in enumerate(self._bw_ints):
            # NEGATED window sums, straight into scratch: one op, no allocation,
            # and `_top_windows` can argsort it ascending as-is.  The gain is
            # recovered by multiplying with the pre-negated `pd_bar` below, which
            # is exact (IEEE negation is exact, so no result changes).
            nwin = np.subtract(cs[:-n], cs[n:], out=nwin_buf[: cs.size - n])
            picks = _top_windows(nwin, n, n_pk)
            # Greedy can return FEWER than n_pk when the blocked spans exhaust the
            # band (a narrow grid, or a very wide bandwidth).  Pad by repeating the
            # last pick: the duplicate rows are exact copies of a real candidate,
            # so the score maximum -- and therefore the chosen action -- is
            # unchanged, and the array shape stays fixed.
            if len(picks) < n_pk:
                picks = picks + [picks[-1]] * (n_pk - len(picks))
            picks_all[j] = picks
            # outer product pd_bar[bw, :] x window_value[:] -> (n_dwell, n_pick).
            # The (n_dwell, 1) column is pre-shaped in reset() and the row side
            # broadcasts as 1-D, so this is two numpy calls rather than five.
            np.multiply(
                self._neg_pd_col[j], nwin[picks_all[j]], out=gain_view[j]
            )

        k_lo = self._k_lo_tpl.copy()
        k_lo[:n_scan].reshape(n_bw, n_dw, n_pk)[:] = picks_all[:, None, :]

        # |f_center - f_last|, built in place: k_lo*cbw + f_base - f_last.
        df = np.multiply(k_lo[:n_scan], self._cbw, out=self._df)
        df += self._f_base
        df -= self.f_last_hz
        np.abs(df, out=df)

        # Kept as a division and in this exact grouping: `df*(1/f_slew)` and any
        # re-association of `(t_settle + x) + dwell` differ from this by an ulp,
        # which is enough to flip a near-tie in the score and change the action.
        t_retune = np.where(df == 0.0, 0.0, self.t_settle + df / self.f_slew)

        cost = self._cost_tpl.copy()
        np.multiply(df, self.L_f, out=cost[:n_scan])
        cost[:n_scan] += self._cost_base

        duration = self._dur_tpl.copy()
        np.add(t_retune, self._dwell_scan, out=duration[:n_scan])

        return (
            CandidateSet(
                self.grid,
                k_lo,
                self._n_ch_col,
                self._dwell_col,
                cost,
                duration,
                self._is_sleep_col,
            ),
            gain,
        )

    def _score(self, cands: CandidateSet, gain: np.ndarray) -> np.ndarray:
        net = gain - cands.cost_j
        if self.score_mode == "raw":
            # The source document's literal form.  Retained so the ablation can
            # show why the rate form wins, rather than merely claiming it.
            return net
        return net / np.maximum(cands.duration_s, 1e-12)

    # ------------------------------------------------------------- log rows
    def _emit_rows(self, action: Action) -> None:
        """Emit one row per channel, to be labelled when next observed."""
        feats = self.belief.feature_matrix(self.t)
        if isinstance(action, Scan):
            chans = self.grid.channels_for(action.f_center_hz, action.bw_hz)
            dwell, bw = action.dwell_s, action.bw_hz
        else:
            chans = np.empty(0, dtype=np.int32)
            dwell, bw = 0.0, 0.0
        t, step = self.t, self.step_index
        dwell, bw = float(dwell), float(bw)
        if chans.size:
            # One .tolist() for the whole block: indexing `feats` with a numpy
            # scalar 16 times per channel is ~20x dearer than converting the
            # submatrix in one go.  Same values, same key order, same schema.
            vals = feats[chans].tolist()
            for c, fv in zip(chans.tolist(), vals):
                row = {
                    "t": t,
                    "step": step,
                    "channel": c,
                    "dwell_s": dwell,
                    "bw_hz": bw,
                    "detected": 0,
                }
                row.update(zip(FEATURE_NAMES, fv))
                self._pending.setdefault(c, []).append(row)

    def _label_pending(self, obs: Obs) -> None:
        """Retroactively label rows for channels this observation just visited."""
        if obs.step_index < 0 or obs.scanned_channels.size == 0:
            return
        hit = {d.channel for d in obs.detections}
        for c in obs.scanned_channels:
            c = int(c)
            rows = self._pending.pop(c, None)
            if not rows:
                continue
            y = 1 if c in hit else 0
            for r in rows:
                r["detected"] = y
                self._rows.append(r)

    def log_rows(self) -> list[dict]:
        return self._rows


class GreedyPolicy(IndexPolicy):
    """Ablation baseline: no belief decay, no staleness, no scheduler.

    Isolates exactly what the Markov propagation and the constrained scheduler
    contribute, which is the only way to claim they earn their place.
    """

    name = "greedy"

    def __init__(self, collect_logs: bool = False):
        super().__init__(greedy=True, collect_logs=collect_logs)
