"""Ground-truth replay: turn an action/detection log + a burst table into numbers.

This module and the simulator are the ONLY things allowed to see ground truth.
It never imports anything from `agent/`, and it takes the burst table as a plain
structured array (`sim.emitters.BURST_DTYPE`) rather than a `World`, so every
metric is unit-testable against a hand-built table.

Every definition here follows DESIGN.md section 6 exactly.  The sharp edges are
deliberate; each one closes a specific way of accidentally lying with the
numbers, and the docstrings say which.

Time conventions, stated once because everything downstream depends on them:

* A scan's *observation window* is ``[t_dwell_start, t_end)`` -- the retune is
  excluded, because the receiver is deaf while the synthesiser settles.
* A detection is credited at ``t_end`` (the moment the report exists), so
  POI@T and TTFI both use the end of the dwell.  Crediting at the start would
  let a 200 ms dwell straddling T count as being inside it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# T values for POI@T.  Fixed by DESIGN.md section 6.
POI_TIMES: tuple[float, ...] = (10.0, 30.0, 60.0)

# Trace-file schema.  Agent D's dashboard consumes this; keep it stable.
TRACE_COLUMNS: tuple[str, ...] = (
    "step",
    "t_start",
    "t_end",
    "kind",
    "f_center_hz",
    "bw_hz",
    "dwell_s",
    "energy_j",
    "n_det",
    "det_channels",
    "best_score",
    "chosen_reason",
    "energy_spent_total",
)


# --------------------------------------------------------------------- records
@dataclass(slots=True)
class StepRecord:
    """One executed action, as the evaluator needs to see it.

    Deliberately holds the *executed* action (the env may truncate a dwell at the
    horizon), not the requested one -- replaying the request would credit
    coverage the receiver never had.
    """

    step: int
    kind: str                 # "scan" | "sleep"
    t_start: float            # clock when the action began (retune included)
    t_end: float              # clock when it finished
    dwell_s: float = 0.0      # 0.0 for Sleep
    retune_s: float = 0.0
    f_center_hz: float = 0.0
    bw_hz: float = 0.0
    k_lo: int = 0             # first scanned channel
    k_hi: int = 0             # one past the last scanned channel (EXCLUSIVE)
    energy_j: float = 0.0
    energy_spent_total: float = 0.0
    det_channels: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    det_snr_db: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    best_score: float = float("nan")
    chosen_reason: str = ""

    @property
    def t_dwell_start(self) -> float:
        """Start of the observation window.  Retune does not count as coverage."""
        return self.t_end - self.dwell_s

    @property
    def n_channels(self) -> int:
        return max(0, self.k_hi - self.k_lo)

    def trace_row(self) -> dict:
        return {
            "step": self.step,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "kind": self.kind,
            "f_center_hz": self.f_center_hz,
            "bw_hz": self.bw_hz,
            "dwell_s": self.dwell_s,
            "energy_j": self.energy_j,
            "n_det": int(self.det_channels.size),
            # space-separated so the field never needs CSV quoting
            "det_channels": " ".join(str(int(c)) for c in self.det_channels),
            "best_score": self.best_score,
            "chosen_reason": self.chosen_reason,
            "energy_spent_total": self.energy_spent_total,
        }


class EpisodeLog:
    """Accumulates `StepRecord`s from `Obs` objects and totals the energy.

    The energy breakdown is recomputed from the *timing* model rather than read
    out of `Obs.energy_cost`, which is what makes `sum(breakdown) == total` an
    independent check that the two models have not drifted (DESIGN.md section 1).
    """

    __slots__ = ("steps", "energy", "horizon_s", "n_channels", "_e_total")

    def __init__(self, energy: dict | None = None, horizon_s: float = 60.0,
                 n_channels: int = 200):
        self.steps: list[StepRecord] = []
        # Defaults are the frozen constants from configs/*.yaml so a hand-built
        # test log does not have to supply them.
        self.energy = dict(
            L_d_w=1.0, L_0_j=2.0e-3, L_f_j_per_hz=2.0e-11, L_sleep_w=0.01,
        )
        if energy:
            self.energy.update({k: float(v) for k, v in energy.items()
                                if k in self.energy})
        self.horizon_s = float(horizon_s)
        self.n_channels = int(n_channels)
        self._e_total = 0.0

    # ------------------------------------------------------------------ build
    def add(self, rec: StepRecord) -> StepRecord:
        self.steps.append(rec)
        self._e_total = max(self._e_total, rec.energy_spent_total)
        return rec

    def add_step(self, **kw) -> StepRecord:
        """Convenience constructor used by the hand-built ground-truth tests."""
        rec = StepRecord(**kw)
        if rec.energy_spent_total == 0.0:
            rec.energy_spent_total = self._e_total + rec.energy_j
        return self.add(rec)

    def record_obs(self, obs, best_score: float = float("nan"),
                   chosen_reason: str = "") -> StepRecord:
        """Build a `StepRecord` from a `sim.contract.Obs`.

        Works identically against `StubEnv` and the real `World`: it prefers
        `info["t_dwell_start"]` when present and otherwise derives the window
        from `t_end - dwell`, which is the same quantity.
        """
        act = obs.action
        kind = "scan" if getattr(act, "bw_hz", None) is not None else "sleep"
        info = obs.info or {}
        if kind == "sleep":
            rec = StepRecord(
                step=int(obs.step_index), kind="sleep",
                t_start=float(obs.t_start), t_end=float(obs.t),
                energy_j=float(obs.energy_cost),
                energy_spent_total=float(obs.energy_total),
                chosen_reason=chosen_reason, best_score=float(best_score),
            )
            return self.add(rec)

        dwell = float(act.dwell_s)
        chans = np.asarray(obs.scanned_channels, dtype=np.int64)
        k_lo = int(chans[0]) if chans.size else 0
        k_hi = int(chans[-1]) + 1 if chans.size else 0
        t_dwell_start = float(info.get("t_dwell_start", float(obs.t) - dwell))
        rec = StepRecord(
            step=int(obs.step_index), kind="scan",
            t_start=float(obs.t_start), t_end=float(obs.t),
            dwell_s=dwell,
            retune_s=float(info.get("t_retune", t_dwell_start - float(obs.t_start))),
            f_center_hz=float(act.f_center_hz), bw_hz=float(act.bw_hz),
            k_lo=k_lo, k_hi=k_hi,
            energy_j=float(obs.energy_cost),
            energy_spent_total=float(obs.energy_total),
            det_channels=np.asarray([d.channel for d in obs.detections],
                                    dtype=np.int32),
            det_snr_db=np.asarray([d.snr_db for d in obs.detections],
                                  dtype=np.float64),
            best_score=float(best_score), chosen_reason=chosen_reason,
        )
        return self.add(rec)

    # ------------------------------------------------------------- accounting
    def totals(self) -> dict:
        scans = [s for s in self.steps if s.kind == "scan"]
        sleeps = [s for s in self.steps if s.kind == "sleep"]
        e = self.energy
        dwell = sum(s.dwell_s for s in scans)
        retune = sum(s.retune_s for s in scans)
        sleep_t = sum(s.t_end - s.t_start for s in sleeps)
        # |df| is recoverable from the retune duration: t_retune = t_settle +
        # |df|/f_slew and L_f = L_d/f_slew, so L_f*|df| == L_d*(t_retune -
        # t_settle).  Recomputing it from the *recorded* energy avoids needing
        # t_settle here and keeps the breakdown exact.
        e_fixed = e["L_0_j"] * len(scans)
        e_dwell = e["L_d_w"] * dwell
        e_sleep = e["L_sleep_w"] * sleep_t
        e_scan_recorded = sum(s.energy_j for s in scans)
        e_retune = e_scan_recorded - e_fixed - e_dwell
        return {
            "n_steps": len(self.steps),
            "n_scans": len(scans),
            "n_sleeps": len(sleeps),
            "dwell_time_s": dwell,
            "retune_time_s": retune,
            "sleep_time_s": sleep_t,
            "t_end_s": self.steps[-1].t_end if self.steps else 0.0,
            "energy_total_j": self._e_total,
            "energy_scan_j": e_dwell,
            "energy_retune_j": e_retune,
            "energy_fixed_j": e_fixed,
            "energy_sleep_j": e_sleep,
            "n_channel_dwells": sum(s.n_channels for s in scans if s.dwell_s > 0.0),
        }

    def trace_rows(self) -> list[dict]:
        return [s.trace_row() for s in self.steps]


# ----------------------------------------------------------- interval algebra
def _merge_intervals(s: np.ndarray, e: np.ndarray):
    """Union of half-open intervals, returned sorted and disjoint.

    Union rather than sum: two scans overlapping the same channel-second are one
    covered second, not two.  Summing would let a policy inflate coverage by
    re-scanning.
    """
    if s.size == 0:
        return s, e
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    cmax = np.maximum.accumulate(e)
    keep = np.empty(s.size, dtype=bool)
    keep[0] = True
    keep[1:] = s[1:] > cmax[:-1]
    idx = np.flatnonzero(keep)
    ends = cmax[np.append(idx[1:], s.size) - 1]
    return s[idx], ends


def _covered_before(starts: np.ndarray, ends: np.ndarray, cum: np.ndarray,
                    x: np.ndarray) -> np.ndarray:
    """Total covered measure in `[0, x)` for a merged interval set."""
    x = np.asarray(x, dtype=np.float64)
    if starts.size == 0:
        return np.zeros(x.shape, dtype=np.float64)
    i = np.searchsorted(starts, x, side="right") - 1
    out = np.zeros(x.shape, dtype=np.float64)
    ok = i >= 0
    if np.any(ok):
        ii = i[ok]
        out[ok] = cum[ii + 1] - np.maximum(0.0, ends[ii] - x[ok])
    return out


class _Coverage:
    """Per-channel merged dwell intervals plus a prefix-sum lookup."""

    __slots__ = ("starts", "ends", "cum", "n_channels")

    def __init__(self, log: EpisodeLog, n_channels: int, horizon_s: float):
        self.n_channels = n_channels
        scans = [s for s in log.steps if s.kind == "scan" and s.dwell_s > 0.0
                 and s.n_channels > 0]
        t0 = np.array([s.t_dwell_start for s in scans], dtype=np.float64)
        t1 = np.array([s.t_end for s in scans], dtype=np.float64)
        klo = np.array([s.k_lo for s in scans], dtype=np.int64)
        khi = np.array([s.k_hi for s in scans], dtype=np.int64)
        np.clip(t0, 0.0, horizon_s, out=t0)
        np.clip(t1, 0.0, horizon_s, out=t1)

        self.starts: list = [None] * n_channels
        self.ends: list = [None] * n_channels
        self.cum: list = [None] * n_channels
        for c in range(n_channels):
            if t0.size:
                m = (klo <= c) & (c < khi) & (t1 > t0)
                s, e = _merge_intervals(t0[m], t1[m])
            else:
                s = e = np.empty(0, dtype=np.float64)
            self.starts[c] = s
            self.ends[c] = e
            self.cum[c] = np.concatenate(([0.0], np.cumsum(e - s)))

    def covered(self, c: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Covered measure of `[a, b)` on channel `c`."""
        s, e, cu = self.starts[c], self.ends[c], self.cum[c]
        return _covered_before(s, e, cu, b) - _covered_before(s, e, cu, a)

    def gaps(self, c: int, horizon_s: float) -> np.ndarray:
        """Gaps between consecutive coverings, including 0->first and last->T."""
        s, e = self.starts[c], self.ends[c]
        if s.size == 0:
            return np.array([horizon_s], dtype=np.float64)
        return np.concatenate((
            [s[0]], s[1:] - e[:-1], [max(0.0, horizon_s - e[-1])],
        ))


# ------------------------------------------------------------ truth-positives
def _flatten_detections(log: EpisodeLog):
    """(channel, window_start, window_end) for every reported detection."""
    ch, a, b = [], [], []
    for s in log.steps:
        if s.kind != "scan" or s.det_channels.size == 0:
            continue
        w0, w1 = s.t_dwell_start, s.t_end
        for c in s.det_channels:
            ch.append(int(c))
            a.append(w0)
            b.append(w1)
    return (np.asarray(ch, dtype=np.int64),
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64))


def _true_positives(bursts: np.ndarray, d_ch, d_t0, d_t1):
    """Match detections to bursts.

    A detection over `[t0,t1)` on channel `c` is a true positive for emitter `e`
    iff some burst of `e` covers `c` and overlaps `[t0,t1)`  (DESIGN.md s.6).
    Returns `(is_fa, pairs)` where `pairs` is a list of
    `(det_index, emitter_id, activation_id, priority)`.
    """
    n_det = d_ch.size
    is_fa = np.ones(n_det, dtype=bool)
    pairs: list[tuple[int, int, int, int]] = []
    if n_det == 0 or bursts.size == 0:
        return is_fa, pairs

    b_lo = bursts["ch_lo"].astype(np.int64)
    b_hi = bursts["ch_hi"].astype(np.int64)
    b_on = bursts["t_on"].astype(np.float64)
    b_off = bursts["t_off"].astype(np.float64)
    b_em = bursts["emitter_id"].astype(np.int64)
    b_act = bursts["activation_id"].astype(np.int64)
    b_pri = bursts["priority"].astype(np.int64)

    # Group detections by channel: bursts are indexed by channel once, then one
    # vectorised overlap test per channel rather than per detection.
    order = np.argsort(d_ch, kind="stable")
    ch_sorted = d_ch[order]
    bounds = np.searchsorted(ch_sorted, np.unique(ch_sorted), side="left")
    uniq = np.unique(ch_sorted)
    bounds = np.append(bounds, n_det)

    for u, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        c = int(uniq[u])
        sel = np.flatnonzero((b_lo <= c) & (c < b_hi))
        if sel.size == 0:
            continue
        di = order[lo:hi]
        # (n_det_c, n_burst_c) overlap matrix
        hit = (b_on[sel][None, :] < d_t1[di][:, None]) & \
              (b_off[sel][None, :] > d_t0[di][:, None])
        any_hit = hit.any(axis=1)
        is_fa[di[any_hit]] = False
        for r, j in zip(*np.nonzero(hit)):
            k = sel[j]
            pairs.append((int(di[r]), int(b_em[k]), int(b_act[k]), int(b_pri[k])))
    return is_fa, pairs


# -------------------------------------------------------------- the metrics
def _pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def compute_metrics(bursts: np.ndarray, log: EpisodeLog, mission=None,
                    horizon_s: float | None = None,
                    n_channels: int | None = None,
                    poi_times: tuple[float, ...] = POI_TIMES) -> dict:
    """Replay `log` against `bursts` and return every metric in DESIGN.md s.6.

    `bursts` is a plain structured array of `sim.emitters.BURST_DTYPE`; no
    `World` is needed, which is what makes the hand-built ground-truth test
    possible.
    """
    horizon_s = float(horizon_s if horizon_s is not None else log.horizon_s)
    n_channels = int(n_channels if n_channels is not None else log.n_channels)
    tot = log.totals()
    out: dict = dict(tot)
    energy_total = tot["energy_total_j"]

    d_ch, d_t0, d_t1 = _flatten_detections(log)
    is_fa, pairs = _true_positives(bursts, d_ch, d_t0, d_t1)

    # ---------------------------------------------------------------- POI@T
    emitters = np.unique(bursts["emitter_id"]) if bursts.size else np.empty(0, np.int64)
    prio_of: dict[int, int] = {}
    first_on: dict[int, float] = {}
    if bursts.size:
        for em, pr, ton in zip(bursts["emitter_id"], bursts["priority"], bursts["t_on"]):
            em = int(em)
            prio_of[em] = int(pr)
            first_on[em] = min(first_on.get(em, math.inf), float(ton))

    # A detection is credited at the END of the dwell -- the moment the report
    # exists.  See the module docstring.
    first_det: dict[int, float] = {}
    unique_det: set[tuple[int, int]] = set()
    unique_det_p1: set[tuple[int, int]] = set()
    for di, em, act, pri in pairs:
        t = float(d_t1[di])
        if t < first_det.get(em, math.inf):
            first_det[em] = t
        unique_det.add((em, act))
        if pri == 1:
            unique_det_p1.add((em, act))

    n_em = int(emitters.size)
    p1_emitters = [int(e) for e in emitters if prio_of.get(int(e)) == 1]
    for T in poi_times:
        key = f"poi_{int(T)}"
        if n_em == 0:
            out[key] = float("nan")
            out[f"poi_p1_{int(T)}"] = float("nan")
            continue
        n_hit = sum(1 for e in emitters if first_det.get(int(e), math.inf) <= T)
        out[key] = n_hit / n_em
        if p1_emitters:
            n1 = sum(1 for e in p1_emitters if first_det.get(e, math.inf) <= T)
            out[f"poi_p1_{int(T)}"] = n1 / len(p1_emitters)
        else:
            out[f"poi_p1_{int(T)}"] = float("nan")

    # ------------------------------------------------------------ TTFI prio-1
    # Censored at the horizon: an emitter never intercepted contributes
    # `horizon_s`, NOT a dropped row.  Reported alongside n_intercepted/n_total
    # so a policy cannot win TTFI by ignoring the hard emitters.
    ttfi = []
    n_icept = 0
    for e in p1_emitters:
        t0 = first_on.get(e, 0.0)
        td = first_det.get(e, math.inf)
        if math.isfinite(td):
            ttfi.append(max(0.0, td - t0))
            n_icept += 1
        else:
            ttfi.append(horizon_s)
    ttfi_a = np.asarray(ttfi, dtype=np.float64)
    out["ttfi_p1_median_s"] = _pct(ttfi_a, 50.0)
    out["ttfi_p1_p90_s"] = _pct(ttfi_a, 90.0)
    out["ttfi_p1_n_intercepted"] = n_icept
    out["ttfi_p1_n_total"] = len(p1_emitters)
    out["ttfi_p1_frac"] = (n_icept / len(p1_emitters)) if p1_emitters else float("nan")

    # ------------------------------------------------- emitter-time coverage
    # GEOMETRIC: was the receiver pointed there, regardless of whether the
    # detector fired.  Separates "looked in the right place" from "dwelt long
    # enough"; retune time is excluded by construction (see StepRecord).
    cov = _Coverage(log, n_channels, horizon_s)
    num = den = 0.0
    num1 = den1 = 0.0
    if bursts.size:
        b_on = np.clip(bursts["t_on"].astype(np.float64), 0.0, horizon_s)
        b_off = np.clip(bursts["t_off"].astype(np.float64), 0.0, horizon_s)
        b_lo = bursts["ch_lo"].astype(np.int64)
        b_hi = bursts["ch_hi"].astype(np.int64)
        b_pri = bursts["priority"].astype(np.int64)
        for c in range(n_channels):
            m = (b_lo <= c) & (c < b_hi) & (b_off > b_on)
            if not np.any(m):
                continue
            dur = b_off[m] - b_on[m]
            got = cov.covered(c, b_on[m], b_off[m])
            num += float(got.sum())
            den += float(dur.sum())
            m1 = b_pri[m] == 1
            if np.any(m1):
                num1 += float(got[m1].sum())
                den1 += float(dur[m1].sum())
    out["coverage_frac"] = (num / den) if den > 0 else float("nan")
    out["coverage_p1_frac"] = (num1 / den1) if den1 > 0 else float("nan")

    # --------------------------------------------- ENERGY PER DETECTION (headline)
    # unique == distinct (emitter_id, activation_id): a hopper's 20 hops inside
    # one ON period count ONCE, and re-detecting one burst 50x does not inflate
    # the denominator.
    n_uni = len(unique_det)
    n_uni1 = len(unique_det_p1)
    out["n_unique_detections"] = n_uni
    out["n_unique_p1_detections"] = n_uni1
    out["energy_per_detection_j"] = (energy_total / n_uni) if n_uni else float("inf")
    out["energy_per_prio1_detection_j"] = (
        (energy_total / n_uni1) if n_uni1 else float("inf")
    )

    # ------------------------------------------------- max staleness prio-1
    # Over prio-1 MISSION channels (agent-visible tasking), not over emitters:
    # the scheduler's revisit deadline is what this is meant to audit.
    if mission is not None:
        p1_ch = np.flatnonzero(np.asarray(mission.priority) == 1)
    else:
        p1_ch = np.empty(0, dtype=np.int64)
    if p1_ch.size:
        worst = 0.0
        acc = []
        for c in p1_ch:
            g = cov.gaps(int(c), horizon_s)
            worst = max(worst, float(g.max()))
            acc.append(float(g.max()))
        out["max_staleness_p1_s"] = worst
        out["mean_staleness_p1_s"] = float(np.mean(acc))
    else:
        out["max_staleness_p1_s"] = float("nan")
        out["mean_staleness_p1_s"] = float("nan")

    # ------------------------------------------------------- false alarm rate
    # Per channel-dwell it is directly comparable to P_fa, so it doubles as a
    # detector calibration check.
    n_det = int(d_ch.size)
    n_fa = int(is_fa.sum())
    n_cd = tot["n_channel_dwells"]
    elapsed = max(tot["t_end_s"], 1e-9)
    out["n_detections"] = n_det
    out["n_true_positive_dets"] = n_det - n_fa
    out["n_false_alarms"] = n_fa
    out["false_alarm_rate_per_dwell"] = (n_fa / n_cd) if n_cd else float("nan")
    out["false_alarm_rate_per_s"] = n_fa / elapsed
    out["n_emitters"] = n_em
    out["n_activations_total"] = (
        int(np.unique(np.stack([bursts["emitter_id"], bursts["activation_id"]]),
                      axis=1).shape[1]) if bursts.size else 0
    )
    return out


# Keys `compute_metrics` is guaranteed to produce.  `eval/runner.py` asserts the
# results header is a superset of this, so a renamed metric fails loudly.
METRIC_KEYS: tuple[str, ...] = (
    "n_steps", "n_scans", "n_sleeps", "dwell_time_s", "retune_time_s",
    "sleep_time_s", "t_end_s", "energy_total_j", "energy_scan_j",
    "energy_retune_j", "energy_fixed_j", "energy_sleep_j", "n_channel_dwells",
    "poi_10", "poi_30", "poi_60", "poi_p1_10", "poi_p1_30", "poi_p1_60",
    "ttfi_p1_median_s", "ttfi_p1_p90_s", "ttfi_p1_n_intercepted",
    "ttfi_p1_n_total", "ttfi_p1_frac", "coverage_frac", "coverage_p1_frac",
    "n_unique_detections", "n_unique_p1_detections", "energy_per_detection_j",
    "energy_per_prio1_detection_j", "max_staleness_p1_s", "mean_staleness_p1_s",
    "n_detections", "n_true_positive_dets", "n_false_alarms",
    "false_alarm_rate_per_dwell", "false_alarm_rate_per_s", "n_emitters",
    "n_activations_total",
)

__all__ = [
    "POI_TIMES", "TRACE_COLUMNS", "METRIC_KEYS",
    "StepRecord", "EpisodeLog", "compute_metrics",
]
