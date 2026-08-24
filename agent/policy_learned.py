"""Rung 2 -- the learned activity model.  See DESIGN.md section 8.

The model refines the analytic rung-1 belief; it never replaces it.  Three
independent guarantees keep it from regressing rung 1 (all implemented below):

  1. `beta` defaults to 0.0 -- the learned path is OFF unless explicitly enabled.
  2. Gated out below `min_visits_for_model = 3`, the cold-start regime where
     rung 1 is provably right (with no visits, the analytic belief IS the prior).
  3. An automatic Brier gate at load: if the model does not beat rung 1 on the
     held-out log rows, `beta` is forced to 0.0 and the demo path becomes
     bit-identical to rung 1.

FIREWALL: this module imports ONLY `sim.contract` and `sim.config` from `sim`.
It never touches the simulator, and it deliberately reimplements `pd_curve`
rather than importing `sim.receiver` (DESIGN.md section 2 -- duplication is
cheaper than a firewall breach).  A cross-check test asserts the two agree.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agent.base import FEATURE_NAMES, N_FEATURES, TRAIN_EXTRA_NAMES

# `sim.config` is on the permitted side of the firewall: it holds the candidate
# action sets and the loader, no ground truth.
from sim.config import BW_CANDIDATES_MHZ, DWELL_CANDIDATES_MS

LOG = logging.getLogger("agent.policy_learned")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "activity_hgb.joblib"

# Training set, frozen by DESIGN.md section 8.  `agile` is the HELD-OUT scenario
# and appears nowhere in this list: training on it would destroy the
# generalisation claim, which is the entire point of rung 2.
TRAIN_SCENARIOS: tuple[str, ...] = ("sparse", "dense")
TRAIN_SEEDS: tuple[int, ...] = tuple(range(100, 120))
HELD_OUT_SCENARIO: str = "agile"

# Probabilities are clipped away from {0, 1} before they are handed to a Bayes
# update, so a single confident-and-wrong prediction cannot pin the posterior.
P_CLIP_LO: float = 1e-4
P_CLIP_HI: float = 1.0 - 1e-4

# The model must beat rung 1 by more than this to be allowed to run.  A tie goes
# to rung 1: the simpler path is the one we can prove correct.
BRIER_GATE_MARGIN: float = 1e-4

# HistGradientBoosting hyper-parameters, verbatim from DESIGN.md section 8.
# lightgbm is NOT installed; sklearn only.
HGB_PARAMS: dict = dict(
    max_iter=300,
    learning_rate=0.08,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
)

# Column indices into a `FEATURE_NAMES`-ordered matrix.  Resolved by name so a
# re-freeze of the contract cannot silently shift them.
IDX_P_RUNG1: int = FEATURE_NAMES.index("p_rung1")
IDX_N_VISITS: int = FEATURE_NAMES.index("n_visits")


# ---------------------------------------------------------------------------
# Detector model -- reimplemented agent-side (see module docstring).
# ---------------------------------------------------------------------------
def _ndtr(x):
    """Standard normal CDF.  scipy if available, erf otherwise."""
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except ImportError:  # pragma: no cover - scipy is a stated dependency
        from math import erf, sqrt

        vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))
        return vec(np.asarray(x, dtype=np.float64))


def _ndtri(p):
    """Standard normal inverse CDF."""
    try:
        from scipy.special import ndtri

        return ndtri(p)
    except ImportError:  # pragma: no cover
        from statistics import NormalDist

        vec = np.vectorize(lambda v: NormalDist().inv_cdf(v))
        return vec(np.asarray(p, dtype=np.float64))


def pd_curve(snr_eff_db, dwell_s, channel_bw_hz: float = 1.0e6, pfa: float = 1e-3):
    """Urkowitz energy detector, Gaussian approximation.  DESIGN.md section 1.

        N   = dwell_s * channel_bw_hz          complex samples
        s   = 10**(snr_eff_db/10)              linear SNR
        P_d = Q((Q^-1(pfa) - sqrt(N)*s) / (1+s))

    `s = 0` yields P_d = pfa automatically, so there is no separate false-alarm
    branch anywhere in the project.  `channel_bw_hz` is the PER-CHANNEL
    bandwidth (1 MHz), not the scan bandwidth: scanning wide costs sensitivity
    through the bandwidth penalty, not through the sample count.
    """
    s = np.power(10.0, np.asarray(snr_eff_db, dtype=np.float64) / 10.0)
    n_samples = np.asarray(dwell_s, dtype=np.float64) * float(channel_bw_hz)
    thresh = -_ndtri(np.asarray(pfa, dtype=np.float64))
    return _ndtr(-((thresh - np.sqrt(n_samples) * s) / (1.0 + s)))


def bw_penalty_db(bw_hz, db_per_octave: float = 1.0):
    """Sensitivity lost by scanning wide.  1 dB/octave -> 20 MHz is 4.32 dB down."""
    return db_per_octave * np.log2(np.asarray(bw_hz, dtype=np.float64) / 1.0e6)


@dataclass(slots=True)
class PdBar:
    """Marginal P_d on a MISS: the detector curve integrated over the agent's
    ASSUMED SNR distribution (its spec sheet, deliberately not truth).

    On a miss there is no reported SNR to condition on, so the likelihood must
    marginalise: `pd_bar[bw, dwell] = E_snr[ P_d(snr_eff, dwell) ]` with
    `snr ~ N(mu, sigma)`.  A 1 ms miss barely moves the belief; a 100 ms miss
    crushes it.  Gauss-Hermite quadrature, 32 nodes -- the integrand is smooth
    and monotone in snr, so this is exact to ~1e-12 and costs nothing.

    This is the same quantity agent B's `Belief` exposes.  `ActivityModel`
    prefers the belief's own table when one is handed to it (so the two can
    never drift); this class is the standalone fallback used for training,
    reporting and tests.
    """

    pfa: float = 1.0e-3
    channel_bw_hz: float = 1.0e6
    mu_db: float = -15.0
    sigma_db: float = 5.0
    db_per_octave: float = 1.0
    n_nodes: int = 32
    _nodes: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x, w = np.polynomial.hermite.hermgauss(int(self.n_nodes))
        self._nodes = x
        self._weights = w / np.sqrt(np.pi)  # normalise so weights sum to 1

    def __call__(self, dwell_s, bw_hz, gain_db=0.0):
        """Broadcasting marginal P_d.  Returns an array shaped like the inputs."""
        dwell = np.asarray(dwell_s, dtype=np.float64)
        bw = np.asarray(bw_hz, dtype=np.float64)
        gain = np.asarray(gain_db, dtype=np.float64)
        dwell, bw, gain = np.broadcast_arrays(dwell, bw, gain)
        # snr_eff_db = snr - 1.0*log2(bw/1e6) + gain      (DESIGN.md section 4)
        snr_samples = self.mu_db + np.sqrt(2.0) * self.sigma_db * self._nodes
        eff = (
            snr_samples.reshape((-1,) + (1,) * dwell.ndim)
            - bw_penalty_db(bw, self.db_per_octave)[None, ...]
            + gain[None, ...]
        )
        pd = pd_curve(eff, dwell[None, ...], self.channel_bw_hz, self.pfa)
        out = np.tensordot(self._weights, pd, axes=(0, 0))
        return out if out.ndim else float(out)

    def table(self) -> dict:
        """`{(bw_mhz, dwell_ms): pd_bar}` over the frozen candidate action sets."""
        return {
            (int(bw), float(dw)): float(self(dw * 1e-3, bw * 1e6))
            for bw in BW_CANDIDATES_MHZ
            for dw in DWELL_CANDIDATES_MS
        }

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PdBar":
        """Build from a loaded scenario config (`sim.config.load_config`)."""
        rx = cfg.get("receiver", {})
        ag = cfg.get("agent", {})
        grid = cfg.get("grid", {})
        return cls(
            pfa=float(rx.get("pfa", 1e-3)),
            channel_bw_hz=float(grid.get("channel_bw_hz", 1.0e6)),
            mu_db=float(ag.get("assumed_snr_mu_db", -15.0)),
            sigma_db=float(ag.get("assumed_snr_sigma_db", 5.0)),
            db_per_octave=float(rx.get("bw_penalty_db_per_octave", 1.0)),
        )


# ---------------------------------------------------------------------------
# The deconvolution -- the three most load-bearing lines in this file.
# ---------------------------------------------------------------------------
def p_det_from_p_active(p_active, pd_bar_next, pfa: float = 1e-3):
    """Forward map: P(detect next) implied by a belief P(active).

        p_det = p_act * P_d + (1 - p_act) * P_fa

    This is what rung 1 predicts for the label, and it is the exact inverse of
    `p_active_from_p_det`.  Having both directions in one place is what makes
    the "we are not just relearning our own detector" claim checkable.
    """
    p = np.asarray(p_active, dtype=np.float64)
    pdb = np.asarray(pd_bar_next, dtype=np.float64)
    return p * pdb + (1.0 - p) * float(pfa)


def p_active_from_p_det(p_det_hat, pd_bar_next, pfa: float = 1e-3):
    """Invert the detector out of the model's output.  DESIGN.md section 8.

    The label is `P(detect next)`, NOT `P(active)`.  Feeding the raw model
    output into a Bayes update would apply the detector twice -- once inside the
    learned prediction and once in the likelihood -- and systematically
    overstate the evidence.  So we deconvolve:

        p_active_hat = (p_det_hat - P_fa) / (pd_bar[bw_next, tau_next] - P_fa)

    clipped into (0,1) because the output feeds Bayes and must be a probability.
    `pd_bar` is the MARGINAL P_d for the dwell/bandwidth the next observation
    will use -- which is precisely why `tau_next_log` and `bw_next_log` are
    given to the model as inputs at train time (`TRAIN_EXTRA_NAMES`) instead of
    letting it average over an unknown dwell.

    THIS IS THE ANSWER to "how do you know your ML isn't just relearning your
    detector?": the detector is divided back out analytically, so what the model
    contributes is only the residual structure -- burst timing, neighbour
    activity, mission context.
    """
    p = np.asarray(p_det_hat, dtype=np.float64)
    # An observation whose marginal P_d barely exceeds P_fa carries almost no
    # information, and the inversion is ill-conditioned there.  Floor the
    # denominator rather than dividing by ~0; callers gate such rows out anyway
    # via `min_visits_for_model`.
    denom = np.maximum(np.asarray(pd_bar_next, dtype=np.float64) - float(pfa), 1e-12)
    return np.clip((p - float(pfa)) / denom, P_CLIP_LO, P_CLIP_HI)


def brier(p, y) -> float:
    """Mean squared error of a probabilistic forecast.  Lower is better."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Training-log ingestion
# ---------------------------------------------------------------------------
# Agent C's `eval/runner.py --collect` writes one row per CHANNEL-OBSERVATION:
# the FEATURE_NAMES columns as they stood immediately BEFORE the observation,
# plus what that observation actually did.  Column names are resolved through
# these aliases so a small schema difference does not break the build.
_ALIASES: dict[str, tuple[str, ...]] = {
    "run_id": ("run_id", "run", "episode", "episode_id"),
    "channel": ("channel", "ch", "chan"),
    "t": ("t", "time", "t_end", "t_s"),
    "detected": ("detected", "det", "is_det", "hit", "y_now"),
    "dwell_s": ("dwell_s", "dwell", "tau_s", "tau"),
    "bw_hz": ("bw_hz", "bw", "bandwidth_hz"),
    "scenario": ("scenario", "config", "cfg", "name"),
    "seed": ("seed",),
}


class LogsUnavailable(RuntimeError):
    """No collected rung-1 logs on disk.  Not a bug -- run the collector first."""


def _resolve(columns, key: str) -> str | None:
    lower = {str(c).lower(): str(c) for c in columns}
    for cand in _ALIASES[key]:
        if cand in lower:
            return lower[cand]
    return None


def load_log_frame(
    paths: "list[Path] | None" = None,
    log_dir: "Path | None" = None,
    scenarios: "tuple[str, ...]" = TRAIN_SCENARIOS,
    seeds: "tuple[int, ...]" = TRAIN_SEEDS,
):
    """Read collected rung-1 logs into one pandas DataFrame.

    Accepts CSV or parquet.  Filters to `scenarios` x `seeds` when those columns
    exist -- this is where `agile` is excluded and where the seed disjointness
    (collection 100-119 vs evaluation 0-9) is enforced.

    Raises `LogsUnavailable` with an actionable message when nothing is on disk,
    so `--train` degrades into an instruction rather than a traceback.
    """
    import pandas as pd

    if paths is None:
        log_dir = Path(log_dir) if log_dir is not None else ROOT / "data" / "logs"
        paths = sorted(
            [p for p in log_dir.glob("*.csv")]
            + [p for p in log_dir.glob("*.parquet")]
        )
        if not paths:
            # Second look: agent C may write the collection under results/.
            alt = ROOT / "results" / "logs"
            paths = sorted(list(alt.glob("*.csv")) + list(alt.glob("*.parquet")))
    paths = [Path(p) for p in paths]
    if not paths:
        raise LogsUnavailable(
            "no rung-1 training logs found in data/logs/ or results/logs/.\n"
            "Collect them first:\n"
            "    python -m eval.runner --collect --scenarios sparse dense "
            "--seeds 100-119\n"
            "(`agile` is HELD OUT and must never be collected for training.)"
        )

    frames = []
    for p in paths:
        if p.suffix == ".parquet":
            frames.append(pd.read_parquet(p))
        else:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)

    sc_col = _resolve(df.columns, "scenario")
    if sc_col is not None:
        keep = df[sc_col].astype(str).str.lower().isin([s.lower() for s in scenarios])
        if (~keep).any():
            dropped = sorted(set(df.loc[~keep, sc_col].astype(str)))
            LOG.info("dropping held-out/unused scenarios from training: %s", dropped)
        df = df.loc[keep].copy()
    seed_col = _resolve(df.columns, "seed")
    if seed_col is not None and seeds:
        df = df.loc[df[seed_col].astype(int).isin(list(seeds))].copy()
    if df.empty:
        raise LogsUnavailable(
            f"logs were found but none matched scenarios={list(scenarios)} "
            f"seeds={seeds[0]}-{seeds[-1]}."
        )
    return df


def build_training_matrix(
    df,
    pd_bar: "PdBar | None" = None,
    pfa: float = 1e-3,
    label_mode: str = "pre_labelled",
):
    """Turn per-observation log rows into `(X, y, meta)` for supervised training.

    The label is defined by DESIGN.md section 8: `y = 1` iff the NEXT
    observation of that channel reports a detection.

    `label_mode` says how far the log already got toward that:

    ``"pre_labelled"`` (DEFAULT, and what `agent/policy_index.py` actually
        emits).  `log_rows()` writes a row at DECISION time -- features as they
        stood before the action, `dwell_s`/`bw_hz` of the action about to be
        taken -- and then fills `detected` in retroactively when that action's
        observation lands.  Each row is therefore ALREADY the
        `(features, tau_next, bw_next, y_next)` tuple rung 2 wants, and shifting
        it again would label a row with the outcome two observations away.

    ``"shift"``  A raw log where `detected` describes the observation the row's
        own features came after.  Rows are grouped by `(run_id, channel)`,
        ordered in time, and labelled from the *following* row; the last row of
        each group has no successor and is dropped.

    Getting this wrong is silent -- the matrix still trains, it just learns the
    wrong conditional -- so the mode is explicit rather than sniffed.

    `X` has exactly `len(FEATURE_NAMES) + 2` columns: FEATURE_NAMES in order,
    then `tau_next_log`, `bw_next_log` -- the dwell and bandwidth the NEXT
    observation used.  Giving the model those means it learns the dwell
    dependence of the label instead of averaging over it.

    Returns `(X, y, meta)` where `meta` carries `pd_bar_next` and `p_rung1` per
    row, which is everything the Brier gate needs.
    """
    import pandas as pd

    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(
            f"training log is missing {len(missing)} contract features: {missing}. "
            "The collector must emit every name in agent.base.FEATURE_NAMES."
        )

    run_col = _resolve(df.columns, "run_id")
    ch_col = _resolve(df.columns, "channel")
    t_col = _resolve(df.columns, "t")
    det_col = _resolve(df.columns, "detected")
    dwell_col = _resolve(df.columns, "dwell_s")
    bw_col = _resolve(df.columns, "bw_hz")
    for name, col in (("channel", ch_col), ("detected", det_col),
                      ("dwell_s", dwell_col), ("bw_hz", bw_col)):
        if col is None:
            raise ValueError(
                f"training log has no column for {name!r}; tried {_ALIASES[name]}"
            )

    d = df.copy()
    if run_col is None:
        # A single-episode log is still trainable; treat it as one run.
        run_col = "_run_id"
        d[run_col] = 0
    if t_col is None:
        t_col = "_t"
        d[t_col] = np.arange(len(d), dtype=np.float64)

    if label_mode not in ("pre_labelled", "shift"):
        raise ValueError(f"label_mode must be 'pre_labelled' or 'shift', got {label_mode!r}")

    d = d.sort_values([run_col, ch_col, t_col], kind="mergesort")

    if label_mode == "shift":
        grp = d.groupby([run_col, ch_col], sort=False)
        # The label and the next observation's action parameters, all shifted by
        # -1 within the channel's own observation sequence.
        y_next = grp[det_col].shift(-1)
        tau_next = grp[dwell_col].shift(-1)
        bw_next = grp[bw_col].shift(-1)
        valid = y_next.notna() & tau_next.notna() & bw_next.notna()
        d = d.loc[valid].copy()
        y = (pd.to_numeric(y_next.loc[valid]).to_numpy() > 0).astype(np.int8)
        tau_next = pd.to_numeric(tau_next.loc[valid]).to_numpy(dtype=np.float64)
        bw_next = pd.to_numeric(bw_next.loc[valid]).to_numpy(dtype=np.float64)
    else:
        # Already aligned by the collector; a sleep row (bw = 0) has no "next
        # observation" attached to it and carries no label, so drop those.
        tau_all = pd.to_numeric(d[dwell_col], errors="coerce")
        bw_all = pd.to_numeric(d[bw_col], errors="coerce")
        y_all = pd.to_numeric(d[det_col], errors="coerce")
        valid = y_all.notna() & tau_all.notna() & bw_all.notna() & (bw_all > 0.0) & (tau_all > 0.0)
        d = d.loc[valid].copy()
        y = (y_all.loc[valid].to_numpy() > 0).astype(np.int8)
        tau_next = tau_all.loc[valid].to_numpy(dtype=np.float64)
        bw_next = bw_all.loc[valid].to_numpy(dtype=np.float64)

    feats = d[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    extra = np.column_stack([np.log1p(tau_next), np.log1p(bw_next / 1.0e6)])
    X = np.column_stack([feats, extra])

    if pd_bar is None:
        pd_bar = PdBar(pfa=pfa)
    meta = {
        "pd_bar_next": np.asarray(pd_bar(tau_next, bw_next), dtype=np.float64),
        "p_rung1": feats[:, IDX_P_RUNG1].copy(),
        "n_visits": feats[:, IDX_N_VISITS].copy(),
        "tau_next": tau_next,
        "bw_next": bw_next,
        "run_id": d[run_col].to_numpy(),
        "channel": d[ch_col].to_numpy(),
        "pfa": float(pfa),
    }
    return X, y, meta


def assert_no_channel_leakage(feature_names) -> None:
    """The raw channel index must never become a feature.  DESIGN.md section 8.

    Absolute channel position would let the model memorise emitter locations
    from `sparse` and score well there while generalising to nothing -- exactly
    the failure the held-out `agile` scenario exists to catch.  Only *relative*
    spectral context (`nbr_recent_hits`, `band_activity`) is permitted; the only
    position-derived input allowed is the mission weight, which is intel, not
    truth.
    """
    banned = {"channel", "ch", "chan", "channel_idx", "channel_index",
              "f_center_hz", "f_hz", "k", "k_lo"}
    hits = sorted({str(n) for n in feature_names} & banned)
    if hits:
        raise ValueError(
            f"channel-index leakage: {hits} must not be model features "
            "(DESIGN.md section 8 -- destroys generalisation to `agile`)."
        )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
@dataclass
class ActivityModel:
    """A calibrated `P(detect next)` predictor, wired for safe Bayes use.

    Agent B's `Belief.attach_model(model, beta)` takes one of these.  The only
    methods the belief needs are `refine()` and the two public attributes
    `beta` / `min_visits_for_model`.
    """

    estimator: object = None                 # fitted CalibratedClassifierCV
    feature_names: tuple[str, ...] = FEATURE_NAMES + TRAIN_EXTRA_NAMES
    beta: float = 0.0                        # GUARANTEE 1: off unless enabled
    min_visits_for_model: int = 3            # GUARANTEE 2: no cold-start meddling
    pfa: float = 1.0e-3
    manifest: dict = field(default_factory=dict)
    pd_bar: "PdBar | None" = None
    gate_ok: bool = True                     # GUARANTEE 3: set by the Brier gate
    gate_reason: str = "not evaluated"

    # Reference action used when the caller has no candidate in hand -- i.e. the
    # `Belief.p_effective` path, which asks "how active is this channel?" with no
    # particular observation in mind.  1 MHz / 10 ms is the centre of the frozen
    # candidate sets and the row DESIGN.md section 1 tabulates.
    infer_dwell_s: float = 0.010
    infer_bw_hz: float = 1.0e6

    # Set by `attach_to`: the Belief's own `pd_bar_for(bw_hz, dwell_s)`.  The
    # deconvolution divides by pd_bar and the belief multiplies by it, so the two
    # MUST be the same numbers; this module's standalone `PdBar` uses a different
    # quadrature and differs by up to ~3e-3, which is enough to bias the
    # inversion.  Prefer the belief's table whenever one is available.
    _pd_bar_fn: object = None

    # -- prediction ---------------------------------------------------------
    def predict_p_det(self, X) -> np.ndarray:
        """Calibrated P(the next observation of this channel reports a detection)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} columns "
                f"({len(FEATURE_NAMES)} contract features + "
                f"{len(TRAIN_EXTRA_NAMES)} train extras), got {X.shape[1]}"
            )
        if self.estimator is None:
            raise RuntimeError("ActivityModel has no fitted estimator")
        return np.clip(
            np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64),
            P_CLIP_LO, P_CLIP_HI,
        )

    def predict_p_active(self, X, pd_bar_next) -> np.ndarray:
        """Model output with the detector divided back out.  See DESIGN.md s8."""
        return p_active_from_p_det(self.predict_p_det(X), pd_bar_next, self.pfa)

    def _pd_bar_for(self, dwell_s, bw_hz) -> np.ndarray:
        """Marginal P_d for one (dwell, bw).  Belief's table wins if attached."""
        if self._pd_bar_fn is not None:
            # NOTE the argument order flip: Belief.pd_bar_for is (bw_hz, dwell_s).
            return np.asarray(
                float(self._pd_bar_fn(float(bw_hz), float(dwell_s))), dtype=np.float64
            )
        pdb = self.pd_bar if self.pd_bar is not None else PdBar(pfa=self.pfa)
        return np.asarray(pdb(dwell_s, bw_hz), dtype=np.float64)

    def refine(
        self,
        feature_matrix,
        dwell_s: float,
        bw_hz: float,
        pd_bar_next=None,
        beta: "float | None" = None,
    ) -> np.ndarray:
        """The single call agent B's `Belief` makes.  Returns a blended P(active).

        `feature_matrix` is `(n_channels, len(FEATURE_NAMES))`.  `dwell_s` and
        `bw_hz` are the CANDIDATE action's parameters, held fixed across all
        channels -- at inference there is no "next observation" yet, so we ask
        the model "if I were to look with this dwell, how likely is a detection?"
        That is what makes the train-time extras honest rather than leakage.

        `pd_bar_next` may be supplied by the belief's own table so the two
        components can never drift; otherwise the local `PdBar` is used.
        """
        X = np.asarray(feature_matrix, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"feature_matrix must have {N_FEATURES} columns, got {X.shape[1]}"
            )
        p_rung1 = X[:, IDX_P_RUNG1]
        n_visits = X[:, IDX_N_VISITS]

        b = float(self.beta if beta is None else beta)
        # GUARANTEE 1 + 3: with beta == 0 nothing below can change the answer, so
        # return the analytic belief UNTOUCHED -- bit-identical to rung 1, not
        # merely numerically close.
        if b == 0.0 or self.estimator is None:
            return p_rung1.copy()

        if pd_bar_next is None:
            pd_bar_next = self._pd_bar_for(dwell_s, bw_hz)
        pd_bar_next = np.broadcast_to(
            np.asarray(pd_bar_next, dtype=np.float64), p_rung1.shape
        )

        extra = np.column_stack([
            np.full(X.shape[0], np.log1p(float(dwell_s))),
            np.full(X.shape[0], np.log1p(float(bw_hz) / 1.0e6)),
        ])
        p_active_hat = self.predict_p_active(np.column_stack([X, extra]), pd_bar_next)

        # GUARANTEE 2: below `min_visits_for_model` the analytic belief is the
        # prior, which is provably the right answer with no evidence.  The model
        # has nothing to add there and everything to lose.
        w = np.where(n_visits >= self.min_visits_for_model, b, 0.0)
        return np.clip((1.0 - w) * p_rung1 + w * p_active_hat, P_CLIP_LO, P_CLIP_HI)

    def p_active_hat(self, feature_matrix, t: "float | None" = None) -> np.ndarray:
        """Adapter for `Belief.attach_model` / `Belief._model_p_active`.

        The frozen `BeliefLike` hook is `model.p_active_hat(features, t) -> (N,)`
        **on the activity scale**, and the belief does its own `beta` blend and
        its own `min_visits_for_model` gate afterwards.  So this returns the RAW
        deconvolved `p_active_hat` -- blending here as well would apply beta
        twice and quietly halve the model's influence.

        Without this method the belief's duck-typed hook finds nothing (an
        `ActivityModel` is a dataclass, not a callable), falls through to
        `return None`, and the learned path silently does nothing at any beta.
        """
        X = np.asarray(feature_matrix, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"feature_matrix must have {N_FEATURES} columns, got {X.shape[1]}"
            )
        if self.estimator is None:
            return None  # belief falls back to rung 1
        dwell, bw = float(self.infer_dwell_s), float(self.infer_bw_hz)
        pd_bar_next = np.broadcast_to(
            self._pd_bar_for(dwell, bw), (X.shape[0],)
        )
        extra = np.column_stack([
            np.full(X.shape[0], np.log1p(dwell)),
            np.full(X.shape[0], np.log1p(bw / 1.0e6)),
        ])
        return self.predict_p_active(np.column_stack([X, extra]), pd_bar_next)

    # -- the Brier gate -----------------------------------------------------
    def evaluate_gate(self, X, y, pd_bar_next, p_rung1=None) -> dict:
        """GUARANTEE 3.  Compare against rung 1 on the SAME held-out rows.

        Rung 1's forecast of this label is its analytic belief pushed FORWARD
        through the detector: `p_rung1*pd_bar + (1-p_rung1)*P_fa`.  That is the
        exact inverse of the deconvolution, so the two forecasts are strictly
        comparable and the comparison is fair by construction.

        If the model fails to beat it, `beta` is forced to 0.0 and the demo path
        becomes bit-identical to rung 1.  A tie loses: the simpler path wins.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if p_rung1 is None:
            p_rung1 = X[:, IDX_P_RUNG1]
        p_model = self.predict_p_det(X)
        p_r1 = np.clip(
            p_det_from_p_active(p_rung1, pd_bar_next, self.pfa), P_CLIP_LO, P_CLIP_HI
        )
        b_model, b_r1 = brier(p_model, y), brier(p_r1, y)
        ok = bool(np.isfinite(b_model) and b_model < b_r1 - BRIER_GATE_MARGIN)
        self.gate_ok = ok
        if ok:
            self.gate_reason = (
                f"model Brier {b_model:.5f} beats rung-1 {b_r1:.5f} "
                f"by {b_r1 - b_model:.5f} on {len(y)} held-out rows"
            )
        else:
            self.gate_reason = (
                f"model Brier {b_model:.5f} does NOT beat rung-1 {b_r1:.5f} "
                f"(margin {BRIER_GATE_MARGIN:g}); beta forced to 0.0"
            )
            if self.beta != 0.0:
                msg = "BRIER GATE TRIPPED: " + self.gate_reason
                LOG.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                print(msg, file=sys.stderr)
            self.beta = 0.0
        return {
            "brier_model": b_model,
            "brier_rung1": b_r1,
            "gate_ok": ok,
            "n_rows": int(len(y)),
            "reason": self.gate_reason,
        }

    # -- persistence --------------------------------------------------------
    @staticmethod
    def manifest_path(path) -> Path:
        return Path(path).with_suffix(".manifest.json")

    def save(self, path=DEFAULT_MODEL_PATH) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_names": tuple(self.feature_names),
                "pfa": self.pfa,
                "min_visits_for_model": self.min_visits_for_model,
            },
            path,
        )
        man = dict(self.manifest)
        man.setdefault("feature_names", list(self.feature_names))
        man.setdefault("contract_features", list(FEATURE_NAMES))
        man.setdefault("train_extra_names", list(TRAIN_EXTRA_NAMES))
        man.setdefault("min_visits_for_model", self.min_visits_for_model)
        man.setdefault("pfa", self.pfa)
        with open(self.manifest_path(path), "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, sort_keys=True, default=str)
        return path

    @classmethod
    def load(
        cls,
        path=DEFAULT_MODEL_PATH,
        beta: float = 0.0,
        holdout: "tuple | None" = None,
        strict: bool = False,
    ) -> "ActivityModel":
        """Load, then run the Brier gate BEFORE the model can influence anything.

        `beta` defaults to 0.0 (GUARANTEE 1): a caller must opt in explicitly.
        `holdout` is `(X, y, pd_bar_next)`; when omitted the gate falls back to
        the Brier scores recorded in the manifest at training time.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path}. Train one with:\n"
                "    python -m agent.policy_learned --train"
            )
        blob = joblib.load(path)
        man = {}
        mp = cls.manifest_path(path)
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as fh:
                man = json.load(fh)

        names = tuple(blob.get("feature_names", FEATURE_NAMES + TRAIN_EXTRA_NAMES))
        expected = FEATURE_NAMES + TRAIN_EXTRA_NAMES
        if names != expected:
            raise ValueError(
                "feature contract mismatch: model was trained on\n"
                f"  {names}\nbut agent.base now declares\n  {expected}\n"
                "Retrain before using this model."
            )
        assert_no_channel_leakage(names)

        obj = cls(
            estimator=blob["estimator"],
            feature_names=names,
            beta=float(beta),
            min_visits_for_model=int(
                blob.get("min_visits_for_model", man.get("min_visits_for_model", 3))
            ),
            pfa=float(blob.get("pfa", man.get("pfa", 1e-3))),
            manifest=man,
        )
        obj.pd_bar = PdBar(pfa=obj.pfa)

        if holdout is not None:
            obj.evaluate_gate(*holdout)
        else:
            b_m = man.get("brier_model")
            b_r = man.get("brier_rung1")
            if b_m is None or b_r is None:
                obj.gate_ok = False
                obj.gate_reason = (
                    "manifest carries no held-out Brier scores; cannot prove the "
                    "model beats rung 1, so beta forced to 0.0"
                )
                if obj.beta != 0.0:
                    LOG.warning("BRIER GATE TRIPPED: %s", obj.gate_reason)
                    print("BRIER GATE TRIPPED: " + obj.gate_reason, file=sys.stderr)
                obj.beta = 0.0
            elif float(b_m) >= float(b_r) - BRIER_GATE_MARGIN:
                obj.gate_ok = False
                obj.gate_reason = (
                    f"manifest Brier {float(b_m):.5f} does not beat rung-1 "
                    f"{float(b_r):.5f}; beta forced to 0.0"
                )
                if obj.beta != 0.0:
                    LOG.warning("BRIER GATE TRIPPED: %s", obj.gate_reason)
                    print("BRIER GATE TRIPPED: " + obj.gate_reason, file=sys.stderr)
                obj.beta = 0.0
            else:
                obj.gate_ok = True
                obj.gate_reason = (
                    f"manifest Brier {float(b_m):.5f} beats rung-1 {float(b_r):.5f}"
                )
        if strict and not obj.gate_ok:
            raise RuntimeError("Brier gate failed: " + obj.gate_reason)
        return obj

    def attach_to(self, belief, beta: "float | None" = None) -> None:
        """Hand this model to agent B's `Belief` via the frozen `BeliefLike` API.

        Also borrows the belief's own `pd_bar_for`, so the P_d the deconvolution
        divides out is bit-identical to the P_d the belief multiplies back in.
        """
        fn = getattr(belief, "pd_bar_for", None)
        if callable(fn):
            self._pd_bar_fn = fn
        b = float(self.beta if beta is None else beta)
        # The gate is the last word: a model that failed it never runs, however
        # the caller asks (GUARANTEE 3).
        if not self.gate_ok:
            b = 0.0
        belief.attach_model(self, b)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_estimator(seed: int = 0, **overrides):
    """The frozen rung-2 estimator.  DESIGN.md section 8 -- do not substitute.

    Calibration is MANDATORY, not decorative: the output is deconvolved and fed
    into a Bayes update, so it must be an honest probability rather than a
    ranking score.  Isotonic because we have tens of thousands of rows and no
    reason to assume the sigmoid shape Platt scaling imposes.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = dict(HGB_PARAMS)
    params.update(overrides)
    base = HistGradientBoostingClassifier(random_state=int(seed), **params)
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)


def train_activity_model(
    X,
    y,
    pd_bar_next,
    seed: int = 0,
    holdout_frac: float = 0.2,
    model_kwargs: "dict | None" = None,
    manifest_extra: "dict | None" = None,
    beta: float = 0.0,
    pfa: float = 1e-3,
    min_visits_for_model: int = 3,
    groups=None,
) -> ActivityModel:
    """Fit, calibrate, and gate.  Returns a model whose `beta` is already safe.

    The held-out split is GROUPED by run when `groups` is supplied, so rows from
    the same episode cannot straddle the split -- otherwise the Brier gate would
    be comparing against a leaked-in-time baseline and would pass too easily.
    """
    import sklearn

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.int8).ravel()
    pd_bar_next = np.asarray(pd_bar_next, dtype=np.float64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}")
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(f"training matrix contains non-finite values at {bad[:5]}")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "training labels are all one class -- the collector produced no "
            "usable positive/negative contrast"
        )

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if groups is not None:
        g = np.asarray(groups)
        uniq = np.unique(g)
        rng.shuffle(uniq)
        n_hold = max(1, int(round(holdout_frac * len(uniq))))
        hold_groups = set(uniq[:n_hold].tolist())
        test_mask = np.array([v in hold_groups for v in g])
    else:
        idx = rng.permutation(n)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[idx[: max(1, int(round(holdout_frac * n)))]] = True
    train_mask = ~test_mask
    # Degenerate splits happen on the tiny matrices used in tests; fall back to
    # scoring on everything rather than raising.
    if train_mask.sum() < 10 or len(np.unique(y[train_mask])) < 2:
        train_mask = np.ones(n, dtype=bool)
        test_mask = np.ones(n, dtype=bool)
    if len(np.unique(y[test_mask])) < 2:
        test_mask = np.ones(n, dtype=bool)

    est = make_estimator(seed=seed, **(model_kwargs or {}))
    est.fit(X[train_mask], y[train_mask])

    model = ActivityModel(
        estimator=est,
        feature_names=FEATURE_NAMES + TRAIN_EXTRA_NAMES,
        beta=float(beta),
        min_visits_for_model=int(min_visits_for_model),
        pfa=float(pfa),
        pd_bar=PdBar(pfa=float(pfa)),
    )
    gate = model.evaluate_gate(
        X[test_mask], y[test_mask], pd_bar_next[test_mask],
        p_rung1=X[test_mask][:, IDX_P_RUNG1],
    )

    model.manifest = {
        "feature_names": list(FEATURE_NAMES + TRAIN_EXTRA_NAMES),
        "contract_features": list(FEATURE_NAMES),
        "train_extra_names": list(TRAIN_EXTRA_NAMES),
        "n_features": len(FEATURE_NAMES) + len(TRAIN_EXTRA_NAMES),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "training_scenarios": list(TRAIN_SCENARIOS),
        "training_seeds": list(TRAIN_SEEDS),
        "held_out_scenario": HELD_OUT_SCENARIO,
        "n_rows_total": int(n),
        "n_rows_train": int(train_mask.sum()),
        "n_rows_holdout": int(test_mask.sum()),
        "positive_rate": float(np.mean(y)),
        "brier_model": gate["brier_model"],
        "brier_rung1": gate["brier_rung1"],
        "brier_gate_margin": BRIER_GATE_MARGIN,
        "gate_ok": gate["gate_ok"],
        "gate_reason": gate["reason"],
        "estimator": "CalibratedClassifierCV(HistGradientBoostingClassifier, "
                     "method=isotonic, cv=3)",
        "hgb_params": {**HGB_PARAMS, **(model_kwargs or {}), "random_state": seed},
        "min_visits_for_model": int(min_visits_for_model),
        "pfa": float(pfa),
        "seed": int(seed),
    }
    if manifest_extra:
        model.manifest.update(manifest_extra)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_train(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        df = load_log_frame(log_dir=args.log_dir)
    except LogsUnavailable as exc:
        print(f"\n[rung 2] cannot train: {exc}\n", file=sys.stderr)
        return 2
    except ImportError as exc:  # pragma: no cover
        print(f"[rung 2] pandas unavailable: {exc}", file=sys.stderr)
        return 2

    assert_no_channel_leakage(FEATURE_NAMES)
    X, y, meta = build_training_matrix(df, pfa=args.pfa, label_mode=args.label_mode)
    print(f"[rung 2] {len(df)} log rows -> {X.shape[0]} labelled rows, "
          f"{X.shape[1]} columns, positive rate {np.mean(y):.4f}")

    model = train_activity_model(
        X, y, meta["pd_bar_next"],
        seed=args.seed, pfa=args.pfa, beta=args.beta,
        min_visits_for_model=args.min_visits, groups=meta["run_id"],
    )
    path = model.save(args.out)
    print(f"[rung 2] saved {path}")
    print(f"[rung 2] manifest {ActivityModel.manifest_path(path)}")
    print(f"[rung 2] brier model={model.manifest['brier_model']:.5f} "
          f"rung1={model.manifest['brier_rung1']:.5f}")
    print(f"[rung 2] gate: {'PASS' if model.gate_ok else 'FAIL'} -- {model.gate_reason}")
    return 0


def _cmd_report(args) -> int:
    path = Path(args.out)
    mp = ActivityModel.manifest_path(path)
    if not mp.exists():
        print(f"[rung 2] no manifest at {mp}; run `--train` first.", file=sys.stderr)
        return 2
    with open(mp, "r", encoding="utf-8") as fh:
        man = json.load(fh)

    print("=" * 68)
    print("RUNG 2 -- learned activity model report")
    print("=" * 68)
    for k in ("estimator", "sklearn_version", "training_scenarios", "training_seeds",
              "held_out_scenario", "n_rows_total", "n_rows_train", "n_rows_holdout",
              "positive_rate", "min_visits_for_model", "pfa"):
        if k in man:
            v = man[k]
            if isinstance(v, list) and len(v) > 6:
                v = f"[{v[0]} .. {v[-1]}] ({len(v)})"
            print(f"  {k:<22} {v}")
    print("-" * 68)
    bm, br = man.get("brier_model"), man.get("brier_rung1")
    if bm is not None and br is not None:
        print(f"  brier (model)          {bm:.5f}")
        print(f"  brier (rung 1)         {br:.5f}")
        print(f"  improvement            {br - bm:+.5f}")
    print(f"  gate                   {'PASS' if man.get('gate_ok') else 'FAIL'}")
    print(f"  {man.get('gate_reason', '')}")
    print("-" * 68)
    print("  anti-regression guarantees:")
    print("    1. beta defaults to 0.0 (learned path off unless opted in)")
    print(f"    2. gated below n_visits < {man.get('min_visits_for_model', 3)}")
    print("    3. Brier gate at load forces beta=0.0 unless rung 1 is beaten")
    print("=" * 68)

    print("\nmarginal P_d table (pd_bar[bw, dwell]) -- the detector we divide out:")
    pdb = PdBar(pfa=float(man.get("pfa", 1e-3)))
    header = "  bw\\dwell " + "".join(f"{d:>8.0f}ms" for d in DWELL_CANDIDATES_MS)
    print(header)
    for bw in BW_CANDIDATES_MHZ:
        row = "".join(
            f"{float(pdb(d * 1e-3, bw * 1e6)):>10.4f}" for d in DWELL_CANDIDATES_MS
        )
        print(f"  {bw:>3d} MHz {row}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agent.policy_learned",
        description="Rung 2: train / report the learned activity model.",
    )
    ap.add_argument("--train", action="store_true", help="fit from collected logs")
    ap.add_argument("--report", action="store_true", help="print the model manifest")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="GUARANTEE 1: learned path is off at 0.0 (the default)")
    ap.add_argument("--min-visits", type=int, default=3)
    ap.add_argument(
        "--label-mode", choices=("pre_labelled", "shift"), default="pre_labelled",
        help="pre_labelled: rows already carry the next observation's outcome "
             "(what agent/policy_index.py log_rows() emits). shift: label each "
             "row from the following row of the same channel.",
    )
    args = ap.parse_args(argv)

    if args.train:
        rc = _cmd_train(args)
        if rc or not args.report:
            return rc
    if args.report:
        return _cmd_report(args)
    if not args.train:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
