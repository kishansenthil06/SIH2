"""Rung-1 state estimator: the layer between the observation engine and the scheduler.

A scan result is *one bit about one band at one instant*.  On its own it cannot
drive a scheduler, because the world keeps changing while we are not looking: this
is a restless bandit, and belief must decay toward the prior between visits.  This
module turns the bit-stream into a calibrated `(N,)` posterior over "is channel c
active right now", plus the per-channel sufficient statistics that rung 2 consumes
as features.

Three ideas carry the whole file:

1. **Closed-form decay.**  For a 2-state CTMC with rates `lam_on` (off->on) and
   `lam_off` (on->off), the exact transient is
   `p(t+dt) = pi + (p(t) - pi)*exp(-Lam*dt)` with `pi = lam_on/Lam`,
   `Lam = lam_on + lam_off`.  Verified identical to `scipy.linalg.expm(Q*dt)` to
   1e-12 (see `eval/tests/test_agent_belief.py`), so there is no matrix exponential
   at runtime.  Note it decays *upward* from below the prior too: a channel we are
   confident is empty drifts back up toward `pi` as our information ages.

2. **Observations are likelihoods, not facts.**  A miss means one of four things --
   nothing there, below sensitivity, silent during the dwell, or outside the tuned
   bandwidth -- so it must not be recorded as "empty".  Everything enters through
   `p*L1 / (p*L1 + (1-p)*L0)`.

3. **Two P_d's** (DESIGN.md section 4).  On a *detection* we hold the reported SNR,
   so we evaluate the detector curve exactly there: a marginal -20 dB / 2 ms
   detection has `P_d ~ 0.004` against `P_fa = 1e-3`, a likelihood ratio of only 4,
   so belief rises but does not saturate; a -10 dB / 5 ms detection has LR ~ 1000
   and pins it.  The agent distrusts marginal detections straight out of the maths.
   On a *miss* there is no SNR to condition on, so we use a **marginal**
   `pd_bar[bw, dwell]`, precomputed once by integrating the curve over the agent's
   assumed `snr ~ N(assumed_snr_mu_db, assumed_snr_sigma_db)` -- its spec-sheet
   belief, deliberately **not** the truth.

FIREWALL: this module imports only `sim.contract` and `sim.config`.  `pd_curve` is
**deliberately reimplemented** here rather than imported from `sim.receiver`; a
cross-check test asserts the two agree to 1e-9.  Duplication is cheaper than a
firewall breach.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from agent.base import (
    FEATURE_NAMES,
    N_FEATURES,
    SENTINEL_NO_IDI,
    SENTINEL_NO_SNR,
)
from sim.contract import ChannelGrid, Mission, Obs, Scan

# Posterior clip.  Never allow certainty: a belief of exactly 0 can never be
# revived by any finite likelihood ratio, which would make one unlucky miss
# permanent.
P_CLIP_LO: float = 1e-6
P_CLIP_HI: float = 1.0 - 1e-6

# Ring buffer of recent observations per channel.  8 is enough to cover the
# 1.0 s window used by `nbr_recent_hits` at realistic revisit rates, and keeps the
# whole thing in an (N, 8) array we can reduce with one vectorised op.
RING: int = 8

# EMA smoothing constants, frozen by the feature contract in `agent/base.py`.
EMA_FAST: float = 0.30
EMA_SLOW: float = 0.05

# Window over which "recent" is defined for the two spectral-context features.
RECENT_WINDOW_S: float = 1.0

# Quadrature for the marginal P_d integral.  41 points over +/-5 sigma of a
# Gaussian is trapezoid-exact to ~1e-9 for this smooth integrand -- far tighter
# than the modelling error in "assumed SNR ~ N(-15, 5)" itself.
_QUAD_N: int = 41
_QUAD_SIGMAS: float = 5.0

# Sentinel for "no observation in this ring slot yet".  A large negative time
# rather than -inf so that `t - ring_t` never produces inf/nan.
_NO_TIME: float = -1.0e9


# --------------------------------------------------------------- detector model
def pd_curve(
    snr_eff_db,
    dwell_s,
    pfa: float = 1.0e-3,
    channel_bw_hz: float = 1.0e6,
):
    """Urkowitz energy detector, Gaussian approximation.  DESIGN.md section 1.

    `N = dwell_s * channel_bw_hz` complex samples; `s` is linear *effective* SNR
    (i.e. already carrying the bandwidth penalty and any gain).

    `s = 0` yields `Q(Q^-1(pfa)) = pfa` automatically, so there is no separate
    false-alarm branch anywhere in the project.

    REIMPLEMENTED, not imported: `sim.receiver` is on the far side of the
    firewall.  `test_agent_belief.py` cross-checks the two to 1e-9 when the real
    receiver exists.
    """
    n = np.asarray(dwell_s, dtype=np.float64) * float(channel_bw_hz)
    s = 10.0 ** (np.asarray(snr_eff_db, dtype=np.float64) / 10.0)
    return norm.sf((norm.isf(pfa) - np.sqrt(n) * s) / (1.0 + s))


def snr_eff_db(
    snr_db,
    bw_hz,
    gain_db: float = 0.0,
    bw_penalty_db_per_octave: float = 1.0,
    gain_nf_improvement_db: float = 6.0,
    channel_bw_hz: float = 1.0e6,
):
    """Effective SNR seen by the detector.  DESIGN.md section 4.

    The bandwidth penalty is load-bearing, not cosmetic: without it the widest
    scan strictly dominates (same time, same energy, 20x the channels) and the
    bandwidth knob is degenerate.  At 1 dB/octave a 20 MHz scan is 4.3 dB less
    sensitive than a 1 MHz one, so it needs ~7x the dwell to reach the same weak
    emitter -- which is what creates "wide-and-fast to explore, narrow-and-long to
    confirm".
    """
    octaves = np.log2(np.asarray(bw_hz, dtype=np.float64) / float(channel_bw_hz))
    boost = gain_nf_improvement_db if gain_db > 0.0 else 0.0
    return np.asarray(snr_db, dtype=np.float64) - bw_penalty_db_per_octave * octaves + boost


def bayes_posterior(p, l1, l0):
    """`p*L1 / (p*L1 + (1-p)*L0)`, clipped.  Vectorised, safe at the extremes.

    If `l1 == l0` the observation is uninformative and this is the identity to
    machine precision -- the property that proves we are doing inference rather
    than bookkeeping.
    """
    p = np.asarray(p, dtype=np.float64)
    num = p * l1
    den = num + (1.0 - p) * l0
    out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), p)
    return np.clip(out, P_CLIP_LO, P_CLIP_HI)


def marginal_pd_table(
    bw_candidates_hz,
    dwell_candidates_s,
    mu_db: float,
    sigma_db: float,
    pfa: float,
    bw_penalty_db_per_octave: float = 1.0,
    channel_bw_hz: float = 1.0e6,
) -> np.ndarray:
    """`pd_bar[bw_idx, dwell_idx]` -- P_d marginalised over the assumed SNR prior.

    This is the likelihood used on a **miss**, where there is no reported SNR to
    condition on.  A 1 ms miss barely moves the belief; a 100 ms miss crushes it.

    Public because agent D needs exactly this table to invert the rung-2 label
    (`p_det_hat -> p_active_hat`), and the two must use the same numbers.
    """
    bw = np.asarray(bw_candidates_hz, dtype=np.float64)
    dwell = np.asarray(dwell_candidates_s, dtype=np.float64)

    z = np.linspace(-_QUAD_SIGMAS, _QUAD_SIGMAS, _QUAD_N)
    wq = norm.pdf(z)
    wq /= wq.sum()  # renormalise so the truncated tails do not bias the mean down

    snr = mu_db + sigma_db * z                       # (Q,)
    eff = snr_eff_db(
        snr[None, :], bw[:, None],
        bw_penalty_db_per_octave=bw_penalty_db_per_octave,
        channel_bw_hz=channel_bw_hz,
    )                                                # (B, Q)
    curve = pd_curve(
        eff[:, :, None], dwell[None, None, :], pfa=pfa, channel_bw_hz=channel_bw_hz
    )                                                # (B, Q, D)
    return np.tensordot(wq, curve, axes=([0], [1]))  # -> (B, D)


# ---------------------------------------------------------------------- belief
def _expand_by_priority(by_prio, scalar_default: float, mission: Mission, n: int) -> np.ndarray:
    """Expand a {priority: value} mapping into an (N,) per-channel array.

    Falls back to `scalar_default` everywhere when the mapping is absent, so a
    config written before per-class priors existed still loads and behaves
    exactly as it did.  Channels whose priority is not named in the mapping
    (including untasked priority-0 channels) also keep the scalar default.
    """
    out = np.full(n, float(scalar_default), dtype=np.float64)
    if not by_prio:
        return out
    for prio, val in dict(by_prio).items():
        out[mission.priority == int(prio)] = float(val)
    return out


class Belief:
    """Per-channel occupancy posterior + rung-2 sufficient statistics.

    Satisfies `agent.base.BeliefLike`.  All state is `(N,)` numpy; there is no
    Python loop over channels anywhere in the update path.

    `mode`:
      - ``"bayes"``   -- the real thing: Markov decay + likelihood updates.
      - ``"laplace"`` -- the `greedy` baseline's belief: a raw Laplace hit rate
        `(n_det+1)/(n_visits+2)` that **never decays**.  This genuinely bypasses
        both the propagation and the Bayes update, which is the point: it isolates
        exactly what the belief layer contributes (DESIGN.md section 6).
    """

    def __init__(self, grid: ChannelGrid, mission: Mission, cfg: dict, mode: str = "bayes"):
        if mode not in ("bayes", "laplace"):
            raise ValueError(f"mode must be 'bayes' or 'laplace', got {mode!r}")
        self.mode = mode
        self.grid = grid
        self.mission = mission
        self.n = int(grid.n_channels)
        n = self.n

        # `cfg` is the full loaded config; tolerate being handed just the agent
        # sub-dict so the belief can be constructed standalone in tests.
        acfg = cfg.get("agent", cfg) if isinstance(cfg, dict) else {}
        rx = cfg.get("receiver", {}) if isinstance(cfg, dict) else {}
        self.cfg = cfg

        self.pfa = float(rx.get("pfa", 1.0e-3))
        self.bw_penalty = float(rx.get("bw_penalty_db_per_octave", 1.0))
        self.gain_nf_db = float(rx.get("gain_nf_improvement_db", 6.0))
        self.channel_bw_hz = float(grid.channel_bw_hz)
        self.horizon_s = float(cfg.get("horizon_s", 60.0)) if isinstance(cfg, dict) else 60.0

        # The agent's assumed dynamics.  DELIBERATELY not the truth -- the agent
        # is given a spec sheet, not the simulator's parameters.
        #
        # The prior is PER PRIORITY CLASS, not uniform, and that matters more than
        # it looks.  With a uniform prior, `value = p*w` is dominated by `w`, so a
        # threat channel (w=1.0 J) outranks a routine one (w=0.1 J) by 10x -- even
        # though threat emitters are on ~2% of the time and routine ones ~40%.
        # Measured consequence of the uniform version: 69.6% of the energy budget
        # (27.3 s of dwell) went into a band that was empty 98% of the time, and
        # the policy lost to a round-robin sweep by 8x on energy per detection.
        #
        # Per-class priors are legitimately agent-visible intel, exactly like the
        # mission priority bands -- an ESM operator knows threat emitters are rare
        # but valuable while routine emitters chatter constantly.  It is NOT a
        # truth leak: the numbers come from config as standing assumptions and are
        # never fitted to the running scenario.
        self.pi_on = _expand_by_priority(
            acfg.get("prior_pi_on_by_priority"),
            float(acfg.get("prior_pi_on", 0.05)),
            mission,
            n,
        )
        self.lam_sum = _expand_by_priority(
            acfg.get("prior_lam_sum_by_priority"),
            float(acfg.get("prior_lam_sum", 1.0)),
            mission,
            n,
        )

        self.assumed_snr_mu_db = float(acfg.get("assumed_snr_mu_db", -15.0))
        self.assumed_snr_sigma_db = float(acfg.get("assumed_snr_sigma_db", 5.0))

        # Per-priority-class assumed SNR.  Same intel argument as the per-class
        # prior above, and it fixes a distinct failure: with ONE assumed SNR
        # distribution the agent believed a 2 ms scan had P_d ~ 0.28 everywhere,
        # when for a -20 dB threat emitter the true P_d at 2 ms is 0.004 -- 70x
        # optimistic.  It therefore never chose a dwell above 5 ms and intercepted
        # exactly zero priority-1 emitters.  Telling it that threat emitters are
        # weak is what makes long dwells rational where they are actually needed.
        self.assumed_snr_mu_by_priority = acfg.get("assumed_snr_mu_db_by_priority") or {}
        # 0 disables the adaptive prior and restores the static one exactly.
        self._prior_pseudocount = float(acfg.get("prior_pseudocount", 0.0))

        learned = acfg.get("learned", {}) or {}
        self.min_visits_for_model = int(learned.get("min_visits_for_model", 3))

        # Candidate sets, needed so `pd_bar` is indexable by the same indices the
        # policy uses when it builds candidates.
        bw_mhz = tuple(acfg.get("bw_candidates_mhz", (1, 2, 5, 10, 20)))
        dwell_ms = tuple(acfg.get("dwell_candidates_ms", (1, 2, 5, 10, 20, 50, 100, 200)))
        self.bw_candidates_hz = np.asarray(bw_mhz, dtype=np.float64) * 1.0e6
        self.dwell_candidates_s = np.asarray(dwell_ms, dtype=np.float64) * 1.0e-3

        # PUBLIC: agent D reads this for the rung-2 deconvolution.
        self.pd_bar: np.ndarray = marginal_pd_table(
            self.bw_candidates_hz,
            self.dwell_candidates_s,
            self.assumed_snr_mu_db,
            self.assumed_snr_sigma_db,
            self.pfa,
            bw_penalty_db_per_octave=self.bw_penalty,
            channel_bw_hz=self.channel_bw_hz,
        )

        # pd_bar_by_class[k, bw, dwell] for priority class k in 0..3, where class 0
        # is "untasked" and falls back to the scalar assumed SNR.  The policy uses
        # this so the expected gain of a candidate reflects how detectable the
        # emitters in THAT band actually are, rather than one band-wide average.
        self.priority_classes = (0, 1, 2, 3)
        self.pd_bar_by_class: np.ndarray = np.stack(
            [
                marginal_pd_table(
                    self.bw_candidates_hz,
                    self.dwell_candidates_s,
                    float(
                        self.assumed_snr_mu_by_priority.get(
                            k, self.assumed_snr_mu_by_priority.get(str(k), self.assumed_snr_mu_db)
                        )
                    ),
                    self.assumed_snr_sigma_db,
                    self.pfa,
                    bw_penalty_db_per_octave=self.bw_penalty,
                    channel_bw_hz=self.channel_bw_hz,
                )
                for k in self.priority_classes
            ]
        )
        # (N,) index into pd_bar_by_class for every channel.
        self.class_index = np.zeros(n, dtype=np.int64)
        for i, k in enumerate(self.priority_classes):
            self.class_index[np.asarray(mission.priority) == k] = i

        self.w = np.asarray(mission.w, dtype=np.float64).copy()

        self._model = None
        self._beta = 0.0

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Cold start.  Every channel sits at the prior with no history."""
        n = self.n
        self.t_now: float = 0.0
        self.p = self.pi_on.copy()

        self.t_last_visit = np.zeros(n, dtype=np.float64)
        self.visited = np.zeros(n, dtype=bool)
        self.n_visits = np.zeros(n, dtype=np.float64)
        self.n_detections = np.zeros(n, dtype=np.float64)
        self.t_last_detect = np.zeros(n, dtype=np.float64)
        self.misses_since_detect = np.zeros(n, dtype=np.float64)
        self.dwell_sum = np.zeros(n, dtype=np.float64)
        self.snr_sum = np.zeros(n, dtype=np.float64)
        self.hit_ema_fast = np.zeros(n, dtype=np.float64)
        self.hit_ema_slow = np.zeros(n, dtype=np.float64)

        # Running moments of the inter-detection interval, so `idi_mean` and
        # `idi_std` are O(1) per update instead of keeping every timestamp.
        self.idi_n = np.zeros(n, dtype=np.float64)
        self.idi_sum = np.zeros(n, dtype=np.float64)
        self.idi_sumsq = np.zeros(n, dtype=np.float64)

        self.ring_t = np.full((n, RING), _NO_TIME, dtype=np.float64)
        self.ring_hit = np.zeros((n, RING), dtype=np.float64)
        self.ring_ptr = np.zeros(n, dtype=np.int64)

    # ------------------------------------------------------------- propagation
    def propagate_to(self, t: float) -> None:
        """Advance the posterior to time `t`.  Mutating; forward only.

        One line, no matrix exponential.  `dt = 0` is exactly the identity and
        `dt -> inf` drives any `p` to `pi`, both by construction.
        """
        t = float(t)
        if self.mode == "laplace":
            # The greedy baseline's belief does not decay -- that is the whole
            # point of the ablation.
            self.t_now = max(self.t_now, t)
            return
        dt = t - self.t_now
        if dt <= 0.0:
            return
        pi = self._prior_target()
        self.p = pi + (self.p - pi) * np.exp(-self.lam_sum * dt)
        self.t_now = t

    def _prior_target(self) -> np.ndarray:
        """What the belief decays TOWARD.  Static config prior, or learned.

        A static prior throws away the most valuable thing the agent learns.
        `p` answers "is this channel radiating right now" -- transient, and it
        should decay.  But a detection also proves an emitter LIVES here, which is
        permanent.  Decaying to a band-wide constant discards that, so the agent
        re-derives the spectrum's layout from scratch every couple of seconds and
        can never do better than sweeping.

        The fix is one Beta-Bernoulli posterior per channel: blend the config
        prior with this channel's own observed hit rate under `prior_pseudocount`
        pseudo-observations.  A channel that has produced detections decays to a
        HIGH floor and keeps earning revisits; a channel scanned repeatedly with
        nothing decays below the prior and is left alone.  That asymmetry is the
        mechanism by which an adaptive policy beats a sweep rather than merely
        matching it -- and it is still strictly the agent's own logs, no truth.
        """
        if self._prior_pseudocount <= 0.0:
            return self.pi_on
        a = self._prior_pseudocount
        return (self.n_detections + a * self.pi_on) / (self.n_visits + a)

    def _propagated(self, t: float) -> np.ndarray:
        """Non-mutating propagation, so scoring at a hypothetical `t` is safe."""
        if self.mode == "laplace":
            return (self.n_detections + 1.0) / (self.n_visits + 2.0)
        pi = self._prior_target()
        dt = float(t) - self.t_now
        if dt <= 0.0:
            return self.p
        # Must decay toward the SAME target as `propagate_to`.  This previously
        # computed `pi` and then used `self.pi_on` anyway, so the non-mutating
        # scoring path decayed toward the static config prior while the mutating
        # path decayed toward the learned Beta-Bernoulli floor -- the mechanism
        # DESIGN.md 11.4 relies on.  It was inert only because the live loop
        # always calls this with dt <= 0 (which short-circuits above); it would
        # have bitten the moment anything scored at a hypothetical future `t`,
        # which is the reason this method exists at all.
        return pi + (self.p - pi) * np.exp(-self.lam_sum * dt)

    # -------------------------------------------------------------- the update
    def update(self, obs: Obs) -> None:
        """Fold one `Obs` into the belief: propagate to `obs.t_start`, then Bayes."""
        self.propagate_to(obs.t_start)

        chans = np.asarray(obs.scanned_channels, dtype=np.int64)
        if chans.size == 0:
            # Sleep (or a truncated scan): no information, but the clock moved.
            self.propagate_to(obs.t)
            return

        act = obs.action
        if not isinstance(act, Scan):
            self.propagate_to(obs.t)
            return

        dwell = float(act.dwell_s)
        bw = float(act.bw_hz)
        t_obs = float(obs.t_start)

        # ---- likelihoods -------------------------------------------------
        # Default every scanned channel to the MISS branch, then overwrite the
        # channels that reported a detection.  `pd_bar` is the marginal over the
        # assumed SNR prior; there is no reported SNR to condition on for a miss.
        pd_miss = self.pd_bar_for(bw, dwell)
        l1 = np.full(chans.size, 1.0 - pd_miss, dtype=np.float64)
        l0 = np.full(chans.size, 1.0 - self.pfa, dtype=np.float64)

        hit = np.zeros(chans.size, dtype=np.float64)
        snr_hit = np.zeros(chans.size, dtype=np.float64)
        if obs.detections:
            det_ch = np.fromiter((d.channel for d in obs.detections), dtype=np.int64,
                                 count=len(obs.detections))
            det_snr = np.fromiter((d.snr_db for d in obs.detections), dtype=np.float64,
                                  count=len(obs.detections))
            # Map detection channels onto positions within `chans`.  `chans` is a
            # contiguous ascending run, so searchsorted is exact and O(k log n).
            pos = np.searchsorted(chans, det_ch)
            ok = (pos < chans.size) & (chans[np.minimum(pos, chans.size - 1)] == det_ch)
            pos, det_snr = pos[ok], det_snr[ok]
            if pos.size:
                # The reported SNR is an estimate of the EFFECTIVE (post-penalty)
                # SNR the detector actually saw, so the curve is evaluated there
                # directly -- no second bandwidth penalty.
                l1[pos] = pd_curve(det_snr, dwell, pfa=self.pfa,
                                   channel_bw_hz=self.channel_bw_hz)
                l0[pos] = self.pfa
                hit[pos] = 1.0
                snr_hit[pos] = det_snr

        self.p[chans] = bayes_posterior(self.p[chans], l1, l0)

        # ---- sufficient statistics --------------------------------------
        det_mask = hit > 0.0
        # `t_last_visit` is the END of the dwell: the evaluator defines a revisit
        # gap as next_dwell_start - prev_dwell_end, and the scheduler's deadline
        # bound is only provable if the belief uses the same convention.
        self.t_last_visit[chans] = float(obs.t)
        self.visited[chans] = True
        self.n_visits[chans] += 1.0
        self.dwell_sum[chans] += dwell

        self.hit_ema_fast[chans] += EMA_FAST * (hit - self.hit_ema_fast[chans])
        self.hit_ema_slow[chans] += EMA_SLOW * (hit - self.hit_ema_slow[chans])

        self.misses_since_detect[chans] = np.where(
            det_mask, 0.0, self.misses_since_detect[chans] + 1.0
        )

        if det_mask.any():
            # `chans` is a contiguous ascending run and the env reports at most
            # one detection per channel, so `dch` is unique -- plain fancy-index
            # assignment is correct here and faster than np.add.at.
            dch = chans[det_mask]
            # Inter-detection interval, only once a channel has a previous one.
            prev = self.n_detections[dch] >= 1.0
            if prev.any():
                pch = dch[prev]
                gaps = np.maximum(t_obs - self.t_last_detect[pch], 0.0)
                self.idi_n[pch] += 1.0
                self.idi_sum[pch] += gaps
                self.idi_sumsq[pch] += gaps * gaps
            self.n_detections[dch] += 1.0
            self.t_last_detect[dch] = t_obs
            self.snr_sum[dch] += snr_hit[det_mask]

        # ---- ring buffer -------------------------------------------------
        ptr = self.ring_ptr[chans]
        self.ring_t[chans, ptr] = t_obs
        self.ring_hit[chans, ptr] = hit
        self.ring_ptr[chans] = (ptr + 1) % RING

        self.propagate_to(obs.t)

    # ------------------------------------------------------------- accessors
    def pd_bar_for(self, bw_hz: float, dwell_s: float) -> float:
        """Nearest-candidate lookup into `pd_bar`.  Public for agent D."""
        i = int(np.argmin(np.abs(self.bw_candidates_hz - float(bw_hz))))
        j = int(np.argmin(np.abs(self.dwell_candidates_s - float(dwell_s))))
        return float(self.pd_bar[i, j])

    def p_active(self, t: float) -> np.ndarray:
        """Rung-1 analytic posterior at time `t`.  Never mutates."""
        return self._propagated(t)

    def p_effective(self, t: float) -> np.ndarray:
        """The rung-2 plug-in point.

        With no model attached this returns `p_active(t)` **exactly** -- that is
        guarantee #1 that rung 2 cannot regress rung 1 (DESIGN.md section 8).
        With a model, blend `(1-beta)*p1 + beta*p2`, but only where the channel
        has at least `min_visits_for_model` visits; below that we are in the
        cold-start regime where rung 1 is provably right (it is the prior).
        """
        p1 = self._propagated(t)
        if self._model is None or self._beta <= 0.0:
            return p1
        p2 = self._model_p_active(t)
        if p2 is None:
            return p1
        p2 = np.clip(np.nan_to_num(np.asarray(p2, dtype=np.float64), nan=0.0),
                     P_CLIP_LO, P_CLIP_HI)
        blend = (1.0 - self._beta) * p1 + self._beta * p2
        return np.where(self.n_visits >= self.min_visits_for_model, blend, p1)

    def _model_p_active(self, t: float):
        """Duck-typed hook.  The model itself lives in agent D's file.

        Accepts either `model.p_active_hat(features, t)` or a plain callable
        `model(features)`; both must return `(N,)` in [0, 1] on the ACTIVITY
        scale (agent D does the `p_det -> p_active` deconvolution on its side,
        because only it knows `bw_next`/`tau_next`).
        """
        feats = self.feature_matrix(t)
        fn = getattr(self._model, "p_active_hat", None)
        if fn is not None:
            return fn(feats, t)
        if callable(self._model):
            return self._model(feats)
        return None

    def attach_model(self, model, beta: float) -> None:
        """Enable the rung-2 path.  `beta` defaults to 0 everywhere else."""
        self._model = model
        self._beta = float(beta)

    def staleness(self, t: float) -> np.ndarray:
        """Seconds since this channel was last inside a completed dwell."""
        return np.maximum(float(t) - self.t_last_visit, 0.0)

    # --------------------------------------------------------------- features
    def feature_matrix(self, t: float) -> np.ndarray:
        """`(n_channels, len(FEATURE_NAMES))` float64, in EXACTLY that column order.

        Guaranteed finite even on a cold-start belief -- agent D consumes this
        directly and a single NaN would poison a whole training run.
        """
        t = float(t)
        n = self.n
        out = np.empty((n, N_FEATURES), dtype=np.float64)

        nv = self.n_visits
        nd = self.n_detections
        has_det = nd > 0.0

        stale = np.maximum(t - self.t_last_visit, 0.0)
        since_det = np.where(has_det, np.maximum(t - self.t_last_detect, 0.0), t)

        # Recent-hit counts from the ring buffer, in one reduction.
        recent = self.ring_hit * ((t - self.ring_t) <= RECENT_WINDOW_S)
        cnt = recent.sum(axis=1)
        nbr = np.zeros(n, dtype=np.float64)
        for d in (-2, -1, 1, 2):  # only 4 shifts; the inner op stays vectorised
            if d < 0:
                nbr[:d] += cnt[-d:]
            else:
                nbr[d:] += cnt[:-d]

        band_act = float(np.mean(has_det & (t - self.t_last_detect <= RECENT_WINDOW_S)))

        idi_mean = np.where(self.idi_n >= 1.0, self.idi_sum / np.maximum(self.idi_n, 1.0),
                            SENTINEL_NO_IDI)
        var = (self.idi_sumsq / np.maximum(self.idi_n, 1.0)
               - (self.idi_sum / np.maximum(self.idi_n, 1.0)) ** 2)
        idi_std = np.where(self.idi_n >= 2.0, np.sqrt(np.maximum(var, 0.0)),
                           SENTINEL_NO_IDI)

        out[:, 0] = self._propagated(t)                              # p_rung1
        out[:, 1] = np.log1p(stale)                                  # log_staleness
        out[:, 2] = np.log1p(since_det)                              # log_since_detect
        out[:, 3] = nv                                               # n_visits
        out[:, 4] = (nd + 1.0) / (nv + 2.0)                          # emp_rate (Laplace)
        out[:, 5] = self.hit_ema_fast
        out[:, 6] = self.hit_ema_slow
        out[:, 7] = self.misses_since_detect
        out[:, 8] = np.log1p(self.dwell_sum / np.maximum(nv, 1.0))   # mean_dwell_log
        out[:, 9] = np.where(has_det, self.snr_sum / np.maximum(nd, 1.0), SENTINEL_NO_SNR)
        out[:, 10] = idi_mean
        out[:, 11] = idi_std
        out[:, 12] = nbr
        out[:, 13] = band_act
        out[:, 14] = self.w
        out[:, 15] = t / max(self.horizon_s, 1e-12)

        # Belt and braces: agent D's training run must never see a NaN.
        return np.nan_to_num(out, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:
        return (f"Belief(n={self.n}, mode={self.mode!r}, t={self.t_now:.4f}, "
                f"mean_p={float(self.p.mean()):.4f}, visits={int(self.n_visits.sum())})")


FEATURE_ORDER: tuple[str, ...] = FEATURE_NAMES  # re-exported for agent D's convenience
