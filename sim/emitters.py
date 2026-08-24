"""Emitter population and the canonical ground-truth burst table.

Truth is **pre-generated in full at reset()**, never advanced step-by-step.  Three
independent reasons, all load-bearing:

1. the `oracle` baseline needs the whole future at t=0;
2. truth becomes reproducible *independently of the agent's actions*, which is
   what makes "same seed, two policies" a fair comparison rather than two
   different worlds;
3. the receiver can integrate an arbitrary continuous window `[t0, t1)` with no
   quantisation error, so a 1 ms dwell and a 200 ms dwell are treated by exactly
   the same code path.

The canonical representation is a **burst table**: one row per contiguous
(time, channel-block) interval during which an emitter is radiating.  An agile
frequency hopper is expressed purely as *more rows sharing one `activation_id`*
-- there is no hopper special case anywhere downstream, which is why
`sim/channel.py`, `sim/receiver.py` and `eval/metrics.py` never mention hopping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.config import build_mission

# ---------------------------------------------------------------------------
# The canonical truth record.  `ch_hi` is EXCLUSIVE, matching
# `ChannelGrid.channels_for` which returns `arange(k_lo, k_lo + n)`.
#
# `activation_id` counts logical ON periods of the emitter; `burst_id` is the
# unique row index.  For a fixed or pulsed emitter the two are 1:1; for an agile
# hopper one activation spans several rows, which is exactly what lets the
# evaluator count "a hopper's 20 hops" as ONE unique detection opportunity.
# ---------------------------------------------------------------------------
BURST_DTYPE = np.dtype(
    [
        ("emitter_id", "i4"),
        ("activation_id", "i4"),
        ("burst_id", "i4"),
        ("t_on", "f8"),
        ("t_off", "f8"),
        ("ch_lo", "i4"),
        ("ch_hi", "i4"),      # EXCLUSIVE
        ("snr_db", "f8"),     # true SNR of THIS burst, after shadowing
        ("priority", "i4"),
        ("w_p", "f8"),        # mission weight in JOULES of the occupied channel
    ]
)

EMPTY_BURSTS = np.empty(0, dtype=BURST_DTYPE)


@dataclass(frozen=True, slots=True)
class Emitter:
    """One concrete emitter, drawn from a YAML `emitters:` spec.

    The two-state CTMC convention matches DESIGN.md section 1 exactly:
    `lam_off` is the ON->OFF rate (so mean ON time = 1/lam_off) and `lam_on` is
    the OFF->ON rate (so mean OFF time = 1/lam_on).  Getting these the wrong way
    round is the single easiest mistake in the project, hence the explicit
    `pi_on` / `lam_sum` properties that the belief's decay law also uses.
    """

    emitter_id: int
    kind: str                 # "fixed" | "pulsed" | "agile"
    channel: int              # lowest occupied channel (inclusive)
    n_channels: int           # block width; occupies [channel, channel + n_channels)
    snr_db: float             # nominal SNR before per-burst shadowing
    priority: int
    w_p: float                # mission weight (JOULES) of the home channel
    lam_on: float             # OFF -> ON rate  (1 / mean_off_s)
    lam_off: float            # ON  -> OFF rate (1 / mean_on_s)
    hop_set: tuple = ()       # agile only: channels it may occupy
    hop_dwell_s: float = 0.0  # agile only: re-draw interval while ON
    hop_pattern: str = ""     # agile only: "random" | "sequential"
    snr_sigma_db: float = 0.0

    @property
    def pi_on(self) -> float:
        """Stationary P(active) = lam_on / (lam_on + lam_off)."""
        return self.lam_on / (self.lam_on + self.lam_off)

    @property
    def lam_sum(self) -> float:
        """Mixing rate Lambda; belief decays as exp(-lam_sum * dt) toward pi_on."""
        return self.lam_on + self.lam_off

    @property
    def ch_hi(self) -> int:
        return self.channel + self.n_channels


# --------------------------------------------------------------- scenario build
def build_emitters(cfg: dict, rng: np.random.Generator) -> list[Emitter]:
    """Expand the YAML `emitters:` list into concrete `Emitter`s.

    `w_p` is taken from the MISSION weight of the channel the emitter occupies,
    not from a separate table.  That is deliberate: the agent's objective and the
    evaluator's metric then reference literally the same number, so "the policy
    optimised the thing we scored" needs no argument.
    """
    mission = build_mission(cfg)
    n_grid = int(cfg["grid"]["n_channels"])
    w = mission.w
    prio_map = mission.priority

    out: list[Emitter] = []
    eid = 0
    for spec in cfg["emitters"]:
        kind = str(spec.get("kind", "fixed"))
        count = int(spec["count"])
        lo, hi = (int(spec["channel_range"][0]), int(spec["channel_range"][1]))
        n_ch = int(spec.get("n_channels", 1))
        snr_lo, snr_hi = float(spec["snr_db"][0]), float(spec["snr_db"][1])
        # DESIGN.md section 1: mean ON = 1/lam_off, mean OFF = 1/lam_on.
        lam_off = 1.0 / float(spec["mean_on_s"])
        lam_on = 1.0 / float(spec["mean_off_s"])
        sigma = float(spec.get("snr_sigma_db", 0.0))
        priority = int(spec.get("priority", 0))

        hi_start = min(hi, n_grid) - n_ch + 1
        if hi_start <= lo:
            raise ValueError(
                f"emitter spec {spec!r}: channel_range too narrow for "
                f"n_channels={n_ch}"
            )

        hop_set: tuple = ()
        hop_dwell = 0.0
        hop_pattern = ""
        if kind == "agile":
            hop_lo = int(spec.get("hop_lo", lo))
            hop_hi = int(spec.get("hop_hi", hi))          # exclusive
            hop_n = int(spec.get("hop_n", 8))
            hop_dwell = float(spec.get("hop_dwell_s", 0.05))
            hop_pattern = str(spec.get("hop_pattern", "random"))
            pool = np.arange(hop_lo, min(hop_hi, n_grid) - n_ch + 1, dtype=np.int64)
            if pool.size == 0:
                raise ValueError(f"emitter spec {spec!r}: empty hop pool")
            hop_n = min(hop_n, pool.size)

        for _ in range(count):
            ch = int(rng.integers(lo, hi_start))
            snr = float(rng.uniform(snr_lo, snr_hi))
            if kind == "agile":
                # Each hopper gets its OWN hop set: two hoppers sharing a set
                # would be trivially correlated and would flatter the neighbour
                # feature that is supposed to catch them.
                hop_set = tuple(
                    int(v) for v in np.sort(rng.choice(pool, size=hop_n, replace=False))
                )
                ch = int(hop_set[int(rng.integers(0, len(hop_set)))])
            blk = slice(ch, ch + n_ch)
            out.append(
                Emitter(
                    emitter_id=eid,
                    kind=kind,
                    channel=ch,
                    n_channels=n_ch,
                    snr_db=snr,
                    priority=priority,
                    w_p=float(np.max(w[blk])) if n_ch else 0.0,
                    lam_on=lam_on,
                    lam_off=lam_off,
                    hop_set=hop_set,
                    hop_dwell_s=hop_dwell,
                    hop_pattern=hop_pattern,
                    snr_sigma_db=sigma,
                )
            )
            eid += 1
    del prio_map  # priority comes from the SPEC (intel), not from the mission map
    return out


# ------------------------------------------------------------- truth generation
def generate_bursts(
    emitters: list[Emitter],
    horizon_s: float,
    rng: np.random.Generator,
    rng_shadow: np.random.Generator,
    mission_w: np.ndarray | None = None,
) -> np.ndarray:
    """Sample the full ON/OFF history of every emitter over `[0, horizon_s)`.

    Sojourn times are exponential, so the process is the 2-state CTMC whose
    closed-form decay `agent/belief.py` assumes.  The initial state is drawn from
    the stationary distribution `pi_on`, so the ensemble is stationary from t=0
    and the empirical duty cycle is unbiased (no burn-in transient to explain
    away in the write-up).

    A burst that begins before the horizon is emitted in FULL, un-clipped: the
    window integrator in `sim/channel.py` never looks past `horizon_s` anyway,
    and clipping would bias the measured mean ON time downward.
    """
    rows: list[tuple] = []
    sigmas: list[float] = []

    for em in emitters:
        mean_on = 1.0 / em.lam_off
        mean_off = 1.0 / em.lam_on
        agile = em.kind == "agile" and em.hop_dwell_s > 0.0 and len(em.hop_set) > 0

        t = 0.0
        # Stationary start: with prob pi_on the emitter is already radiating.
        # Exponentials are memoryless, so the residual sojourn is a full draw.
        if rng.random() >= em.pi_on:
            t += float(rng.exponential(mean_off))

        hop_i = int(rng.integers(0, len(em.hop_set))) if agile else 0
        act = 0
        while t < horizon_s:
            on_dur = float(rng.exponential(mean_on))
            t_on, t_end = t, t + on_dur

            if agile:
                # One activation, many rows.  Nothing downstream knows.
                seg = t_on
                while seg < t_end:
                    seg_end = min(seg + em.hop_dwell_s, t_end)
                    if em.hop_pattern == "sequential":
                        ch = em.hop_set[hop_i % len(em.hop_set)]
                        hop_i += 1
                    else:
                        ch = em.hop_set[int(rng.integers(0, len(em.hop_set)))]
                    rows.append(
                        (em.emitter_id, act, 0, seg, seg_end, ch, ch + em.n_channels,
                         em.snr_db, em.priority, _w_for(mission_w, ch, em.n_channels, em.w_p))
                    )
                    sigmas.append(em.snr_sigma_db)
                    seg = seg_end
            else:
                rows.append(
                    (em.emitter_id, act, 0, t_on, t_end, em.channel, em.ch_hi,
                     em.snr_db, em.priority, em.w_p)
                )
                sigmas.append(em.snr_sigma_db)

            act += 1
            t = t_end + float(rng.exponential(mean_off))

    if not rows:
        return EMPTY_BURSTS.copy()

    bursts = np.array(rows, dtype=BURST_DTYPE)
    # Per-burst log-normal shadowing: the same emitter is not equally audible on
    # every transmission.  Drawn from its OWN stream so that changing the
    # shadowing model does not reshuffle the ON/OFF timeline.
    sig = np.asarray(sigmas, dtype=np.float64)
    bursts["snr_db"] += rng_shadow.standard_normal(bursts.size) * sig

    # Canonical order: chronological, ties broken by emitter.  `burst_id` is then
    # the row index, so a burst table can be compared byte-for-byte across runs.
    order = np.lexsort((bursts["emitter_id"], bursts["activation_id"], bursts["t_on"]))
    bursts = bursts[order]
    bursts["burst_id"] = np.arange(bursts.size, dtype=np.int32)
    return bursts


def _w_for(mission_w, ch: int, n_ch: int, fallback: float) -> float:
    """Weight of the channel this particular hop landed on (not the home channel)."""
    if mission_w is None:
        return fallback
    return float(np.max(mission_w[ch: ch + n_ch]))
