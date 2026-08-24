"""FROZEN AT PHASE 0 -- the interface every other module is built against.

This module is the ONLY thing in `sim/` that `agent/` and `app/` are permitted to
import.  It contains pure types and grid arithmetic; it holds no ground truth and
has no reference path to a World.  See `eval/tests/test_firewall.py`.

Changing anything here after the freeze requires an explicit re-freeze
announcement, not an edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------- unit aliases
# Documentation only -- Python does not enforce these, but every signature in the
# project uses them so units are never ambiguous at a call site.
HZ = float      # frequency, hertz
SEC = float     # time, seconds of SIM CLOCK (never wall clock)
JOULE = float   # energy, joules


class GridError(ValueError):
    """A Scan that does not align to the channel grid, or falls outside the band."""


class FirewallViolation(RuntimeError):
    """Agent-side code attempted to read ground truth."""


# --------------------------------------------------------------------- actions
@dataclass(frozen=True, slots=True)
class Scan:
    """Tune the receiver and listen.

    The agent must NOT construct this directly -- it calls
    `ChannelGrid.action_for`, which guarantees grid alignment by construction.
    """

    f_center_hz: HZ
    bw_hz: HZ
    dwell_s: SEC
    gain_db: float = 0.0


@dataclass(frozen=True, slots=True)
class Sleep:
    """Do nothing for `dt_s`.  The only action that meaningfully saves energy."""

    dt_s: SEC


Action = Scan | Sleep


# ---------------------------------------------------------------- observations
@dataclass(frozen=True, slots=True)
class Detection:
    """One reported detection.  May be a false alarm -- the agent cannot tell."""

    channel: int        # grid index
    f_hz: HZ            # grid.center_hz(channel)
    bw_hz: HZ           # grid.channel_bw_hz
    snr_db: float       # ESTIMATED post-detection SNR (noisy), never truth


@dataclass(frozen=True, slots=True)
class Obs:
    """Everything the agent learns from one step.

    The first four fields are those frozen by the source doc section 7; the rest
    were agreed at the same freeze and are purely additive.
    """

    t: SEC                              # sim clock AFTER the action completed
    action: Action                      # what was ACTUALLY executed (may be truncated)
    detections: tuple[Detection, ...]   # sorted by channel ascending
    energy_cost: JOULE                  # energy for THIS step only

    t_start: SEC                        # clock when the action began
    scanned_channels: np.ndarray        # int32 (k,); empty for Sleep
    energy_total: JOULE                 # cumulative since reset
    step_index: int                     # 0-based; reset() returns -1
    done: bool                          # horizon reached or budget exhausted
    info: dict = field(default_factory=dict)   # diagnostics; agent MUST ignore

    @property
    def duration_s(self) -> SEC:
        return self.t - self.t_start


def null_obs() -> Obs:
    """The Obs returned by reset(): a zero-length Sleep that cost nothing.

    Having reset() return a well-formed Obs means Policy.act() needs no special
    first-step branch.
    """
    return Obs(
        t=0.0,
        action=Sleep(0.0),
        detections=(),
        energy_cost=0.0,
        t_start=0.0,
        scanned_channels=np.empty(0, dtype=np.int32),
        energy_total=0.0,
        step_index=-1,
        done=False,
        info={},
    )


# -------------------------------------------------------------------- the grid
@dataclass(frozen=True, slots=True)
class ChannelGrid:
    """Discrete 1 MHz channel grid; the only mapping to continuous frequency.

    Channel k occupies [f_start + k*cbw, f_start + (k+1)*cbw).
    """

    f_start_hz: HZ = 2.000e9
    n_channels: int = 200
    channel_bw_hz: HZ = 1.0e6

    def center_hz(self, k):
        return self.f_start_hz + (np.asarray(k) + 0.5) * self.channel_bw_hz

    @property
    def f_stop_hz(self) -> HZ:
        return self.f_start_hz + self.n_channels * self.channel_bw_hz

    @property
    def span_hz(self) -> HZ:
        return self.n_channels * self.channel_bw_hz

    def channels_for(self, f_center_hz: HZ, bw_hz: HZ) -> np.ndarray:
        """Channels covered by a scan.  Single source of truth for env + evaluator.

        Raises GridError on misalignment or out-of-band, so an illegal action can
        never silently do something reasonable-looking.
        """
        n = bw_hz / self.channel_bw_hz
        k_lo_f = (f_center_hz - bw_hz / 2.0 - self.f_start_hz) / self.channel_bw_hz
        n_i = int(round(n))
        k_lo = int(round(k_lo_f))
        if abs(n - n_i) > 1e-6:
            raise GridError(
                f"bw_hz={bw_hz!r} is not an integer multiple of "
                f"channel_bw_hz={self.channel_bw_hz!r}"
            )
        if abs(k_lo_f - k_lo) > 1e-6:
            raise GridError(
                f"f_center_hz={f_center_hz!r} with bw_hz={bw_hz!r} does not align to "
                f"the channel grid (lower edge falls at channel {k_lo_f!r})"
            )
        if n_i < 1:
            raise GridError(f"bw_hz={bw_hz!r} covers no channels")
        if k_lo < 0 or k_lo + n_i > self.n_channels:
            raise GridError(
                f"scan covers channels [{k_lo}, {k_lo + n_i}) which is outside "
                f"[0, {self.n_channels})"
            )
        return np.arange(k_lo, k_lo + n_i, dtype=np.int32)

    def action_for(
        self, k_lo: int, n_ch: int, dwell_s: SEC, gain_db: float = 0.0
    ) -> Scan:
        """Canonical Scan constructor.  The agent uses ONLY this.

        Odd n_ch puts f_center on a channel centre, even n_ch on a channel
        boundary -- both exactly representable in float64.
        """
        if n_ch < 1:
            raise GridError(f"n_ch={n_ch} must be >= 1")
        if k_lo < 0 or k_lo + n_ch > self.n_channels:
            raise GridError(
                f"channels [{k_lo}, {k_lo + n_ch}) outside [0, {self.n_channels})"
            )
        bw = n_ch * self.channel_bw_hz
        f_lo = self.f_start_hz + k_lo * self.channel_bw_hz
        return Scan(f_lo + bw / 2.0, bw, dwell_s, gain_db)


# --------------------------------------------------------------------- mission
@dataclass(frozen=True, slots=True)
class Mission:
    """Agent-visible tasking.  Priorities come from intel, NEVER from truth.

    This is what keeps `w_b` on the agent side of the firewall: the agent is told
    "channels 40-60 are the threat band", not "emitter 3 is priority 1".
    """

    priority: np.ndarray      # int32 (N,), value in {1,2,3}; 0 = not tasked
    w: np.ndarray             # float64 (N,), priority weight in JOULES
    deadlines_s: dict         # {priority: max revisit gap, seconds}
    watch_list: np.ndarray    # int32 (k,), channels with their own deadline
    watch_deadline_s: SEC

    def deadline_for(self) -> np.ndarray:
        """(N,) float64 per-channel hard revisit deadline; inf where untasked."""
        out = np.full(self.priority.shape, np.inf, dtype=np.float64)
        for prio, d in self.deadlines_s.items():
            out[self.priority == int(prio)] = float(d)
        if self.watch_list.size:
            out[self.watch_list] = np.minimum(
                out[self.watch_list], self.watch_deadline_s
            )
        return out


# -------------------------------------------------------------------- protocol
@runtime_checkable
class ScanEnv(Protocol):
    """What the agent may assume about its environment.  Note: no truth()."""

    grid: ChannelGrid
    horizon_s: SEC
    mission: Mission

    def reset(self, scenario: str | dict, seed: int) -> Obs: ...
    def step(self, action: Action) -> Obs: ...
