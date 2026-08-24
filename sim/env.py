"""The world: clock, energy ledger, ground truth, and the agent-facing firewall.

`World` owns everything.  `World.agent_view()` hands out an `AgentEnv` that owns
*nothing* -- see the class docstring for why that is a structural guarantee
rather than a convention.

Three properties this module exists to guarantee:

* **Truth depends only on `(scenario, seed)`.**  The burst table is generated in
  full at `reset()` from its own RNG stream, before the policy has acted even
  once.  No action can perturb it, so two policies on the same seed are compared
  on literally the same world.
* **Detector noise depends only on `(seed, step_index)`.**  Counter-based Philox
  means two policies issuing the same scan at the same step index get the same
  coin flips.  That is a variance-reduction trick (common random numbers), and
  it materially tightens the confidence interval on the headline energy ratio.
* **Timing and energy cannot drift apart.**  Both are computed from `|df|` and
  the executed dwell in the same few lines; `sim/config.py` already asserts
  `L_f == L_d / f_slew`, so `E = L_0 + L_d*dwell + L_f*|df|` holds to 1e-15.
"""
from __future__ import annotations

import sys

import numpy as np

from sim.channel import rasterize_occupancy, rasterize_power, window_rho
from sim.config import build_grid, build_mission, load_config
from sim.contract import (
    ChannelGrid,
    Detection,
    FirewallViolation,
    Mission,
    Obs,
    Scan,
    Sleep,
    null_obs,
)
from sim.emitters import EMPTY_BURSTS, build_emitters, generate_bursts
from sim.receiver import Receiver

# How far up the stack to look for an agent-side caller.  12 frames comfortably
# covers policy -> scheduler -> belief -> helper chains without making the check
# expensive; truth() is called once per episode by the evaluator, not per step.
_STACK_DEPTH = 12
_FORBIDDEN_ROOTS = ("agent", "app")

# Per-step Philox substream stride.  CORRECTION to the brief, which specified
# `counter=step_index` directly: Philox is a counter-based generator that
# ADVANCES its counter as it emits, so `counter=i` and `counter=i+1` produce
# streams offset by a single 4x64 block -- i.e. almost entirely overlapping.
# A 200-channel scan draws 600 doubles per step, so with a stride of 1 the
# uniform used for channel 0 at step i+1 is literally the uniform used for
# channel 4 at step i, and one unlucky draw becomes ~196 consecutive false
# alarms.  Measured: 350 false alarms where 200 +/- 42 were expected.
# 2**64 guarantees the substreams cannot meet (a step would have to draw 7e19
# values first) while keeping the noise a pure function of (seed, step_index),
# which is the property the common-random-numbers comparison actually needs.
_PHILOX_STRIDE = 1 << 64


def _forbid_agent_callers() -> None:
    """Raise if anything in `agent/` or `app/` is anywhere in the call stack.

    This is the RUNTIME third of the firewall (DESIGN.md section 2); the other
    two are structural (`AgentEnv.__slots__`) and static (the AST scan in
    `eval/tests/test_firewall.py`).  Belt, braces, and a second pair of braces:
    a firewall breach silently inflates every number in the write-up, so it is
    worth three independent mechanisms.
    """
    frame = sys._getframe(1)
    for _ in range(_STACK_DEPTH):
        frame = frame.f_back
        if frame is None:
            return
        mod = frame.f_globals.get("__name__", "")
        root = mod.split(".", 1)[0]
        if root in _FORBIDDEN_ROOTS:
            raise FirewallViolation(
                f"ground truth was requested from {mod!r}; `agent/` and `app/` "
                f"may import only `sim.contract`"
            )


class World:
    """The full simulator.  Never hand one of these to a policy."""

    def __init__(self, scenario: str | dict = "sparse", seed: int = 0):
        self.cfg: dict = {}
        self._scenario_key = None
        self.reset(scenario, seed)

    # ------------------------------------------------------------------ setup
    def _configure(self, scenario: str | dict) -> None:
        self.cfg = load_config(scenario)
        self.grid: ChannelGrid = build_grid(self.cfg)
        self.mission: Mission = build_mission(self.cfg)
        self.horizon_s: float = float(self.cfg["horizon_s"])
        self.receiver = Receiver.from_config(self.cfg)

        rx, en = self.cfg["receiver"], self.cfg["energy"]
        self.pfa = float(rx["pfa"])
        self.t_settle_s = float(rx["t_settle_s"])
        self.f_slew = float(rx["f_slew_hz_per_s"])
        self.L_d = float(en["L_d_w"])
        self.L_0 = float(en["L_0_j"])
        self.L_f = float(en["L_f_j_per_hz"])
        self.L_sleep = float(en["L_sleep_w"])
        self.budget_j = float(en["budget_j"])

    def reset(self, scenario: str | dict | None = None, seed: int = 0) -> Obs:
        """Rebuild the world.  Returns `contract.null_obs()`.

        The four RNG streams are spawned from one `SeedSequence` so they are
        provably independent, and each has exactly one job.  In particular
        `rng_emitters` is consumed entirely *inside this method*, which is what
        makes truth independent of the policy.
        """
        if scenario is not None:
            key = scenario if isinstance(scenario, str) else id(scenario)
            if not self.cfg or key != self._scenario_key or isinstance(scenario, dict):
                self._configure(scenario)
                self._scenario_key = key
        elif not self.cfg:
            self._configure("sparse")

        self.seed = int(seed)
        ss = np.random.SeedSequence(self.seed)
        s_em, s_noise, s_shadow, s_policy = ss.spawn(4)
        self.rng_emitters = np.random.default_rng(s_em)
        # Reserved by design: per-step detector noise uses counter-based Philox
        # (below) so that it is addressable by step index rather than by
        # consumption order.  Kept in the spawn so the stream layout is stable.
        self.rng_noise = np.random.default_rng(s_noise)
        self.rng_shadowing = np.random.default_rng(s_shadow)
        self.rng_policy = np.random.default_rng(s_policy)

        self.emitters = build_emitters(self.cfg, self.rng_emitters)
        self._bursts = generate_bursts(
            self.emitters,
            self.horizon_s,
            self.rng_emitters,
            self.rng_shadowing,
            mission_w=self.mission.w,
        )

        self.t = 0.0
        self.f_last_hz = float(self.grid.center_hz(0))
        self.energy_total = 0.0
        self.step_index = -1
        self._truth_cache: dict = {}
        return null_obs()

    # ------------------------------------------------------------------- step
    def step(self, action) -> Obs:
        t_start = self.t
        self.step_index += 1

        if isinstance(action, Sleep):
            return self._step_sleep(action, t_start)
        if isinstance(action, Scan):
            return self._step_scan(action, t_start)
        raise TypeError(f"expected Scan or Sleep, got {type(action)!r}")

    def _step_sleep(self, action: Sleep, t_start: float) -> Obs:
        # Truncate rather than reject: a policy is allowed to ask for more than
        # the horizon has left, and `Obs.action` reports what actually happened.
        dt = min(max(0.0, float(action.dt_s)), max(0.0, self.horizon_s - self.t))
        self.t += dt
        cost = self.L_sleep * dt
        self.energy_total += cost
        # The VCO stays parked while asleep, so the next scan pays retune from
        # wherever we last tuned -- sleeping is never a free way to move.
        return Obs(
            t=self.t,
            action=Sleep(dt),
            detections=(),
            energy_cost=cost,
            t_start=t_start,
            scanned_channels=np.empty(0, dtype=np.int32),
            energy_total=self.energy_total,
            step_index=self.step_index,
            done=self.done,
            info={"kind": "sleep", "t_retune": 0.0},
        )

    def _step_scan(self, action: Scan, t_start: float) -> Obs:
        chans = self.grid.channels_for(action.f_center_hz, action.bw_hz)
        df = abs(float(action.f_center_hz) - self.f_last_hz)
        t_retune = 0.0 if df == 0.0 else self.t_settle_s + df / self.f_slew

        t0 = self.t + t_retune
        dwell = min(float(action.dwell_s), max(0.0, self.horizon_s - t0))
        executed = Scan(action.f_center_hz, action.bw_hz, dwell, action.gain_db)

        self.t = t0 + dwell
        self.f_last_hz = float(action.f_center_hz)

        # Detection integrates truth over [t0, t0+dwell) ONLY: the receiver is
        # deaf while the synthesiser settles, which is precisely why hopping is
        # expensive in mission time as well as in joules.
        rho = window_rho(self._bursts, t0, self.t, self.grid.n_channels)[chans]

        rng = np.random.Generator(
            np.random.Philox(key=self.seed, counter=self.step_index * _PHILOX_STRIDE)
        )
        det, snr_rep = self.receiver.observe(
            rho, dwell, float(action.bw_hz), rng, gain_db=float(action.gain_db)
        )

        mult = self.receiver.energy_mult(float(action.gain_db))
        cost = (self.L_0 + self.L_d * dwell + self.L_f * df) * mult
        self.energy_total += cost

        hits = chans[det]
        detections = tuple(
            Detection(
                channel=int(c),
                f_hz=float(self.grid.center_hz(int(c))),
                bw_hz=self.grid.channel_bw_hz,
                snr_db=float(s),
            )
            for c, s in zip(hits, snr_rep[det])
        )
        return Obs(
            t=self.t,
            action=executed,
            detections=detections,
            energy_cost=cost,
            t_start=t_start,
            scanned_channels=chans,
            energy_total=self.energy_total,
            step_index=self.step_index,
            done=self.done,
            info={"kind": "scan", "t_retune": t_retune, "t_dwell_start": t0},
        )

    @property
    def done(self) -> bool:
        return self.t >= self.horizon_s or self.energy_total >= self.budget_j

    # ------------------------------------------------------------------ truth
    # EVALUATOR ONLY.  Each of these calls `_forbid_agent_callers()`.
    def truth_bursts(self) -> np.ndarray:
        """The canonical burst table (BURST_DTYPE).  A read-only view."""
        _forbid_agent_callers()
        v = self._bursts.view()
        v.flags.writeable = False
        return v

    def truth(self, dt_s: float = 1e-3) -> np.ndarray:
        """(T_bins, n_channels) bool occupancy raster.  Cached per episode."""
        _forbid_agent_callers()
        key = ("occ", float(dt_s))
        if key not in self._truth_cache:
            self._truth_cache[key] = rasterize_occupancy(
                self._bursts, float(dt_s), self._n_bins(dt_s), self.grid.n_channels
            )
        return self._truth_cache[key]

    def truth_power(self, dt_s: float = 1e-3) -> np.ndarray:
        """(T_bins, n_channels) float32 raster of summed LINEAR SNR.

        The `oracle` baseline reads this.  Label it a REFERENCE CEILING, not an
        optimum -- it is myopic over one action (DESIGN.md section 6).
        """
        _forbid_agent_callers()
        key = ("pow", float(dt_s))
        if key not in self._truth_cache:
            self._truth_cache[key] = rasterize_power(
                self._bursts, float(dt_s), self._n_bins(dt_s), self.grid.n_channels
            )
        return self._truth_cache[key]

    def _n_bins(self, dt_s: float) -> int:
        return int(np.ceil(self.horizon_s / float(dt_s) - 1e-9))

    # --------------------------------------------------------------- firewall
    def agent_view(self) -> "AgentEnv":
        return AgentEnv(self)


class AgentEnv:
    """The ONLY environment object a policy ever sees.

    Why this is a structural guarantee and not a naming convention:

    * `__slots__` means there is no `__dict__`, so nothing can be attached later
      and no attribute exists beyond the six named here;
    * the two callables stored are **bound methods**.  A bound method exposes
      `__self__`, but nothing on the documented `ScanEnv` surface reaches it, and
      there is no `self.world`, `self._world` or `self.bursts` to walk;
    * the remaining four attributes are the grid, the horizon, the mission and
      the candidate action sets -- all of which the agent is *supposed* to know,
      and none of which contains a reference to a burst table.

    So the shortest path from a policy to ground truth is a deliberate
    `__self__` dereference, which the AST scan in `eval/tests/test_firewall.py`
    catches and the runtime stack check in `_forbid_agent_callers` catches
    again.
    """

    __slots__ = ("_step", "_reset", "grid", "horizon_s", "mission", "action_space")

    def __init__(self, world: World):
        self._step, self._reset = world.step, world.reset   # BOUND METHODS ONLY
        self.grid, self.horizon_s, self.mission = (
            world.grid,
            world.horizon_s,
            world.mission,
        )
        a = world.cfg.get("agent", {})
        self.action_space = {
            "bw_candidates_mhz": tuple(a.get("bw_candidates_mhz", ())),
            "dwell_candidates_ms": tuple(a.get("dwell_candidates_ms", ())),
            "sleep_candidates_ms": tuple(a.get("sleep_candidates_ms", ())),
        }

    def reset(self, scenario: str | dict, seed: int) -> Obs:
        return self._reset(scenario, seed)

    def step(self, action) -> Obs:
        return self._step(action)


def make_world(cfg: str | dict, seed: int = 0) -> World:
    """Preferred constructor.  `cfg` is a scenario name, a path, or a dict."""
    return World(cfg, seed)


__all__ = ["World", "AgentEnv", "make_world", "EMPTY_BURSTS"]
