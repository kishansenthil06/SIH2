"""FROZEN AT PHASE 0 -- a minimal environment satisfying the ScanEnv protocol.

Agents B (brain), C (proof) and D (learn/show) build against this so that none of
them is blocked on agent A finishing the real simulator.  This is what makes the
CP1 integration checkpoint an event rather than a disaster.

It is deliberately simple but NOT pure noise: timing and energy accounting are
exactly the real ones, and three hard-coded emitters blink on and off, so a policy
built against it exercises real code paths and a belief tracker has something to
actually learn.  It holds no burst table and offers no truth() -- swapping in the
real `World.agent_view()` at CP1 must be a one-line change.
"""
from __future__ import annotations

import numpy as np

from sim.config import build_grid, build_mission, load_config
from sim.contract import (
    ChannelGrid,
    Detection,
    Mission,
    Obs,
    Scan,
    Sleep,
    null_obs,
)

# (channel, mean_on_s, mean_off_s, snr_db)
_STUB_EMITTERS: tuple[tuple[int, float, float, float], ...] = (
    (50, 0.20, 8.0, -19.0),   # priority-1 band, rare and weak
    (130, 0.60, 3.0, -15.0),  # priority-2 band, moderate
    (17, 2.00, 3.0, -10.0),   # routine, strong and mostly on
)


class StubEnv:
    """Satisfies `sim.contract.ScanEnv`.  No ground truth is exposed."""

    def __init__(self, scenario: str | dict = "sparse"):
        self.cfg = load_config(scenario)
        self.grid: ChannelGrid = build_grid(self.cfg)
        self.mission: Mission = build_mission(self.cfg)
        self.horizon_s: float = float(self.cfg["horizon_s"])

        rx, en = self.cfg["receiver"], self.cfg["energy"]
        self.pfa = float(rx["pfa"])
        self.t_settle_s = float(rx["t_settle_s"])
        self.f_slew = float(rx["f_slew_hz_per_s"])
        self.L_d = float(en["L_d_w"])
        self.L_0 = float(en["L_0_j"])
        self.L_f = float(en["L_f_j_per_hz"])
        self.L_sleep = float(en["L_sleep_w"])
        self.budget_j = float(en["budget_j"])

        self._rng = np.random.default_rng(0)
        self.t = 0.0
        self.f_last_hz = self.grid.center_hz(0)
        self.energy_total = 0.0
        self.step_index = -1

    # ------------------------------------------------------------------ api
    def reset(self, scenario: str | dict, seed: int) -> Obs:
        if scenario is not None and scenario != self.cfg.get("name"):
            self.__init__(scenario)
        self._rng = np.random.default_rng(seed)
        self.t = 0.0
        self.f_last_hz = self.grid.center_hz(0)
        self.energy_total = 0.0
        self.step_index = -1
        # Random phase per emitter so different seeds are genuinely different.
        self._phase = self._rng.random(len(_STUB_EMITTERS))
        return null_obs()

    def step(self, action) -> Obs:
        t_start = self.t
        self.step_index += 1

        if isinstance(action, Sleep):
            dt = max(0.0, float(action.dt_s))
            dt = min(dt, max(0.0, self.horizon_s - self.t))
            self.t += dt
            cost = self.L_sleep * dt
            self.energy_total += cost
            return Obs(
                t=self.t,
                action=Sleep(dt),
                detections=(),
                energy_cost=cost,
                t_start=t_start,
                scanned_channels=np.empty(0, dtype=np.int32),
                energy_total=self.energy_total,
                step_index=self.step_index,
                done=self._done(),
                info={"kind": "sleep"},
            )

        if not isinstance(action, Scan):
            raise TypeError(f"expected Scan or Sleep, got {type(action)!r}")

        chans = self.grid.channels_for(action.f_center_hz, action.bw_hz)
        df = abs(action.f_center_hz - self.f_last_hz)
        t_retune = 0.0 if df == 0.0 else self.t_settle_s + df / self.f_slew

        dwell = float(action.dwell_s)
        remaining = max(0.0, self.horizon_s - (self.t + t_retune))
        dwell = min(dwell, remaining)
        executed = Scan(action.f_center_hz, action.bw_hz, dwell, action.gain_db)

        t0 = self.t + t_retune
        self.t = t0 + dwell
        self.f_last_hz = action.f_center_hz

        cost = self.L_0 + self.L_d * dwell + self.L_f * df
        self.energy_total += cost

        dets = self._detect(chans, t0, dwell, action.bw_hz)
        return Obs(
            t=self.t,
            action=executed,
            detections=dets,
            energy_cost=cost,
            t_start=t_start,
            scanned_channels=chans,
            energy_total=self.energy_total,
            step_index=self.step_index,
            done=self._done(),
            info={"kind": "scan", "t_retune": t_retune},
        )

    # -------------------------------------------------------------- internals
    def _done(self) -> bool:
        return self.t >= self.horizon_s or self.energy_total >= self.budget_j

    def _active(self, i: int, t: float) -> bool:
        """Deterministic square wave per emitter -- cheap, and genuinely periodic
        so a belief tracker built against the stub has real structure to find."""
        ch, on, off, _snr = _STUB_EMITTERS[i]
        period = on + off
        phase = (t / period + self._phase[i]) % 1.0
        return phase < on / period

    def _detect(self, chans, t0: float, dwell: float, bw_hz: float):
        """Crude but structurally correct: Pd rises with dwell, Pfa otherwise."""
        out = []
        for c in chans:
            p = self.pfa
            snr_rep = self._rng.uniform(-24.0, -19.0)
            for i, (ech, _on, _off, esnr) in enumerate(_STUB_EMITTERS):
                if int(c) == ech and self._active(i, t0 + dwell * 0.5):
                    s = 10.0 ** (esnr / 10.0)
                    n = dwell * 1.0e6
                    p = float(np.clip(0.5 * (1.0 + np.tanh(0.5 * (np.sqrt(n) * s - 3.0))), self.pfa, 1.0))
                    snr_rep = esnr + self._rng.normal(0.0, 1.5)
                    break
            if self._rng.random() < p:
                out.append(
                    Detection(
                        channel=int(c),
                        f_hz=float(self.grid.center_hz(int(c))),
                        bw_hz=self.grid.channel_bw_hz,
                        snr_db=float(snr_rep),
                    )
                )
        return tuple(out)
