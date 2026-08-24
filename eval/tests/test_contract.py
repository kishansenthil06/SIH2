"""Phase-0 contract tests.  These must pass before any lane starts work."""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from sim.config import (
    BW_CANDIDATES_MHZ,
    SCENARIOS,
    build_grid,
    build_mission,
    load_config,
    validate_config,
)
from sim.contract import (
    ChannelGrid,
    GridError,
    Obs,
    ScanEnv,
    Scan,
    Sleep,
    null_obs,
)
from sim.stub_env import StubEnv


class TestChannelGrid(unittest.TestCase):
    def setUp(self):
        self.g = ChannelGrid()

    def test_round_trip_all_positions(self):
        """action_for -> channels_for is exact for every legal (k_lo, n)."""
        for n in BW_CANDIDATES_MHZ:
            for k in range(0, self.g.n_channels - n + 1):
                a = self.g.action_for(k, n, 0.01)
                cs = self.g.channels_for(a.f_center_hz, a.bw_hz)
                self.assertEqual(cs[0], k)
                self.assertEqual(len(cs), n)
                self.assertEqual(cs[-1], k + n - 1)

    def test_misaligned_center_raises(self):
        with self.assertRaises(GridError):
            self.g.channels_for(2.0e9 + 0.4e6, 1e6)

    def test_non_integer_bandwidth_raises(self):
        with self.assertRaises(GridError):
            self.g.channels_for(2.0e9 + 0.75e6, 1.5e6)

    def test_out_of_band_raises(self):
        with self.assertRaises(GridError):
            self.g.channels_for(1.9e9, 1e6)
        with self.assertRaises(GridError):
            self.g.channels_for(self.g.f_stop_hz, 1e6)

    def test_action_for_rejects_out_of_range(self):
        with self.assertRaises(GridError):
            self.g.action_for(195, 20, 0.01)
        with self.assertRaises(GridError):
            self.g.action_for(-1, 1, 0.01)

    def test_odd_and_even_widths_are_exact(self):
        """Odd n centres on a channel, even n on a boundary; both exact."""
        for n in (1, 2, 5, 10, 20):
            a = self.g.action_for(7, n, 0.01)
            self.assertEqual(
                (a.f_center_hz - a.bw_hz / 2.0 - self.g.f_start_hz) % self.g.channel_bw_hz,
                0.0,
            )


class TestFrozenTypes(unittest.TestCase):
    def test_scan_is_frozen(self):
        s = Scan(2.0e9, 1e6, 0.01)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.dwell_s = 0.02

    def test_obs_is_frozen(self):
        o = null_obs()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            o.t = 1.0

    def test_null_obs_shape(self):
        o = null_obs()
        self.assertEqual(o.step_index, -1)
        self.assertEqual(o.duration_s, 0.0)
        self.assertFalse(o.done)
        self.assertEqual(len(o.detections), 0)
        self.assertEqual(o.scanned_channels.dtype, np.int32)


class TestConfigs(unittest.TestCase):
    def test_all_scenarios_load(self):
        for s in SCENARIOS:
            cfg = load_config(s)
            self.assertEqual(cfg["name"], s)
            self.assertIsInstance(cfg["grid"]["f_start_hz"], float)
            self.assertIsInstance(cfg["horizon_s"], float)

    def test_energy_timing_consistency_enforced(self):
        """L_f must equal L_d/f_slew or the headline energy number is a lie."""
        cfg = load_config("sparse")
        cfg["energy"]["L_f_j_per_hz"] = 3.0e-11
        with self.assertRaises(Exception):
            validate_config(cfg)

    def test_mission_expansion(self):
        cfg = load_config("sparse")
        m = build_mission(cfg)
        self.assertEqual(m.priority[500], 1)    # threat band 400-600
        self.assertEqual(m.priority[1300], 2)   # 1200-1500
        self.assertEqual(m.priority[50], 3)     # catch-all
        self.assertAlmostEqual(m.w[500], 1.000)
        self.assertAlmostEqual(m.w[1300], 0.300)
        self.assertAlmostEqual(m.w[50], 0.100)

    def test_deadlines(self):
        m = build_mission(load_config("sparse"))
        d = m.deadline_for()
        # Deadlines were reset so total mandated dwell fits the budget; the old
        # {1: 0.5} was infeasible by 20x.  See DESIGN.md 11.6.
        self.assertAlmostEqual(d[500], 30.0)
        self.assertAlmostEqual(d[1300], 8.0)
        self.assertAlmostEqual(d[50], 20.0)

    def test_config_hash_is_stable_and_discriminating(self):
        a = load_config("sparse")["config_hash"]
        b = load_config("sparse")["config_hash"]
        c = load_config("dense")["config_hash"]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_bad_priority_band_rejected(self):
        cfg = load_config("sparse")
        cfg["mission"]["priority_bands"].append({"ch_lo": 1900, "ch_hi": 5000, "priority": 1})
        with self.assertRaises(Exception):
            validate_config(cfg)


class TestStubEnv(unittest.TestCase):
    def test_satisfies_protocol(self):
        self.assertIsInstance(StubEnv("sparse"), ScanEnv)

    def test_no_truth_exposed(self):
        """The stub must not tempt anyone into reading ground truth."""
        e = StubEnv("sparse")
        for attr in ("truth", "truth_bursts", "truth_power", "emitters"):
            self.assertFalse(hasattr(e, attr), f"StubEnv should not expose {attr}")

    def test_energy_identity(self):
        e = StubEnv("sparse")
        e.reset("sparse", 1)
        a = e.grid.action_for(10, 5, 0.01)
        df = abs(a.f_center_hz - e.grid.center_hz(0))
        o = e.step(a)
        self.assertAlmostEqual(o.energy_cost, e.L_0 + e.L_d * 0.01 + e.L_f * df, places=15)

    def test_same_frequency_rescan_is_free_to_retune(self):
        e = StubEnv("sparse")
        e.reset("sparse", 1)
        a = e.grid.action_for(10, 5, 0.01)
        e.step(a)
        o = e.step(a)
        self.assertEqual(o.info["t_retune"], 0.0)

    def test_sleep_cost_and_clock(self):
        e = StubEnv("sparse")
        e.reset("sparse", 1)
        o = e.step(Sleep(0.1))
        self.assertAlmostEqual(o.energy_cost, e.L_sleep * 0.1, places=15)
        self.assertAlmostEqual(o.duration_s, 0.1, places=12)

    def test_episode_terminates(self):
        e = StubEnv("sparse")
        o = e.reset("sparse", 0)
        steps = 0
        while not o.done and steps < 200_000:
            o = e.step(e.grid.action_for(steps % 196, 5, 0.01))
            steps += 1
        self.assertTrue(o.done)


if __name__ == "__main__":
    unittest.main(verbosity=2)
