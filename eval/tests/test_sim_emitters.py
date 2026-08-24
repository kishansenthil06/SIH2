"""Ground truth: determinism, policy-independence, CTMC statistics, hopping.

These are the tests that make every later comparison meaningful.  If truth is not
reproducible from `(scenario, seed)` alone, then "policy A used half the energy of
policy B" is not a measurement, it is a coincidence.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from sim.channel import window_rho
from sim.config import build_mission, load_config
from sim.contract import Sleep
from sim.emitters import BURST_DTYPE, Emitter, build_emitters, generate_bursts
from sim.env import make_world


def base_cfg(**over) -> dict:
    """A minimal VALID config dict.  Kept tiny so tests stay well under 2 s."""
    cfg = dict(
        name="unit",
        horizon_s=5.0,
        grid=dict(f_start_hz=2.0e9, n_channels=20, channel_bw_hz=1.0e6),
        receiver=dict(
            pfa=1.0e-3, t_settle_s=0.5e-3, f_slew_hz_per_s=50.0e9,
            bw_penalty_db_per_octave=1.0, snr_est_sigma_db=1.5,
            gain_enabled=False, gain_db_high=10.0, gain_nf_improvement_db=6.0,
            gain_energy_mult=1.6, gain_saturation_snr_db=-5.0,
            gain_fa_mult_on_saturation=10.0,
        ),
        energy=dict(L_d_w=1.0, L_0_j=2.0e-3, L_f_j_per_hz=2.0e-11,
                    L_sleep_w=0.01, budget_j=1.0e9),
        mission=dict(
            priority_bands=[dict(ch_lo=0, ch_hi=5, priority=1),
                            dict(ch_lo=0, ch_hi=20, priority=3)],
            weights={"1": 0.100, "3": 0.010},
            deadlines_s={"1": 0.5, "3": 10.0},
            watch_list=[], watch_deadline_s=0.3,
        ),
        emitters=[dict(kind="fixed", count=3, channel_range=[0, 20], n_channels=1,
                       snr_db=[-12.0, -8.0], priority=3,
                       mean_on_s=0.5, mean_off_s=1.5, snr_sigma_db=1.0)],
        agent=dict(prior_pi_on=0.05, prior_lam_sum=1.0,
                   bw_candidates_mhz=[1, 2, 5, 10, 20],
                   dwell_candidates_ms=[1, 10], sleep_candidates_ms=[10]),
    )
    cfg.update(copy.deepcopy(over))
    return cfg


class TestEmitterExpansion(unittest.TestCase):
    def test_sojourn_rate_conversion(self):
        """mean_on_s -> lam_off and mean_off_s -> lam_on.  The classic sign error."""
        cfg = base_cfg()
        cfg["emitters"][0].update(mean_on_s=0.25, mean_off_s=4.0)
        ems = build_emitters(load_config(cfg), np.random.default_rng(0))
        for e in ems:
            self.assertAlmostEqual(e.lam_off, 1.0 / 0.25)
            self.assertAlmostEqual(e.lam_on, 1.0 / 4.0)
            self.assertAlmostEqual(1.0 / e.lam_off, 0.25, msg="mean ON time")
            self.assertAlmostEqual(1.0 / e.lam_on, 4.0, msg="mean OFF time")
            self.assertAlmostEqual(e.pi_on, 0.25 / (0.25 + 4.0))
            self.assertAlmostEqual(e.lam_sum, e.lam_on + e.lam_off)

    def test_wp_comes_from_the_mission_weight_of_the_occupied_channel(self):
        """The agent's objective and the evaluator's metric are the SAME number."""
        cfg = load_config(base_cfg())
        mission = build_mission(cfg)
        ems = build_emitters(cfg, np.random.default_rng(3))
        for e in ems:
            self.assertEqual(e.w_p, float(np.max(mission.w[e.channel: e.ch_hi])))
            self.assertIn(e.w_p, (0.100, 0.010))

        bursts = generate_bursts(ems, 5.0, np.random.default_rng(3),
                                 np.random.default_rng(4), mission_w=mission.w)
        for row in bursts:
            self.assertEqual(row["w_p"], float(np.max(mission.w[row["ch_lo"]: row["ch_hi"]])))

    def test_emitters_respect_channel_range_and_block_width(self):
        cfg = base_cfg()
        cfg["emitters"] = [dict(kind="pulsed", count=6, channel_range=[10, 18],
                                n_channels=2, snr_db=[-17.0, -14.0], priority=2,
                                mean_on_s=0.6, mean_off_s=3.0, snr_sigma_db=2.0)]
        cfg["mission"]["weights"]["2"] = 0.03
        cfg["mission"]["deadlines_s"]["2"] = 2.0
        ems = build_emitters(load_config(cfg), np.random.default_rng(11))
        self.assertEqual(len(ems), 6)
        for e in ems:
            self.assertEqual(e.n_channels, 2)
            self.assertGreaterEqual(e.channel, 10)
            self.assertLessEqual(e.ch_hi, 18)

    def test_burst_table_dtype_and_invariants(self):
        w = make_world("sparse", 0)
        b = w.truth_bursts()
        self.assertEqual(b.dtype, BURST_DTYPE)
        self.assertTrue(np.all(b["t_off"] > b["t_on"]))
        self.assertTrue(np.all(b["ch_hi"] > b["ch_lo"]))     # ch_hi is EXCLUSIVE
        self.assertTrue(np.all(b["ch_hi"] <= w.grid.n_channels))
        self.assertTrue(np.all(np.diff(b["t_on"]) >= 0.0), "canonical order is by t_on")
        np.testing.assert_array_equal(b["burst_id"], np.arange(b.size))
        self.assertFalse(b.flags.writeable, "truth must be handed out read-only")


class TestDeterminism(unittest.TestCase):
    def test_same_scenario_and_seed_give_byte_identical_truth(self):
        a = make_world("sparse", 7).truth_bursts()
        b = make_world("sparse", 7).truth_bursts()
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_different_seed_gives_different_truth(self):
        a = make_world("sparse", 7).truth_bursts()
        b = make_world("sparse", 8).truth_bursts()
        self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_truth_is_independent_of_the_action_sequence(self):
        """THE property that makes two policies comparable on one seed.

        Truth is generated in full at reset() from its own RNG stream, so no
        amount of scanning, sleeping or hopping can perturb it.
        """
        w1 = make_world("sparse", 5)
        for k in range(30):
            w1.step(w1.grid.action_for(k * 3 % 180, 5, 0.01))

        w2 = make_world("sparse", 5)
        for k in range(30):
            w2.step(Sleep(0.02) if k % 2 else w2.grid.action_for(199 - k, 1, 0.2))

        w3 = make_world("sparse", 5)   # never stepped at all

        self.assertEqual(w1.truth_bursts().tobytes(), w2.truth_bursts().tobytes())
        self.assertEqual(w1.truth_bursts().tobytes(), w3.truth_bursts().tobytes())
        np.testing.assert_array_equal(w1.truth(1e-2), w2.truth(1e-2))
        np.testing.assert_array_equal(w1.truth_power(1e-2), w2.truth_power(1e-2))

    def test_reset_reproduces_the_same_world(self):
        w = make_world("dense", 2)
        first = w.truth_bursts().tobytes()
        w.step(w.grid.action_for(0, 20, 0.05))
        w.reset("dense", 2)
        self.assertEqual(w.truth_bursts().tobytes(), first)
        self.assertEqual(w.t, 0.0)
        self.assertEqual(w.energy_total, 0.0)
        self.assertEqual(w.step_index, -1)


class TestMarkovStatistics(unittest.TestCase):
    """Empirical sojourn statistics must match the CTMC the belief assumes."""

    @classmethod
    def setUpClass(cls):
        cls.mean_on, cls.mean_off = 0.5, 1.5
        cls.horizon = 10000.0
        cfg = base_cfg(horizon_s=cls.horizon)
        cfg["emitters"] = [dict(kind="fixed", count=5, channel_range=[0, 20],
                                n_channels=1, snr_db=[-12.0, -12.0], priority=3,
                                mean_on_s=cls.mean_on, mean_off_s=cls.mean_off,
                                snr_sigma_db=0.0)]
        cls.w = make_world(cfg, 42)
        cls.b = cls.w.truth_bursts()

    def test_mean_on_time(self):
        on = self.b["t_off"] - self.b["t_on"]
        self.assertGreater(on.size, 10000, "need a big sample for a 3% band")
        self.assertAlmostEqual(float(on.mean()) / self.mean_on, 1.0, delta=0.03)

    def test_mean_off_time(self):
        gaps = []
        for eid in np.unique(self.b["emitter_id"]):
            sub = self.b[self.b["emitter_id"] == eid]
            gaps.append(sub["t_on"][1:] - sub["t_off"][:-1])
        off = np.concatenate(gaps)
        self.assertTrue(np.all(off > 0.0))
        self.assertAlmostEqual(float(off.mean()) / self.mean_off, 1.0, delta=0.03)

    def test_duty_cycle_matches_pi_on(self):
        pi_on = self.mean_on / (self.mean_on + self.mean_off)
        for eid in np.unique(self.b["emitter_id"]):
            sub = self.b[self.b["emitter_id"] == eid]
            covered = np.minimum(sub["t_off"], self.horizon) - np.maximum(sub["t_on"], 0.0)
            duty = float(np.clip(covered, 0.0, None).sum()) / self.horizon
            self.assertAlmostEqual(duty / pi_on, 1.0, delta=0.03)

    def test_sojourns_are_exponential_not_fixed(self):
        """A constant-duration burst would also pass the mean test; this rules it out."""
        on = self.b["t_off"] - self.b["t_on"]
        # For Exp(lambda), std == mean.
        self.assertAlmostEqual(float(on.std()) / float(on.mean()), 1.0, delta=0.05)


class TestAgileHopper(unittest.TestCase):
    """A hopper is *only* extra rows sharing one activation_id -- no special case."""

    @classmethod
    def setUpClass(cls):
        cfg = base_cfg(horizon_s=40.0)
        cfg["grid"]["n_channels"] = 100
        cfg["mission"]["priority_bands"] = [dict(ch_lo=0, ch_hi=100, priority=1)]
        cfg["mission"]["weights"] = {"1": 0.100}
        cfg["mission"]["deadlines_s"] = {"1": 0.5}
        cfg["emitters"] = [dict(kind="agile", count=3, channel_range=[10, 30],
                                n_channels=1, snr_db=[-19.0, -16.0], priority=1,
                                mean_on_s=1.5, mean_off_s=2.0, snr_sigma_db=1.0,
                                hop_lo=10, hop_hi=90, hop_n=16,
                                hop_dwell_s=0.05, hop_pattern="random")]
        cls.w = make_world(cfg, 9)
        cls.b = cls.w.truth_bursts()

    def test_activation_spans_several_channels_and_is_contiguous_in_time(self):
        checked = 0
        for eid in np.unique(self.b["emitter_id"]):
            sub = self.b[self.b["emitter_id"] == eid]
            for act in np.unique(sub["activation_id"]):
                rows = np.sort(sub[sub["activation_id"] == act], order="t_on")
                if rows.size < 2:
                    continue   # an ON period shorter than one hop dwell
                self.assertGreaterEqual(len(set(rows["ch_lo"].tolist())), 2,
                                        "hopper must visit >= 2 distinct channels")
                np.testing.assert_allclose(rows["t_on"][1:], rows["t_off"][:-1],
                                           rtol=0, atol=1e-12)
                checked += 1
        self.assertGreater(checked, 20, "not enough multi-hop activations to test")

    def test_hop_dwell_is_respected(self):
        dur = self.b["t_off"] - self.b["t_on"]
        self.assertTrue(np.all(dur <= 0.05 + 1e-12), "no row may exceed hop_dwell_s")

    def test_each_hopper_gets_its_own_hop_set_inside_bounds(self):
        sets = []
        for e in self.w.emitters:
            self.assertEqual(e.kind, "agile")
            self.assertEqual(len(e.hop_set), 16)
            self.assertEqual(len(set(e.hop_set)), 16, "hop channels must be distinct")
            self.assertTrue(all(10 <= c < 90 for c in e.hop_set))
            sets.append(frozenset(e.hop_set))
        self.assertEqual(len(set(sets)), 3, "hoppers must not share a hop set")


class TestChannelIntegration(unittest.TestCase):
    """window_rho: time-averaged linear SNR, from which misses become emergent."""

    def _bursts(self, rows):
        b = np.zeros(len(rows), dtype=BURST_DTYPE)
        for i, (t_on, t_off, lo, hi, snr) in enumerate(rows):
            b[i] = (0, i, i, t_on, t_off, lo, hi, snr, 3, 0.01)
        return b

    def test_full_coverage_gives_exactly_the_linear_snr(self):
        b = self._bursts([(0.0, 10.0, 3, 5, -10.0)])
        rho = window_rho(b, 1.0, 1.1, 8)
        np.testing.assert_allclose(rho[3:5], 10.0 ** (-1.0), rtol=1e-12)
        np.testing.assert_array_equal(rho[[0, 1, 2, 5, 6, 7]], 0.0)

    def test_half_coverage_halves_rho(self):
        b = self._bursts([(0.0, 0.05, 2, 3, 0.0)])
        rho = window_rho(b, 0.0, 0.1, 4)
        self.assertAlmostEqual(rho[2], 0.5, places=12)

    def test_dwell_landing_in_an_off_gap_sees_nothing(self):
        """A miss produced by the physics, not by a rule."""
        b = self._bursts([(0.0, 0.01, 1, 2, 0.0), (0.5, 0.6, 1, 2, 0.0)])
        np.testing.assert_array_equal(window_rho(b, 0.02, 0.03, 4), 0.0)

    def test_overlapping_bursts_add_in_linear_power(self):
        b = self._bursts([(0.0, 1.0, 1, 3, 0.0), (0.0, 1.0, 2, 4, 0.0)])
        rho = window_rho(b, 0.0, 1.0, 5)
        np.testing.assert_allclose(rho, [0.0, 1.0, 2.0, 1.0, 0.0], rtol=1e-12)

    def test_rho_is_never_negative_after_cancellation(self):
        b = self._bursts([(0.0, 1.0, 0, 200, -8.0)] * 40)
        self.assertTrue(np.all(window_rho(b, 0.0, 1.0, 200) >= 0.0))

    def test_raster_and_window_agree(self):
        w = make_world("sparse", 12)
        occ = w.truth(1e-3)
        b = w.truth_bursts()
        rho = window_rho(b, 0.100, 0.101, w.grid.n_channels)
        np.testing.assert_array_equal(occ[100] & (rho > 0), rho > 0)


if __name__ == "__main__":
    unittest.main()
