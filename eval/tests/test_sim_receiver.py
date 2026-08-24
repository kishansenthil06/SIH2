"""The Urkowitz energy detector: the curve, the identity, and the two knobs.

`pd_curve` is the most safety-critical function in `sim/`.  Every headline number
in the write-up is downstream of it, and `agent/belief.py` **deliberately
reimplements** it rather than importing this module (DESIGN.md section 2), so a
silent drift here would not crash anything -- it would just quietly make the
belief wrong and the results meaningless.

Four things are pinned here, in descending order of how much rests on them:

1. **The verified table** (DESIGN.md section 1) -- reproduced to 3 decimals.  If
   this fails, either the detector changed or the table is stale; nothing else in
   this file matters until it passes.
2. **`s = 0` gives exactly `P_fa`**, which is why there is no separate
   false-alarm branch anywhere in the simulator.  `P_fa` calibration and `P_d`
   calibration therefore exercise the same three lines.
3. **The bandwidth penalty is real** (DESIGN.md section 4) -- 1 dB/octave.
   Without it the widest scan strictly dominates (same time, same energy, 20x the
   channels) and the bandwidth knob is degenerate.  So the penalty is asserted in
   dB *and* in the observable that matters: a 20 MHz scan needs ~7x the dwell of
   a 1 MHz one on the same emitter.
4. **The closed form and the Bernoulli draw agree empirically** -- the analytic
   curve is worthless if `observe()` samples from something else.  Both
   calibrations run through a real `World`, so they also check the plumbing
   between `window_rho`, `Receiver.observe` and the per-step Philox stream.

The gain path is exercised once even though `gain_enabled` is false in every
shipped config: an untested knob is a knob that does not work.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from sim.env import make_world
from sim.receiver import Receiver, bw_penalty_db, pd_curve, pd_from_linear

PFA = 1.0e-3
CH_BW = 1.0e6

BW_CANDIDATES_MHZ = (1, 2, 5, 10, 20)
DWELL_CANDIDATES_MS = (1, 2, 5, 10, 20, 50, 100, 200)

# DESIGN.md section 1, "Verified numbers -- use these exactly".  Transcribed, NOT
# recomputed: the point of the table is to catch a change in the detector, and a
# table generated from the detector could not do that.
# {snr_db: {dwell_ms: P_d}} at P_fa = 1e-3, B = 1 MHz.
VERIFIED_PD = {
    -10: {1: 0.526, 10: 1.000, 50: 1.000, 100: 1.000, 200: 1.000},
    -15: {1: 0.021, 10: 0.528, 50: 1.000, 100: 1.000, 200: 1.000},
    -18: {1: 0.005, 10: 0.069, 50: 0.672, 100: 0.971, 200: 1.000},
    -20: {1: 0.003, 10: 0.019, 50: 0.199, 100: 0.528, 200: 0.914},
}


def base_cfg(**over) -> dict:
    """A minimal VALID config dict.  Kept tiny so every test stays under 2 s.

    Emitter-free by default -- the false-alarm calibration needs a world where
    every detection is, by construction, junk.
    """
    cfg = dict(
        name="unit-rx",
        horizon_s=5.0,
        grid=dict(f_start_hz=2.0e9, n_channels=20, channel_bw_hz=1.0e6),
        receiver=dict(
            pfa=PFA, t_settle_s=0.5e-3, f_slew_hz_per_s=50.0e9,
            bw_penalty_db_per_octave=1.0, snr_est_sigma_db=1.5,
            gain_enabled=False, gain_db_high=10.0, gain_nf_improvement_db=6.0,
            gain_energy_mult=1.6, gain_saturation_snr_db=-5.0,
            gain_fa_mult_on_saturation=10.0,
        ),
        energy=dict(L_d_w=1.0, L_0_j=2.0e-3, L_f_j_per_hz=2.0e-11,
                    L_sleep_w=0.01, budget_j=1.0e9),
        mission=dict(
            priority_bands=[dict(ch_lo=0, ch_hi=20, priority=3)],
            weights={"3": 0.010}, deadlines_s={"3": 10.0},
            watch_list=[], watch_deadline_s=0.3,
        ),
        emitters=[],
        agent=dict(prior_pi_on=0.05, prior_lam_sum=1.0,
                   bw_candidates_mhz=[1, 2, 5, 10, 20],
                   dwell_candidates_ms=[1, 10], sleep_candidates_ms=[10]),
    )
    cfg.update(copy.deepcopy(over))
    return cfg


def always_on_emitter(snr_db: float, n_channels: int, grid_n: int) -> dict:
    """One emitter that is radiating for the whole horizon, at an exact SNR.

    `mean_on_s` enormous and `mean_off_s` tiny puts `pi_on` at 1 - 1e-18, so the
    stationary start is ON with certainty and the first ON sojourn outlives any
    horizon a unit test would use.  `snr_sigma_db = 0` removes per-burst
    shadowing, so the SNR the detector sees is the SNR written here -- which is
    what lets the measured rate be compared against a *specific* table entry
    rather than an average over a shadowing distribution.
    """
    return dict(kind="fixed", count=1, channel_range=[0, grid_n],
                n_channels=n_channels, snr_db=[snr_db, snr_db], priority=3,
                mean_on_s=1.0e9, mean_off_s=1.0e-9, snr_sigma_db=0.0)


class TestVerifiedPdTable(unittest.TestCase):
    """The single most important assertion in this file.

    5 dB of SNR is about one decade of dwell -- that relationship is what makes
    "narrow-and-long to confirm" a rational strategy rather than a hand-coded
    one, and it is the reason the scenario SNRs in DESIGN.md section 5 are placed
    where they are.
    """

    def test_matches_design_md_to_three_decimals(self):
        for snr_db, row in VERIFIED_PD.items():
            for dwell_ms, expected in row.items():
                with self.subTest(snr_db=snr_db, dwell_ms=dwell_ms):
                    got = float(pd_curve(snr_db, dwell_ms * 1e-3, CH_BW, PFA))
                    self.assertAlmostEqual(got, expected, places=3)

    def test_five_db_is_about_one_decade_of_dwell(self):
        """The diagonal of the table: (-15 dB, 10 ms) ~ (-20 dB, 100 ms) ~ 0.528."""
        a = float(pd_curve(-15.0, 10e-3, CH_BW, PFA))
        b = float(pd_curve(-20.0, 100e-3, CH_BW, PFA))
        self.assertAlmostEqual(a, 0.528, places=3)
        self.assertAlmostEqual(b, 0.528, places=3)
        self.assertAlmostEqual(a, b, delta=0.002)

    def test_pd_curve_broadcasts_over_snr_and_dwell(self):
        snr = np.array([-10.0, -15.0, -18.0, -20.0])
        got = pd_curve(snr, 10e-3, CH_BW, PFA)
        self.assertEqual(got.shape, (4,))
        np.testing.assert_allclose(
            got, [1.000, 0.528, 0.069, 0.019], atol=5e-4
        )
        dwells = np.array([1e-3, 10e-3, 50e-3, 100e-3, 200e-3])
        got = pd_curve(-18.0, dwells, CH_BW, PFA)
        np.testing.assert_allclose(
            got, [0.005, 0.069, 0.672, 0.971, 1.000], atol=5e-4
        )


class TestSilentChannelIsExactlyPfa(unittest.TestCase):
    """`s = 0` -> `P_d = Q(Q^-1(P_fa)) = P_fa`.

    This identity is why no module in `sim/` contains a false-alarm branch.  If
    it ever stopped holding exactly, a silent channel would fire at some other
    rate and the measured false-alarm rate in `eval/metrics.py` would stop being
    comparable to the configured `pfa` -- which is the only reason that metric
    doubles as a calibration check (DESIGN.md section 6).
    """

    def test_zero_linear_snr_returns_pfa(self):
        for pfa in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
            for dwell_s in (1e-3, 10e-3, 200e-3, 1.0):
                for bw in (1e6, 5e6, 20e6):
                    with self.subTest(pfa=pfa, dwell_s=dwell_s, bw=bw):
                        got = float(pd_from_linear(0.0, dwell_s, bw, pfa))
                        self.assertAlmostEqual(got, pfa, delta=1e-12)

    def test_minus_inf_db_returns_pfa(self):
        """The dB entry point must degrade to the same identity."""
        got = float(pd_curve(-np.inf, 10e-3, CH_BW, PFA))
        self.assertAlmostEqual(got, PFA, delta=1e-12)

    def test_receiver_reports_pfa_on_a_silent_channel_at_every_bandwidth(self):
        """The penalty scales the SIGNAL, so it cannot move a silent channel."""
        rx = Receiver(pfa=PFA, channel_bw_hz=CH_BW)
        rho = np.zeros(4)
        for bw_mhz in BW_CANDIDATES_MHZ:
            for dwell_ms in DWELL_CANDIDATES_MS:
                with self.subTest(bw_mhz=bw_mhz, dwell_ms=dwell_ms):
                    p = rx.detect_probability(rho, dwell_ms * 1e-3, bw_mhz * 1e6)
                    np.testing.assert_allclose(p, PFA, atol=1e-12)

    def test_zero_length_dwell_reports_nothing(self):
        """A scan truncated to nothing collects N = 0 samples.

        The Gaussian approximation degenerates there (it would report P_d > P_fa
        off zero evidence), so `observe` short-circuits.  Asserted because a
        truncated action at the horizon takes exactly this path.
        """
        rx = Receiver(pfa=PFA, channel_bw_hz=CH_BW)
        det, snr = rx.observe(np.full(8, 0.1), 0.0, 1e6, np.random.default_rng(0))
        self.assertEqual(det.shape, (8,))
        self.assertFalse(bool(det.any()))
        np.testing.assert_array_equal(snr, 0.0)


class TestMonotonicity(unittest.TestCase):
    """Across the FULL candidate cross-product the two knobs must not invert.

    The index policy assumes exactly this shape when it prices a candidate: more
    dwell can only help, more bandwidth can only hurt sensitivity.  A local
    inversion anywhere in the grid would make the score non-comparable across
    candidates and the "wide-and-fast to explore, narrow-and-long to confirm"
    behaviour would stop being derivable from the physics.
    """

    SNRS = (-30.0, -25.0, -22.0, -20.0, -18.0, -15.0, -12.0, -10.0, -5.0, 0.0)

    def _grid(self, snr_db: float) -> np.ndarray:
        rx = Receiver(pfa=PFA, channel_bw_hz=CH_BW)
        rho = np.array([10.0 ** (snr_db / 10.0)])
        return np.array(
            [
                [
                    float(rx.detect_probability(rho, d * 1e-3, bw * 1e6)[0])
                    for d in DWELL_CANDIDATES_MS
                ]
                for bw in BW_CANDIDATES_MHZ
            ]
        )

    def test_pd_is_non_decreasing_in_dwell(self):
        for snr_db in self.SNRS:
            with self.subTest(snr_db=snr_db):
                d = np.diff(self._grid(snr_db), axis=1)
                self.assertGreaterEqual(float(d.min()), -1e-15)

    def test_pd_is_non_increasing_in_bandwidth(self):
        for snr_db in self.SNRS:
            with self.subTest(snr_db=snr_db):
                d = np.diff(self._grid(snr_db), axis=0)
                self.assertLessEqual(float(d.max()), 1e-15)

    def test_pd_is_bounded_by_pfa_below_and_one_above(self):
        for snr_db in self.SNRS:
            g = self._grid(snr_db)
            self.assertGreaterEqual(float(g.min()), PFA - 1e-12)
            self.assertLessEqual(float(g.max()), 1.0)

    def test_the_knobs_actually_move_something(self):
        """Guards against a degenerate grid passing monotonicity trivially."""
        g = self._grid(-18.0)
        self.assertGreater(g[0, -1] - g[0, 0], 0.5, "dwell must matter at 1 MHz")
        self.assertGreater(g[0, 5] - g[-1, 5], 0.3, "bandwidth must matter at 50 ms")


class TestBandwidthPenalty(unittest.TestCase):
    """DESIGN.md section 4: load-bearing, not cosmetic.

    Without it a 20 MHz scan is 20 channels for the price of one and the widest
    bandwidth strictly dominates every other candidate -- the knob would be
    degenerate and there would be no explore/confirm trade-off to demonstrate.
    """

    def test_one_db_per_octave_in_db(self):
        for bw_mhz, octaves in ((1, 0), (2, 1), (4, 2), (8, 3), (16, 4)):
            with self.subTest(bw_mhz=bw_mhz):
                self.assertAlmostEqual(
                    bw_penalty_db(bw_mhz * 1e6, 1.0), float(octaves), places=12
                )

    def test_twenty_mhz_is_4_3_db_less_sensitive_than_one_mhz(self):
        """The number DESIGN.md section 4 quotes: log2(20) = 4.3219 dB."""
        pen = bw_penalty_db(20e6, 1.0)
        self.assertAlmostEqual(pen, 4.321928094887363, places=12)
        self.assertAlmostEqual(pen - bw_penalty_db(1e6, 1.0), 4.3, delta=0.03)

    def test_the_penalty_is_exactly_an_snr_shift(self):
        """A wide scan on a strong emitter == a narrow scan on a weaker one.

        Asserting the dB relationship *through the receiver* rather than only
        through `bw_penalty_db` is what proves the constant is actually wired
        into the detector rather than merely defined next to it.
        """
        rx = Receiver(pfa=PFA, channel_bw_hz=CH_BW, bw_penalty_db_per_octave=1.0)
        for bw_mhz in BW_CANDIDATES_MHZ:
            pen = float(np.log2(bw_mhz))
            for snr_db in (-20.0, -15.0, -10.0):
                for dwell_ms in (1, 10, 100):
                    with self.subTest(bw_mhz=bw_mhz, snr_db=snr_db, dwell_ms=dwell_ms):
                        wide = float(
                            rx.detect_probability(
                                np.array([10.0 ** (snr_db / 10.0)]),
                                dwell_ms * 1e-3, bw_mhz * 1e6,
                            )[0]
                        )
                        narrow = float(
                            pd_curve(snr_db - pen, dwell_ms * 1e-3, CH_BW, PFA)
                        )
                        self.assertAlmostEqual(wide, narrow, places=12)

    def test_scanning_wide_needs_materially_more_dwell(self):
        """The observable consequence, in the units the policy actually chooses.

        DESIGN.md section 4 quotes ~7x.  Measured here by bisection on dwell:
        the 20 MHz dwell needed to match a 1 MHz / 10 ms look is 70-76 ms across
        the scenario SNR range -- so a wide scan is never a free lunch, and the
        candidate set has a real trade-off in it.
        """
        rx = Receiver(pfa=PFA, channel_bw_hz=CH_BW)
        for snr_db in (-12.0, -15.0, -18.0, -20.0):
            with self.subTest(snr_db=snr_db):
                rho = np.array([10.0 ** (snr_db / 10.0)])
                target = float(rx.detect_probability(rho, 10e-3, 1e6)[0])

                lo, hi = 10e-3, 10.0
                for _ in range(60):     # 60 halvings: exact to ~1e-17 s
                    mid = 0.5 * (lo + hi)
                    if float(rx.detect_probability(rho, mid, 20e6)[0]) < target:
                        lo = mid
                    else:
                        hi = mid
                ratio = hi / 10e-3
                self.assertGreater(ratio, 5.0, "20 MHz must cost real dwell")
                self.assertAlmostEqual(ratio, 7.0, delta=1.0)

    def test_widest_bandwidth_does_not_dominate(self):
        """The degeneracy the penalty exists to prevent, stated directly."""
        rx = Receiver(pfa=PFA, channel_bw_hz=CH_BW)
        rho = np.array([10.0 ** (-18.0 / 10.0)])
        narrow = float(rx.detect_probability(rho, 50e-3, 1e6)[0])
        wide = float(rx.detect_probability(rho, 50e-3, 20e6)[0])
        self.assertAlmostEqual(narrow, 0.672, places=3)
        self.assertLess(wide, 0.5 * narrow,
                        "a 20 MHz look must be much worse per channel")


class TestEmpiricalPfaCalibration(unittest.TestCase):
    """The Bernoulli draw must fire at the rate the closed form promises.

    Run through a real `World` on an emitter-free scenario, so this also checks
    `window_rho` returns exactly 0.0 for an empty burst table and that the
    per-step Philox substreams do not correlate (DESIGN.md's `_PHILOX_STRIDE`
    note records the bug where they did: 350 false alarms where 200 +/- 42 were
    expected).

    2000 channels x 100 steps = 200k channel-dwells in ~0.1 s: many channels per
    dwell rather than many steps, so the sample size costs almost no wall clock.
    """

    N_STEPS = 100

    @classmethod
    def setUpClass(cls):
        cfg = base_cfg(horizon_s=5.0)
        cfg["grid"]["n_channels"] = 2000
        cfg["mission"]["priority_bands"] = [dict(ch_lo=0, ch_hi=2000, priority=3)]
        world = make_world(cfg, 0)
        assert world.truth_bursts().size == 0, "this scenario must be emitter-free"

        action = world.grid.action_for(0, 2000, 1e-3)
        cls.n_dwells = 0
        cls.n_fa = 0
        cls.hits = np.zeros(2000, dtype=np.int64)
        for _ in range(cls.N_STEPS):
            obs = world.step(action)
            cls.n_dwells += int(obs.scanned_channels.size)
            cls.n_fa += len(obs.detections)
            for d in obs.detections:
                cls.hits[d.channel] += 1

    def test_sample_size_is_what_the_test_claims(self):
        self.assertEqual(self.n_dwells, 200_000)

    def test_measured_false_alarm_rate_is_within_three_sigma_of_pfa(self):
        rate = self.n_fa / self.n_dwells
        sigma = float(np.sqrt(PFA * (1.0 - PFA) / self.n_dwells))
        self.assertAlmostEqual(
            rate, PFA, delta=3.0 * sigma,
            msg=f"measured P_fa={rate:.6f} from {self.n_fa} alarms in "
                f"{self.n_dwells} channel-dwells; 3 sigma = {3 * sigma:.6f}",
        )

    def test_false_alarms_are_spread_over_the_band(self):
        """A correlated Philox stream shows up as a CLUMP, not as a wrong rate.

        The historical bug (see `_PHILOX_STRIDE` in `sim/env.py`) reused one
        step's uniforms in the next, turning a single unlucky draw into ~196
        consecutive false alarms.  So a rate test alone is not enough: check
        that ~200 alarms landed on ~200 distinct channels and that no single
        channel hogged them.
        """
        self.assertGreater(self.n_fa, 100, "need enough alarms to talk about spread")
        distinct = int((self.hits > 0).sum())
        self.assertGreater(distinct, 0.8 * self.n_fa,
                           f"{self.n_fa} alarms on only {distinct} channels")
        self.assertLessEqual(int(self.hits.max()), 4,
                             "no channel may hog the false alarms")
        # Alarms in the top half of the band, i.e. not all in one contiguous run.
        self.assertGreater(int(self.hits[1000:].sum()), 0.25 * self.n_fa)

    def test_false_alarm_snr_looks_like_a_marginal_detection(self):
        """A false alarm the belief could trivially filter out would be useless.

        `_FA_SNR_LO_DB/-HI_DB` put it in -24..-19 dB, i.e. exactly where a real
        weak threat emitter lives, so the agent has to reason about it rather
        than threshold it away.
        """
        rx = Receiver(pfa=0.5, channel_bw_hz=CH_BW)   # fire often, cheaply
        det, snr = rx.observe(np.zeros(4000), 10e-3, 1e6, np.random.default_rng(7))
        reported = snr[det]
        self.assertGreater(reported.size, 100)
        self.assertGreaterEqual(float(reported.min()), -24.0)
        self.assertLessEqual(float(reported.max()), -19.0)


class TestEmpiricalPdCalibration(unittest.TestCase):
    """The other half of the same three lines, on a permanently-lit channel.

    -15 dB at 10 ms is the table's most-quoted cell (0.528) and sits right on the
    steep part of the curve, where a wrong `sqrt(N)` or a wrong `(1 + s)` would
    be most visible.  1 MHz so the bandwidth penalty is exactly zero and the
    measurement is against the verified table entry itself, not a derived value.
    """

    N_STEPS = 2000
    EXPECTED = 0.528

    @classmethod
    def setUpClass(cls):
        cfg = base_cfg(horizon_s=30.0)
        cfg["emitters"] = [always_on_emitter(-15.0, n_channels=20, grid_n=20)]
        world = make_world(cfg, 3)
        bursts = world.truth_bursts()
        assert bursts.size == 1 and bursts["t_on"][0] == 0.0
        assert bursts["t_off"][0] > 30.0, "emitter must be lit for the whole horizon"
        assert abs(float(bursts["snr_db"][0]) + 15.0) < 1e-12, "no shadowing"

        action = world.grid.action_for(5, 1, 10e-3)   # 1 MHz => zero bw penalty
        cls.n_dwells = 0
        cls.n_det = 0
        for _ in range(cls.N_STEPS):
            obs = world.step(action)
            assert obs.action.dwell_s == 10e-3, "horizon must not truncate this"
            cls.n_dwells += int(obs.scanned_channels.size)
            cls.n_det += len(obs.detections)

    def test_measured_detection_rate_is_within_three_sigma(self):
        rate = self.n_det / self.n_dwells
        sigma = float(np.sqrt(self.EXPECTED * (1.0 - self.EXPECTED) / self.n_dwells))
        self.assertAlmostEqual(
            rate, self.EXPECTED, delta=3.0 * sigma,
            msg=f"measured P_d={rate:.4f} from {self.n_det} detections in "
                f"{self.n_dwells} dwells; 3 sigma = {3 * sigma:.4f}",
        )

    def test_it_is_not_trivially_zero_or_one(self):
        """0.528 is the interesting answer; 0 and 1 would both pass a sloppy test."""
        rate = self.n_det / self.n_dwells
        self.assertGreater(rate, 0.4)
        self.assertLess(rate, 0.65)

    def test_a_one_ms_look_at_the_same_emitter_almost_never_fires(self):
        """The same channel, 10x less dwell: 0.021, i.e. 25x worse.

        This is DESIGN.md section 11.8's finding in miniature -- an action too
        short to detect is not coverage, however good the staleness looks.
        """
        cfg = base_cfg(horizon_s=10.0)
        cfg["emitters"] = [always_on_emitter(-15.0, n_channels=20, grid_n=20)]
        world = make_world(cfg, 3)
        action = world.grid.action_for(5, 1, 1e-3)
        n_det = sum(len(world.step(action).detections) for _ in range(2000))
        rate = n_det / 2000.0
        sigma = float(np.sqrt(0.021 * 0.979 / 2000.0))
        self.assertAlmostEqual(rate, 0.021, delta=3.0 * sigma,
                               msg=f"measured P_d={rate:.4f} at 1 ms")


class TestGainPath(unittest.TestCase):
    """`gain_enabled` is false in every shipped config, so exercise it once here.

    An untested knob is a knob that does not work, and the gain path is the one
    place in the receiver where `P_fa` is NOT a constant -- so leaving it
    unexercised would also leave the only exception to the `s = 0 -> P_fa`
    identity unverified.
    """

    def _rx(self, **over) -> Receiver:
        kw = dict(pfa=PFA, channel_bw_hz=CH_BW, gain_enabled=True,
                  gain_db_high=10.0, gain_nf_improvement_db=6.0,
                  gain_energy_mult=1.6, gain_saturation_snr_db=-5.0,
                  gain_fa_mult_on_saturation=10.0)
        kw.update(over)
        return Receiver(**kw)

    def test_gain_is_inactive_unless_enabled_and_requested(self):
        self.assertFalse(self._rx(gain_enabled=False).gain_active(10.0))
        self.assertFalse(self._rx().gain_active(9.999))
        self.assertFalse(self._rx().gain_active(0.0))
        self.assertTrue(self._rx().gain_active(10.0))
        self.assertTrue(self._rx().gain_active(12.0))

    def test_high_gain_buys_exactly_the_nf_improvement(self):
        """+6 dB of effective SNR, not 6 dB of anything else.

        Checked at -28 dB where the curve is still steep, so a wrong sign or a
        wrong factor cannot hide in the saturated top of the curve.
        """
        rx = self._rx()
        for snr_db in (-30.0, -28.0, -25.0, -22.0):
            for dwell_ms in (10, 50, 200):
                with self.subTest(snr_db=snr_db, dwell_ms=dwell_ms):
                    rho = np.array([10.0 ** (snr_db / 10.0)])
                    on = float(rx.detect_probability(rho, dwell_ms * 1e-3, 1e6, 10.0)[0])
                    off = float(rx.detect_probability(rho, dwell_ms * 1e-3, 1e6, 0.0)[0])
                    boosted = float(pd_curve(snr_db + 6.0, dwell_ms * 1e-3, CH_BW, PFA))
                    self.assertAlmostEqual(on, boosted, places=12)
                    self.assertGreater(on, off, "gain must help, not merely differ")

    def test_gain_composes_with_the_bandwidth_penalty(self):
        """Both are dB shifts on the same effective SNR: -1*log2(bw/1e6) + 6."""
        rx = self._rx()
        rho = np.array([10.0 ** (-25.0 / 10.0)])
        got = float(rx.detect_probability(rho, 20e-3, 20e6, 10.0)[0])
        want = float(pd_curve(-25.0 - np.log2(20.0) + 6.0, 20e-3, CH_BW, PFA))
        self.assertAlmostEqual(got, want, places=12)

    def test_high_gain_multiplies_energy(self):
        rx = self._rx()
        self.assertEqual(rx.energy_mult(10.0), 1.6)
        self.assertEqual(rx.energy_mult(0.0), 1.0)
        self.assertEqual(self._rx(gain_enabled=False).energy_mult(10.0), 1.0)

    def test_world_charges_the_gain_multiplier(self):
        """The knob has to be wired through `World._step_scan`, not just defined."""
        cfg = base_cfg(horizon_s=5.0)
        cfg["receiver"]["gain_enabled"] = True
        world = make_world(cfg, 0)
        plain = world.step(world.grid.action_for(0, 1, 10e-3, 0.0))
        boosted = world.step(world.grid.action_for(0, 1, 10e-3, 10.0))
        self.assertAlmostEqual(plain.energy_cost, 2.0e-3 + 1.0 * 10e-3, delta=1e-15)
        self.assertAlmostEqual(boosted.energy_cost, 1.6 * plain.energy_cost, delta=1e-15)

    def test_saturation_raises_pfa_across_the_whole_comb(self):
        """A strong in-band signal desensitises the front end, not just its channel.

        The ONE place where a silent channel does not report exactly `pfa` -- so
        it is asserted explicitly rather than left as an accident.
        """
        rx = self._rx()
        rho = np.array([1.0, 0.0, 0.0])     # 0 dB is well above the -5 dB threshold
        p = rx.detect_probability(rho, 10e-3, 1e6, 10.0)
        np.testing.assert_allclose(p[1:], PFA * 10.0, atol=1e-12)

        quiet = np.array([10.0 ** (-2.0), 0.0])   # -20 dB: below saturation
        p = rx.detect_probability(quiet, 10e-3, 1e6, 10.0)
        self.assertAlmostEqual(float(p[1]), PFA, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
