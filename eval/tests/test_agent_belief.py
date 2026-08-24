"""Tests for `agent/belief.py` -- the rung-1 state estimator.

Four properties carry the project's credibility and all four are asserted here
mechanically rather than by inspection:

1. The closed-form Markov decay is the *exact* transient of the 2-state CTMC, not
   an approximation of it.  This is the restless-bandit dynamic; if it is wrong,
   every claim about "beating a sweep" is wrong with it.
2. Observations enter as likelihoods, so an uninformative observation is exactly
   the identity and no finite sequence of updates can push the posterior out of
   (0, 1).
3. The **two P_d's** of DESIGN.md section 4 really do make the agent distrust a
   marginal detection: -20 dB / 2 ms carries a likelihood ratio of ~4 and barely
   moves the belief, while -10 dB / 5 ms carries ~1000 and pins it.  That is the
   difference between inference and bookkeeping, and it falls straight out of the
   detector curve.
4. `feature_matrix` is finite and correctly shaped even on a cold start, because
   agent D trains on it and one NaN poisons a whole run.

Test files may import `sim.env` / `sim.stub_env`; `agent/` source files may not.
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm
from scipy.stats import norm

from agent.base import FEATURE_NAMES, N_FEATURES, SENTINEL_NO_SNR
from agent.belief import (
    P_CLIP_HI,
    P_CLIP_LO,
    Belief,
    bayes_posterior,
    marginal_pd_table,
    pd_curve,
    snr_eff_db,
)
from sim.config import build_grid, build_mission, load_config
from sim.contract import Detection, Obs, null_obs

PFA = 1.0e-3


def _cfg():
    return load_config("sparse")


def _belief(cfg=None, mode="bayes"):
    cfg = cfg or _cfg()
    return Belief(build_grid(cfg), build_mission(cfg), cfg, mode=mode)


def _scan_obs(grid, ch, t_start, dwell_s, snr_db=None, step=0):
    """One single-channel Obs, with or without a detection."""
    act = grid.action_for(int(ch), 1, float(dwell_s))
    dets = ()
    if snr_db is not None:
        dets = (
            Detection(
                channel=int(ch),
                f_hz=float(grid.center_hz(int(ch))),
                bw_hz=float(grid.channel_bw_hz),
                snr_db=float(snr_db),
            ),
        )
    return Obs(
        t=float(t_start) + float(dwell_s),
        action=act,
        detections=dets,
        energy_cost=0.0,
        t_start=float(t_start),
        scanned_channels=np.asarray([int(ch)], dtype=np.int32),
        energy_total=0.0,
        step_index=int(step),
        done=False,
    )


# ---------------------------------------------------------------------------
class TestClosedFormDecay(unittest.TestCase):
    """(1) `pi + (p-pi)*exp(-Lam*dt)` IS `expm(Q*dt)`, to 1e-12."""

    def test_matches_matrix_exponential(self):
        b = _belief()
        worst = 0.0
        for lam_on in (0.02, 0.125, 1.0, 7.5):
            for lam_off in (0.05, 0.5, 3.0, 11.0):
                # Generator for state order (off, on), row-stochastic convention:
                # row `off` leaks at lam_on into `on`, row `on` at lam_off.
                q = np.array([[-lam_on, lam_on], [lam_off, -lam_off]])
                lam = lam_on + lam_off
                pi = lam_on / lam
                for dt in (0.0, 1e-4, 0.013, 0.5, 3.0, 40.0):
                    for p0 in (0.0, 0.05, 0.5, 0.9999, 1.0):
                        b.t_now = 0.0
                        b.pi_on[:] = pi
                        b.lam_sum[:] = lam
                        b.p[:] = p0
                        b.propagate_to(dt)
                        got = float(b.p[0])

                        v0 = np.array([1.0 - p0, p0])
                        want = float((v0 @ expm(q * dt))[1])
                        worst = max(worst, abs(got - want))
                        self.assertAlmostEqual(got, want, delta=1e-12)
        # Guard against the assertion silently passing on a trivial grid.
        self.assertLess(worst, 1e-12)

    def test_snr_eff_bandwidth_penalty(self):
        # 1 dB/octave: a 20 MHz scan is log2(20) = 4.32 dB less sensitive.
        self.assertAlmostEqual(float(snr_eff_db(-15.0, 1.0e6)), -15.0, places=12)
        self.assertAlmostEqual(
            float(snr_eff_db(-15.0, 20.0e6)), -15.0 - np.log2(20.0), places=12
        )

    def test_pd_curve_reproduces_design_table(self):
        """DESIGN.md section 1's verified table, to its printed 3 decimals."""
        table = {
            (-10, 0.001): 0.526, (-10, 0.002): 0.895, (-10, 0.005): 1.000,
            (-15, 0.001): 0.021, (-15, 0.010): 0.528, (-15, 0.020): 0.910,
            (-18, 0.050): 0.672, (-18, 0.100): 0.971,
            (-20, 0.002): 0.004, (-20, 0.100): 0.528, (-20, 0.200): 0.914,
            (-22, 0.200): 0.395,
        }
        for (snr, dwell), want in table.items():
            got = float(pd_curve(float(snr), dwell, pfa=PFA))
            self.assertAlmostEqual(got, want, places=3, msg=f"{snr} dB / {dwell} s")

    def test_agrees_with_receiver_across_the_firewall(self):
        """`agent.belief.pd_curve` is a deliberate reimplementation of
        `sim.receiver.pd_curve` (DESIGN.md section 2).  Duplication is only
        cheaper than a firewall breach while the two stay identical."""
        from sim.receiver import pd_curve as rx_pd_curve  # test-only import

        snr = np.linspace(-30.0, 5.0, 71)
        for dwell in (1e-3, 5e-3, 2e-2, 1e-1, 2e-1):
            mine = np.asarray(pd_curve(snr, dwell, pfa=PFA, channel_bw_hz=1.0e6))
            theirs = np.asarray(rx_pd_curve(snr, dwell, 1.0e6, PFA))
            self.assertLess(float(np.max(np.abs(mine - theirs))), 1e-9)


class TestFixedPoint(unittest.TestCase):
    """(2) pi is a fixed point; dt=0 is the identity; dt->inf collapses to pi."""

    def setUp(self):
        self.b = _belief()

    def test_prior_is_a_fixed_point_for_any_dt(self):
        for dt in (0.0, 1e-6, 0.1, 7.0, 1e4):
            self.b.t_now = 0.0
            self.b.p[:] = self.b.pi_on
            self.b.propagate_to(dt)
            np.testing.assert_allclose(self.b.p, self.b.pi_on, rtol=0, atol=1e-15)

    def test_zero_dt_is_the_identity(self):
        rng = np.random.default_rng(0)
        p0 = rng.uniform(0.0, 1.0, self.b.n)
        self.b.t_now = 3.5
        self.b.p[:] = p0
        self.b.propagate_to(3.5)
        np.testing.assert_array_equal(self.b.p, p0)
        # ... and propagation is forward-only: going backwards is a no-op.
        self.b.propagate_to(1.0)
        np.testing.assert_array_equal(self.b.p, p0)

    def test_infinite_dt_drives_everything_to_the_prior(self):
        for p0 in (0.0, 0.5, 1.0):
            self.b.t_now = 0.0
            self.b.p[:] = p0
            self.b.propagate_to(1.0e4)   # Lam = 1/s, so 1e4 mixing times
            np.testing.assert_allclose(self.b.p, self.b.pi_on, atol=1e-12)

    def test_decays_upward_from_below_the_prior(self):
        """A channel we are confident is EMPTY drifts back up toward pi.  Missing
        this is the classic restless-bandit bug: information ages in both
        directions, and a policy that forgets that never revisits."""
        self.b.t_now = 0.0
        self.b.p[:] = 1.0e-4
        before = self.b.p.copy()
        self.b.propagate_to(0.5)
        self.assertTrue(np.all(self.b.p > before))
        self.assertTrue(np.all(self.b.p < self.b.pi_on))


class TestBayesSanity(unittest.TestCase):
    """(3) Observations are likelihoods.  The posterior is well-behaved."""

    def test_detection_raises_and_miss_lowers(self):
        p = np.array([0.01, 0.05, 0.5, 0.9])
        pd_det = float(pd_curve(-12.0, 0.005, pfa=PFA))
        up = bayes_posterior(p, pd_det, PFA)
        self.assertTrue(np.all(up > p), f"detection did not raise belief: {up}")

        pd_bar = 0.5256                       # pd_bar[1 MHz, 10 ms], see DESIGN
        down = bayes_posterior(p, 1.0 - pd_bar, 1.0 - PFA)
        self.assertTrue(np.all(down < p), f"miss did not lower belief: {down}")

    def test_uninformative_observation_is_the_identity(self):
        """P_d == P_fa carries no information and must move nothing.  This is the
        property that proves we are doing inference, not bookkeeping."""
        p = np.linspace(1e-3, 1.0 - 1e-3, 501)
        for like in (PFA, 0.5, 1.0 - PFA, 0.123456):
            out = bayes_posterior(p, like, like)
            self.assertLess(float(np.max(np.abs(out - p))), 1e-12)

    def test_zero_snr_detection_is_exactly_pfa(self):
        """`s = 0` gives Q(Q^-1(P_fa)) = P_fa, so no separate false-alarm branch
        is needed anywhere in the project (DESIGN.md section 1)."""
        for dwell in (1e-3, 1e-2, 2e-1):
            self.assertAlmostEqual(
                float(pd_curve(-np.inf, dwell, pfa=PFA)), PFA, places=12
            )

    def test_stays_strictly_inside_the_unit_interval(self):
        """10 000 random updates: 200 channels x 50 rounds."""
        rng = np.random.default_rng(7)
        p = rng.uniform(1e-4, 1.0 - 1e-4, 200)
        for _ in range(50):
            l1 = rng.uniform(0.0, 1.0, 200)
            l0 = rng.uniform(0.0, 1.0, 200)
            p = bayes_posterior(p, l1, l0)
            self.assertTrue(np.all(np.isfinite(p)))
            self.assertTrue(np.all(p >= P_CLIP_LO))
            self.assertTrue(np.all(p <= P_CLIP_HI))
        # The clip is what makes one unlucky miss recoverable: a belief of exactly
        # zero can never be revived by any finite likelihood ratio.
        self.assertGreater(P_CLIP_LO, 0.0)
        self.assertLess(P_CLIP_HI, 1.0)


class TestTwoPdDistinction(unittest.TestCase):
    """(4) DESIGN.md section 4 -- the agent distrusts marginal detections."""

    def test_likelihood_ratios_match_the_design_note(self):
        lr_marginal = float(pd_curve(-20.0, 0.002, pfa=PFA)) / PFA
        lr_strong = float(pd_curve(-10.0, 0.005, pfa=PFA)) / PFA
        # DESIGN.md: "likelihood ratio 4" and "LR ~ 1000".
        self.assertGreater(lr_marginal, 3.0)
        self.assertLess(lr_marginal, 6.0)
        self.assertGreater(lr_strong, 500.0)
        self.assertLess(lr_strong, 1100.0)
        self.assertGreater(lr_strong / lr_marginal, 100.0)

    def test_marginal_detection_moves_belief_only_slightly(self):
        b = _belief()
        grid = b.grid
        ch = 50
        p0 = float(b.p[ch])

        b.update(_scan_obs(grid, ch, 0.0, 0.002, snr_db=-20.0))
        p_marginal = float(b.p[ch])

        b2 = _belief()
        b2.update(_scan_obs(grid, ch, 0.0, 0.005, snr_db=-10.0))
        p_strong = float(b2.p[ch])

        # Ordering: both rise, the strong one far more.
        self.assertGreater(p_marginal, p0)
        self.assertGreater(p_strong, p_marginal)
        # Rough magnitudes, from p0 = 0.05 with LR 4.4 and LR 1000.
        self.assertLess(p_marginal, 0.30, "marginal detection saturated the belief")
        self.assertGreater(p_strong, 0.97, "strong detection failed to pin the belief")

    def test_miss_uses_the_marginal_pd_not_the_reported_snr(self):
        """A 1 ms miss barely moves the belief; a 200 ms miss crushes it.

        The Bayes step is checked separately from the state after `update`,
        because `update` also propagates across the dwell -- and over a 200 ms
        dwell the decay pulls a crushed belief measurably back toward pi.  Both
        halves matter, so both are asserted.
        """
        b = _belief()
        p0 = float(b.pi_on[10])
        # --- the likelihood step alone -----------------------------------
        post = [
            float(bayes_posterior(p0, 1.0 - b.pd_bar_for(1.0e6, dw), 1.0 - PFA))
            for dw in (0.001, 0.010, 0.200)
        ]
        self.assertTrue(post[0] > post[1] > post[2], f"not monotone in dwell: {post}")
        self.assertGreater(post[0], 0.8 * p0, "a 1 ms miss should barely register")
        self.assertLess(post[2], 0.15 * p0, "a 200 ms miss should crush the belief")

        # --- and end to end, decay included -------------------------------
        ratios = []
        for dw in (0.001, 0.010, 0.200):
            bb = _belief()
            bb.update(_scan_obs(bb.grid, 10, 0.0, dw))
            ratios.append(float(bb.p[10]) / p0)
        self.assertTrue(
            ratios[0] > ratios[1] > ratios[2], f"not monotone in dwell: {ratios}"
        )
        self.assertGreater(ratios[0], 0.7)
        self.assertLess(ratios[2], 0.35)


class TestConvergence(unittest.TestCase):
    """(5) Repeated evidence wins against the decay."""

    def test_strong_permanent_emitter_converges_up(self):
        b = _belief()
        grid = b.grid
        ch = 50
        t = 0.0
        for i in range(10):
            b.update(_scan_obs(grid, ch, t, 0.005, snr_db=-10.0, step=i))
            t += 0.010
        self.assertGreater(float(b.p_active(t)[ch]), 0.99)
        # Untouched channels are still sitting at the prior.
        self.assertAlmostEqual(float(b.p_active(t)[0]), float(b.pi_on[0]), places=6)

    def test_permanently_empty_channel_is_held_below_the_prior(self):
        b = _belief()
        grid = b.grid
        ch = 7
        pi = float(b.pi_on[ch])
        t = 0.0
        seen = []
        for i in range(12):
            b.update(_scan_obs(grid, ch, t, 0.100, step=i))
            t += 0.120
            seen.append(float(b.p_active(t)[ch]))
        self.assertTrue(
            all(v < pi for v in seen),
            f"empty channel drifted back to/above the prior: {seen}",
        )
        # It is held down, but never to certainty -- so it stays revisitable.
        self.assertGreater(seen[-1], 0.0)
        self.assertLess(seen[-1], 0.5 * pi)


class TestMarginalPdOrdering(unittest.TestCase):
    """(6) pd_bar rises with dwell and falls with bandwidth."""

    def setUp(self):
        self.bw = np.array([1, 2, 5, 10, 20], dtype=float) * 1e6
        self.dwell = np.array([1, 2, 5, 10, 20, 50, 100, 200], dtype=float) * 1e-3
        self.tbl = marginal_pd_table(self.bw, self.dwell, -15.0, 5.0, PFA)

    def test_shape_and_range(self):
        self.assertEqual(self.tbl.shape, (self.bw.size, self.dwell.size))
        self.assertTrue(np.all(self.tbl > PFA))
        self.assertTrue(np.all(self.tbl < 1.0))

    def test_monotone_in_dwell_and_bandwidth(self):
        self.assertTrue(np.all(np.diff(self.tbl, axis=1) > 0.0), "not rising with dwell")
        self.assertTrue(np.all(np.diff(self.tbl, axis=0) < 0.0), "not falling with bw")

    def test_bandwidth_penalty_is_load_bearing(self):
        """Without a real penalty the widest scan strictly dominates and the
        bandwidth knob is degenerate.  At 1 dB/octave, 20 MHz needs roughly an
        order of magnitude more dwell to match 1 MHz."""
        narrow_1ms = self.tbl[0, 0]
        wide = self.tbl[4, :]
        # 20 MHz never catches 1 MHz at the same dwell ...
        self.assertTrue(np.all(wide < self.tbl[0, :]))
        # ... and needs several steps up the dwell ladder to match it.
        self.assertLess(self.tbl[4, 1], narrow_1ms)

    def test_lookup_picks_the_nearest_candidate(self):
        b = _belief()
        self.assertAlmostEqual(
            b.pd_bar_for(1.0e6, 0.010), float(b.pd_bar[0, 3]), places=12
        )
        self.assertAlmostEqual(
            b.pd_bar_for(19.0e6, 0.19), float(b.pd_bar[4, 7]), places=12
        )


class TestFeatureContract(unittest.TestCase):
    """(7) The (N, F) matrix agent D trains on."""

    def _check(self, feats, n):
        self.assertEqual(feats.shape, (n, N_FEATURES))
        self.assertEqual(feats.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(feats)), "non-finite feature")

    def test_cold_start(self):
        b = _belief()
        self._check(b.feature_matrix(0.0), b.n)
        self._check(b.feature_matrix(59.999), b.n)

    def test_after_a_mixed_hit_and_miss_sequence(self):
        b = _belief()
        grid = b.grid
        rng = np.random.default_rng(3)
        t = 0.0
        for i in range(120):
            ch = int(rng.integers(0, b.n))
            snr = float(rng.uniform(-22.0, -8.0)) if rng.random() < 0.4 else None
            dwell = float(rng.choice([0.001, 0.005, 0.02, 0.1]))
            b.update(_scan_obs(grid, ch, t, dwell, snr_db=snr, step=i))
            t += dwell + 0.002
        self._check(b.feature_matrix(t), b.n)

    def test_column_order_and_semantics(self):
        b = _belief()
        grid = b.grid
        b.update(_scan_obs(grid, 50, 0.0, 0.005, snr_db=-11.0))
        f = b.feature_matrix(1.0)
        col = {name: i for i, name in enumerate(FEATURE_NAMES)}

        np.testing.assert_allclose(f[:, col["p_rung1"]], b.p_active(1.0), atol=1e-12)
        self.assertEqual(float(f[50, col["n_visits"]]), 1.0)
        self.assertEqual(float(f[0, col["n_visits"]]), 0.0)
        # Never-detected channels carry the sentinel, not a silent zero.
        self.assertEqual(float(f[0, col["mean_snr_db"]]), SENTINEL_NO_SNR)
        self.assertAlmostEqual(float(f[50, col["mean_snr_db"]]), -11.0, places=9)
        np.testing.assert_allclose(f[:, col["w_channel"]], b.mission.w, atol=0)
        self.assertAlmostEqual(
            float(f[0, col["t_frac"]]), 1.0 / b.horizon_s, places=12
        )

    def test_laplace_mode_never_decays(self):
        """The `greedy` ablation's belief: a raw Laplace hit rate.  If this
        quietly kept the Markov propagation, the ablation would prove nothing."""
        b = _belief(mode="laplace")
        p0 = b.p_active(0.0).copy()
        np.testing.assert_allclose(p0, 0.5, atol=1e-12)   # (0+1)/(0+2)
        np.testing.assert_allclose(b.p_active(1e6), p0, atol=0)
        b.update(_scan_obs(b.grid, 3, 0.0, 0.005, snr_db=-10.0))
        self.assertGreater(float(b.p_active(0.005)[3]), 0.5)
        self.assertAlmostEqual(float(b.p_active(1e6)[3]), float(b.p_active(0.005)[3]))

    def test_null_obs_is_absorbed_without_special_casing(self):
        b = _belief()
        before = b.p.copy()
        b.update(null_obs())
        np.testing.assert_array_equal(b.p, before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
