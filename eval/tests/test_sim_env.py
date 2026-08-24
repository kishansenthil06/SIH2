"""`World`: the clock, the energy ledger, the firewall, and the guarantee that
the policy cannot touch the world it is being measured in.

Every row in `results/runs.csv` is a comparison of two policies on "the same
seed".  That phrase only means something if truth is a pure function of
`(scenario, seed)` -- if any action could perturb it, "index used 66% less energy
than round_robin" would be a comparison of two different worlds and the headline
number would be an artefact.  So the first two classes here are not hygiene
tests; they are the precondition for the whole evaluation.

The rest pins the arithmetic that the objective is written in:

* **clock** -- `duration = t_retune + dwell`, and the receiver is deaf during the
  retune, which is why hopping costs mission time as well as joules;
* **energy** -- `E = L_0 + L_d*dwell + L_f*|df|` to 1e-15, on the *executed*
  action.  `sim/config.py` already asserts `L_f == L_d / f_slew`, so timing and
  energy cannot drift apart; this checks the other half, that the ledger
  actually charges what the model says;
* **truncation** -- an over-long action is truncated rather than rejected, and
  `Obs.action` reports what was ACTUALLY executed (contract, `Obs.action`).  A
  metric that replayed the *requested* dwell would over-count coverage at the
  horizon;
* **firewall** -- structurally and at runtime.  `eval/tests/test_firewall.py`
  covers this from the eval side too; a breach silently inflates every number in
  the write-up, so cheap duplication is the right trade.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from sim.contract import ChannelGrid, FirewallViolation, Mission, Scan, Sleep
from sim.env import AgentEnv, World, make_world

L_D = 1.0
L_F = 2.0e-11
L_0 = 2.0e-3
L_SLEEP = 0.01
T_SETTLE = 0.5e-3
F_SLEW = 50.0e9


def base_cfg(**over) -> dict:
    """A minimal VALID config dict.  Deliberately tiny: 20 channels and a 1 s
    horizon, so a test that steps thousands of times still finishes in
    milliseconds.  The constants are the shipped ones (DESIGN.md section 1) --
    only the grid, the horizon and the budget are shrunk.
    """
    cfg = dict(
        name="unit-env",
        horizon_s=1.0,
        grid=dict(f_start_hz=2.0e9, n_channels=20, channel_bw_hz=1.0e6),
        receiver=dict(
            pfa=1.0e-3, t_settle_s=T_SETTLE, f_slew_hz_per_s=F_SLEW,
            bw_penalty_db_per_octave=1.0, snr_est_sigma_db=1.5,
            gain_enabled=False, gain_db_high=10.0, gain_nf_improvement_db=6.0,
            gain_energy_mult=1.6, gain_saturation_snr_db=-5.0,
            gain_fa_mult_on_saturation=10.0,
        ),
        energy=dict(L_d_w=L_D, L_0_j=L_0, L_f_j_per_hz=L_F,
                    L_sleep_w=L_SLEEP, budget_j=1.0e9),
        mission=dict(
            priority_bands=[dict(ch_lo=0, ch_hi=5, priority=1),
                            dict(ch_lo=0, ch_hi=20, priority=3)],
            weights={"1": 1.000, "3": 0.100},
            deadlines_s={"1": 30.0, "3": 20.0},
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


def obs_fingerprint(obs) -> tuple:
    """Everything an evaluator or a policy could possibly read from one step.

    Compared field by field rather than by `==` on the dataclass, because `Obs`
    holds a numpy array (ambiguous truth value) and an `info` dict the agent is
    contractually required to ignore.
    """
    return (
        obs.t, obs.t_start, obs.energy_cost, obs.energy_total,
        obs.step_index, obs.done,
        tuple(int(c) for c in obs.scanned_channels),
        tuple((d.channel, d.f_hz, d.bw_hz, d.snr_db) for d in obs.detections),
        type(obs.action).__name__,
        getattr(obs.action, "dwell_s", getattr(obs.action, "dt_s", None)),
    )


def run(world: World, actions) -> list[tuple]:
    return [obs_fingerprint(world.step(a)) for a in actions]


def scan_plan(grid: ChannelGrid) -> list:
    """A deliberately varied action list: hops, rescans, sleeps, wide and narrow."""
    return [
        grid.action_for(0, 1, 0.010),
        grid.action_for(0, 1, 0.010),      # rescan, same frequency
        grid.action_for(15, 5, 0.002),     # long hop, wide
        Sleep(0.020),
        grid.action_for(15, 5, 0.020),     # same f as before the sleep
        grid.action_for(3, 2, 0.001),
        Sleep(0.005),
        grid.action_for(10, 10, 0.050),
    ]


class TestDeterminism(unittest.TestCase):
    """Same `(scenario, seed)` -> the same world, and the same observations.

    Detector noise is counter-based Philox keyed on `(seed, step_index)`, not on
    consumption order, so two policies issuing the same scan at the same step
    index see the same coin flips.  That is common random numbers, and it is
    what tightens the confidence interval on the headline energy ratio -- so it
    has to be an exact property, not an approximate one.
    """

    def test_truth_bursts_are_byte_identical(self):
        a = make_world(base_cfg(), 7).truth_bursts()
        b = make_world(base_cfg(), 7).truth_bursts()
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(a.dtype, b.dtype)

    def test_shipped_scenarios_are_byte_identical_too(self):
        """The tiny config could hide a bug that only bites at 2000 channels."""
        for scenario in ("sparse", "dense"):
            with self.subTest(scenario=scenario):
                a = make_world(scenario, 3).truth_bursts()
                b = make_world(scenario, 3).truth_bursts()
                self.assertEqual(a.tobytes(), b.tobytes())
                self.assertGreater(a.size, 0)

    def test_different_seeds_give_different_worlds(self):
        a = make_world(base_cfg(), 7).truth_bursts()
        b = make_world(base_cfg(), 8).truth_bursts()
        self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_identical_actions_give_identical_obs_sequences(self):
        w1, w2 = make_world(base_cfg(), 11), make_world(base_cfg(), 11)
        self.assertEqual(run(w1, scan_plan(w1.grid)), run(w2, scan_plan(w2.grid)))

    def test_reset_returns_the_null_obs_and_rewinds_everything(self):
        w = make_world(base_cfg(), 4)
        before = run(w, scan_plan(w.grid))
        obs = w.reset(base_cfg(), 4)
        self.assertEqual(obs.step_index, -1)
        self.assertEqual(obs.t, 0.0)
        self.assertEqual(obs.energy_total, 0.0)
        self.assertFalse(obs.done)
        self.assertEqual(w.t, 0.0)
        self.assertEqual(w.energy_total, 0.0)
        self.assertEqual(w.step_index, -1)
        self.assertEqual(w.f_last_hz, float(w.grid.center_hz(0)))
        self.assertEqual(before, run(w, scan_plan(w.grid)))

    def test_detector_noise_is_keyed_on_step_index_not_on_history(self):
        """Two policies issuing the same scan at the same step index get the
        same coin flips, whatever they did before.

        This is common random numbers, and it is what tightens the confidence
        interval on the headline energy ratio.  Tested on an emitter-free world
        with an inflated `pfa`, so every detection comes from the RNG and
        nothing else: if the two lists match, the streams match.

        `_PHILOX_STRIDE` is why this works.  Philox ADVANCES its counter as it
        emits, so `counter=step_index` (the brief's literal wording) would give
        neighbouring steps almost entirely overlapping streams -- measured at
        350 false alarms where 200 +/- 42 were expected.  The 2**64 stride keeps
        the noise a pure function of `(seed, step_index)` without that overlap.
        """
        cfg = base_cfg(horizon_s=5.0)
        cfg["emitters"] = []
        cfg["grid"]["n_channels"] = 400
        cfg["mission"]["priority_bands"] = [dict(ch_lo=0, ch_hi=400, priority=3)]
        cfg["receiver"]["pfa"] = 0.2          # fire often, so the test is not vacuous

        w1, w2 = make_world(cfg, 21), make_world(cfg, 21)
        probe = w1.grid.action_for(0, 400, 0.010)

        # Step 0: wildly different histories, including different RNG consumption.
        w1.step(w1.grid.action_for(100, 200, 0.050))
        w2.step(Sleep(0.500))
        # Step 1: the same scan from both.
        o1, o2 = w1.step(probe), w2.step(probe)

        self.assertEqual(o1.step_index, o2.step_index)
        self.assertNotAlmostEqual(o1.t, o2.t, places=6, msg="clocks must differ")
        self.assertGreater(len(o1.detections), 50, "need a real sample to compare")
        self.assertEqual([d.channel for d in o1.detections],
                         [d.channel for d in o2.detections])
        self.assertEqual([d.snr_db for d in o1.detections],
                         [d.snr_db for d in o2.detections])

    def test_neighbouring_steps_do_not_share_a_noise_stream(self):
        """The `_PHILOX_STRIDE` regression, stated as the property it protects.

        Consecutive steps must draw genuinely different uniforms; the historical
        bug made step i+1's channel 0 reuse step i's channel 4.
        """
        cfg = base_cfg(horizon_s=5.0)
        cfg["emitters"] = []
        cfg["grid"]["n_channels"] = 400
        cfg["mission"]["priority_bands"] = [dict(ch_lo=0, ch_hi=400, priority=3)]
        cfg["receiver"]["pfa"] = 0.2

        w = make_world(cfg, 21)
        probe = w.grid.action_for(0, 400, 0.010)
        a = [d.channel for d in w.step(probe).detections]
        b = [d.channel for d in w.step(probe).detections]
        self.assertGreater(len(a), 50)
        self.assertNotEqual(a, b)
        overlap = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
        self.assertLess(overlap, 0.5, "consecutive steps must not be correlated")


class TestTruthIsIndependentOfThePolicy(unittest.TestCase):
    """THE property that makes cross-policy comparison on one seed valid.

    The burst table is generated in full inside `reset()` from `rng_emitters`,
    which is consumed entirely before the policy has acted once.  So no action
    -- no matter how many, how long, or in what order -- can move an emitter.
    Every number in `results/runs.csv` depends on this.
    """

    def _worlds(self):
        w_scan = make_world(base_cfg(), 5)
        for k in range(40):
            w_scan.step(w_scan.grid.action_for(k % 20, 1, 0.001))

        w_mixed = make_world(base_cfg(), 5)
        for k in range(40):
            w_mixed.step(
                Sleep(0.002) if k % 3 else w_mixed.grid.action_for(19 - k % 20, 1, 0.02)
            )

        w_wide = make_world(base_cfg(), 5)
        for _ in range(10):
            w_wide.step(w_wide.grid.action_for(0, 20, 0.05))

        w_idle = make_world(base_cfg(), 5)          # never stepped at all
        return w_scan, w_mixed, w_wide, w_idle

    def test_burst_table_is_unchanged_by_any_action_sequence(self):
        ws = self._worlds()
        ref = ws[-1].truth_bursts().tobytes()
        self.assertGreater(ws[-1].truth_bursts().size, 0)
        for w in ws:
            self.assertEqual(w.truth_bursts().tobytes(), ref)

    def test_rasters_are_unchanged_too(self):
        ws = self._worlds()
        occ, pwr = ws[-1].truth(1e-3), ws[-1].truth_power(1e-3)
        for w in ws:
            np.testing.assert_array_equal(w.truth(1e-3), occ)
            np.testing.assert_array_equal(w.truth_power(1e-3), pwr)

    def test_the_clocks_really_did_diverge(self):
        """Guards against the previous two tests passing because nothing ran."""
        w_scan, w_mixed, w_wide, w_idle = self._worlds()
        self.assertEqual(w_idle.t, 0.0)
        self.assertGreater(w_scan.t, 0.0)
        self.assertNotAlmostEqual(w_scan.t, w_mixed.t, places=6)
        self.assertNotAlmostEqual(w_scan.energy_total, w_wide.energy_total, places=6)


class TestClock(unittest.TestCase):
    """`duration = t_retune + dwell`, exactly.

    `t_retune = t_settle + |df|/f_slew`, and the receiver is DEAF while the
    synthesiser settles -- detection integrates `[t0, t0+dwell)` only.  That is
    why hopping is expensive in mission time as well as in joules, and it is the
    reason `score_rate` divides by `t_retune + dwell` rather than by dwell.
    """

    def setUp(self):
        self.w = make_world(base_cfg(horizon_s=10.0), 2)
        self.grid = self.w.grid

    def test_duration_is_retune_plus_dwell(self):
        for k, n, dwell in ((0, 1, 0.010), (19, 1, 0.002), (5, 10, 0.050),
                            (5, 10, 0.001), (0, 20, 0.020)):
            with self.subTest(k=k, n=n, dwell=dwell):
                obs = self.w.step(self.grid.action_for(k, n, dwell))
                want = obs.info["t_retune"] + obs.action.dwell_s
                self.assertAlmostEqual(obs.duration_s, want, delta=1e-15)
                self.assertAlmostEqual(obs.t - obs.t_start, want, delta=1e-15)

    def test_retune_matches_the_settle_plus_slew_model(self):
        self.w.step(self.grid.action_for(0, 1, 0.001))       # park at channel 0
        obs = self.w.step(self.grid.action_for(10, 1, 0.001))
        df = abs(float(self.grid.center_hz(10)) - float(self.grid.center_hz(0)))
        self.assertAlmostEqual(df, 10.0e6, delta=1e-6)
        self.assertAlmostEqual(obs.info["t_retune"], T_SETTLE + df / F_SLEW, delta=1e-15)

    def test_a_same_frequency_rescan_pays_no_retune(self):
        a = self.grid.action_for(7, 5, 0.005)
        self.w.step(self.grid.action_for(0, 1, 0.001))
        first = self.w.step(a)
        second = self.w.step(a)
        self.assertGreater(first.info["t_retune"], 0.0)
        self.assertEqual(second.info["t_retune"], 0.0)
        self.assertAlmostEqual(second.duration_s, second.action.dwell_s, delta=1e-15)

    def test_the_first_scan_of_an_episode_starts_parked_at_channel_zero(self):
        obs = self.w.step(self.grid.action_for(0, 1, 0.001))
        self.assertEqual(obs.info["t_retune"], 0.0)

    def test_sleep_leaves_f_last_unchanged(self):
        """The VCO stays parked while asleep -- sleeping is never a free hop."""
        self.w.step(self.grid.action_for(12, 1, 0.001))
        f_before = self.w.f_last_hz
        obs = self.w.step(Sleep(0.100))
        self.assertEqual(self.w.f_last_hz, f_before)
        self.assertEqual(obs.info["t_retune"], 0.0)
        self.assertEqual(obs.duration_s, 0.100)
        # ... and the next scan at that same frequency still pays nothing.
        self.assertEqual(self.w.step(self.grid.action_for(12, 1, 0.001)).info["t_retune"],
                         0.0)
        # ... while a different frequency pays from where it was parked, not
        # from wherever it drifted.
        obs = self.w.step(self.grid.action_for(2, 1, 0.001))
        df = abs(float(self.grid.center_hz(2)) - f_before)
        self.assertAlmostEqual(obs.info["t_retune"], T_SETTLE + df / F_SLEW, delta=1e-15)

    def test_the_clock_is_the_running_sum_of_durations(self):
        t = 0.0
        for a in scan_plan(self.grid) * 3:
            obs = self.w.step(a)
            self.assertAlmostEqual(obs.t_start, t, delta=1e-12)
            t = obs.t
        self.assertAlmostEqual(self.w.t, t, delta=1e-15)

    def test_detection_window_excludes_the_retune(self):
        """`info["t_dwell_start"]` is when the receiver actually opened its ears."""
        self.w.step(self.grid.action_for(0, 1, 0.001))
        obs = self.w.step(self.grid.action_for(19, 1, 0.010))
        self.assertAlmostEqual(obs.info["t_dwell_start"],
                               obs.t_start + obs.info["t_retune"], delta=1e-15)
        self.assertAlmostEqual(obs.t - obs.info["t_dwell_start"],
                               obs.action.dwell_s, delta=1e-15)


class TestEnergyLedger(unittest.TestCase):
    """`E = L_0 + L_d*dwell + L_f*|df|`, on the EXECUTED action.

    `sim/config.py` asserts `L_f == L_d / f_slew` so the energy model and the
    timing model are the same physics twice.  This asserts the ledger charges it.
    """

    def setUp(self):
        self.w = make_world(base_cfg(horizon_s=10.0), 6)
        self.grid = self.w.grid

    def test_scan_energy_identity_to_1e_15(self):
        f_prev = float(self.grid.center_hz(0))
        for a in scan_plan(self.grid) * 4:
            obs = self.w.step(a)
            if isinstance(obs.action, Sleep):
                continue                   # a sleep does not move the VCO
            df = abs(float(obs.action.f_center_hz) - f_prev)
            want = L_0 + L_D * obs.action.dwell_s + L_F * df
            self.assertAlmostEqual(obs.energy_cost, want, delta=1e-15)
            f_prev = float(obs.action.f_center_hz)

    def test_the_verified_constants_land_on_the_verified_numbers(self):
        """DESIGN.md section 1's check column, reproduced through the ledger."""
        obs = self.w.step(self.grid.action_for(0, 1, 0.010))
        self.assertAlmostEqual(obs.energy_cost, L_0 + 0.010, delta=1e-15)
        self.assertAlmostEqual(obs.energy_cost - L_0, 10.0e-3, delta=1e-15,
                               msg="10 ms dwell = 10 mJ")

        w = make_world(base_cfg(horizon_s=10.0, grid=dict(
            f_start_hz=2.0e9, n_channels=400, channel_bw_hz=1.0e6)), 6)
        w.step(w.grid.action_for(0, 1, 0.0))                 # park at channel 0
        obs = w.step(w.grid.action_for(200, 1, 0.0))
        df = abs(float(w.grid.center_hz(200)) - float(w.grid.center_hz(0)))
        self.assertAlmostEqual(df, 200.0e6, delta=1e-6)
        self.assertAlmostEqual(L_F * df, 4.0e-3, delta=1e-15, msg="200 MHz hop = 4 mJ")
        self.assertAlmostEqual(df / F_SLEW, 4.0e-3, delta=1e-15, msg="200 MHz hop = 4 ms")
        self.assertAlmostEqual(obs.energy_cost, L_0 + 4.0e-3, delta=1e-15)

    def test_any_scan_costs_at_least_l0(self):
        obs = self.w.step(self.grid.action_for(0, 1, 0.0))
        self.assertAlmostEqual(obs.energy_cost, L_0, delta=1e-15)

    def test_sleep_costs_l_sleep_times_dt(self):
        for dt in (0.010, 0.050, 0.100, 0.200):
            with self.subTest(dt=dt):
                obs = self.w.step(Sleep(dt))
                self.assertAlmostEqual(obs.energy_cost, L_SLEEP * dt, delta=1e-15)
        obs = self.w.step(Sleep(0.100))
        self.assertAlmostEqual(obs.energy_cost, 1.0e-3, delta=1e-15,
                               msg="100 ms sleep = 1 mJ = 1/12 of one 10 ms scan")

    def test_sleeping_is_twelve_times_cheaper_than_the_scan_it_replaces(self):
        """The number that makes `Sleep` the only meaningful saving."""
        w = make_world(base_cfg(horizon_s=10.0), 6)
        scan = w.step(w.grid.action_for(0, 1, 0.010)).energy_cost
        sleep = w.step(Sleep(0.100)).energy_cost
        self.assertAlmostEqual(scan / sleep, 12.0, delta=1e-9)

    def test_energy_total_is_the_running_sum(self):
        total = 0.0
        for a in scan_plan(self.grid) * 5:
            obs = self.w.step(a)
            total += obs.energy_cost
            self.assertAlmostEqual(obs.energy_total, total, delta=1e-15)
        self.assertAlmostEqual(self.w.energy_total, total, delta=1e-15)

    def test_energy_is_never_negative_and_never_free(self):
        for a in scan_plan(self.grid):
            obs = self.w.step(a)
            self.assertGreater(obs.energy_cost, 0.0)


class TestTruncation(unittest.TestCase):
    """An over-long action is truncated, not rejected -- and `Obs.action` says so.

    The contract calls `Obs.action` "what was ACTUALLY executed".  `eval/metrics.py`
    replays the recorded action log against the burst table to score coverage, so
    if the log carried the *requested* dwell every policy would be credited with
    coverage past the horizon that it never had.
    """

    def test_a_scan_overrunning_the_horizon_is_truncated(self):
        w = make_world(base_cfg(horizon_s=0.1), 1)
        w.step(Sleep(0.090))
        requested = w.grid.action_for(10, 1, 0.050)
        obs = w.step(requested)

        self.assertLess(obs.action.dwell_s, requested.dwell_s)
        self.assertAlmostEqual(obs.t, 0.1, delta=1e-15)
        # Assert against the RETURNED action, never the requested one.
        self.assertAlmostEqual(
            obs.action.dwell_s, 0.1 - obs.t_start - obs.info["t_retune"], delta=1e-15
        )
        self.assertEqual(obs.action.f_center_hz, requested.f_center_hz)
        self.assertEqual(obs.action.bw_hz, requested.bw_hz)

    def test_the_energy_charged_matches_the_truncated_dwell(self):
        """Truncation must not be a way to buy 50 ms of dwell for 9 ms of joules,
        nor a way to be charged for time the receiver never spent listening."""
        w = make_world(base_cfg(horizon_s=0.1), 1)
        w.step(Sleep(0.090))
        f_prev = w.f_last_hz
        obs = w.step(w.grid.action_for(10, 1, 0.050))
        df = abs(float(obs.action.f_center_hz) - f_prev)
        self.assertAlmostEqual(obs.energy_cost,
                               L_0 + L_D * obs.action.dwell_s + L_F * df, delta=1e-15)

    def test_a_sleep_overrunning_the_horizon_is_truncated(self):
        w = make_world(base_cfg(horizon_s=0.1), 1)
        obs = w.step(Sleep(5.0))
        self.assertAlmostEqual(obs.action.dt_s, 0.1, delta=1e-15)
        self.assertAlmostEqual(obs.t, 0.1, delta=1e-15)
        self.assertAlmostEqual(obs.energy_cost, L_SLEEP * 0.1, delta=1e-15)

    def test_actions_at_or_past_the_horizon_execute_as_nothing(self):
        w = make_world(base_cfg(horizon_s=0.1), 1)
        w.step(Sleep(5.0))                       # already at the horizon
        obs = w.step(Sleep(1.0))
        self.assertEqual(obs.action.dt_s, 0.0)
        self.assertEqual(obs.energy_cost, 0.0)

        obs = w.step(w.grid.action_for(0, 1, 0.050))
        self.assertEqual(obs.action.dwell_s, 0.0)
        self.assertEqual(obs.detections, (), "a zero-length dwell collects N=0 samples")
        self.assertAlmostEqual(obs.energy_cost, L_0, delta=1e-15)

    def test_a_negative_sleep_is_clamped_to_zero(self):
        w = make_world(base_cfg(horizon_s=1.0), 1)
        obs = w.step(Sleep(-3.0))
        self.assertEqual(obs.action.dt_s, 0.0)
        self.assertEqual(obs.t, 0.0)
        self.assertEqual(obs.energy_cost, 0.0)

    def test_retune_may_carry_the_clock_just_past_the_horizon(self):
        """DOCUMENTED BEHAVIOUR, not an accident: only the DWELL is truncated.

        A retune already in flight is not abandoned, so `t` can exceed
        `horizon_s` by at most one `t_retune` (here 0.78 ms on a 20 MHz grid;
        at most 40.5 ms on the shipped 2 GHz one).  The dwell truncates to
        exactly zero, so no sensing happens after the horizon and the metrics --
        which are censored at `T` anyway -- are unaffected.  Asserted so that a
        future change to either half is noticed rather than absorbed.
        """
        w = make_world(base_cfg(horizon_s=0.1), 1)
        w.step(w.grid.action_for(0, 1, 0.001))
        w.step(Sleep(1.0))                        # sit exactly on the horizon
        obs = w.step(w.grid.action_for(19, 1, 0.050))
        self.assertEqual(obs.action.dwell_s, 0.0)
        self.assertEqual(obs.detections, ())
        self.assertGreater(obs.t, 0.1)
        self.assertLessEqual(obs.t - 0.1, obs.info["t_retune"] + 1e-15)
        self.assertLess(obs.t - 0.1, 1.0e-3)

    def test_an_out_of_band_scan_still_raises_rather_than_truncating(self):
        """Truncation is for TIME.  A misaligned or out-of-band scan is a bug in
        the policy and must not silently do something reasonable-looking."""
        from sim.contract import GridError

        w = make_world(base_cfg(), 1)
        with self.assertRaises(GridError):
            w.step(Scan(f_center_hz=2.0e9 + 500e6, bw_hz=1e6, dwell_s=0.001))

    def test_a_non_action_is_rejected(self):
        w = make_world(base_cfg(), 1)
        with self.assertRaises(TypeError):
            w.step("scan please")


class TestDoneFlag(unittest.TestCase):
    """`done` iff `t >= horizon_s` OR `energy_total >= budget_j`.

    Both paths are verified independently, because `budget_j = 6.0` over a 60 s
    horizon is an average of 0.1 W -- roughly a 10% duty cycle -- so the budget
    path is the one that actually fires in practice.  A policy that ignores it
    exhausts the budget at t = 5.26 s, at 8.8% of the horizon (DESIGN.md
    section 1), which is why `done` has to be honest about which limit bit.
    """

    def test_done_on_the_horizon_with_an_unreachable_budget(self):
        w = make_world(base_cfg(horizon_s=0.1), 1)     # budget_j = 1e9
        obs = w.step(w.grid.action_for(0, 1, 0.010))
        self.assertFalse(obs.done)
        obs = w.step(Sleep(5.0))
        self.assertTrue(obs.done)
        self.assertGreaterEqual(obs.t, 0.1)
        self.assertLess(obs.energy_total, w.budget_j, "the BUDGET must not be why")

    def test_done_on_the_budget_with_an_unreachable_horizon(self):
        cfg = base_cfg(horizon_s=60.0)
        cfg["energy"]["budget_j"] = 0.05               # ~4 scans of 10 ms
        w = make_world(cfg, 1)
        self.assertEqual(w.budget_j, 0.05)

        seen = False
        for _ in range(20):
            obs = w.step(w.grid.action_for(0, 1, 0.010))
            if obs.done:
                seen = True
                break
        self.assertTrue(seen, "the budget must bind")
        self.assertGreaterEqual(obs.energy_total, 0.05)
        self.assertLess(obs.t, 60.0, "the HORIZON must not be why")

    def test_done_is_a_live_property_not_a_latched_flag(self):
        """`reset` must clear it, or a second episode would end immediately."""
        cfg = base_cfg(horizon_s=0.02)
        w = make_world(cfg, 1)
        self.assertTrue(w.step(Sleep(1.0)).done)
        self.assertTrue(w.done)
        w.reset(cfg, 1)
        self.assertFalse(w.done)

    def test_stepping_past_done_is_allowed_and_costs_nothing_extra(self):
        """The runner stops on `done`; the env does not enforce it.  Making that
        explicit stops anyone relying on an exception that is not thrown."""
        w = make_world(base_cfg(horizon_s=0.05), 1)
        w.step(Sleep(1.0))
        obs = w.step(Sleep(1.0))
        self.assertTrue(obs.done)
        self.assertEqual(obs.energy_cost, 0.0)


class TestFirewall(unittest.TestCase):
    """Structural and runtime halves of DESIGN.md section 2, from the sim side.

    `eval/tests/test_firewall.py` covers the same ground plus the static AST
    scan.  The duplication is deliberate and cheap: a breach does not crash
    anything, it silently inflates every number in the write-up.
    """

    def setUp(self):
        self.world = make_world(base_cfg(), 0)
        self.env = self.world.agent_view()

    def test_agent_view_returns_an_agent_env(self):
        self.assertIsInstance(self.env, AgentEnv)
        self.assertNotIsInstance(self.env, World)

    def test_agent_env_has_no_dict_to_attach_anything_to(self):
        self.assertFalse(hasattr(self.env, "__dict__"))
        with self.assertRaises(AttributeError):
            self.env.smuggled = self.world

    def test_agent_env_exposes_no_truth_attribute(self):
        for name in ("truth", "truth_bursts", "truth_power", "emitters",
                     "_bursts", "_world", "world", "cfg", "receiver", "seed"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.env, name),
                                 f"AgentEnv must not expose {name!r}")

    def test_no_slot_holds_a_world(self):
        self.assertEqual(
            AgentEnv.__slots__,
            ("_step", "_reset", "grid", "horizon_s", "mission", "action_space"),
        )
        for name in AgentEnv.__slots__:
            with self.subTest(name=name):
                self.assertNotIsInstance(getattr(self.env, name), World)

    def test_what_agent_env_does_expose_is_only_what_the_agent_may_know(self):
        self.assertIsInstance(self.env.grid, ChannelGrid)
        self.assertIsInstance(self.env.mission, Mission)
        self.assertEqual(self.env.horizon_s, self.world.horizon_s)
        self.assertEqual(set(self.env.action_space),
                         {"bw_candidates_mhz", "dwell_candidates_ms",
                          "sleep_candidates_ms"})

    def test_agent_env_still_works_as_an_environment(self):
        obs = self.env.step(self.env.grid.action_for(0, 1, 0.001))
        self.assertAlmostEqual(obs.t, 0.001, delta=1e-15)
        self.assertEqual(obs.step_index, 0)
        self.assertEqual(self.env.reset(base_cfg(), 0).step_index, -1)

    def _call_from_agent_module(self, method: str):
        """Execute `world.<method>()` inside a globals dict whose `__name__`
        starts with `agent.`.

        `_forbid_agent_callers` walks the stack looking at `f_globals["__name__"]`,
        so this is exactly the situation it exists to catch -- without planting a
        real breach in a real file.
        """
        ns = {"__name__": "agent.env_probe", "world": self.world}
        exec(compile(f"def probe():\n    return world.{method}()\n",
                     "<agent.env_probe>", "exec"), ns)
        return ns["probe"]()

    def test_truth_raises_firewall_violation_for_an_agent_caller(self):
        for method in ("truth", "truth_bursts", "truth_power"):
            with self.subTest(method=method):
                with self.assertRaises(FirewallViolation):
                    self._call_from_agent_module(method)

    def test_an_app_caller_is_blocked_too(self):
        ns = {"__name__": "app.server", "world": self.world}
        exec(compile("def probe():\n    return world.truth()\n",
                     "<app.server>", "exec"), ns)
        with self.assertRaises(FirewallViolation):
            ns["probe"]()

    def test_an_indirect_agent_caller_is_blocked(self):
        """The check walks 12 frames, so a helper chain cannot launder the call."""
        ns = {"__name__": "agent.policy_probe", "world": self.world}
        src = ("def outer():\n    return middle()\n"
               "def middle():\n    return inner()\n"
               "def inner():\n    return world.truth_bursts()\n")
        exec(compile(src, "<agent.policy_probe>", "exec"), ns)
        with self.assertRaises(FirewallViolation):
            ns["outer"]()

    def test_the_evaluator_side_may_read_truth(self):
        """This module is `eval.tests.test_sim_env` -- the evaluator side."""
        self.assertGreater(self.world.truth_bursts().size, 0)
        self.assertEqual(self.world.truth(1e-2).ndim, 2)

    def test_truth_bursts_is_handed_out_read_only(self):
        b = self.world.truth_bursts()
        self.assertFalse(b.flags.writeable)
        with self.assertRaises(ValueError):
            b["t_on"][0] = -1.0


class TestTruthRasterisation(unittest.TestCase):
    """`truth`/`truth_power` must agree with the burst table they are built from.

    `eval/metrics.py` scores coverage off the burst table while the `oracle`
    reads `truth_power`; if the two disagreed, the reference ceiling would be
    measured against a different world than the policies were.
    """

    @classmethod
    def setUpClass(cls):
        cls.horizon = 2.0
        cls.cfg = base_cfg(horizon_s=cls.horizon)
        cls.world = make_world(cls.cfg, 1)
        cls.bursts = cls.world.truth_bursts()
        assert cls.bursts.size > 2, "need a few bursts to check against"

    def _reference(self, dt_s: float, n_bins: int):
        """Brute-force raster, built the slow obvious way from the burst table."""
        n_ch = self.world.grid.n_channels
        occ = np.zeros((n_bins, n_ch), dtype=bool)
        pwr = np.zeros((n_bins, n_ch), dtype=np.float64)
        edges = np.arange(n_bins + 1) * dt_s
        for row in self.bursts:
            i_lo = int(np.searchsorted(edges, row["t_on"], side="right")) - 1
            i_hi = int(np.ceil(row["t_off"] / dt_s))
            i_lo = max(i_lo, 0)
            i_hi = min(i_hi, n_bins)
            if i_hi <= i_lo:
                continue
            occ[i_lo:i_hi, row["ch_lo"]:row["ch_hi"]] = True
            pwr[i_lo:i_hi, row["ch_lo"]:row["ch_hi"]] += 10.0 ** (row["snr_db"] / 10.0)
        return occ, pwr

    def test_shape_is_ceil_horizon_over_dt_by_n_channels(self):
        for dt_s in (1e-3, 2e-3, 3e-3, 1e-2, 0.04):
            with self.subTest(dt_s=dt_s):
                n_bins = int(np.ceil(self.horizon / dt_s - 1e-9))
                self.assertEqual(self.world.truth(dt_s).shape,
                                 (n_bins, self.world.grid.n_channels))
                self.assertEqual(self.world.truth_power(dt_s).shape,
                                 (n_bins, self.world.grid.n_channels))

    def test_a_non_dividing_dt_rounds_up(self):
        """2.0 s at 3 ms is 666.67 bins, so the last bin is a partial one."""
        self.assertEqual(self.world.truth(3e-3).shape[0], 667)

    def test_dtypes_are_bool_and_float32(self):
        self.assertEqual(self.world.truth(1e-3).dtype, np.bool_)
        self.assertEqual(self.world.truth_power(1e-3).dtype, np.float32)

    def test_occupancy_agrees_with_the_burst_table_cell_by_cell(self):
        for dt_s in (1e-3, 1e-2):
            with self.subTest(dt_s=dt_s):
                n_bins = int(np.ceil(self.horizon / dt_s - 1e-9))
                occ_ref, _ = self._reference(dt_s, n_bins)
                np.testing.assert_array_equal(self.world.truth(dt_s), occ_ref)
                self.assertTrue(occ_ref.any(), "the reference must not be empty")

    def test_power_agrees_with_the_burst_table_cell_by_cell(self):
        for dt_s in (1e-3, 1e-2):
            with self.subTest(dt_s=dt_s):
                n_bins = int(np.ceil(self.horizon / dt_s - 1e-9))
                _, pwr_ref = self._reference(dt_s, n_bins)
                np.testing.assert_allclose(
                    self.world.truth_power(dt_s), pwr_ref.astype(np.float32),
                    rtol=1e-6, atol=1e-9,
                )

    def test_a_hand_checked_window(self):
        """One burst, read straight off the table and looked up in both rasters.

        Deliberately not vectorised: if the reference above and the rasteriser
        ever shared a bug, this is the assertion that would still catch it.
        """
        dt_s = 1e-3
        raster = self.world.truth(dt_s)
        power = self.world.truth_power(dt_s)
        row = self.bursts[0]
        ch = int(row["ch_lo"])
        lin = 10.0 ** (float(row["snr_db"]) / 10.0)

        # A bin strictly inside the burst, on a channel it occupies.
        i_mid = int((float(row["t_on"]) + float(row["t_off"])) / 2.0 / dt_s)
        self.assertTrue(bool(raster[i_mid, ch]))
        self.assertGreaterEqual(float(power[i_mid, ch]), lin * (1.0 - 1e-6))

        # The same instant, one channel outside the burst block (this scenario's
        # emitters are 1 channel wide and randomly placed, so check it is genuinely
        # unoccupied before asserting).
        outside = int(row["ch_hi"]) % self.world.grid.n_channels
        covered = self.bursts[
            (self.bursts["ch_lo"] <= outside) & (self.bursts["ch_hi"] > outside)
            & (self.bursts["t_on"] < (i_mid + 1) * dt_s)
            & (self.bursts["t_off"] > i_mid * dt_s)
        ]
        self.assertEqual(covered.size, 0,
                         "fixed seed: this cell must be genuinely unoccupied")
        self.assertFalse(bool(raster[i_mid, outside]))
        self.assertEqual(float(power[i_mid, outside]), 0.0)

    def test_occupancy_is_exactly_where_power_is_positive(self):
        occ = self.world.truth(1e-3)
        pwr = self.world.truth_power(1e-3)
        np.testing.assert_array_equal(occ, pwr > 0.0)

    def test_rasters_are_cached_per_episode_and_cleared_on_reset(self):
        w = make_world(self.cfg, 1)
        first = w.truth(1e-3)
        self.assertIs(w.truth(1e-3), first, "rasterising twice is pure waste")
        w.reset(self.cfg, 1)
        self.assertIsNot(w.truth(1e-3), first, "a new episode needs a new raster")

    def test_an_emitter_free_world_rasterises_to_all_false(self):
        cfg = base_cfg(horizon_s=0.5)
        cfg["emitters"] = []
        w = make_world(cfg, 0)
        self.assertEqual(w.truth_bursts().size, 0)
        occ, pwr = w.truth(1e-3), w.truth_power(1e-3)
        self.assertEqual(occ.shape, (500, 20))
        self.assertFalse(bool(occ.any()))
        self.assertEqual(pwr.dtype, np.float32)
        np.testing.assert_array_equal(pwr, 0.0)


if __name__ == "__main__":
    unittest.main()
