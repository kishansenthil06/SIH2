"""Tests for the `epsilon_greedy` baseline and the vendored bandit it wraps.

Two things need pinning here, and they are different in kind:

* The **vendored** `MLScheduler` is someone else's code kept byte-for-byte. These
  tests characterise it rather than judge it, so that if it is ever re-synced
  from source a behavioural change shows up as a failure instead of a silent
  drift in a baseline everyone has stopped looking at.
* The **wrapper** exists to make the comparison fair. Every fairness property it
  supplies -- same action as the tuned sweep, one band per legal scan, horizon
  survival, the shared reward -- is a property someone could reasonably accuse us
  of rigging, so each one is asserted explicitly.
"""
from __future__ import annotations

import random
import unittest

from agent.policy_bandit import EpsilonGreedyPolicy
from agent.vendor.ml_scheduler import MLScheduler
from sim.config import build_grid, build_mission, load_config
from sim.contract import Scan, Sleep
from sim.env import make_world


def _short_cfg(name: str = "sparse", horizon: float = 6.0) -> dict:
    cfg = load_config(name)
    cfg["horizon_s"] = horizon
    # Scale the budget with the horizon so the 0.1 W average -- the thing the
    # pacing has to respect -- is preserved rather than accidentally relaxed.
    cfg["energy"]["budget_j"] = 0.1 * horizon
    return cfg


class TestVendoredScheduler(unittest.TestCase):
    """Characterisation of the vendored bandit, unmodified."""

    def test_hit_rate_is_a_lifetime_average_with_no_decay(self):
        """This is the whole reason it underperforms; pin it explicitly.

        A band busy early keeps its score forever, which is the stationarity
        assumption DESIGN.md 11.13 identifies as the failure mechanism.
        """
        s = MLScheduler([0, 1], epsilon=0.0)
        for _ in range(5):
            s.update(0, reward=1.0, hit=True)
        self.assertEqual(s.get_hit_rate(0), 1.0)
        for _ in range(5):
            s.update(0, reward=0.0, hit=False)
        # 5 hits in 10 scans -- the early hits still count exactly as much as the
        # recent misses.  A decaying estimator would be below 0.5 here.
        self.assertEqual(s.get_hit_rate(0), 0.5)

    def test_unscanned_band_scores_zero_not_optimistic(self):
        """Zero, not infinity: an unvisited band is never preferred on novelty."""
        s = MLScheduler([0, 1], epsilon=0.0)
        self.assertEqual(s.get_hit_rate(1), 0.0)
        s.update(0, reward=1.0, hit=True)
        self.assertEqual(s.choose_band(), 0)   # exploits, never explores at eps=0

    def test_epsilon_one_is_pure_exploration(self):
        s = MLScheduler(list(range(50)), epsilon=1.0)
        random.seed(0)
        picks = {s.choose_band() for _ in range(200)}
        self.assertGreater(len(picks), 1)

    def test_statistics_reports_every_band(self):
        s = MLScheduler([0, 1, 2], epsilon=0.2)
        s.update(1, reward=0.5, hit=True)
        st = s.get_statistics()
        self.assertEqual(set(st), {0, 1, 2})
        self.assertEqual(st[1], {"scans": 1, "hits": 1, "hit_rate": 1.0, "reward": 0.5})


class TestWrapperFairness(unittest.TestCase):
    """Each assertion here answers "did you rig the baseline?" with a test."""

    def setUp(self):
        self.cfg = _short_cfg()
        self.grid = build_grid(self.cfg)
        self.mission = build_mission(self.cfg)
        self.p = EpsilonGreedyPolicy()
        self.p.reset(self.grid, self.mission, self.cfg["horizon_s"], 0, self.cfg)

    def test_uses_the_same_action_as_the_tuned_sweep(self):
        rr = self.cfg["baselines"]["round_robin"]
        self.assertEqual(self.p.n_ch, int(rr["bw_mhz"]))
        self.assertAlmostEqual(self.p.dwell_s, float(rr["dwell_ms"]) * 1e-3)

    def test_one_band_is_exactly_one_legal_scan(self):
        """Every arm must be reachable in a single action, or the bandit is being
        asked to point at something it cannot scan."""
        self.assertEqual(self.p.n_bands, self.grid.n_channels // self.p.n_ch)
        for band in (0, self.p.n_bands // 2, self.p.n_bands - 1):
            act = self.grid.action_for(band * self.p.n_ch, self.p.n_ch, self.p.dwell_s)
            covered = self.grid.channels_for(act.f_center_hz, act.bw_hz)
            self.assertEqual(len(covered), self.p.n_ch)

    def test_pacing_accounts_for_retune_not_just_dwell(self):
        """Retune dominates because the bandit hops without regard to locality.

        Costing only the scan under-estimated the true step cost by ~2x and the
        episode died at t=39 s with the budget gone -- which flatters a baseline
        by truncating its energy-per-detection denominator.
        """
        self.assertGreater(self.p.sleep_between_s, 0.0)
        scan_only = self.p.L_0 + self.p.L_d * self.p.dwell_s
        with_retune = scan_only + self.p.L_f * (self.grid.span_hz / 3.0)
        self.assertGreater(with_retune, 1.5 * scan_only)


class TestWrapperEndToEnd(unittest.TestCase):
    def _run(self, scenario="sparse", seed=0, horizon=6.0):
        cfg = _short_cfg(scenario, horizon)
        w = make_world(cfg, seed=seed)
        env = w.agent_view()
        p = EpsilonGreedyPolicy()
        o = env.reset(cfg, seed)
        p.reset(env.grid, env.mission, env.horizon_s, seed, cfg)
        n = 0
        while not o.done and n < 100_000:
            o = env.step(p.act(o))
            n += 1
        return p, o, cfg

    def test_reaches_the_horizon_within_budget(self):
        """A baseline that dies early looks artificially efficient."""
        _p, o, cfg = self._run()
        self.assertGreaterEqual(o.t, cfg["horizon_s"] - 1e-6)
        self.assertLessEqual(o.energy_total, cfg["energy"]["budget_j"] + 1e-9)

    def test_emits_only_legal_actions(self):
        cfg = _short_cfg()
        w = make_world(cfg, seed=0)
        env = w.agent_view()
        p = EpsilonGreedyPolicy()
        o = env.reset(cfg, 0)
        p.reset(env.grid, env.mission, env.horizon_s, 0, cfg)
        n = 0
        while not o.done and n < 100_000:
            a = p.act(o)
            if isinstance(a, Scan):
                env.grid.channels_for(a.f_center_hz, a.bw_hz)   # raises if illegal
                self.assertAlmostEqual(a.dwell_s, p.dwell_s)
            else:
                self.assertIsInstance(a, Sleep)
                self.assertGreater(a.dt_s, 0.0)
            o = env.step(a)
            n += 1

    def test_it_actually_learns_when_there_is_something_to_learn(self):
        """Guards against the wrapper silently never calling `update`.

        `dense`/seed 1 is chosen deliberately: seed 0 lands **zero** hits across
        78 scans on the same scenario, which is not a wrapper bug but the
        baseline's own poor exploration (78 scans spread over 400 bands, with
        epsilon-greedy camping on whatever it found first).  Asserting the
        credit path works needs a seed where there is something to credit.
        """
        p, _o, _cfg = self._run(scenario="dense", seed=1, horizon=20.0)
        st = p.statistics()
        self.assertGreater(sum(v["scans"] for v in st.values()), 0)
        self.assertGreater(sum(v["hits"] for v in st.values()), 0,
                           "the bandit was never credited with a detection")

    def test_a_zero_hit_episode_is_the_baseline_not_a_bug(self):
        """Pins the finding in DESIGN.md 11.13 rather than leaving it folklore.

        The bandit detects nothing on 9 of 15 full-length episodes.  If a future
        change makes this pass trivially, the baseline has silently become a
        different (better) algorithm and the comparison in the README no longer
        describes what is in the repo.
        """
        p, _o, _cfg = self._run(scenario="dense", seed=0, horizon=20.0)
        st = p.statistics()
        self.assertGreater(sum(v["scans"] for v in st.values()), 50)
        self.assertEqual(sum(v["hits"] for v in st.values()), 0)

    def test_reproducible_on_seed(self):
        a, _, _ = self._run(seed=1)
        b, _, _ = self._run(seed=1)
        self.assertEqual(a.statistics(), b.statistics())


if __name__ == "__main__":
    unittest.main(verbosity=2)
