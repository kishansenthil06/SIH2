"""Tests for `agent/policy_index.py` -- the rung-1 reward-rate index policy.

This is the policy the headline result belongs to, so the properties asserted
here are the ones that would make the headline meaningless if they were false:

1. **Every emitted action is legal.**  `IndexPolicy` never constructs a `Scan`
   directly; it goes through `ChannelGrid.action_for`, so alignment is meant to
   hold by construction.  "By construction" is a claim, and 5000 actions passed
   back through `ChannelGrid.channels_for` is the check.  Dwell and bandwidth
   must also come from the configured candidate sets -- an action outside them
   would be scored against a `pd_bar` entry that does not describe it.
2. **Sleep is reachable, and for the stated reason.**  DESIGN.md section 3 says
   `Sleep` needs no threshold because `score_rate(Sleep) = -L_sleep = -0.010 W`
   independent of `dt`, while a hopeless scan scores about -1.1 W.  So sleep is
   selected *exactly* when every scan has a negative reward rate.  That
   relationship is asserted numerically rather than inferred from behaviour.
3. **`L_f` is wired into the score.**  Two channels of equal value, one at the
   current frequency and one 150 MHz away: the near one must win, and the
   preference must flip when the receiver is retuned.  A one-sided test would
   pass on candidate ordering alone.
4. **Budget respected AND horizon reached.**  DESIGN.md section 1 is explicit
   that a policy which exhausts its budget at t = 5 s is a failed policy, not a
   frugal one.  Both halves are asserted together; either alone can be gamed.
5. **`score_mode: "raw"` still executes.**  The source document's literal form is
   the ablation in DESIGN.md 11.8 (POI 0.083, 16% coverage).  It must stay
   runnable so the ablation stays reproducible -- but nothing here asserts it
   performs well, because it measurably does not.
6. **`log_rows()` schema is pinned.**  `agent/policy_learned.py` labels these
   rows retroactively and trains on them; one renamed key or one NaN silently
   poisons rung 2.

Test files may import `sim.env` / `sim.stub_env`; `agent/` source files may not.
"""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from agent.base import FEATURE_NAMES, EnergyState
from agent.policy_index import IndexPolicy
from agent.scheduler import MIN_SLEEP_S
from sim.config import load_config
from sim.contract import GridError, Scan, Sleep
from sim.stub_env import StubEnv

# One parse of the YAML for the whole module; every config below is a copy.
_BASE = load_config("sparse")

# A 200-channel / 200 MHz band with the sparse mission's shape scaled 10:1.
# Full-size episodes on the real 2000-channel grid run 1500-4600 decisions at
# ~1.3 ms each, which no single test can afford; the band is narrowed rather
# than the action space, so every bandwidth and dwell candidate stays reachable
# and 150 MHz of retune separation still fits (see TestRetuneLocality).
SMALL_N = 200
SMALL_HORIZON = 8.0
# 0.1 W, exactly the binding average of DESIGN.md section 1.
NOMINAL_W = 0.1


def small_cfg(
    horizon_s: float = SMALL_HORIZON,
    n_channels: int = SMALL_N,
    budget_j: float | None = None,
    **agent_overrides,
) -> dict:
    """A scaled `sparse`.  Everything but the band width and the clock is real."""
    cfg = copy.deepcopy(_BASE)
    cfg.pop("config_hash", None)
    cfg["name"] = "sparse_small"
    cfg["horizon_s"] = float(horizon_s)
    cfg["grid"]["n_channels"] = int(n_channels)
    cfg["energy"]["budget_j"] = (
        NOMINAL_W * float(horizon_s) if budget_j is None else float(budget_j)
    )
    cfg["mission"]["priority_bands"] = [
        {"ch_lo": 40, "ch_hi": 60, "priority": 1},       # threat band
        {"ch_lo": 120, "ch_hi": 150, "priority": 2},
        {"ch_lo": 0, "ch_hi": int(n_channels), "priority": 3},
    ]
    cfg["emitters"][0]["channel_range"] = [40, 60]
    cfg["emitters"][1]["channel_range"] = [120, 150]
    cfg["emitters"][2]["channel_range"] = [0, int(n_channels)]
    cfg["agent"].update(agent_overrides)
    return load_config(cfg)                              # revalidate and re-hash


def run_episode(cfg: dict, seed: int, policy=None, env=None):
    """Drive one episode; return `(policy, actions, final_obs)`.

    Deliberately a local loop rather than `eval.runner.run_episode`: these tests
    are about what the POLICY emits, so the action list must be the actions the
    policy returned, not the (possibly truncated) ones the environment executed.
    """
    env = env or StubEnv(cfg)
    policy = policy or IndexPolicy()
    obs = env.reset(cfg, seed)
    policy.reset(env.grid, env.mission, env.horizon_s, seed, cfg)
    actions = []
    while not obs.done:
        action = policy.act(obs)
        actions.append(action)
        obs = env.step(action)
    return policy, actions, obs


def action_key(action):
    """A hashable, exact identity for an action -- used for determinism."""
    if isinstance(action, Scan):
        return ("scan", action.f_center_hz, action.bw_hz, action.dwell_s, action.gain_db)
    return ("sleep", action.dt_s)


# ===========================================================================
# 1. Action legality
# ===========================================================================
class TestActionLegality(unittest.TestCase):
    """5000 actions from `IndexPolicy` against `StubEnv`.

    The 5000 actions are collected once in `setUpClass` (~3.5 s) and each test
    below is then an assertion over that list, so no single test method runs an
    episode.  Seeds 0-19 are consumed in order until the quota is met.
    """

    N_ACTIONS = 5000

    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.env = StubEnv(cls.cfg)
        cls.grid = cls.env.grid
        cls.actions = []
        cls.n_episodes = 0
        for seed in range(20):
            _, actions, _ = run_episode(cls.cfg, seed, env=cls.env)
            cls.actions.extend(actions)
            cls.n_episodes += 1
            if len(cls.actions) >= cls.N_ACTIONS:
                break
        cls.actions = cls.actions[: cls.N_ACTIONS]
        cls.scans = [a for a in cls.actions if isinstance(a, Scan)]
        cls.sleeps = [a for a in cls.actions if isinstance(a, Sleep)]
        cls.bw_ok = {float(b) * 1e6 for b in cls.cfg["agent"]["bw_candidates_mhz"]}
        cls.dwell_ok = {float(d) * 1e-3 for d in cls.cfg["agent"]["dwell_candidates_ms"]}
        cls.sleep_ok = {float(d) * 1e-3 for d in cls.cfg["agent"]["sleep_candidates_ms"]}

    def test_collected_the_full_quota(self):
        self.assertEqual(len(self.actions), self.N_ACTIONS)
        self.assertGreater(len(self.scans), 0)
        self.assertGreater(len(self.sleeps), 0)

    def test_every_action_is_a_scan_or_a_sleep(self):
        for i, action in enumerate(self.actions):
            if not isinstance(action, (Scan, Sleep)):
                self.fail(f"action {i} is {type(action)!r}, not Scan or Sleep")

    def test_every_scan_passes_channels_for(self):
        """`GridError` on misalignment or out-of-band means an illegal action can
        never silently do something reasonable-looking -- so not raising, across
        every scan, is the legality proof."""
        for i, scan in enumerate(self.scans):
            try:
                chans = self.grid.channels_for(scan.f_center_hz, scan.bw_hz)
            except GridError as exc:                       # pragma: no cover
                self.fail(f"scan {i} {scan!r} is not grid-legal: {exc}")
            self.assertGreater(chans.size, 0)
            self.assertGreaterEqual(int(chans[0]), 0)
            self.assertLessEqual(int(chans[-1]), self.grid.n_channels - 1)
            self.assertEqual(int(chans.size), int(round(scan.bw_hz / 1e6)))

    def test_every_bandwidth_comes_from_the_candidate_set(self):
        seen = {a.bw_hz for a in self.scans}
        self.assertTrue(
            seen <= self.bw_ok, f"bandwidths outside the candidate set: {seen - self.bw_ok}"
        )
        self.assertGreaterEqual(len(seen), 2, "bandwidth knob is degenerate")

    def test_every_dwell_comes_from_the_candidate_set(self):
        seen = {a.dwell_s for a in self.scans}
        self.assertTrue(
            seen <= self.dwell_ok, f"dwells outside the candidate set: {seen - self.dwell_ok}"
        )
        self.assertGreaterEqual(len(seen), 2, "dwell knob is degenerate")

    def test_every_sleep_is_positive_and_bounded(self):
        """Sleep `dt` is NOT required to be a candidate value: the scheduler's
        layer-5 clamp shortens it to the next deadline, floored at 1 ms.  What
        must hold is that it never exceeds the longest candidate and never
        reaches zero, which would spin the episode loop without moving the
        clock."""
        longest = max(self.sleep_ok)
        for i, sleep in enumerate(self.sleeps):
            self.assertGreaterEqual(sleep.dt_s, MIN_SLEEP_S - 1e-12, f"sleep {i}")
            self.assertLessEqual(sleep.dt_s, longest + 1e-12, f"sleep {i}")

    def test_no_gain_is_requested_while_gain_is_disabled(self):
        for scan in self.scans:
            self.assertEqual(scan.gain_db, 0.0)


# ===========================================================================
# 2. Sleep is reachable, and for the reason DESIGN.md section 3 gives
# ===========================================================================
class TestSleepIsReachableAndCorrect(unittest.TestCase):
    """DESIGN.md section 3: "Sleep needs no threshold."

    `Sleep(dt)` has `gain = 0`, `cost = L_sleep*dt`, `duration = dt`, so
    `score_rate(Sleep) = -L_sleep` for every `dt`.  A hopeless scan pays
    `L_0 + L_d*dwell` over a duration of at most `dwell`, which is far more
    negative.  Sleep is therefore chosen exactly when every scan candidate has a
    negative reward rate -- the correct answer to "is doing nothing optimal right
    now?" rather than a tuned threshold.
    """

    def setUp(self):
        self.cfg = small_cfg()
        self.env = StubEnv(self.cfg)
        self.pol = IndexPolicy()
        self.env.reset(self.cfg, 0)
        self.pol.reset(self.env.grid, self.env.mission, self.env.horizon_s, 0, self.cfg)
        self.l_sleep = float(self.cfg["energy"]["L_sleep_w"])

    def _score_at_zero_belief(self):
        """Freeze the belief at p = 0 everywhere and score at t = 0.

        `Belief._propagated` returns `self.p` unchanged when `dt <= 0`, so
        pinning `t_now` to the scoring time makes `p_effective` exactly the array
        set here -- no decay, nothing to disentangle.
        """
        self.pol.belief.p[:] = 0.0
        self.pol.belief.t_now = 0.0
        np.testing.assert_allclose(self.pol.belief.p_effective(0.0), 0.0)
        cands, gain = self.pol._enumerate(0.0)
        return cands, gain, self.pol._score(cands, gain)

    def test_sleep_rate_is_minus_l_sleep_independent_of_dt(self):
        cands, _gain, scores = self._score_at_zero_belief()
        sleep_scores = scores[cands.is_sleep]
        sleep_dts = cands.dwell_s[cands.is_sleep]
        self.assertGreaterEqual(sleep_dts.size, 3)
        self.assertGreater(float(sleep_dts.max() / sleep_dts.min()), 5.0,
                           "need well-separated dt values for 'independent of dt'")
        for dt, s in zip(sleep_dts.tolist(), sleep_scores.tolist()):
            with self.subTest(dt=dt):
                self.assertAlmostEqual(s, -self.l_sleep, places=15)

    def test_every_hopeless_scan_scores_below_sleep(self):
        """With zero belief every scan has gain 0, so its rate is
        `-(L_0 + L_d*dwell + L_f*|df|) / duration`.  The best of them is about
        -1.01 W against sleep's -0.010 W -- two orders of magnitude apart, which
        is why no threshold is needed."""
        cands, gain, scores = self._score_at_zero_belief()
        scan_scores = scores[~cands.is_sleep]
        np.testing.assert_allclose(gain[~cands.is_sleep], 0.0)
        self.assertLess(float(scan_scores.max()), -self.l_sleep)
        # The numbers DESIGN.md section 3 quotes, to a loose tolerance.
        self.assertLess(float(scan_scores.max()), -1.0)
        self.assertGreater(float(scan_scores.max()), -1.2)

    def test_zero_belief_makes_the_policy_sleep(self):
        cands, _gain, scores = self._score_at_zero_belief()
        action, reason = self.pol.scheduler.select(
            cands, scores, self.pol.belief, 0.0, self.pol.energy
        )
        self.assertIsInstance(action, Sleep)
        self.assertEqual(reason, "sleep")

    def test_one_certain_channel_makes_the_policy_scan_it(self):
        """p = 1.0 on a threat channel (w = 1.0 J).  The gain then dwarfs the
        cost of the 50 ms dwell that class needs, so a covering scan must win."""
        target = 50                       # inside the 40-60 prio-1 band
        self.assertEqual(int(self.env.mission.priority[target]), 1)
        self.assertAlmostEqual(float(self.env.mission.w[target]), 1.0)

        self.pol.belief.p[:] = 0.0
        self.pol.belief.p[target] = 1.0
        self.pol.belief.t_now = 0.0
        cands, gain = self.pol._enumerate(0.0)
        scores = self.pol._score(cands, gain)
        action, reason = self.pol.scheduler.select(
            cands, scores, self.pol.belief, 0.0, self.pol.energy
        )
        self.assertIsInstance(action, Scan)
        self.assertEqual(reason, "index")
        self.assertIn(
            target, self.env.grid.channels_for(action.f_center_hz, action.bw_hz).tolist()
        )
        self.assertGreater(float(scores.max()), -self.l_sleep)


# ===========================================================================
# 3. Retune locality -- L_f is in the score, not decoration
# ===========================================================================
class TestRetuneLocality(unittest.TestCase):
    """Two channels of identical value, 150 MHz apart.

    The far one costs an extra `L_f*150e6 = 3.0 mJ` and an extra
    `t_settle + 150e6/f_slew = 3.5 ms` of duration, so it must lose on the
    reward RATE.  The test is run in both directions from the same candidate set:
    a one-sided version would also pass if the winner were decided by candidate
    ordering, which is exactly the failure mode worth excluding.
    """

    NEAR_FAR = (20, 170)          # 150 channels = 150 MHz apart

    def setUp(self):
        self.cfg = small_cfg()
        self.env = StubEnv(self.cfg)
        a, b = self.NEAR_FAR
        # Both must be priority 3 so their mission weights are identical.
        self.assertEqual(int(self.env.mission.priority[a]), 3)
        self.assertEqual(int(self.env.mission.priority[b]), 3)
        self.assertAlmostEqual(
            float(self.env.mission.w[a]), float(self.env.mission.w[b])
        )

    def _winner_channels(self, tuned_to: int):
        pol = IndexPolicy()
        self.env.reset(self.cfg, 0)
        pol.reset(self.env.grid, self.env.mission, self.env.horizon_s, 0, self.cfg)
        pol.belief.p[:] = 0.0
        for ch in self.NEAR_FAR:
            pol.belief.p[ch] = 0.3
        pol.belief.t_now = 0.0
        pol.f_last_hz = float(self.env.grid.center_hz(tuned_to))
        cands, gain = pol._enumerate(0.0)
        scores = pol._score(cands, gain)
        best = int(np.argmax(scores))
        action = cands.action(best)
        self.assertIsInstance(action, Scan)
        return set(
            self.env.grid.channels_for(action.f_center_hz, action.bw_hz).tolist()
        )

    def test_the_near_channel_wins_from_either_end(self):
        a, b = self.NEAR_FAR
        covered = self._winner_channels(tuned_to=a)
        self.assertIn(a, covered)
        self.assertNotIn(b, covered)

        covered = self._winner_channels(tuned_to=b)
        self.assertIn(b, covered)
        self.assertNotIn(a, covered)

    def test_the_retune_penalty_is_the_size_the_config_says(self):
        """Guard on the premise: if `L_f` were negligible against the score
        spread the test above would prove nothing."""
        l_f = float(self.cfg["energy"]["L_f_j_per_hz"])
        l_0 = float(self.cfg["energy"]["L_0_j"])
        self.assertAlmostEqual(l_f * 150e6, 3.0e-3)
        self.assertGreater(l_f * 150e6, l_0)      # bigger than a whole scan's fixed cost


# ===========================================================================
# 4. Budget respected AND horizon reached
# ===========================================================================
class TestBudgetAndHorizon(unittest.TestCase):
    """DESIGN.md section 1: "every policy, baselines included, must pace itself
    with `Sleep` to survive the horizon."  A policy that runs out of energy early
    is a failed policy, not a frugal one, so both halves are asserted together.
    """

    def _assert_both_halves(self, cfg, seed=0):
        _pol, actions, obs = run_episode(cfg, seed)
        budget = float(cfg["energy"]["budget_j"])
        horizon = float(cfg["horizon_s"])
        self.assertLessEqual(
            obs.energy_total, budget + 1e-9,
            f"spent {obs.energy_total:.6f} J of a {budget:.6f} J budget",
        )
        self.assertGreaterEqual(
            obs.t, horizon - 1e-9,
            f"episode ended at t={obs.t:.6f} s, short of the {horizon:.1f} s horizon "
            f"having spent {obs.energy_total:.6f}/{budget:.6f} J",
        )
        return actions, obs

    def test_nominal_budget(self):
        cfg = small_cfg()
        pol, actions, obs = run_episode(cfg, 0)
        budget = float(cfg["energy"]["budget_j"])
        self.assertLessEqual(obs.energy_total, budget + 1e-9)
        self.assertGreaterEqual(obs.t, float(cfg["horizon_s"]) - 1e-9)
        self.assertTrue(any(isinstance(a, Scan) for a in actions))
        # The scheduler's pacing decisions are only as good as the energy it is
        # shown, and `EnergyState.spent_j` is copied straight from
        # `obs.energy_total` at the top of every `act`.
        self.assertIsInstance(pol.energy, EnergyState)
        self.assertAlmostEqual(pol.energy.budget_j, budget)
        self.assertLessEqual(pol.energy.spent_j, obs.energy_total + 1e-12)

    def test_budget_cut_to_ten_percent(self):
        """The stress case, and note the arithmetic: the nominal budget is 0.1 W
        against `L_sleep = 0.01 W`, so 10% of nominal is *exactly* the energy
        needed to stand by for the whole horizon.  The correct behaviour at that
        point is to sleep through and finish the episode with the budget spent to
        the joule -- not to spend it on scans and strand itself.  The scheduler's
        standby reserve is what produces that, and both halves still hold."""
        cfg = small_cfg(budget_j=0.10 * NOMINAL_W * SMALL_HORIZON)
        _actions, obs = self._assert_both_halves(cfg)
        standby = float(cfg["energy"]["L_sleep_w"]) * SMALL_HORIZON
        self.assertAlmostEqual(float(cfg["energy"]["budget_j"]), standby, places=12)
        self.assertAlmostEqual(obs.energy_total, standby, places=9)

    def test_budget_cut_to_a_quarter_still_scans(self):
        """Non-degenerate frugality: 25% of nominal leaves room above standby, so
        the policy must both scan and survive."""
        cfg = small_cfg(budget_j=0.25 * NOMINAL_W * SMALL_HORIZON)
        actions, _ = self._assert_both_halves(cfg)
        self.assertTrue(
            any(isinstance(a, Scan) for a in actions),
            "at 25% of nominal there is budget above standby; a policy that never "
            "scans is not frugal, it is broken",
        )


# ===========================================================================
# 5. score_mode: "raw" -- the source document's literal form
# ===========================================================================
class TestRawScoreMode(unittest.TestCase):
    """DESIGN.md 11.8 ablated `score_mode: raw` and rejected it (POI 0.083,
    J/detection inf, 16% coverage).  Nothing here asserts it performs well.  What
    it must do is keep RUNNING, so that ablation stays reproducible rather than
    becoming a claim in a document nobody can re-derive.
    """

    def test_raw_mode_runs_and_emits_legal_actions(self):
        cfg = small_cfg(score_mode="raw")
        pol, actions, obs = run_episode(cfg, 0)
        self.assertEqual(pol.score_mode, "raw")
        self.assertGreater(len(actions), 0)
        self.assertAlmostEqual(obs.t, float(cfg["horizon_s"]), places=6)

        grid = StubEnv(cfg).grid
        bw_ok = {float(b) * 1e6 for b in cfg["agent"]["bw_candidates_mhz"]}
        dwell_ok = {float(d) * 1e-3 for d in cfg["agent"]["dwell_candidates_ms"]}
        scans = [a for a in actions if isinstance(a, Scan)]
        self.assertGreater(len(scans), 0)
        for scan in scans:
            grid.channels_for(scan.f_center_hz, scan.bw_hz)   # raises if illegal
            self.assertIn(scan.bw_hz, bw_ok)
            self.assertIn(scan.dwell_s, dwell_ok)

    def test_raw_is_gain_minus_cost_with_no_duration_divisor(self):
        """Pin the form itself, not its performance: `score_raw = gain - cost`
        while `score_rate = (gain - cost)/duration`."""
        cfg = small_cfg(score_mode="raw")
        env = StubEnv(cfg)
        pol = IndexPolicy()
        env.reset(cfg, 0)
        pol.reset(env.grid, env.mission, env.horizon_s, 0, cfg)
        cands, gain = pol._enumerate(0.0)
        np.testing.assert_allclose(pol._score(cands, gain), gain - cands.cost_j)

        pol.score_mode = "rate"
        np.testing.assert_allclose(
            pol._score(cands, gain),
            (gain - cands.cost_j) / np.maximum(cands.duration_s, 1e-12),
        )

    def test_raw_prefers_a_longer_dwell_than_rate_on_the_same_candidates(self):
        """The mechanism behind DESIGN.md 11.8's "rate under-dwells; raw
        over-dwells": on one identical candidate set the two modes pick
        different dwells, and raw's is the longer."""
        cfg = small_cfg()
        env = StubEnv(cfg)
        pol = IndexPolicy()
        env.reset(cfg, 0)
        pol.reset(env.grid, env.mission, env.horizon_s, 0, cfg)
        pol.belief.p[:] = 0.02
        pol.belief.p[50] = 0.9
        pol.belief.t_now = 0.0
        cands, gain = pol._enumerate(0.0)

        pol.score_mode = "rate"
        dwell_rate = float(cands.dwell_s[int(np.argmax(pol._score(cands, gain)))])
        pol.score_mode = "raw"
        dwell_raw = float(cands.dwell_s[int(np.argmax(pol._score(cands, gain)))])
        self.assertGreater(dwell_raw, dwell_rate)


# ===========================================================================
# 6. log_rows() schema -- rung 2 depends on this exact shape
# ===========================================================================
class TestLogRowSchema(unittest.TestCase):
    EXPECTED_KEYS = frozenset(
        {"t", "step", "channel", "dwell_s", "bw_hz", "detected"} | set(FEATURE_NAMES)
    )

    @classmethod
    def setUpClass(cls):
        # A 5 s clock is plenty: rows are released on the SECOND visit to a
        # channel, and there are thousands of those well before the horizon.
        cls.cfg = small_cfg(horizon_s=5.0)
        pol, _actions, _obs = run_episode(cls.cfg, 1, policy=IndexPolicy(collect_logs=True))
        cls.rows = pol.log_rows()

    def test_rows_are_emitted(self):
        """Rows are only released once the channel is observed AGAIN -- that is
        how the retroactive label is obtained -- so an empty log would mean the
        labelling path never ran."""
        self.assertGreater(len(self.rows), 100)

    def test_every_row_has_exactly_the_expected_keys(self):
        for i, row in enumerate(self.rows):
            keys = frozenset(row)
            if keys != self.EXPECTED_KEYS:
                self.fail(
                    f"row {i} schema drift: missing={sorted(self.EXPECTED_KEYS - keys)}, "
                    f"unexpected={sorted(keys - self.EXPECTED_KEYS)}"
                )

    def test_no_nan_or_inf_anywhere(self):
        """One NaN poisons a whole training run, and
        `HistGradientBoostingClassifier` will happily ingest it."""
        for i, row in enumerate(self.rows):
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        self.fail(f"row {i} key {key!r} is {value!r}")

    def test_label_and_identity_columns_are_well_formed(self):
        n_channels = int(self.cfg["grid"]["n_channels"])
        horizon = float(self.cfg["horizon_s"])
        for i, row in enumerate(self.rows):
            self.assertIn(row["detected"], (0, 1), f"row {i}")
            self.assertTrue(0 <= int(row["channel"]) < n_channels, f"row {i}")
            self.assertTrue(0.0 <= float(row["t"]) <= horizon + 1e-9, f"row {i}")
            self.assertGreaterEqual(int(row["step"]), 0, f"row {i}")
            # Rows are emitted for SCAN decisions only, so both are non-zero.
            self.assertGreater(float(row["dwell_s"]), 0.0, f"row {i}")
            self.assertGreater(float(row["bw_hz"]), 0.0, f"row {i}")

    def test_one_row_per_channel_per_decision(self):
        """The shape `agent/policy_learned.py` assumes: a decision that scanned
        `k` channels contributes `k` rows sharing one `(t, step)`, one per
        distinct channel."""
        by_step: dict = {}
        for row in self.rows:
            by_step.setdefault((row["t"], row["step"]), []).append(row)
        self.assertGreater(len(by_step), 10)
        for key, group in by_step.items():
            channels = [r["channel"] for r in group]
            self.assertEqual(len(channels), len(set(channels)), f"duplicate channel at {key}")
            self.assertEqual(len({r["bw_hz"] for r in group}), 1, f"mixed bw at {key}")
            self.assertEqual(len({r["dwell_s"] for r in group}), 1, f"mixed dwell at {key}")

    def test_feature_values_are_in_feature_name_order(self):
        """`row.update(zip(FEATURE_NAMES, fv))` is order-sensitive; a reordered
        `FEATURE_NAMES` would silently mislabel every column."""
        row = self.rows[0]
        self.assertEqual(
            [k for k in row if k in set(FEATURE_NAMES)], list(FEATURE_NAMES)
        )

    def test_logging_off_by_default(self):
        pol, _actions, _obs = run_episode(self.cfg, 1)
        self.assertEqual(pol.log_rows(), [])


# ===========================================================================
# 7. Determinism
# ===========================================================================
class TestDeterminism(unittest.TestCase):
    """DESIGN.md section 10: every result carries its `config_hash` and seed.
    That is only worth anything if the pair reproduces the run.
    """

    @classmethod
    def setUpClass(cls):
        # A 4 s clock still runs a few hundred decisions -- ample to diverge if
        # anything were non-deterministic, and a quarter of the cost.
        cls.cfg = small_cfg(horizon_s=4.0)
        cfg = cls.cfg
        _p, cls.a1, cls.obs1 = run_episode(cfg, 7, env=StubEnv(cfg))
        _p, cls.a2, cls.obs2 = run_episode(cfg, 7, env=StubEnv(cfg))
        _p, cls.other_seed, _o = run_episode(cfg, 8, env=StubEnv(cfg))
        # Same seed again, but on an env object that has already run a DIFFERENT
        # seed -- the runner reuses environments, so leaked state here would make
        # `runs.csv` unreproducible.
        reused = StubEnv(cfg)
        run_episode(cfg, 11, env=reused)
        _p, cls.a3, _o = run_episode(cfg, 7, env=reused)

    def test_same_seed_and_scenario_give_an_identical_action_sequence(self):
        self.assertGreater(len(self.a1), 50)
        self.assertEqual(
            [action_key(a) for a in self.a1], [action_key(a) for a in self.a2]
        )
        self.assertEqual(self.obs1.t, self.obs2.t)
        self.assertEqual(self.obs1.energy_total, self.obs2.energy_total)

    def test_a_different_seed_gives_a_different_sequence(self):
        """Guard against the test above passing on a policy that ignores the
        environment entirely."""
        self.assertNotEqual(
            [action_key(a) for a in self.a1], [action_key(a) for a in self.other_seed]
        )

    def test_reusing_one_env_object_does_not_change_the_sequence(self):
        self.assertEqual(
            [action_key(a) for a in self.a1], [action_key(a) for a in self.a3]
        )


if __name__ == "__main__":
    unittest.main()
