"""Tests for `agent/scheduler.py` -- the hard-constraint layer.

Two of the project's load-bearing claims live in this file, and neither was
asserted anywhere before it existed.

**Claim 1 -- separation (DESIGN.md section 7).**  `Scheduler.select` receives an
opaque `(M,)` score array it did not compute and cannot recompute; the learner
proposes value, the scheduler picks under hard constraints.  Until now that was
enforced only by the type signature, which is a convention, not a guarantee.  The
mechanical form of the claim is: *with no constraint active, `select` returns the
argmax of whatever scores it was handed* -- for every index, including indices
whose candidate is the most expensive and longest-running in the set.  If the
scheduler ever folded a cost, a duration or a coverage term into its choice, that
property would break.  It is asserted here across all M positions.

**Claim 2 -- deadline hardness (DESIGN.md section 7, metric in section 6).**
"Max staleness is hard-bounded" is quoted as a *provable* property, so the bound
`D_1 + max_dwell + max_retune` is asserted directly against a full episode.  Two
sub-properties come with it:

* a deadline names *which* channel, but the reward-RATE score still picks the
  dwell, and rate always prefers the shortest action because duration is its
  denominator (DESIGN.md 11.5).  `deadline_min_pd` exists so a deadline visit is
  long enough to actually detect the class it is checking -- 50 ms on a threat
  channel, not the 2 ms the rate index would pick.  Coverage with too short a
  dwell is not coverage, so the min-dwell requirement is asserted too;
* a deadline override must beat budget pacing, and budget pacing must beat the
  index.  Both branches are exercised.

The episode-level bound test currently **FAILS**.  That is deliberate and it is
the point of the test: see its docstring for the measured numbers and the two
mechanisms behind it.  The unit tests below isolate both mechanisms and pass, so
the diagnosis travels with the failure.

Test files may import `sim.env` / `sim.stub_env`; `agent/` source files may not.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np

from agent.base import REASONS, EnergyState
from agent.policy_index import IndexPolicy
from agent.scheduler import MIN_SLEEP_S, CandidateSet, Scheduler
from sim.config import load_config
from sim.contract import ChannelGrid, Mission, Scan, Sleep
from sim.stub_env import StubEnv

# Energy/timing constants, taken from configs/*.yaml so a hand-built candidate
# costs exactly what the real one would.  DESIGN.md section 1.
L_0 = 2.0e-3        # J, fixed per-scan cost
L_D = 1.0           # W while dwelling
L_F = 2.0e-11       # J/Hz of retune
L_SLEEP = 0.01      # W standby
T_SETTLE = 0.5e-3   # s
F_SLEW = 50.0e9     # Hz/s
HORIZON = 60.0

# Small synthetic band for the unit tests.  64 channels is enough to hold three
# priority classes and a 20-channel-wide candidate, and small enough that every
# construction below is readable by eye.
N_CH = 64
GRID = ChannelGrid(f_start_hz=2.0e9, n_channels=N_CH, channel_bw_hz=1.0e6)
P1_BAND = (8, 12)
P2_BAND = (20, 28)


def _mission(deadlines=None, watch=(), watch_deadline_s: float = 0.3) -> Mission:
    """Mirror of `sim.config.build_mission`: catch-all first, specific bands win."""
    prio = np.full(N_CH, 3, dtype=np.int32)
    prio[P2_BAND[0]:P2_BAND[1]] = 2
    prio[P1_BAND[0]:P1_BAND[1]] = 1
    w = np.zeros(N_CH, dtype=np.float64)
    w[prio == 1], w[prio == 2], w[prio == 3] = 1.0, 0.3, 0.1
    return Mission(
        priority=prio,
        w=w,
        deadlines_s=dict(deadlines or {1: 30.0, 2: 8.0, 3: 20.0}),
        watch_list=np.asarray(watch, dtype=np.int32),
        watch_deadline_s=float(watch_deadline_s),
    )


class FakeBelief:
    """The scheduler reads exactly two things off a belief: `staleness(t)` and
    `t_last_visit`.  Supplying only those keeps every construction below explicit
    about the one quantity under test."""

    def __init__(self, t_last_visit):
        self.t_last_visit = np.asarray(t_last_visit, dtype=np.float64)

    def staleness(self, t: float) -> np.ndarray:
        return np.maximum(float(t) - self.t_last_visit, 0.0)


def _fresh(t_last: float = 0.0) -> FakeBelief:
    return FakeBelief(np.full(N_CH, float(t_last)))


def _cands(
    scans=(),
    sleeps=(),
    f_last_hz: float | None = None,
    cost_override: dict | None = None,
    duration_override: dict | None = None,
) -> CandidateSet:
    """Build a columnar candidate set with the REAL energy/timing formulas.

    `scans` is a sequence of `(k_lo, n_ch, dwell_s)`; `sleeps` a sequence of
    `dt_s`.  Sleep rows carry `k_lo = -1, n_ch = 0` and their `dt` in `dwell_s`,
    exactly as `agent/policy_index.py` builds them.
    """
    f_last = float(GRID.center_hz(0)) if f_last_hz is None else float(f_last_hz)
    k_lo, n_ch, dwell, cost, dur, is_sleep = [], [], [], [], [], []
    for k, n, d in scans:
        f_c = GRID.f_start_hz + k * GRID.channel_bw_hz + n * GRID.channel_bw_hz / 2.0
        df = abs(f_c - f_last)
        t_retune = 0.0 if df == 0.0 else T_SETTLE + df / F_SLEW
        k_lo.append(k)
        n_ch.append(n)
        dwell.append(d)
        cost.append(L_0 + L_D * d + L_F * df)
        dur.append(t_retune + d)
        is_sleep.append(False)
    for dt in sleeps:
        k_lo.append(-1)
        n_ch.append(0)
        dwell.append(dt)
        cost.append(L_SLEEP * dt)
        dur.append(dt)
        is_sleep.append(True)

    cost_a = np.asarray(cost, dtype=np.float64)
    dur_a = np.asarray(dur, dtype=np.float64)
    for i, v in (cost_override or {}).items():
        cost_a[i] = float(v)
    for i, v in (duration_override or {}).items():
        dur_a[i] = float(v)
    return CandidateSet(
        GRID,
        np.asarray(k_lo, dtype=np.int32),
        np.asarray(n_ch, dtype=np.int32),
        np.asarray(dwell, dtype=np.float64),
        cost_a,
        dur_a,
        np.asarray(is_sleep, dtype=bool),
    )


def _sched(mission: Mission | None = None, **kw) -> Scheduler:
    return Scheduler(GRID, mission or _mission(), HORIZON, l_sleep_w=L_SLEEP, **kw)


def _reason_kind(reason: str) -> str:
    """`deadline:ch=9` -> `deadline`, so it can be checked against REASONS."""
    return reason.split(":", 1)[0]


def _reason_channel(reason: str) -> int:
    return int(reason.split("=", 1)[1])


# ===========================================================================
# CLAIM 1 -- the scheduler picks; it never scores.
# ===========================================================================
class TestSeparationOfScoringAndSelection(unittest.TestCase):
    """DESIGN.md section 7's architectural claim, made mechanical.

    The scheduler is handed a score array it could not have produced.  With no
    constraint active its output must be a pure function of that array -- the
    argmax and nothing else.
    """

    def setUp(self):
        # t = 0 with every channel just visited: staleness 0, so no deadline is
        # overdue; energy untouched, so pacing is not engaged.  Layer 1 passes
        # trivially (2 ms scans against a 60 s / 6 J episode).
        self.sched = _sched()
        self.belief = _fresh()
        self.energy = EnergyState(spent_j=0.0, budget_j=6.0)
        self.cands = _cands(
            scans=[(k, 1, 0.002) for k in range(0, 60, 5)],
            sleeps=(0.01, 0.05),
        )

    def test_returns_argmax_of_supplied_scores_at_every_index(self):
        """For EVERY position i, making i the maximum makes i the choice.

        This is the strongest available form of "does no scoring of its own": the
        map from scores to choice is exactly `argmax`, over the whole index set,
        not merely on one convenient example.
        """
        m = len(self.cands)
        self.assertGreater(m, 4, "need a non-trivial candidate set")
        for i in range(m):
            with self.subTest(i=i):
                scores = np.full(m, -1.0)
                scores[i] = 5.0
                action, reason = self.sched.select(
                    self.cands, scores, self.belief, 0.0, self.energy
                )
                self.assertEqual(action, self.cands.action(i))
                self.assertIn(_reason_kind(reason), REASONS)
                # The reason distinguishes only the KIND of action here, because
                # no constraint fired.
                self.assertEqual(
                    reason, "sleep" if bool(self.cands.is_sleep[i]) else "index"
                )

    def test_choice_ignores_cost_and_duration(self):
        """Give the top score to the dearest, longest candidate; it must win.

        A scheduler that quietly re-derived value from `cost_j` / `duration_s` --
        the only two score-shaped columns it can see -- would refuse this pick.
        Reversing the scores must reverse the choice for the same reason.
        """
        scans = [(k, 1, d) for k, d in zip((0, 10, 20, 30, 40), (0.001, 0.002, 0.005, 0.02, 0.2))]
        cands = _cands(scans=scans)
        m = len(cands)
        # Cost and duration both increase monotonically with the index.
        self.assertTrue(np.all(np.diff(cands.cost_j) > 0.0))
        self.assertTrue(np.all(np.diff(cands.duration_s) > 0.0))

        ascending = np.arange(m, dtype=np.float64)
        action, reason = self.sched.select(
            cands, ascending, self.belief, 0.0, self.energy
        )
        self.assertEqual(action, cands.action(m - 1))   # dearest AND longest
        self.assertEqual(reason, "index")

        action, _ = self.sched.select(
            cands, ascending[::-1].copy(), self.belief, 0.0, self.energy
        )
        self.assertEqual(action, cands.action(0))       # cheapest AND shortest

    def test_permuting_the_scores_permutes_the_choice(self):
        """Permute scores across a FIXED candidate set; the choice must follow.

        Concretely: if `perm` maps positions and `scores_permuted = scores[perm]`,
        then the chosen candidate is `perm[argmax(scores)]`-shaped -- i.e. the
        selection tracks the score vector, not the candidate layout.
        """
        rng = np.random.default_rng(11)
        m = len(self.cands)
        base = rng.permutation(m).astype(np.float64)
        for trial in range(8):
            with self.subTest(trial=trial):
                perm = rng.permutation(m)
                scores = base[perm]
                action, _ = self.sched.select(
                    self.cands, scores, self.belief, 0.0, self.energy
                )
                self.assertEqual(action, self.cands.action(int(np.argmax(scores))))

    def test_rejects_a_score_array_of_the_wrong_shape(self):
        """The (M,) contract is checked, not assumed -- a silent broadcast here
        would mean the scheduler was ranking something other than the candidates
        it was given."""
        with self.assertRaises(ValueError):
            self.sched.select(
                self.cands, np.zeros(len(self.cands) + 1), self.belief, 0.0, self.energy
            )


# ===========================================================================
# CLAIM 2 -- deadline hardness, end to end.
# ===========================================================================
def _episode_coverage(cfg: dict, seed: int):
    """Run one episode and return per-channel worst coverage gap + reason tally.

    Staleness is a GEOMETRIC property of the action log (DESIGN.md section 6:
    "independent of whether the detector fired"), so `StubEnv` is sufficient and
    ~2x faster than a real world.  A gap is measured to the END of a dwell, which
    is the same convention `eval/metrics.py` uses, and the two boundary gaps
    (`0 -> first` and `last -> T`) are included.
    """
    env = StubEnv(cfg)
    pol = IndexPolicy()
    obs = env.reset(cfg, seed)
    pol.reset(env.grid, env.mission, env.horizon_s, seed, cfg)

    n = env.grid.n_channels
    prio = np.asarray(env.mission.priority)
    last = np.zeros(n, dtype=np.float64)
    worst = np.zeros(n, dtype=np.float64)
    deadline_hits = {1: 0, 2: 0, 3: 0}
    deadline_dwells: list[tuple[int, float]] = []

    while not obs.done:
        action = pol.act(obs)
        reason = pol.last_reason
        if reason.startswith("deadline:"):
            ch = _reason_channel(reason)
            deadline_hits[int(prio[ch])] = deadline_hits.get(int(prio[ch]), 0) + 1
            if isinstance(action, Scan):
                deadline_dwells.append((ch, float(action.dwell_s)))
        obs = env.step(action)
        if isinstance(obs.action, Scan) and obs.scanned_channels.size:
            c = obs.scanned_channels
            worst[c] = np.maximum(worst[c], obs.t - last[c])
            last[c] = obs.t
    worst = np.maximum(worst, env.horizon_s - last)
    return {
        "worst": worst,
        "last": last,
        "prio": prio,
        "horizon_s": env.horizon_s,
        "deadline_hits": deadline_hits,
        "deadline_dwells": deadline_dwells,
        "energy_total": obs.energy_total,
        "t_end": obs.t,
    }


class TestDeadlineHardnessEndToEnd(unittest.TestCase):
    """The plan's "test 22" -- the property that protects the headline.

    Config: the real `sparse` grid (2000 channels / 2 GHz), the real priority
    bands (400-600 / 1200-1500 / catch-all) and the real deadlines
    `{1: 30, 2: 8, 3: 20}` s.  ONLY `horizon_s` and `budget_j` are scaled, and
    they are scaled *together* (45 s / 4.5 J) so the 0.1 W average of DESIGN.md
    section 1 is preserved exactly.  45 s is the shortest horizon on which the
    30 s prio-1 deadline can be tested at all and still leaves 1.5 deadline
    periods; the nominal 60 s episode behaves the same but takes over 2 s.
    """

    @classmethod
    def setUpClass(cls):
        cfg = copy.deepcopy(load_config("sparse"))
        cfg.pop("config_hash", None)
        cfg["horizon_s"] = 45.0
        cfg["energy"]["budget_j"] = 4.5      # unchanged 0.1 W average
        cls.cfg = load_config(cfg)
        cls.ep = _episode_coverage(cls.cfg, seed=3)
        # `load_config` round-trips a dict through JSON, so mapping keys arrive
        # as strings; `sim.config.build_mission` re-integerises them the same way.
        deadlines = {int(k): float(v) for k, v in cls.cfg["mission"]["deadlines_s"].items()}
        cls.d1 = deadlines[1]
        max_dwell = max(cls.cfg["agent"]["dwell_candidates_ms"]) * 1e-3
        # Worst retune is a full-span hop: t_settle + span/f_slew.
        span = cls.cfg["grid"]["n_channels"] * cls.cfg["grid"]["channel_bw_hz"]
        max_retune = (
            cls.cfg["receiver"]["t_settle_s"] + span / cls.cfg["receiver"]["f_slew_hz_per_s"]
        )
        cls.bound = cls.d1 + max_dwell + max_retune

    def test_episode_is_well_formed(self):
        """Guard: the bound test below is only meaningful on a real episode."""
        self.assertAlmostEqual(self.ep["t_end"], self.cfg["horizon_s"], places=6)
        self.assertLessEqual(
            self.ep["energy_total"], self.cfg["energy"]["budget_j"] + 1e-9
        )
        self.assertGreater(int((self.ep["last"] > 0.0).sum()), 100)

    def test_max_staleness_prio1_within_deadline_bound(self):
        """*** KNOWN FAILURE -- documents a real defect, do not silence. ***

        DESIGN.md section 7 states that putting revisit deadlines in the
        scheduler rather than in the score makes "max staleness is hard-bounded"
        a *provable* property.  It is not.  Measured on the config above,
        `sparse`, seeds 0-4, the worst prio-1 coverage gap is 32.9-45.0 s against
        a bound of 30.24 s, and at some seeds prio-1 channels are never visited
        at all.  The real `sim.env` world is worse, not better (60.0 s at the
        nominal 60 s horizon, all 200 prio-1 channels over bound).

        Two mechanisms, both isolated as passing unit tests further down:

        1. `Scheduler._deadline_override` selects the target channel with
           `argmax(staleness - deadline)`, i.e. by overdue-ness in ABSOLUTE
           seconds.  With deadlines `{1: 30, 2: 8, 3: 20}` the class with the
           LONGEST deadline -- priority 1, the threat band -- is structurally the
           last to be picked, so a prio-1 deadline is only ever served once every
           shorter-deadline channel is fresh.  See
           `test_most_overdue_is_absolute_seconds_not_priority`.
        2. The candidate set is enumerated by the index, which is deadline-blind
           (`IndexPolicy._enumerate` shortlists the top `windows_per_bw` windows
           by value).  When nothing in it covers the most-overdue channel,
           `_deadline_override` returns None and the deadline is silently
           dropped.  Instrumented on this config: the override wanted to fire on
           641 of 986 decisions and was dropped for want of a covering candidate
           on 635 of them -- 99.1%.  See
           `test_deadline_is_dropped_when_no_candidate_covers_the_target`.

        The guarantee is therefore best-effort, not hard.  Fixing it is a change
        to `agent/` (make the enumerator inject a covering candidate for the
        most-overdue channel, and rank overdue-ness relative to the deadline
        rather than in absolute seconds) and is out of scope for this test file.
        """
        p1 = np.flatnonzero(self.ep["prio"] == 1)
        self.assertGreater(p1.size, 0)
        worst_p1 = self.ep["worst"][p1]
        n_over = int((worst_p1 > self.bound).sum())
        n_unvisited = int((self.ep["last"][p1] == 0.0).sum())
        self.assertLessEqual(
            float(worst_p1.max()),
            self.bound,
            f"max staleness on prio-1 mission channels is "
            f"{float(worst_p1.max()):.3f} s, over the bound of {self.bound:.4f} s "
            f"(D_1={self.d1} + max_dwell + max_retune). "
            f"{n_over}/{p1.size} prio-1 channels exceed it and {n_unvisited} were "
            f"never visited. Deadline overrides fired by priority: "
            f"{self.ep['deadline_hits']}. See this test's docstring.",
        )

    def test_deadline_triggered_prio1_scans_are_long_enough(self):
        """DESIGN.md 11.5 -- coverage with too short a dwell is not coverage.

        A deadline names *which* channel; the reward-rate score still picks the
        dwell, and rate always prefers the shortest action.  `deadline_min_pd`
        (0.45) is what forces a deadline visit to be long enough to see the class
        it is checking -- 50 ms against the -19 dB threat band.

        End to end this currently vacuously passes, because no prio-1 deadline
        override fires at all (see the test above).  The requirement is asserted
        non-vacuously at the unit level in `TestDeadlineMinDwell`.
        """
        prio = self.ep["prio"]
        pol = IndexPolicy()
        env = StubEnv(self.cfg)
        env.reset(self.cfg, 0)
        pol.reset(env.grid, env.mission, env.horizon_s, 0, self.cfg)
        need = pol._min_dwell_for
        self.assertIsNotNone(need, "deadline_min_pd is configured, so this must exist")
        for ch, dwell in self.ep["deadline_dwells"]:
            if prio[ch] != 1:
                continue
            with self.subTest(channel=ch):
                self.assertGreaterEqual(dwell, float(need[ch]) - 1e-12)


# ===========================================================================
# Layer 2/3 -- deadlines and the watch list, at unit level.
# ===========================================================================
class TestDeadlineOverride(unittest.TestCase):
    def test_deadline_beats_a_higher_scoring_candidate(self):
        """The whole reason deadlines live here and not in the score: a deadline
        in a filter is a guarantee, a deadline in a score is a suggestion."""
        # t = 40: everything visited at 39 except channel 9 (prio 1), never.
        t_last = np.full(N_CH, 39.0)
        t_last[9] = 0.0
        cands = _cands(scans=[(9, 1, 0.05), (25, 1, 0.002)])
        action, reason = _sched().select(
            cands, np.array([1.0, 99.0]), FakeBelief(t_last), 40.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "deadline:ch=9")
        self.assertIsInstance(action, Scan)
        self.assertIn(9, GRID.channels_for(action.f_center_hz, action.bw_hz))

    def test_watch_list_has_its_own_deadline(self):
        """Layer 3.  Channel 40 is prio 3 (deadline 20 s) but watch-listed at
        0.3 s, so it comes due long before its priority class would."""
        mission = _mission(watch=(40,), watch_deadline_s=0.3)
        t_last = np.full(N_CH, 0.9)
        t_last[40] = 0.0                      # stale 1.0 s at t = 1.0
        cands = _cands(scans=[(40, 1, 0.002), (0, 1, 0.002)])
        action, reason = _sched(mission).select(
            cands, np.array([1.0, 99.0]), FakeBelief(t_last), 1.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "watchlist:ch=40")
        self.assertIn(40, GRID.channels_for(action.f_center_hz, action.bw_hz))

    def test_most_overdue_is_absolute_seconds_not_priority(self):
        """CHARACTERISATION of mechanism (1) behind the failing bound test.

        Channel 9 is priority 1 (deadline 30 s) and channel 20 is priority 2
        (deadline 8 s); at t = 40 with nothing ever visited, both are overdue --
        prio 1 by 10 s, prio 2 by 32 s.  `argmax(staleness - deadline)` ranks by
        overdue-ness in absolute seconds, so the prio-2 channel is served even
        though the prio-1 candidate scores 9x higher.

        With the shipped deadlines `{1: 30, 2: 8, 3: 20}` the highest-priority
        class has the LONGEST deadline, so it is systematically last in this
        ordering.  That is why `max_staleness_p1_s` is unbounded in practice.
        """
        cands = _cands(scans=[(9, 1, 0.05), (20, 1, 0.05)])
        action, reason = _sched().select(
            cands, np.array([9.0, 1.0]), _fresh(0.0), 40.0, EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "deadline:ch=20")
        self.assertEqual(_mission().priority[_reason_channel(reason)], 2)

        # And prio 1 IS served the moment it becomes the absolute argmax.
        t_last = np.full(N_CH, 39.0)
        t_last[9] = 0.0
        action, reason = _sched().select(
            cands, np.array([1.0, 9.0]), FakeBelief(t_last), 40.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "deadline:ch=9")

    def test_deadline_is_dropped_when_no_candidate_covers_the_target(self):
        """CHARACTERISATION of mechanism (2) behind the failing bound test.

        Channel 9 is 10 s overdue, but the candidate set -- built by the
        deadline-blind index enumerator -- contains nothing covering it.
        `_deadline_override` returns None and the decision falls through to the
        plain index, with no record that a hard constraint was abandoned.

        So layer 2 is a hard constraint only over the candidates it happens to be
        offered.  On the real 2000-channel grid that condition fails on 99.1% of
        the decisions where a deadline is overdue.
        """
        t_last = np.full(N_CH, 39.0)
        t_last[9] = 0.0
        cands = _cands(scans=[(25, 1, 0.002), (40, 1, 0.002)])
        action, reason = _sched().select(
            cands, np.array([1.0, 2.0]), FakeBelief(t_last), 40.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "index")
        self.assertEqual(action, cands.action(1))
        self.assertNotIn(9, GRID.channels_for(action.f_center_hz, action.bw_hz))


class TestDeadlineMinDwell(unittest.TestCase):
    """DESIGN.md 11.5 at unit level: a deadline visit must be long enough to
    actually detect the class it is checking."""

    def setUp(self):
        self.t_last = np.full(N_CH, 39.0)
        self.t_last[9] = 0.0                      # prio-1 channel, 40 s stale
        # A 1 ms look and a 50 ms look at the same channel; the rate index would
        # always prefer the 1 ms one, so its score is set higher on purpose.
        self.cands = _cands(scans=[(9, 1, 0.001), (9, 1, 0.05)])
        self.scores = np.array([99.0, 1.0])
        self.min_dwell = np.zeros(N_CH)
        self.min_dwell[P1_BAND[0]:P1_BAND[1]] = 0.05   # 50 ms on threat channels

    def test_without_min_dwell_the_score_picks_the_2ms_look(self):
        """The defect DESIGN.md 11.5 records: deadline satisfied, staleness
        healthy, P_d = 0.004, prio-1 POI exactly zero."""
        action, reason = _sched().select(
            self.cands, self.scores, FakeBelief(self.t_last), 40.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "deadline:ch=9")
        self.assertAlmostEqual(action.dwell_s, 0.001)

    def test_min_dwell_forces_the_adequate_dwell(self):
        """With `set_min_dwell_for`, the shorter look is filtered out BEFORE the
        score ranks anything -- so the higher-scoring 1 ms candidate loses."""
        sched = _sched()
        sched.set_min_dwell_for(self.min_dwell)
        action, reason = sched.select(
            self.cands, self.scores, FakeBelief(self.t_last), 40.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "deadline:ch=9")
        self.assertAlmostEqual(action.dwell_s, 0.05)

    def test_min_dwell_falls_through_when_nothing_is_long_enough(self):
        """Documented behaviour: a short look still refreshes staleness, and the
        alternative is silently abandoning the deadline entirely."""
        sched = _sched()
        sched.set_min_dwell_for(self.min_dwell)
        short_only = _cands(scans=[(9, 1, 0.001), (9, 1, 0.002)])
        action, reason = sched.select(
            short_only, np.array([99.0, 1.0]), FakeBelief(self.t_last), 40.0,
            EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "deadline:ch=9")
        self.assertAlmostEqual(action.dwell_s, 0.001)

    def test_min_dwell_applies_only_to_deadline_visits(self):
        """It is a deadline constraint, not a global floor -- the index path is
        free to pick a 1 ms scan."""
        sched = _sched()
        sched.set_min_dwell_for(self.min_dwell)
        action, reason = sched.select(
            self.cands, self.scores, _fresh(0.0), 0.0, EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "index")
        self.assertAlmostEqual(action.dwell_s, 0.001)


# ===========================================================================
# Layer 4 -- budget pacing.
# ===========================================================================
class TestBudgetPacing(unittest.TestCase):
    def test_overspending_forces_a_sleep(self):
        """`allowed(t) = budget*(t/horizon + 0.05)`.  At t = 6 s of a 60 s / 6 J
        episode that is 0.9 J; 2.0 J spent is well over, so scans are suppressed
        and the reason must say so rather than blaming the index."""
        energy = EnergyState(spent_j=2.0, budget_j=6.0)
        self.assertTrue(energy.over_pace(6.0, HORIZON))
        cands = _cands(scans=[(0, 1, 0.002), (10, 1, 0.002)], sleeps=(0.01, 0.2))
        action, reason = _sched().select(
            cands, np.array([9.0, 8.0, -1.0, -2.0]), _fresh(5.0), 6.0, energy,
        )
        self.assertEqual(reason, "budget-pace")
        self.assertIsInstance(action, Sleep)
        # Pacing sleeps for as long as it is allowed to; that is the constraint's
        # own semantics, not a score -- note the LOWEST-scoring sleep row wins.
        self.assertAlmostEqual(action.dt_s, 0.2)

    def test_not_over_pace_leaves_the_index_alone(self):
        energy = EnergyState(spent_j=0.1, budget_j=6.0)
        self.assertFalse(energy.over_pace(6.0, HORIZON))
        cands = _cands(scans=[(0, 1, 0.002), (10, 1, 0.002)], sleeps=(0.01, 0.2))
        action, reason = _sched().select(
            cands, np.array([9.0, 8.0, -1.0, -2.0]), _fresh(5.0), 6.0, energy,
        )
        self.assertEqual(reason, "index")
        self.assertEqual(action, cands.action(0))

    def test_a_deadline_override_beats_budget_pacing(self):
        """Layer 4 is SOFT -- layer 1 is already the hard cap, so a deadline may
        borrow against the pacing curve.  Same energy state as the first test,
        plus one overdue prio-1 channel."""
        energy = EnergyState(spent_j=5.0, budget_j=6.0)
        self.assertTrue(energy.over_pace(40.0, HORIZON))
        t_last = np.full(N_CH, 39.0)
        t_last[9] = 0.0
        cands = _cands(scans=[(9, 1, 0.05), (25, 1, 0.002)], sleeps=(0.01, 0.2))
        action, reason = _sched().select(
            cands, np.array([1.0, 2.0, 3.0, 4.0]), FakeBelief(t_last), 40.0, energy,
        )
        self.assertEqual(reason, "deadline:ch=9")
        self.assertIsInstance(action, Scan)


# ===========================================================================
# Layer 5 -- the sleep clamp.
# ===========================================================================
class TestSleepClamp(unittest.TestCase):
    """`dt = min(dt, next_deadline - t)`, floored at 1 ms.

    All candidates are given durations that overshoot the next deadline so that
    layer 1b's soft narrowing empties and `pool` falls back to plain feasibility
    -- otherwise the long sleep would simply be filtered out rather than clamped,
    and the clamp would never be reached.
    """

    def _belief_with_deadline_at(self, due_t: float) -> FakeBelief:
        # Channel 0 is prio 3 (20 s deadline); last visited at due_t - 20.
        t_last = np.full(N_CH, 24.9)
        t_last[0] = due_t - 20.0
        return FakeBelief(t_last)

    def test_sleep_is_clamped_to_the_next_deadline(self):
        cands = _cands(scans=[(30, 1, 0.2)], sleeps=(0.2,))
        action, reason = _sched().select(
            cands, np.array([1.0, 9.0]), self._belief_with_deadline_at(25.05),
            25.0, EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "sleep")
        self.assertIsInstance(action, Sleep)
        self.assertAlmostEqual(action.dt_s, 0.05, places=9)

    def test_clamp_is_floored_at_min_sleep(self):
        """A zero-length sleep would spin the episode loop without advancing the
        clock, so the clamp floors rather than truncates."""
        cands = _cands(scans=[(30, 1, 0.2)], sleeps=(0.2,))
        action, _ = _sched().select(
            cands, np.array([1.0, 9.0]), self._belief_with_deadline_at(25.0 + 1e-7),
            25.0, EnergyState(0.0, 6.0),
        )
        self.assertIsInstance(action, Sleep)
        self.assertAlmostEqual(action.dt_s, MIN_SLEEP_S, places=12)

    def test_uncontested_sleep_is_not_shortened(self):
        cands = _cands(scans=[(30, 1, 0.2)], sleeps=(0.2,))
        action, reason = _sched().select(
            cands, np.array([1.0, 9.0]), self._belief_with_deadline_at(40.0),
            25.0, EnergyState(0.0, 6.0),
        )
        self.assertEqual(reason, "sleep")
        self.assertAlmostEqual(action.dt_s, 0.2)

    def test_budget_pace_sleep_is_clamped_too(self):
        """The pacing branch goes through the same clamp, so pacing can never be
        the thing that blows a deadline either."""
        cands = _cands(scans=[(30, 1, 0.2)], sleeps=(0.2,))
        action, reason = _sched().select(
            cands, np.array([9.0, 1.0]), self._belief_with_deadline_at(25.05),
            25.0, EnergyState(spent_j=5.0, budget_j=6.0),
        )
        self.assertEqual(reason, "budget-pace")
        self.assertAlmostEqual(action.dt_s, 0.05, places=9)


# ===========================================================================
# Layer 1 -- feasibility.
# ===========================================================================
class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.belief = _fresh(59.0)      # nothing overdue at t = 1.0
        self.energy = EnergyState(spent_j=0.0, budget_j=6.0)

    def test_a_candidate_costing_more_than_remaining_is_never_selected(self):
        cands = _cands(
            scans=[(0, 1, 0.002), (10, 1, 0.002)], cost_override={0: 100.0}
        )
        action, reason = _sched().select(
            cands, np.array([9.0, 1.0]), self.belief, 1.0, self.energy
        )
        self.assertEqual(action, cands.action(1))
        self.assertEqual(reason, "index")

    def test_a_candidate_overrunning_the_horizon_is_never_selected(self):
        cands = _cands(
            scans=[(0, 1, 0.002), (10, 1, 0.002)], duration_override={0: 100.0}
        )
        action, _ = _sched().select(
            cands, np.array([9.0, 1.0]), self.belief, 1.0, self.energy
        )
        self.assertEqual(action, cands.action(1))

    def test_standby_reserve_is_enforced(self):
        """DESIGN.md section 1: a policy that strands itself and terminates on
        the budget at t=5 s is a failed policy, not a frugal one.  A candidate is
        infeasible unless it also leaves `L_sleep*(horizon - t_end)` to stand by
        with -- here 0.59 J of the 6 J budget."""
        remaining = self.energy.remaining_j
        cands = _cands(scans=[(0, 1, 0.002)], cost_override={0: remaining - 1.0e-6})
        action, reason = _sched().select(
            cands, np.array([9.0]), self.belief, 1.0, self.energy
        )
        self.assertEqual(reason, "fallback")
        self.assertIsInstance(action, Sleep)

        # The same candidate, priced to leave the reserve intact, IS feasible.
        reserve = L_SLEEP * (HORIZON - (1.0 + float(cands.duration_s[0])))
        ok = _cands(scans=[(0, 1, 0.002)], cost_override={0: remaining - reserve - 1e-6})
        action, reason = _sched().select(
            ok, np.array([9.0]), self.belief, 1.0, self.energy
        )
        self.assertEqual(reason, "index")

    def test_all_infeasible_gives_a_fallback_sleep(self):
        cands = _cands(scans=[(0, 1, 0.002)], cost_override={0: 100.0})
        action, reason = _sched().select(
            cands, np.array([9.0]), self.belief, 1.0, self.energy
        )
        self.assertEqual(reason, "fallback")
        self.assertEqual(action, Sleep(MIN_SLEEP_S))

    def test_empty_candidate_set_gives_a_fallback_sleep(self):
        empty = _cands()
        action, reason = _sched().select(
            empty, np.empty(0), self.belief, 1.0, self.energy
        )
        self.assertEqual(reason, "fallback")
        self.assertEqual(action, Sleep(MIN_SLEEP_S))


# ===========================================================================
# `enabled=False` -- the `greedy` ablation path.
# ===========================================================================
class TestSchedulerDisabled(unittest.TestCase):
    """`greedy` is `index` with the belief decay removed, the staleness bonus
    zeroed and the scheduler disabled.  For the ablation in DESIGN.md 11.9 to
    measure what the SCHEDULER contributes, `enabled=False` must genuinely bypass
    layers 2-5 and reduce to a pure argmax -- while still keeping layer 1, since
    emitting an action the receiver cannot afford is not policy behaviour, it is
    early termination."""

    def setUp(self):
        self.cands = _cands(
            scans=[(9, 1, 0.05), (20, 1, 0.05), (40, 1, 0.002)], sleeps=(0.2,)
        )
        self.scores = np.array([1.0, 2.0, 9.0, 0.5])

    def test_bypasses_the_deadline_layers(self):
        """Same inputs, both settings: enabled serves the overdue channel,
        disabled takes the argmax."""
        stale = _fresh(0.0)          # at t = 40 everything is overdue
        energy = EnergyState(0.0, 6.0)

        action_on, reason_on = _sched().select(
            self.cands, self.scores, stale, 40.0, energy
        )
        self.assertTrue(reason_on.startswith("deadline:"))

        action_off, reason_off = _sched(enabled=False).select(
            self.cands, self.scores, stale, 40.0, energy
        )
        self.assertEqual(reason_off, "index")
        self.assertEqual(action_off, self.cands.action(int(np.argmax(self.scores))))
        self.assertNotEqual(action_on, action_off)

    def test_bypasses_budget_pacing(self):
        over = EnergyState(spent_j=2.0, budget_j=6.0)
        self.assertTrue(over.over_pace(6.0, HORIZON))
        fresh = _fresh(5.0)

        _, reason_on = _sched().select(self.cands, self.scores, fresh, 6.0, over)
        self.assertEqual(reason_on, "budget-pace")

        action_off, reason_off = _sched(enabled=False).select(
            self.cands, self.scores, fresh, 6.0, over
        )
        self.assertEqual(reason_off, "index")
        self.assertEqual(action_off, self.cands.action(2))

    def test_bypasses_the_sleep_clamp(self):
        t_last = np.full(N_CH, 24.9)
        t_last[0] = 5.05                      # prio-3 deadline due at 25.05
        cands = _cands(scans=[(30, 1, 0.2)], sleeps=(0.2,))
        scores = np.array([1.0, 9.0])

        action_on, _ = _sched().select(
            cands, scores, FakeBelief(t_last), 25.0, EnergyState(0.0, 6.0)
        )
        self.assertAlmostEqual(action_on.dt_s, 0.05, places=9)

        action_off, reason_off = _sched(enabled=False).select(
            cands, scores, FakeBelief(t_last), 25.0, EnergyState(0.0, 6.0)
        )
        self.assertEqual(reason_off, "sleep")
        self.assertAlmostEqual(action_off.dt_s, 0.2)      # unclamped

    def test_is_a_pure_argmax_at_every_index(self):
        m = len(self.cands)
        sched = _sched(enabled=False)
        for i in range(m):
            with self.subTest(i=i):
                scores = np.full(m, -1.0)
                scores[i] = 5.0
                action, reason = sched.select(
                    self.cands, scores, _fresh(0.0), 40.0, EnergyState(0.0, 6.0)
                )
                self.assertEqual(action, self.cands.action(i))
                self.assertEqual(
                    reason, "sleep" if bool(self.cands.is_sleep[i]) else "index"
                )

    def test_layer_1_is_still_enforced(self):
        cands = _cands(
            scans=[(0, 1, 0.002), (10, 1, 0.002)], cost_override={0: 100.0}
        )
        action, _ = _sched(enabled=False).select(
            cands, np.array([9.0, 1.0]), _fresh(59.0), 1.0, EnergyState(0.0, 6.0)
        )
        self.assertEqual(action, cands.action(1))


if __name__ == "__main__":
    unittest.main()
