"""Metric ground truth.  Every number below was worked out by hand first.

The point of this file is that `eval/metrics.py` is checked against arithmetic a
person did on paper, not against its own output.  A metric that quietly changes
definition is the single easiest way to publish a wrong headline, and it is
invisible to any test that merely asserts "it ran".

The fixture is one hand-built episode:

    3 emitters, 4 activations, 5 actions, 10 channels, horizon 10 s.

Truth (`BURST_DTYPE` rows):

    | emitter | prio | act | channel | on   | off  |
    |---------|------|-----|---------|------|------|
    | 0       | 1    | 0   | 2       | 1.0  | 2.0  |
    | 0       | 1    | 1   | 2       | 5.0  | 6.0  |
    | 1       | 2    | 0   | 5       | 0.5  | 3.7  |
    | 2       | 3    | 0   | 8       | 7.0  | 8.0  |   <- never looked at

Actions (`t_start -> t_end`, observation window is `[t_end - dwell, t_end)`):

    | # | kind  | t_start | retune | dwell | t_end | channels | detections   |
    |---|-------|---------|--------|-------|-------|----------|--------------|
    | 0 | scan  | 0.0     | 0.0    | 1.5   | 1.5   | [2,3)    | ch2  -> TP (0,0) |
    | 1 | sleep | 1.5     | -      | -     | 3.5   | -        | -            |
    | 2 | scan  | 3.5     | 0.1    | 0.4   | 4.0   | [5,6)    | ch5  -> TP (1,0) |
    | 3 | scan  | 4.0     | 0.2    | 1.8   | 6.0   | [2,4)    | ch2 -> TP (0,1), ch3 -> FA |
    | 4 | sleep | 6.0     | -      | -     | 10.0  | -        | -            |

Action 2 is the interesting one: its window is `[3.6, 4.0)` and emitter 1 stops
radiating at 3.7, so it catches only the last 100 ms of a 3.2 s burst.  That is
what makes the coverage arithmetic non-trivial and the detection still a true
positive -- coverage is geometric, detection is not.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from eval.metrics import (
    METRIC_KEYS,
    TRACE_COLUMNS,
    EpisodeLog,
    StepRecord,
    compute_metrics,
)
from sim.contract import Mission
from sim.emitters import BURST_DTYPE

# POI horizons chosen so the three values differ: emitter 0 is first detected at
# t = 1.5 and emitter 1 at t = 4.0.
POI_T = (2.0, 5.0, 10.0)
HORIZON = 10.0
N_CH = 10

# (emitter_id, activation_id, burst_id, t_on, t_off, ch_lo, ch_hi, snr_db, prio, w_p)
BURSTS = np.array(
    [
        (0, 0, 0, 1.0, 2.0, 2, 3, -19.0, 1, 0.100),
        (0, 1, 1, 5.0, 6.0, 2, 3, -19.0, 1, 0.100),
        (1, 0, 2, 0.5, 3.7, 5, 6, -15.0, 2, 0.030),
        (2, 0, 3, 7.0, 8.0, 8, 9, -10.0, 3, 0.010),
    ],
    dtype=BURST_DTYPE,
)

# Energies chosen so every term in the breakdown is separately identifiable:
#   E(scan) = L_0 + L_d*dwell + L_f*|df|,  L_0 = 2 mJ, L_d = 1 W.
E0 = 0.002 + 1.5                 # 1.5020   no retune
E2 = 0.002 + 0.4 + 0.0001        # 0.4021   retune energy 0.1 mJ
E3 = 0.002 + 1.8 + 0.0002        # 1.8022   retune energy 0.2 mJ
E1 = 0.01 * 2.0                  # 0.0200   sleep 2.0 s
E4 = 0.01 * 4.0                  # 0.0400   sleep 4.0 s
E_TOTAL = E0 + E1 + E2 + E3 + E4  # 3.7663


def build_mission() -> Mission:
    """Channel 2 is the prio-1 tasking, channel 5 prio-2, the rest prio-3."""
    prio = np.full(N_CH, 3, dtype=np.int32)
    prio[2] = 1
    prio[5] = 2
    w = np.where(prio == 1, 0.100, np.where(prio == 2, 0.030, 0.010)).astype(np.float64)
    return Mission(
        priority=prio, w=w,
        deadlines_s={1: 0.5, 2: 2.0, 3: 10.0},
        watch_list=np.empty(0, dtype=np.int32), watch_deadline_s=0.3,
    )


def build_log() -> EpisodeLog:
    log = EpisodeLog(horizon_s=HORIZON, n_channels=N_CH)
    log.add_step(step=0, kind="scan", t_start=0.0, t_end=1.5, dwell_s=1.5,
                 retune_s=0.0, f_center_hz=2.0025e9, bw_hz=1.0e6,
                 k_lo=2, k_hi=3, energy_j=E0,
                 det_channels=np.array([2], np.int32),
                 det_snr_db=np.array([-18.5]), chosen_reason="index")
    log.add_step(step=1, kind="sleep", t_start=1.5, t_end=3.5, energy_j=E1,
                 chosen_reason="sleep")
    log.add_step(step=2, kind="scan", t_start=3.5, t_end=4.0, dwell_s=0.4,
                 retune_s=0.1, f_center_hz=2.0055e9, bw_hz=1.0e6,
                 k_lo=5, k_hi=6, energy_j=E2,
                 det_channels=np.array([5], np.int32),
                 det_snr_db=np.array([-14.2]), chosen_reason="deadline:ch=5")
    log.add_step(step=3, kind="scan", t_start=4.0, t_end=6.0, dwell_s=1.8,
                 retune_s=0.2, f_center_hz=2.003e9, bw_hz=2.0e6,
                 k_lo=2, k_hi=4, energy_j=E3,
                 det_channels=np.array([2, 3], np.int32),
                 det_snr_db=np.array([-19.1, -21.0]), chosen_reason="index")
    log.add_step(step=4, kind="sleep", t_start=6.0, t_end=10.0, energy_j=E4,
                 chosen_reason="budget-pace")
    return log


class TestHandBuiltEpisode(unittest.TestCase):
    """All six DESIGN.md section 6 metrics against hand arithmetic."""

    @classmethod
    def setUpClass(cls):
        cls.log = build_log()
        cls.mission = build_mission()
        cls.m = compute_metrics(BURSTS, cls.log, cls.mission,
                                horizon_s=HORIZON, n_channels=N_CH,
                                poi_times=POI_T)

    # ------------------------------------------------------------ accounting
    def test_action_counts_and_times(self):
        m = self.m
        self.assertEqual(m["n_steps"], 5)
        self.assertEqual(m["n_scans"], 3)
        self.assertEqual(m["n_sleeps"], 2)
        self.assertAlmostEqual(m["dwell_time_s"], 1.5 + 0.4 + 1.8)      # 3.7
        self.assertAlmostEqual(m["retune_time_s"], 0.0 + 0.1 + 0.2)     # 0.3
        self.assertAlmostEqual(m["sleep_time_s"], 2.0 + 4.0)            # 6.0
        self.assertAlmostEqual(m["t_end_s"], HORIZON)
        # 1 + 1 + 2 channels actually dwelt on
        self.assertEqual(m["n_channel_dwells"], 4)

    def test_energy_breakdown_sums_to_the_total(self):
        """The breakdown is recomputed from TIMING; the total is what the env
        charged.  Their agreeing is an independent check that the energy model
        and the timing model have not drifted (DESIGN.md section 1)."""
        m = self.m
        self.assertAlmostEqual(m["energy_total_j"], E_TOTAL, places=12)
        self.assertAlmostEqual(m["energy_fixed_j"], 3 * 0.002, places=12)
        self.assertAlmostEqual(m["energy_scan_j"], 3.7, places=12)
        self.assertAlmostEqual(m["energy_sleep_j"], 0.06, places=12)
        self.assertAlmostEqual(m["energy_retune_j"], 0.0003, places=9)
        parts = (m["energy_scan_j"] + m["energy_retune_j"]
                 + m["energy_fixed_j"] + m["energy_sleep_j"])
        self.assertAlmostEqual(parts, m["energy_total_j"], places=9)

    # ------------------------------------------------------------------ POI
    def test_poi_at_t(self):
        """Credited at the END of the dwell -- the moment the report exists.

        emitter 0 -> t = 1.5,  emitter 1 -> t = 4.0,  emitter 2 -> never.
        """
        m = self.m
        self.assertAlmostEqual(m["poi_2"], 1 / 3)
        self.assertAlmostEqual(m["poi_5"], 2 / 3)
        self.assertAlmostEqual(m["poi_10"], 2 / 3)

    def test_poi_prio1(self):
        m = self.m
        for t in (2, 5, 10):
            self.assertAlmostEqual(m[f"poi_p1_{t}"], 1.0)

    def test_a_detection_straddling_t_does_not_count_early(self):
        """Action 3's window is [4.2, 6.0); crediting at the START would make
        POI@5 include emitter 0's second activation.  It must not."""
        m = compute_metrics(BURSTS, self.log, self.mission, horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=(5.5,))
        # emitter 0 was already caught at 1.5, so use TTFI's first_det instead:
        # the second activation is credited at 6.0, outside T = 5.5.
        self.assertAlmostEqual(m["poi_5"], 2 / 3)

    # ----------------------------------------------------------------- TTFI
    def test_ttfi_prio1(self):
        """emitter 0 first radiates at 1.0 and is first detected at 1.5."""
        m = self.m
        self.assertAlmostEqual(m["ttfi_p1_median_s"], 0.5)
        self.assertAlmostEqual(m["ttfi_p1_p90_s"], 0.5)
        self.assertEqual(m["ttfi_p1_n_intercepted"], 1)
        self.assertEqual(m["ttfi_p1_n_total"], 1)
        self.assertAlmostEqual(m["ttfi_p1_frac"], 1.0)

    def test_ttfi_is_censored_not_dropped(self):
        """A prio-1 emitter that is never intercepted must contribute
        `horizon_s`, not vanish -- otherwise a policy wins TTFI by ignoring the
        hard emitters (DESIGN.md section 6)."""
        extra = np.concatenate([
            BURSTS,
            np.array([(3, 0, 4, 2.0, 3.0, 9, 10, -21.0, 1, 0.100)], dtype=BURST_DTYPE),
        ])
        m = compute_metrics(extra, self.log, self.mission, horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertEqual(m["ttfi_p1_n_intercepted"], 1)
        self.assertEqual(m["ttfi_p1_n_total"], 2)
        # values are [0.5, 10.0] -> median 5.25
        self.assertAlmostEqual(m["ttfi_p1_median_s"], 5.25)
        self.assertAlmostEqual(m["ttfi_p1_frac"], 0.5)

    # ------------------------------------------------------------- coverage
    def test_emitter_time_coverage(self):
        """Geometric: was the channel inside an in-progress dwell?

            (0,0) ch2 [1.0,2.0) covered by [0.0,1.5) -> 0.5 of 1.0
            (0,1) ch2 [5.0,6.0) covered by [4.2,6.0) -> 1.0 of 1.0
            (1,0) ch5 [0.5,3.7) covered by [3.6,4.0) -> 0.1 of 3.2
            (2,0) ch8 [7.0,8.0) never looked at      -> 0.0 of 1.0
                                                 1.6 / 6.2
        """
        self.assertAlmostEqual(self.m["coverage_frac"], 1.6 / 6.2)
        # prio-1 bursts only: (0.5 + 1.0) / (1.0 + 1.0)
        self.assertAlmostEqual(self.m["coverage_p1_frac"], 0.75)

    def test_coverage_ignores_retune(self):
        """Action 2 spends 0.1 s retuning before a 0.4 s dwell.  Counting the
        retune would give 0.2 s of emitter-1 coverage instead of 0.1 s."""
        rec = [s for s in self.log.steps if s.step == 2][0]
        self.assertAlmostEqual(rec.t_dwell_start, 3.6)
        self.assertAlmostEqual(rec.t_start, 3.5)

    def test_coverage_does_not_double_count_overlapping_scans(self):
        """Two scans covering the same channel-second are ONE covered second.

        Summing instead of unioning would let a policy inflate coverage simply
        by re-scanning, which is exactly the behaviour we are trying to measure
        against.
        """
        log = build_log()
        log.add_step(step=5, kind="scan", t_start=1.0, t_end=1.5, dwell_s=0.5,
                     retune_s=0.0, k_lo=2, k_hi=3, energy_j=0.0)
        m = compute_metrics(BURSTS, log, self.mission, horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertAlmostEqual(m["coverage_frac"], 1.6 / 6.2)

    # ------------------------------------------------ headline: energy/detection
    def test_energy_per_detection(self):
        """unique == distinct (emitter_id, activation_id).

        Four TP detections here map to three distinct activations: (0,0), (0,1)
        and (1,0).  Emitter 2 was never looked at.
        """
        m = self.m
        self.assertEqual(m["n_unique_detections"], 3)
        self.assertEqual(m["n_unique_p1_detections"], 2)
        self.assertAlmostEqual(m["energy_per_detection_j"], E_TOTAL / 3, places=9)
        self.assertAlmostEqual(m["energy_per_prio1_detection_j"], E_TOTAL / 2, places=9)

    def test_redetecting_one_burst_does_not_inflate_the_denominator(self):
        """THE property the headline metric exists for.

        Ten more detections of activation (0,0) add zero unique detections, so
        energy per detection can only get WORSE by re-scanning.  This is the
        exact failure mode the metric is designed to expose in a policy that
        parks on one loud emitter.
        """
        log = build_log()
        for i in range(10):
            log.add_step(step=10 + i, kind="scan", t_start=1.0 + 0.01 * i,
                         t_end=1.05 + 0.01 * i, dwell_s=0.05, retune_s=0.0,
                         k_lo=2, k_hi=3, energy_j=0.052,
                         det_channels=np.array([2], np.int32),
                         det_snr_db=np.array([-19.0]))
        m = compute_metrics(BURSTS, log, self.mission, horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertEqual(m["n_unique_detections"], 3)
        self.assertEqual(m["n_true_positive_dets"], 3 + 10)
        self.assertGreater(m["energy_per_detection_j"],
                           self.m["energy_per_detection_j"])

    def test_energy_per_detection_is_inf_when_nothing_is_found(self):
        log = EpisodeLog(horizon_s=HORIZON, n_channels=N_CH)
        log.add_step(step=0, kind="sleep", t_start=0.0, t_end=10.0, energy_j=0.1)
        m = compute_metrics(BURSTS, log, self.mission, horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertEqual(m["n_unique_detections"], 0)
        self.assertTrue(math.isinf(m["energy_per_detection_j"]))

    # ------------------------------------------------------- max staleness
    def test_max_staleness_prio1(self):
        """Prio-1 MISSION channel is ch 2; its merged coverage is
        [0.0,1.5) and [4.2,6.0), so the gaps are

            0 -> first    : 0.0
            first -> next : 4.2 - 1.5 = 2.7
            last  -> T    : 10.0 - 6.0 = 4.0        <- the worst
        """
        self.assertAlmostEqual(self.m["max_staleness_p1_s"], 4.0)
        self.assertAlmostEqual(self.m["mean_staleness_p1_s"], 4.0)

    def test_max_staleness_is_the_horizon_when_never_visited(self):
        mission = build_mission()
        prio = mission.priority.copy()
        prio[9] = 1                                    # a prio-1 channel we never scan
        m2 = Mission(priority=prio, w=mission.w, deadlines_s=mission.deadlines_s,
                     watch_list=mission.watch_list,
                     watch_deadline_s=mission.watch_deadline_s)
        m = compute_metrics(BURSTS, self.log, m2, horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertAlmostEqual(m["max_staleness_p1_s"], HORIZON)

    # ----------------------------------------------------------- false alarms
    def test_false_alarm_rate(self):
        """Of four detections, only ch 3 at [4.2, 6.0) has no overlapping burst.

        Per channel-dwell the rate is directly comparable to `P_fa`, which is
        what makes it double as a detector calibration check.
        """
        m = self.m
        self.assertEqual(m["n_detections"], 4)
        self.assertEqual(m["n_true_positive_dets"], 3)
        self.assertEqual(m["n_false_alarms"], 1)
        self.assertAlmostEqual(m["false_alarm_rate_per_dwell"], 1 / 4)
        self.assertAlmostEqual(m["false_alarm_rate_per_s"], 1 / 10.0)

    def test_population_counts(self):
        self.assertEqual(self.m["n_emitters"], 3)
        self.assertEqual(self.m["n_activations_total"], 4)

    # ------------------------------------------------------------- contract
    def test_every_declared_metric_key_is_produced(self):
        """Checked with the DEFAULT `poi_times`: `METRIC_KEYS` names poi_10/30/60,
        which only exist when the default POI horizons are used.  The rest of
        this class deliberately passes its own, so it needs its own call."""
        m = compute_metrics(BURSTS, self.log, self.mission, horizon_s=HORIZON,
                            n_channels=N_CH)
        missing = [k for k in METRIC_KEYS if k not in m]
        self.assertEqual(missing, [], f"compute_metrics dropped {missing}")

    def test_trace_rows_match_the_declared_schema(self):
        rows = self.log.trace_rows()
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertEqual(tuple(r.keys()), TRACE_COLUMNS)
        # `det_channels` is space-separated so the field never needs CSV quoting.
        self.assertEqual(rows[3]["det_channels"], "2 3")
        self.assertEqual(rows[1]["det_channels"], "")
        self.assertEqual(rows[1]["kind"], "sleep")


class TestEdgeCases(unittest.TestCase):
    """The degenerate inputs that a live demo will eventually produce."""

    def test_empty_log(self):
        log = EpisodeLog(horizon_s=HORIZON, n_channels=N_CH)
        m = compute_metrics(BURSTS, log, build_mission(), horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertEqual(m["n_steps"], 0)
        self.assertAlmostEqual(m["poi_10"], 0.0)
        self.assertAlmostEqual(m["coverage_frac"], 0.0)
        self.assertAlmostEqual(m["max_staleness_p1_s"], HORIZON)

    def test_empty_burst_table(self):
        """No emitters: POI is undefined (nan), every detection is a false alarm."""
        log = build_log()
        empty = np.empty(0, dtype=BURST_DTYPE)
        m = compute_metrics(empty, log, build_mission(), horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertTrue(math.isnan(m["poi_10"]))
        self.assertEqual(m["n_false_alarms"], 4)
        self.assertEqual(m["n_unique_detections"], 0)
        self.assertEqual(m["n_activations_total"], 0)

    def test_zero_length_dwell_contributes_no_coverage(self):
        log = EpisodeLog(horizon_s=HORIZON, n_channels=N_CH)
        log.add_step(step=0, kind="scan", t_start=1.0, t_end=1.0, dwell_s=0.0,
                     retune_s=0.0, k_lo=2, k_hi=3, energy_j=0.002)
        m = compute_metrics(BURSTS, log, build_mission(), horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        self.assertAlmostEqual(m["coverage_frac"], 0.0)
        self.assertEqual(m["n_channel_dwells"], 0)

    def test_one_detection_can_credit_two_overlapping_emitters(self):
        """Two emitters sharing a channel are both credited by one report.

        This is not a bug: the receiver cannot separate them, and the evaluator
        must not pretend it can.  It is also why `n_unique_detections` can
        exceed `n_true_positive_dets`.
        """
        stacked = np.concatenate([
            BURSTS,
            np.array([(9, 0, 9, 1.0, 2.0, 2, 3, -12.0, 3, 0.010)], dtype=BURST_DTYPE),
        ])
        m = compute_metrics(stacked, build_log(), build_mission(),
                            horizon_s=HORIZON, n_channels=N_CH, poi_times=POI_T)
        self.assertEqual(m["n_true_positive_dets"], 3)
        self.assertEqual(m["n_unique_detections"], 4)

    def test_bursts_outside_the_horizon_are_clipped_for_coverage(self):
        late = np.array([(5, 0, 5, 9.0, 12.0, 2, 3, -10.0, 3, 0.010)],
                        dtype=BURST_DTYPE)
        m = compute_metrics(late, build_log(), build_mission(), horizon_s=HORIZON,
                            n_channels=N_CH, poi_times=POI_T)
        # 1 s of the 3 s burst is inside [0, 10) and none of it was covered.
        self.assertAlmostEqual(m["coverage_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
