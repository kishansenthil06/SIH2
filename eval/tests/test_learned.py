"""Rung-2 tests.  DESIGN.md section 8.

Six properties, each of which is a claim we make out loud in the pitch:

  1. round trip      -- the model can actually learn, and persists exactly.
  2. deconvolution   -- `p_det -> p_active` inverts the detector to 1e-9.  This
                        is the numerical answer to "how do you know your ML
                        isn't just relearning your detector?"
  3. anti-regression -- beta=0 is bit-identical to rung 1, and a useless model
                        is forced back to beta=0 by the Brier gate.
  4. feature contract-- the training matrix is FEATURE_NAMES in order (+2), finite.
  5. no leakage      -- the raw channel index never reaches the estimator.
  6. wiring          -- `Belief.attach_model` actually reaches the model.

The training data is generated HERE, not read from disk: agent C's collector
(`python -m eval.runner --collect --seeds 100-119`) may not exist yet, and these
tests must not depend on it.  `test_logs_absent_degrades_cleanly` covers the
"collector has not run" path explicitly.

Everything is capped to a small `max_iter` so the whole file stays inside the
30 s suite budget; the frozen 300-iteration hyper-parameters are asserted
separately in `test_hyperparameters_are_the_frozen_ones`.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.base import FEATURE_NAMES, N_FEATURES, TRAIN_EXTRA_NAMES
from agent.policy_learned import (
    BRIER_GATE_MARGIN,
    HGB_PARAMS,
    ActivityModel,
    LogsUnavailable,
    P_CLIP_HI,
    P_CLIP_LO,
    PdBar,
    assert_no_channel_leakage,
    brier,
    build_training_matrix,
    load_log_frame,
    p_active_from_p_det,
    p_det_from_p_active,
    train_activity_model,
)

# Small enough that three calibration folds fit inside the per-test budget,
# large enough that isotonic calibration has something to fit.
N_ROWS = 1400
FAST_MODEL = dict(max_iter=30, early_stopping=False, min_samples_leaf=20)


# ---------------------------------------------------------------------------
# Synthetic collector.  Mimics the schema `agent/policy_index.py:log_rows()`
# emits: one row per channel per decision, features as they stood BEFORE the
# action, the action's own dwell/bw, and `detected` filled in retroactively --
# i.e. rows are ALREADY (features, tau_next, bw_next, y_next).
# ---------------------------------------------------------------------------
def synth_log(n: int = N_ROWS, seed: int = 7, deterministic: bool = True):
    """Return a DataFrame in the collector's schema.

    With `deterministic=True` the label is an exact function of exactly two
    features, so a working learner must drive the Brier score near zero and a
    broken one cannot fake it.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    cols = {}
    for name in FEATURE_NAMES:
        cols[name] = rng.normal(0.0, 1.0, n)
    # Keep the two contract-meaningful columns on their real scales, because the
    # rung-1 baseline forecast and the min_visits gate both read them.
    cols["p_rung1"] = rng.uniform(0.01, 0.60, n)
    cols["n_visits"] = rng.integers(0, 15, n).astype(np.float64)
    cols["emp_rate"] = rng.uniform(0.0, 1.0, n)
    cols["hit_ema_fast"] = rng.uniform(0.0, 1.0, n)

    df = pd.DataFrame(cols)
    df["run_id"] = rng.integers(0, 10, n)
    df["channel"] = rng.integers(0, 200, n)
    df["t"] = rng.uniform(0.0, 60.0, n)
    df["step"] = np.arange(n)
    # Only values from the frozen candidate sets, so `pd_bar` lookups are exact.
    df["dwell_s"] = rng.choice([0.001, 0.002, 0.010, 0.050, 0.200], n)
    df["bw_hz"] = rng.choice([1.0e6, 2.0e6, 5.0e6, 20.0e6], n)
    df["scenario"] = "sparse"
    df["seed"] = 100

    if deterministic:
        # y depends on exactly two features and nothing else.
        y = ((df["p_rung1"].to_numpy() > 0.30) & (df["hit_ema_fast"].to_numpy() > 0.40))
        df["detected"] = y.astype(int)
    else:
        # Draw the label through the FORWARD map rung 1 itself uses,
        # `p_det = p_act*pd_bar + (1-p_act)*P_fa`.  That makes rung 1 a
        # well-calibrated forecaster of this label, which is the only setting in
        # which the Brier gate is a meaningful test: a sabotaged model has to
        # lose to a baseline that is actually good, not to a straw man.
        pdb = PdBar()
        pd_bar = np.asarray(
            pdb(df["dwell_s"].to_numpy(), df["bw_hz"].to_numpy()), dtype=np.float64
        )
        p_det = df["p_rung1"].to_numpy() * pd_bar + (1.0 - df["p_rung1"].to_numpy()) * 1e-3
        df["detected"] = (rng.uniform(size=n) < p_det).astype(int)
    return df


def sorted_like_training(df):
    """`build_training_matrix` sorts by (run_id, channel, t); mirror that here."""
    return df.sort_values(["run_id", "channel", "t"], kind="mergesort").reset_index(
        drop=True
    )


class _Sabotage:
    """A model that predicts 0.5 for everything.  Ranks nothing, learns nothing.

    Brier 0.25 on any label, which is far worse than rung 1 on a sparse band, so
    it is exactly the thing the gate exists to catch.
    """

    def predict_proba(self, X):
        p = np.full(np.asarray(X).shape[0], 0.5)
        return np.column_stack([1.0 - p, p])


class _Constant:
    """Predicts a fixed probability.  Used to make `refine` deterministic."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        p = np.full(np.asarray(X).shape[0], self.p)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------
class TestRoundTrip(unittest.TestCase):
    """The model learns a learnable label, and survives save/load bit-exactly."""

    @classmethod
    def setUpClass(cls):
        df = synth_log(deterministic=True)
        cls.X, cls.y, cls.meta = build_training_matrix(df)
        cls.model = train_activity_model(
            cls.X, cls.y, cls.meta["pd_bar_next"],
            seed=0, beta=0.5, groups=cls.meta["run_id"],
            model_kwargs=FAST_MODEL,
        )

    def test_learns_the_label(self):
        p = self.model.predict_p_det(self.X)
        self.assertLess(brier(p, self.y), 0.05,
                        "a label that is a deterministic function of two features "
                        "must be learnable to Brier < 0.05")

    def test_calibrated_output_is_a_probability(self):
        p = self.model.predict_p_det(self.X)
        self.assertTrue(np.all(p >= P_CLIP_LO) and np.all(p <= P_CLIP_HI))
        # Calibration is mandatory because the output feeds Bayes: the mean
        # forecast must track the base rate, not merely rank correctly.
        self.assertAlmostEqual(float(p.mean()), float(self.y.mean()), delta=0.05)

    def test_persist_reload_is_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.model.save(Path(d) / "activity_hgb.joblib")
            self.assertTrue(path.exists())
            man_path = ActivityModel.manifest_path(path)
            self.assertTrue(man_path.exists())
            man = json.loads(man_path.read_text(encoding="utf-8"))
            for key in ("feature_names", "sklearn_version", "training_seeds",
                        "brier_model", "brier_rung1", "n_rows_total"):
                self.assertIn(key, man, f"manifest must record {key!r}")
            self.assertEqual(man["training_seeds"], list(range(100, 120)))
            self.assertEqual(man["held_out_scenario"], "agile")
            self.assertNotIn("agile", man["training_scenarios"])

            reloaded = ActivityModel.load(path, beta=0.5)
            a = self.model.predict_p_det(self.X)
            b = reloaded.predict_p_det(self.X)
            self.assertTrue(np.array_equal(a, b),
                            "reloaded model must give bit-identical predictions")


# ---------------------------------------------------------------------------
# 2. Deconvolution -- the load-bearing three lines
# ---------------------------------------------------------------------------
class TestDeconvolution(unittest.TestCase):
    """`p_active_hat` must invert `p_det = p_act*P_d + (1-p_act)*P_fa` exactly.

    If this fails, the learned belief double-counts the detector: the model has
    already folded P_d into its forecast and the Bayes update would apply it a
    second time.
    """

    def test_inverts_the_forward_map(self):
        pdb = PdBar()
        p_act = np.linspace(1e-4, 1.0 - 1e-4, 257)
        pfa = 1e-3
        for bw_hz in (1.0e6, 2.0e6, 5.0e6, 10.0e6, 20.0e6):
            for dwell_s in (0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200):
                pd_bar = float(pdb(dwell_s, bw_hz))
                p_det = p_det_from_p_active(p_act, pd_bar, pfa)
                back = p_active_from_p_det(p_det, pd_bar, pfa)
                err = float(np.max(np.abs(back - p_act)))
                self.assertLess(
                    err, 1e-9,
                    f"deconvolution error {err:.2e} at bw={bw_hz/1e6:g} MHz, "
                    f"dwell={dwell_s*1e3:g} ms (pd_bar={pd_bar:.5f})",
                )

    def test_pfa_input_maps_to_zero_activity(self):
        # A forecast of exactly P_fa means "the only detections I expect are
        # false alarms", i.e. no activity.  Clipped, not negative.
        pdb = PdBar()
        pd_bar = float(pdb(0.010, 1.0e6))
        self.assertAlmostEqual(
            float(p_active_from_p_det(1e-3, pd_bar, 1e-3)), P_CLIP_LO, places=12
        )

    def test_deconvolution_is_dwell_dependent(self):
        # The same P(detect) means far MORE activity at a short dwell than at a
        # long one -- which is exactly why tau_next/bw_next are model inputs.
        pdb = PdBar()
        short = float(p_active_from_p_det(0.20, float(pdb(0.001, 1.0e6))))
        long_ = float(p_active_from_p_det(0.20, float(pdb(0.200, 1.0e6))))
        self.assertGreater(short, long_)

    def test_uses_the_beliefs_own_pd_bar_when_attached(self):
        """The belief's table and this module's must never drift apart.

        The deconvolution divides by pd_bar and the belief's Bayes update
        multiplies by it, so a disagreement is a systematic bias, not rounding.
        """
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        cfg = load_config("sparse")
        bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
        model = ActivityModel(estimator=None)
        model.attach_to(bel, 0.0)
        self.assertEqual(
            float(model._pd_bar_for(0.010, 1.0e6)), bel.pd_bar_for(1.0e6, 0.010)
        )

    def test_standalone_pdbar_reproduces_the_beliefs_quadrature(self):
        """The STANDALONE table must match too, not just the borrowed one.

        `eval/runner.py` wires the model in with a bare `attach_model`, never
        `attach_to`, so `_pd_bar_fn` is None on the path that actually runs the
        sweep -- which means the standalone `PdBar` IS the divisor there.  The
        old Gauss-Hermite rule disagreed with `agent.belief.marginal_pd_table` by
        up to ~2.9e-3, a systematic bias in every deconvolution.  The default
        quadrature now reproduces the belief's rule exactly.
        """
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        cfg = load_config("sparse")
        bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
        pdb = PdBar.from_cfg(cfg)
        self.assertEqual(pdb.quadrature, "belief", "the matching rule is the default")

        worst = 0.0
        for bw_mhz in (1, 2, 5, 10, 20):
            for dw_ms in (1, 2, 5, 10, 20, 50, 100, 200):
                ref = bel.pd_bar_for(bw_mhz * 1e6, dw_ms * 1e-3)
                got = float(pdb(dw_ms * 1e-3, bw_mhz * 1e6))
                worst = max(worst, abs(got - ref))
        self.assertLess(worst, 1e-12,
                        f"standalone PdBar drifts from Belief by {worst:.2e}")

    def test_class_aware_deconvolution_also_inverts_exactly(self):
        """The opt-in per-class divisor must invert to 1e-9 as well.

        Agent B exposes `pd_bar_by_class` because ONE band-wide assumed SNR made
        a 2 ms scan look like `P_d ~ 0.28` everywhere when for a -20 dB threat
        emitter it is 0.004.  Dividing by the per-class number is a different
        question but the same algebra, and the round trip must still be exact --
        otherwise the flag would quietly double-count the detector.
        """
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        cfg = load_config("sparse")
        bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
        model = ActivityModel(estimator=None, use_class_pd_bar=True)
        model.borrow_tables_from(bel)

        n = bel.n
        pd_vec = model._pd_bar_vector(0.002, 1.0e6, n)
        self.assertEqual(pd_vec.shape, (n,))
        # Per class, not one number everywhere -- that is the whole point.
        self.assertGreater(len(set(np.round(pd_vec, 12).tolist())), 1)

        rng = np.random.default_rng(3)
        p_act = rng.uniform(1e-4, 1.0 - 1e-4, n)
        back = p_active_from_p_det(
            p_det_from_p_active(p_act, pd_vec, 1e-3), pd_vec, 1e-3
        )
        self.assertLess(float(np.max(np.abs(back - p_act))), 1e-9)

    def test_class_aware_divisor_is_smaller_on_a_threat_channel(self):
        """A -19 dB threat channel is far less detectable than a -10 dB routine
        one, so the same P(detect) must deconvolve to MORE activity there."""
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        cfg = load_config("sparse")
        mission = build_mission(cfg)
        bel = Belief(build_grid(cfg), mission, cfg)
        model = ActivityModel(estimator=None, use_class_pd_bar=True)
        model.borrow_tables_from(bel)

        prio = np.asarray(mission.priority)
        pd_vec = model._pd_bar_vector(0.002, 1.0e6, bel.n)
        p1 = pd_vec[prio == 1]
        p3 = pd_vec[prio == 3]
        if not p1.size or not p3.size:
            self.skipTest("scenario has no prio-1 or no prio-3 mission channels")
        self.assertLess(float(p1.mean()), float(p3.mean()),
                        "threat channels must have the LOWER assumed P_d")
        # ...and therefore the larger deconvolved activity for one P(detect).
        a1 = p_active_from_p_det(0.05, float(p1.mean()))
        a3 = p_active_from_p_det(0.05, float(p3.mean()))
        self.assertGreater(float(a1), float(a3))

    def test_band_wide_is_the_default(self):
        """DESIGN.md section 8 writes the band-wide form, and the Brier gate
        compares against rung 1's band-wide forward map, so that is the default;
        per-class is a flag, never a silent change of behaviour."""
        self.assertFalse(ActivityModel().use_class_pd_bar)


# ---------------------------------------------------------------------------
# 3. Anti-regression -- the important one
# ---------------------------------------------------------------------------
class TestAntiRegression(unittest.TestCase):
    """Three independent guarantees that rung 2 cannot make the demo worse."""

    def setUp(self):
        self.df = synth_log(deterministic=False, seed=11)
        self.X, self.y, self.meta = build_training_matrix(self.df)
        self.F = self.X[:, :N_FEATURES]

    # -- guarantee 1 -------------------------------------------------------
    def test_beta_defaults_to_zero(self):
        self.assertEqual(ActivityModel().beta, 0.0)
        self.assertEqual(ActivityModel(estimator=_Constant(0.9)).beta, 0.0)

    def test_beta_zero_is_bit_identical_to_rung1(self):
        model = ActivityModel(estimator=_Constant(0.9), beta=0.0)
        out = model.refine(self.F, 0.010, 1.0e6)
        self.assertTrue(
            np.array_equal(out, self.F[:, FEATURE_NAMES.index("p_rung1")]),
            "beta=0 must return the analytic belief UNTOUCHED -- bit-identical, "
            "not merely close",
        )
        # And explicitly overriding beta to 0 at the call site does the same.
        out2 = ActivityModel(estimator=_Constant(0.9), beta=0.9).refine(
            self.F, 0.010, 1.0e6, beta=0.0
        )
        self.assertTrue(np.array_equal(out2, self.F[:, FEATURE_NAMES.index("p_rung1")]))

    # -- guarantee 2 -------------------------------------------------------
    def test_cold_start_channels_are_gated_out(self):
        model = ActivityModel(estimator=_Constant(0.9), beta=1.0,
                              min_visits_for_model=3)
        out = model.refine(self.F, 0.010, 1.0e6)
        p1 = self.F[:, FEATURE_NAMES.index("p_rung1")]
        cold = self.F[:, FEATURE_NAMES.index("n_visits")] < 3
        self.assertTrue(cold.any(), "synthetic log must contain cold-start rows")
        self.assertTrue(
            np.allclose(out[cold], p1[cold]),
            "below min_visits_for_model the analytic belief IS the prior and the "
            "model must not touch it",
        )
        self.assertFalse(np.allclose(out[~cold], p1[~cold]),
                         "above the gate the model must actually do something")

    # -- guarantee 3 -------------------------------------------------------
    def test_sabotaged_model_trips_the_brier_gate(self):
        model = ActivityModel(estimator=_Sabotage(), beta=0.8)
        with self.assertWarns(RuntimeWarning):
            gate = model.evaluate_gate(
                self.X, self.y, self.meta["pd_bar_next"], p_rung1=self.meta["p_rung1"]
            )
        self.assertFalse(gate["gate_ok"])
        self.assertEqual(model.beta, 0.0, "a failing model must be forced to beta=0")
        self.assertAlmostEqual(gate["brier_model"], 0.25, places=6)
        self.assertLess(gate["brier_rung1"], gate["brier_model"])
        # ...and with beta forced to 0, the refined belief is rung 1 again.
        out = model.refine(self.F, 0.010, 1.0e6)
        self.assertTrue(np.array_equal(out, self.F[:, FEATURE_NAMES.index("p_rung1")]))

    def test_a_tie_loses_to_rung1(self):
        """The gate margin is one-sided on purpose: the simpler path wins ties."""
        # pfa=0 and pd_bar=1 make the forward map the identity, so rung 1's
        # forecast is exactly 0.5 -- the same number the model emits.
        model = ActivityModel(estimator=_Sabotage(), beta=0.5, pfa=0.0)
        p_r1 = np.full(self.y.shape[0], 0.5)
        pd_bar = np.full(self.y.shape[0], 1.0)
        gate = model.evaluate_gate(self.X, self.y, pd_bar, p_rung1=p_r1)
        self.assertEqual(gate["brier_model"], gate["brier_rung1"])
        self.assertAlmostEqual(gate["brier_model"], gate["brier_rung1"], places=3)
        self.assertFalse(gate["gate_ok"])
        self.assertEqual(model.beta, 0.0)
        self.assertGreater(BRIER_GATE_MARGIN, 0.0)

    def test_gate_trips_at_load_without_manifest_scores(self):
        """A model with no recorded held-out Brier cannot prove itself -> beta=0."""
        model = ActivityModel(estimator=_Constant(0.9), beta=0.0)
        model.manifest = {"note": "no brier scores recorded"}
        with tempfile.TemporaryDirectory() as d:
            path = model.save(Path(d) / "activity_hgb.joblib")
            reloaded = ActivityModel.load(path, beta=0.9)
            self.assertEqual(reloaded.beta, 0.0)
            self.assertFalse(reloaded.gate_ok)
            with self.assertRaises(RuntimeError):
                ActivityModel.load(path, beta=0.9, strict=True)

    def test_failed_gate_cannot_be_re_enabled_by_attach(self):
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        cfg = load_config("sparse")
        bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
        model = ActivityModel(estimator=_Constant(0.9), beta=0.0)
        model.gate_ok = False
        model.attach_to(bel, 0.9)
        self.assertTrue(np.array_equal(bel.p_active(1.0), bel.p_effective(1.0)))


# ---------------------------------------------------------------------------
# 4. Feature contract
# ---------------------------------------------------------------------------
class TestFeatureContract(unittest.TestCase):

    def setUp(self):
        # Compared column-by-column below, so hold the frame in the same row
        # order `build_training_matrix` puts it in.
        self.df = sorted_like_training(synth_log(deterministic=False, seed=3))
        self.X, self.y, self.meta = build_training_matrix(self.df)

    def test_column_count_and_order(self):
        self.assertEqual(self.X.shape[1], N_FEATURES + len(TRAIN_EXTRA_NAMES))
        self.assertEqual(len(TRAIN_EXTRA_NAMES), 2)
        for i, name in enumerate(FEATURE_NAMES):
            self.assertTrue(
                np.allclose(self.X[:, i], self.df[name].to_numpy()),
                f"column {i} must be {name!r}, in FEATURE_NAMES order",
            )

    def test_train_extras_are_the_next_actions_parameters(self):
        tau_col = self.X[:, N_FEATURES]
        bw_col = self.X[:, N_FEATURES + 1]
        self.assertTrue(np.allclose(tau_col, np.log1p(self.meta["tau_next"])))
        self.assertTrue(np.allclose(bw_col, np.log1p(self.meta["bw_next"] / 1.0e6)))

    def test_matrix_is_finite(self):
        self.assertTrue(np.isfinite(self.X).all(),
                        "one NaN would poison an entire training run")
        self.assertTrue(np.isfinite(self.meta["pd_bar_next"]).all())
        self.assertTrue(np.all(self.meta["pd_bar_next"] > 0.0))

    def test_model_declares_the_full_column_list(self):
        self.assertEqual(
            tuple(ActivityModel().feature_names), FEATURE_NAMES + TRAIN_EXTRA_NAMES
        )

    def test_wrong_width_is_rejected(self):
        model = ActivityModel(estimator=_Constant(0.5), beta=0.5)
        with self.assertRaises(ValueError):
            model.predict_p_det(self.X[:, :-1])
        with self.assertRaises(ValueError):
            model.refine(self.X, 0.010, 1.0e6)   # train-width matrix at inference

    def test_pre_labelled_is_the_default_and_does_not_double_shift(self):
        """`policy_index.log_rows()` rows are ALREADY (features, tau_next, y_next).

        Shifting them again would label a row with the outcome two observations
        away -- a silent failure that still trains, so it is asserted here.
        """
        _, y_pre, meta_pre = build_training_matrix(self.df)
        self.assertTrue(
            np.array_equal(y_pre, (self.df["detected"].to_numpy() > 0).astype(np.int8))
        )
        self.assertTrue(np.allclose(meta_pre["tau_next"], self.df["dwell_s"].to_numpy()))
        # The `shift` mode remains available for a raw, unlabelled collector.
        X_sh, y_sh, _ = build_training_matrix(self.df, label_mode="shift")
        self.assertLess(X_sh.shape[0], self.X.shape[0],
                        "shift mode drops the last row of every channel group")
        with self.assertRaises(ValueError):
            build_training_matrix(self.df, label_mode="nonsense")

    def test_missing_contract_feature_is_a_loud_error(self):
        bad = self.df.drop(columns=["nbr_recent_hits"])
        with self.assertRaises(ValueError) as ctx:
            build_training_matrix(bad)
        self.assertIn("nbr_recent_hits", str(ctx.exception))

    def test_hyperparameters_are_the_frozen_ones(self):
        # DESIGN.md section 8, verbatim.  Tests override these for speed, so the
        # real values are asserted on the module constant.
        self.assertEqual(HGB_PARAMS["max_iter"], 300)
        self.assertEqual(HGB_PARAMS["learning_rate"], 0.08)
        self.assertEqual(HGB_PARAMS["max_leaf_nodes"], 15)
        self.assertEqual(HGB_PARAMS["min_samples_leaf"], 40)
        self.assertEqual(HGB_PARAMS["l2_regularization"], 1.0)
        self.assertTrue(HGB_PARAMS["early_stopping"])
        self.assertEqual(HGB_PARAMS["validation_fraction"], 0.15)

    def test_calibration_wrapper_is_present(self):
        """Calibration is mandatory: the output feeds Bayes, not a ranking."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier

        from agent.policy_learned import make_estimator

        est = make_estimator(seed=0)
        self.assertIsInstance(est, CalibratedClassifierCV)
        self.assertEqual(est.method, "isotonic")
        self.assertEqual(est.cv, 3)
        self.assertIsInstance(est.estimator, HistGradientBoostingClassifier)


# ---------------------------------------------------------------------------
# 5. No channel-index leakage
# ---------------------------------------------------------------------------
class TestNoChannelLeakage(unittest.TestCase):
    """Absolute channel position would let the model memorise `sparse`'s emitter
    layout and generalise to nothing -- which is precisely what the held-out
    `agile` scenario exists to catch."""

    def test_channel_is_not_a_contract_feature(self):
        self.assertNotIn("channel", FEATURE_NAMES)
        for banned in ("channel", "ch", "chan", "channel_idx", "k_lo", "f_center_hz"):
            self.assertNotIn(banned, FEATURE_NAMES)

    def test_channel_is_not_a_fitting_column(self):
        df = synth_log(deterministic=False, seed=5)
        self.assertIn("channel", df.columns, "the log DOES carry channel, for grouping")
        X, _, meta = build_training_matrix(df)
        fitting_cols = tuple(ActivityModel().feature_names)
        self.assertNotIn("channel", fitting_cols)
        self.assertEqual(X.shape[1], len(fitting_cols))
        # And no fitted column is numerically the channel index either.
        ch = np.asarray(meta["channel"], dtype=np.float64)
        for j in range(X.shape[1]):
            self.assertFalse(np.array_equal(X[:, j], ch),
                             f"column {j} is the raw channel index")

    def test_guard_raises_on_a_leaked_name(self):
        assert_no_channel_leakage(FEATURE_NAMES + TRAIN_EXTRA_NAMES)  # must not raise
        for leak in ("channel", "k_lo", "f_center_hz"):
            with self.assertRaises(ValueError):
                assert_no_channel_leakage(FEATURE_NAMES + (leak,))

    def test_load_rejects_a_model_trained_on_a_different_contract(self):
        model = ActivityModel(estimator=_Constant(0.5))
        model.feature_names = FEATURE_NAMES + ("channel",)
        with tempfile.TemporaryDirectory() as d:
            path = model.save(Path(d) / "activity_hgb.joblib")
            with self.assertRaises(ValueError):
                ActivityModel.load(path)


# ---------------------------------------------------------------------------
# 6. Wiring into the frozen BeliefLike hook
# ---------------------------------------------------------------------------
class TestBeliefWiring(unittest.TestCase):
    """`Belief._model_p_active` duck-types `model.p_active_hat(features, t)`.

    An `ActivityModel` is a dataclass, not a callable, so without that method the
    belief falls through to `return None` and the learned path silently does
    nothing at ANY beta.  This test is the regression guard for that.
    """

    def _belief(self):
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        cfg = load_config("sparse")
        return Belief(build_grid(cfg), build_mission(cfg), cfg)

    def test_model_exposes_the_hook(self):
        self.assertTrue(callable(getattr(ActivityModel, "p_active_hat", None)))

    def test_attached_model_actually_moves_the_belief(self):
        bel = self._belief()
        bel.n_visits[:] = 5.0                      # past the cold-start gate
        model = ActivityModel(estimator=_Constant(0.30), beta=0.5)
        model.attach_to(bel, 0.5)
        p1, pe = bel.p_active(1.0), bel.p_effective(1.0)
        self.assertFalse(np.array_equal(p1, pe),
                         "an attached model at beta>0 must change p_effective")
        # Verify the arithmetic end to end: blend of rung 1 and the DECONVOLVED
        # model output, using the belief's own pd_bar.
        pd_bar = bel.pd_bar_for(model.infer_bw_hz, model.infer_dwell_s)
        p_act = (0.30 - model.pfa) / (pd_bar - model.pfa)
        self.assertAlmostEqual(float(pe[0]), 0.5 * float(p1[0]) + 0.5 * p_act, places=9)

    def test_beta_zero_leaves_the_belief_bit_identical(self):
        bel = self._belief()
        bel.n_visits[:] = 5.0
        model = ActivityModel(estimator=_Constant(0.30), beta=0.0)
        model.attach_to(bel)
        self.assertTrue(np.array_equal(bel.p_active(1.0), bel.p_effective(1.0)))

    def test_unfitted_model_falls_back_to_rung1(self):
        bel = self._belief()
        bel.n_visits[:] = 5.0
        model = ActivityModel(estimator=None, beta=0.9)
        bel.attach_model(model, 0.9)
        self.assertTrue(np.array_equal(bel.p_active(1.0), bel.p_effective(1.0)))


# ---------------------------------------------------------------------------
# Degradation when agent C's collector has not run
# ---------------------------------------------------------------------------
class TestLogsAbsentDegradesCleanly(unittest.TestCase):

    def test_missing_logs_raise_an_actionable_message(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(LogsUnavailable) as ctx:
                load_log_frame(log_dir=Path(d) / "does-not-exist")
        msg = str(ctx.exception)
        self.assertIn("eval.runner", msg)
        self.assertIn("--collect", msg)
        self.assertIn("agile", msg, "the message must name the held-out scenario")

    def test_cli_train_returns_nonzero_without_logs(self):
        from agent.policy_learned import main

        with tempfile.TemporaryDirectory() as d:
            rc = main(["--train", "--log-dir", str(Path(d) / "nope"),
                       "--out", str(Path(d) / "m.joblib")])
        self.assertEqual(rc, 2, "--train must degrade to an instruction, not a crash")

    def test_agile_is_never_trained_on(self):
        import pandas as pd

        from agent.policy_learned import TRAIN_SCENARIOS

        self.assertNotIn("agile", TRAIN_SCENARIOS)
        df = synth_log(deterministic=False, seed=9)
        agile = df.copy()
        agile["scenario"] = "agile"
        with tempfile.TemporaryDirectory() as d:
            pd.concat([df, agile], ignore_index=True).to_csv(
                Path(d) / "logs.csv", index=False
            )
            kept = load_log_frame(log_dir=Path(d))
        self.assertEqual(set(kept["scenario"].unique()), {"sparse"})


# ---------------------------------------------------------------------------
# 7. The REAL collector output, when it is on disk
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
REAL_LOG_DIR = ROOT / "data" / "logs"


def _one_real_log():
    """The smallest real collector CSV, or None.  `data/logs/` is gitignored."""
    if not REAL_LOG_DIR.is_dir():
        return None
    paths = sorted(REAL_LOG_DIR.glob("*.csv"), key=lambda p: p.stat().st_size)
    return paths[0] if paths else None


class TestRealCollectorOutput(unittest.TestCase):
    """Everything above runs on a synthetic frame, deliberately -- the tests must
    not depend on agent C's collector having run.  But a synthetic schema can
    agree with itself and still disagree with reality, so when
    `python -m eval.runner --collect` HAS run, these assert against its actual
    output.  They skip cleanly when it has not.

    Capped to one episode and a slice of its rows so the file stays inside the
    per-test budget on the wide grid (~5300 rows per episode).
    """

    MAX_ROWS = 4000

    @classmethod
    def setUpClass(cls):
        cls.path = _one_real_log()
        if cls.path is None:
            raise unittest.SkipTest(
                "no collected logs in data/logs/; run "
                "`python -m eval.runner --collect --scenarios sparse dense "
                "--seeds 100-119` to exercise these"
            )
        import pandas as pd

        cls.df = pd.read_csv(cls.path, nrows=cls.MAX_ROWS)

    def test_real_log_carries_every_contract_feature(self):
        for name in FEATURE_NAMES:
            self.assertIn(name, self.df.columns,
                          f"collector must emit {name!r}")

    def test_channel_is_present_in_the_log_and_absent_from_the_matrix(self):
        """(a) of the brief: `channel` is bookkeeping and must be dropped.

        This is the assertion made against REAL data rather than a synthetic
        stand-in: the collector genuinely writes `channel` on every row, and the
        fitted matrix genuinely must not contain it.
        """
        self.assertIn("channel", self.df.columns)
        X, _, meta = build_training_matrix(self.df)
        self.assertEqual(X.shape[1], len(FEATURE_NAMES) + len(TRAIN_EXTRA_NAMES))
        self.assertNotIn("channel", tuple(meta["fitting_names"]))

        ch = np.asarray(meta["channel"], dtype=np.float64)
        self.assertGreater(len(np.unique(ch)), 1, "a real log visits many channels")
        for j in range(X.shape[1]):
            col = X[:, j]
            self.assertFalse(np.array_equal(col, ch),
                             f"column {j} IS the raw channel index")
            # Also reject a monotone re-encoding of it (a rank, a scaled copy):
            # anything perfectly correlated with position leaks the same thing.
            if np.std(col) > 0:
                r = abs(float(np.corrcoef(col, ch)[0, 1]))
                self.assertLess(r, 0.99,
                                f"column {j} ({meta['fitting_names'][j]}) is "
                                f"{r:.4f}-correlated with the channel index")

    def test_real_rows_are_finite_and_labelled(self):
        X, y, _ = build_training_matrix(self.df)
        self.assertTrue(np.isfinite(X).all(), "a single NaN poisons the whole fit")
        self.assertEqual(set(np.unique(y).tolist()) - {0, 1}, set())

    def test_deconvolution_inverts_on_real_pd_bar_values(self):
        """(b) of the brief, on the dwell/bandwidth pairs a real episode used."""
        _, _, meta = build_training_matrix(self.df)
        pdb = np.asarray(meta["pd_bar_next"], dtype=np.float64)
        self.assertTrue(np.all(pdb > 1e-3), "every real pd_bar exceeds P_fa")
        rng = np.random.default_rng(0)
        p_act = rng.uniform(1e-4, 1.0 - 1e-4, pdb.size)
        back = p_active_from_p_det(p_det_from_p_active(p_act, pdb, 1e-3), pdb, 1e-3)
        err = float(np.max(np.abs(back - p_act)))
        self.assertLess(err, 1e-9, f"real-data deconvolution error {err:.2e}")

    def test_scenario_and_seed_columns_enforce_the_split(self):
        """`agile` never trained on; collection seeds never overlap evaluation."""
        from agent.policy_learned import TRAIN_SEEDS

        self.assertIn("scenario", self.df.columns)
        self.assertIn("seed", self.df.columns)
        self.assertNotIn("agile", set(self.df["scenario"].astype(str)))
        seeds = {int(s) for s in self.df["seed"]}
        self.assertTrue(seeds.issubset(set(TRAIN_SEEDS)),
                        f"collected seeds {sorted(seeds)} must lie in 100-119")
        self.assertFalse(seeds & set(range(10)),
                         "evaluation seeds 0-9 must never appear in training logs")


class TestTrainedModelOnDisk(unittest.TestCase):
    """The shipped `models/activity_hgb.joblib`, if `--train` has been run.

    This is the check that the manifest a judge would be shown actually says
    what the pitch claims: it beats rung 1, on held-out rows, by a recorded
    margin, with the training set it declares.
    """

    @classmethod
    def setUpClass(cls):
        from agent.policy_learned import DEFAULT_MODEL_PATH

        cls.path = DEFAULT_MODEL_PATH
        cls.man_path = ActivityModel.manifest_path(cls.path)
        if not cls.man_path.exists():
            raise unittest.SkipTest(
                f"no manifest at {cls.man_path}; run "
                "`python -m agent.policy_learned --train`"
            )
        cls.man = json.loads(cls.man_path.read_text(encoding="utf-8"))

    def test_manifest_records_the_required_provenance(self):
        for key in ("feature_names", "sklearn_version", "training_seeds",
                    "brier_model", "brier_rung1", "n_rows_total",
                    "n_rows_holdout", "gate_ok", "hgb_params"):
            self.assertIn(key, self.man, f"manifest must record {key!r}")
        self.assertEqual(self.man["feature_names"],
                         list(FEATURE_NAMES + TRAIN_EXTRA_NAMES))

    def test_the_shipped_model_actually_beats_rung1(self):
        b_m = float(self.man["brier_model"])
        b_r = float(self.man["brier_rung1"])
        self.assertLess(b_m, b_r - BRIER_GATE_MARGIN,
                        f"shipped model Brier {b_m:.5f} does not beat rung-1 "
                        f"{b_r:.5f}; the gate would force beta=0")
        self.assertTrue(self.man["gate_ok"])
        self.assertGreater(int(self.man["n_rows_holdout"]), 1000)

    def test_it_beats_the_base_rate_too_not_just_rung1(self):
        """Rung 1 on a 1.4%-positive label scores about what a constant
        base-rate forecast scores, so 'beats rung 1' could in principle mean
        'beats a straw man'.  It does not: the model also beats the best
        possible CONSTANT forecast, which is the honest floor."""
        r = float(self.man["positive_rate"])
        base = r * (1.0 - r)               # Brier of the constant forecast p = r
        self.assertLess(float(self.man["brier_model"]), base,
                        "the model must beat a constant base-rate forecast")

    def test_it_declares_the_held_out_scenario_and_no_evaluation_seeds(self):
        from agent.policy_learned import TRAIN_SEEDS

        self.assertEqual(self.man["held_out_scenario"], "agile")
        self.assertNotIn("agile", self.man["training_scenarios"])
        observed = self.man.get("observed_scenarios")
        if observed is not None:
            self.assertNotIn("agile", [s.lower() for s in observed])
        seeds = self.man.get("observed_seeds")
        if seeds:
            self.assertTrue(set(seeds).issubset(set(TRAIN_SEEDS)))
            self.assertFalse(set(seeds) & set(range(10)))

    def test_loading_it_defaults_to_beta_zero(self):
        """GUARANTEE 1 on the real artefact: opening the shipped model without
        asking for the learned path leaves the demo bit-identical to rung 1."""
        if not self.path.exists():
            self.skipTest("manifest present but joblib absent")
        model = ActivityModel.load(self.path)
        self.assertEqual(model.beta, 0.0)

    def test_beta_zero_on_the_real_model_reproduces_rung1_exactly(self):
        """The most important assertion in this file.

        Rung 1 currently LOSES to the fair-tuned sweep on POI@60 (DESIGN.md
        section 11.8).  Rung 2 must therefore be provably incapable of masking
        that: at beta = 0 the refined belief is not close to rung 1, it IS rung 1,
        array-equal, on the real fitted estimator rather than a stub.
        """
        if not self.path.exists():
            self.skipTest("manifest present but joblib absent")
        model = ActivityModel.load(self.path, beta=0.0)
        rng = np.random.default_rng(1)
        F = rng.uniform(0.0, 1.0, (256, N_FEATURES))
        F[:, FEATURE_NAMES.index("n_visits")] = rng.integers(0, 20, 256)
        out = model.refine(F, 0.010, 1.0e6)
        self.assertTrue(np.array_equal(out, F[:, FEATURE_NAMES.index("p_rung1")]))

    def test_the_real_model_at_beta_gt_zero_does_move_the_belief(self):
        """...and the gate is not vacuous: with beta > 0 it changes something."""
        if not self.path.exists():
            self.skipTest("manifest present but joblib absent")
        model = ActivityModel.load(self.path, beta=0.6)
        if not model.gate_ok:
            self.skipTest("shipped model failed its own gate; nothing to move")
        self.assertGreater(model.beta, 0.0)
        rng = np.random.default_rng(2)
        F = rng.uniform(0.0, 1.0, (256, N_FEATURES))
        F[:, FEATURE_NAMES.index("n_visits")] = 8.0     # past the cold-start gate
        out = model.refine(F, 0.010, 1.0e6)
        self.assertFalse(np.array_equal(out, F[:, FEATURE_NAMES.index("p_rung1")]))
        self.assertTrue(np.all((out >= P_CLIP_LO) & (out <= P_CLIP_HI)))


class TestSabotagedRealModelTripsTheGate(unittest.TestCase):
    """A saved model whose estimator predicts 0.5 everywhere must be refused at
    LOAD time, through the full save/manifest/load path rather than by calling
    `evaluate_gate` directly."""

    def test_sabotage_is_caught_by_the_manifest_gate_at_load(self):
        df = synth_log(deterministic=False, seed=21)
        X, y, meta = build_training_matrix(df)
        model = ActivityModel(estimator=_Sabotage(), beta=0.9)
        gate = model.evaluate_gate(X, y, meta["pd_bar_next"],
                                   p_rung1=meta["p_rung1"])
        self.assertFalse(gate["gate_ok"])
        model.manifest = {
            "brier_model": gate["brier_model"],
            "brier_rung1": gate["brier_rung1"],
            "gate_ok": gate["gate_ok"],
        }
        with tempfile.TemporaryDirectory() as d:
            path = model.save(Path(d) / "activity_hgb.joblib")
            reloaded = ActivityModel.load(path, beta=0.9)
            self.assertEqual(reloaded.beta, 0.0,
                             "a sabotaged model must load with beta forced to 0")
            self.assertFalse(reloaded.gate_ok)
            # And it stays off however it is attached.
            from agent.belief import Belief
            from sim.config import build_grid, build_mission, load_config

            cfg = load_config("sparse")
            bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
            bel.n_visits[:] = 9.0
            reloaded.attach_to(bel, 0.9)
            self.assertTrue(np.array_equal(bel.p_active(1.0), bel.p_effective(1.0)))


class TestEndToEndAntiRegression(unittest.TestCase):
    """The guarantee that matters, on a REAL episode rather than a stub belief.

    Every other anti-regression test here checks `refine()` in isolation.  This
    one runs the actual `index_learned` policy through the actual evaluator and
    asserts that at `beta = 0` it produces the same trajectory as rung 1 --
    same step count, same energy, same POI, same detections.

    That matters more than usual right now.  Rung 1 currently LOSES to the
    fair-tuned sweep on POI@60 (DESIGN.md section 11.8) and the honest headline
    is the energy result alone.  If rung 2 could perturb the demo path while
    nominally disabled, it could quietly paper over that -- so "disabled" has to
    mean identical, not merely similar.

    Horizon is 4 s, not 60: this is a proof of IDENTITY, and identity does not
    need a long episode.  A full-horizon run is ~7 s and would blow the suite
    budget (DESIGN.md section 10).
    """

    HORIZON_S = 4.0
    KEYS = ("n_steps", "n_scans", "n_sleeps", "energy_total_j", "poi_60",
            "n_unique_detections", "coverage_frac")

    @classmethod
    def setUpClass(cls):
        from agent.policy_learned import DEFAULT_MODEL_PATH

        if not DEFAULT_MODEL_PATH.exists():
            raise unittest.SkipTest(
                f"no trained model at {DEFAULT_MODEL_PATH}; run "
                "`python -m agent.policy_learned --train`"
            )
        import eval.runner as R

        cls.R = R
        cls.model_path = DEFAULT_MODEL_PATH
        cls.rung1 = R.run_episode("index", "sparse", 0, horizon_s=cls.HORIZON_S)
        cls.learned_0 = cls._run_learned(beta=0.0)

    @classmethod
    def _run_learned(cls, beta: float):
        """Run `index_learned` at an explicit beta, restoring the factory after.

        Only ever called with beta=0 from the suite -- at beta>0 the model runs
        on every decision and an episode takes tens of seconds.
        """
        model = ActivityModel.load(cls.model_path, beta=beta)
        orig = cls.R.POLICY_FACTORIES["index_learned"]
        cls.R.POLICY_FACTORIES["index_learned"] = (
            lambda cfg, collect_logs=False, **kw:
            cls.R._LearnedIndexPolicy(model, collect_logs=collect_logs)
        )
        try:
            row = cls.R.run_episode("index_learned", "sparse", 0,
                                    horizon_s=cls.HORIZON_S)
        finally:
            cls.R.POLICY_FACTORIES["index_learned"] = orig
        row["_beta"] = model.beta
        row["_gate_ok"] = model.gate_ok
        return row

    def test_beta_zero_reproduces_rung1_exactly(self):
        for k in self.KEYS:
            self.assertEqual(
                self.rung1[k], self.learned_0[k],
                f"beta=0 changed {k!r}: rung1={self.rung1[k]!r} "
                f"learned={self.learned_0[k]!r} -- the learned path must be a "
                "no-op when disabled",
            )

    def test_beta_zero_is_what_the_default_load_gives(self):
        self.assertEqual(self.learned_0["_beta"], 0.0)

    def test_the_learned_path_is_not_secretly_inert(self):
        """A no-op at beta=0 is only meaningful if beta>0 does something.

        Otherwise this whole class would pass on a model that never ran at all
        -- which is exactly the failure mode `p_active_hat` exists to prevent
        (an `ActivityModel` is a dataclass, not a callable, so without that hook
        `Belief._model_p_active` returns None and rung 2 silently does nothing).

        Checked through the belief rather than by running a second episode: at
        beta=0.6 the model is invoked on every decision and costs ~19 ms a call
        (a 3-fold isotonic-calibrated HGB ensemble has a ~13 ms floor per
        `predict_proba` regardless of row count), so a comparison episode takes
        ~35 s and would blow the whole suite budget on its own.  The full
        episode-level divergence was verified by hand; see the report.
        """
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        model = ActivityModel.load(self.model_path, beta=0.6)
        if not model.gate_ok:
            self.skipTest("shipped model failed its Brier gate; beta pinned to 0")
        self.assertGreater(model.beta, 0.0)

        cfg = load_config("sparse")
        bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
        bel.n_visits[:] = 5.0                     # past the cold-start gate
        model.attach_to(bel, 0.6)
        p1, pe = bel.p_active(1.0), bel.p_effective(1.0)
        self.assertFalse(np.array_equal(p1, pe),
                         "at beta=0.6 the learned belief must move p_effective")

    def test_cold_start_channels_are_untouched_in_a_real_belief(self):
        """GUARANTEE 2 through the real wiring: a channel with no visits keeps
        the analytic prior even at beta=0.6."""
        from agent.belief import Belief
        from sim.config import build_grid, build_mission, load_config

        model = ActivityModel.load(self.model_path, beta=0.6)
        if not model.gate_ok:
            self.skipTest("shipped model failed its Brier gate")
        cfg = load_config("sparse")
        bel = Belief(build_grid(cfg), build_mission(cfg), cfg)
        bel.n_visits[:] = 0.0                     # nothing has been visited yet
        model.attach_to(bel, 0.6)
        self.assertTrue(np.array_equal(bel.p_active(1.0), bel.p_effective(1.0)))


class TestLeakageGuardIsOnTheFittingPath(unittest.TestCase):
    """The guard must protect the CODE PATH, not just the CLI entry point."""

    def test_train_activity_model_rejects_an_extra_column(self):
        df = synth_log(deterministic=False, seed=31)
        X, y, meta = build_training_matrix(df)
        # Smuggle the channel index in as one more column.
        X_bad = np.column_stack([X, np.asarray(meta["channel"], dtype=np.float64)])
        with self.assertRaises(ValueError) as ctx:
            train_activity_model(X_bad, y, meta["pd_bar_next"],
                                 model_kwargs=FAST_MODEL)
        self.assertIn("columns", str(ctx.exception))

    def test_predict_rejects_an_extra_column(self):
        model = ActivityModel(estimator=_Constant(0.4))
        X = np.zeros((4, len(FEATURE_NAMES) + len(TRAIN_EXTRA_NAMES) + 1))
        with self.assertRaises(ValueError):
            model.predict_p_det(X)


if __name__ == "__main__":
    unittest.main()
