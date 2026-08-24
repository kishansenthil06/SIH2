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

        They use different quadratures and disagree by ~3e-3, which is enough to
        bias the inversion, so an attached model borrows the belief's numbers.
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


if __name__ == "__main__":
    unittest.main()
