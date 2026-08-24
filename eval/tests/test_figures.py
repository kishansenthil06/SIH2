"""Figure tests.  The pitch must never depend on a package being present.

The load-bearing test here is `TestMatplotlibForcedImportError`: it hides
`matplotlib` from the import machinery and asserts `eval/figures.py` still
produces its CSV + ASCII artefacts (DESIGN.md section 10).  Everything else
checks that the four charts read a realistic `results/*.csv` without crashing and
that the numbers on them are the numbers in the file.

Synthetic results are generated here rather than read from `results/`, so these
tests do not depend on agent C's runner having been executed.
"""
from __future__ import annotations

import builtins
import csv
import sys
import tempfile
import unittest
from pathlib import Path

from eval import figures as F

POLICIES = ("round_robin", "random", "greedy", "index", "index_learned", "oracle")
SCENARIOS = ("sparse", "dense")
SEEDS = tuple(range(4))

# Hand-chosen so the headline claim is visibly true in the fixture: `index`
# reaches round_robin's POI at roughly half its energy.
_PROFILE = {
    "round_robin":   (61.1, 0.88),
    "random":        (58.0, 0.55),
    "greedy":        (40.0, 0.62),
    "index":         (30.5, 0.89),
    "index_learned": (28.9, 0.92),
    "oracle":        (12.0, 1.00),
}


def write_runs_csv(path: Path) -> list[dict]:
    """A `results/runs.csv` in the shape `eval/metrics.py:METRIC_KEYS` implies."""
    rows = []
    for scen in SCENARIOS:
        for pol in POLICIES:
            e0, p0 = _PROFILE[pol]
            for seed in SEEDS:
                # Deterministic per-seed spread; no RNG, so the fixture is stable.
                jitter = 0.04 * ((seed % 4) - 1.5)
                energy = e0 * (1.0 + jitter)
                poi = min(1.0, max(0.0, p0 * (1.0 + 0.5 * jitter)))
                n_det = max(1.0, 40.0 * poi)
                rows.append({
                    "run_id": f"{scen}_{pol}_s{seed}",
                    "scenario": scen, "policy": pol, "seed": seed,
                    "config_hash": "deadbeef",
                    "energy_total_j": energy,
                    "energy_per_detection_j": energy / n_det,
                    "poi_10": poi * 0.6, "poi_30": poi * 0.85, "poi_60": poi,
                    "ttfi_p1_median_s": 20.0 * (1.05 - poi),
                    "ttfi_p1_p90_s": 45.0 * (1.05 - poi),
                    "ttfi_p1_n_intercepted": round(3 * poi),
                    "ttfi_p1_n_total": 3,
                    "ttfi_p1_frac": poi,
                    "max_staleness_p1_s": 0.5 if pol.startswith("index") else 9.0,
                    "n_unique_detections": n_det,
                })
    _dump(path, rows)
    return rows


def write_ablation_csv(path: Path) -> None:
    rows = [
        {"variant": "index", "seed": s, "energy_per_detection_j": 0.85 + 0.01 * s,
         "poi_60": 0.89, "energy_total_j": 30.5}
        for s in SEEDS
    ] + [
        {"variant": "score_raw", "seed": s, "energy_per_detection_j": 2.10 + 0.01 * s,
         "poi_60": 0.50, "energy_total_j": 79.7}
        for s in SEEDS
    ] + [
        {"variant": "no_deadline", "seed": s, "energy_per_detection_j": 1.20,
         "poi_60": 0.71, "energy_total_j": 34.0}
        for s in SEEDS
    ]
    _dump(path, rows)


def write_steps_csv(path: Path, n: int = 60, period: float = 1.0) -> None:
    """A trace in `eval.metrics.TRACE_COLUMNS` order."""
    rows = []
    t = 0.0
    for i in range(n):
        t0 = i * period
        t1 = t0 + 0.010
        rows.append({
            "step": i, "t_start": t0, "t_end": t1,
            "kind": "scan" if i % 5 else "sleep",
            "f_center_hz": 2.4e9 + (i % 20) * 5e6, "bw_hz": 5e6,
            "dwell_s": 0.010, "energy_j": 0.012,
            "n_det": i % 3, "det_channels": " ".join(str(4 * (i % 20) + k)
                                                     for k in range(i % 3)),
            "best_score": -0.01 * i, "chosen_reason": "index" if i % 7 else "deadline:ch=12",
            "energy_spent_total": 0.012 * (i + 1),
        })
        t = t1
    assert t > 0
    _dump(path, rows)


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class _Fixture(unittest.TestCase):
    """A full synthetic `results/` tree in a temp dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.res = Path(self._tmp.name) / "results"
        self.out = self.res / "figures"
        self.runs = write_runs_csv(self.res / "runs.csv")
        write_ablation_csv(self.res / "ablation.csv")
        for pol in ("index", "round_robin"):
            write_steps_csv(self.res / "steps" / f"sparse_{pol}_s0.csv",
                            period=0.2 if pol == "index" else 1.0)
        self.addCleanup(self._tmp.cleanup)


# ---------------------------------------------------------------------------
# 6. Figures degrade -- the required test
# ---------------------------------------------------------------------------
class TestMatplotlibForcedImportError(_Fixture):
    """With matplotlib unimportable, every chart still lands as CSV + ASCII.

    DESIGN.md section 10: the pitch never depends on a package being present.
    This blocks the import at the machinery level rather than passing a flag, so
    it catches a stray top-level `import matplotlib` anywhere in the module.
    """

    def setUp(self):
        super().setUp()
        self._real_import = builtins.__import__
        self._saved = {k: v for k, v in sys.modules.items()
                       if k == "matplotlib" or k.startswith("matplotlib.")}
        for k in self._saved:
            del sys.modules[k]

        def blocked(name, *a, **kw):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib is not installed (forced by test)")
            return self._real_import(name, *a, **kw)

        builtins.__import__ = blocked
        self.addCleanup(self._restore)

    def _restore(self):
        builtins.__import__ = self._real_import
        sys.modules.update(self._saved)

    def test_import_really_is_blocked(self):
        with self.assertRaises(ImportError):
            import matplotlib  # noqa: F401

    def test_make_all_produces_the_fallback(self):
        rep = F.make_all(self.res, self.out, use_mpl=True)
        text = str(rep)
        self.assertIn("ASCII/CSV fallback", text,
                      "the report must say the fallback path was taken")
        self.assertEqual([], [p for p in rep.written if p.suffix == ".png"],
                         "no PNG can be produced without matplotlib")
        for name in ("energy_vs_poi.csv", "ttfi.csv", "staleness.csv",
                     "ablation.csv", "figures_report.txt"):
            self.assertTrue((self.out / name).exists(),
                            f"fallback must still write {name}")

    def test_fallback_text_carries_the_headline_numbers(self):
        rep = F.make_all(self.res, self.out, use_mpl=True)
        text = str(rep)
        self.assertIn("ENERGY PER DETECTION vs POI", text)
        self.assertIn("round_robin", text)
        self.assertIn("index", text)
        self.assertIn("TTFI", text)
        self.assertIn("ABLATION", text)
        # The claim itself, spelled out rather than left to the reader --
        # and written so it renders honestly whichever way the numbers go.
        self.assertIn("energy/detection:", text)
        self.assertIn("POI@60:", text)

    def test_loader_returns_not_ok(self):
        plt, ok = F._load_mpl(True)
        self.assertIsNone(plt)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# The normal path
# ---------------------------------------------------------------------------
class TestFiguresWithMatplotlib(_Fixture):

    def setUp(self):
        super().setUp()
        try:
            import matplotlib  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("matplotlib not installed")

    def test_all_four_pngs_are_written(self):
        rep = F.make_all(self.res, self.out, use_mpl=True)
        for name in ("energy_vs_poi.png", "ttfi.png", "staleness.png", "ablation.png"):
            p = self.out / name
            self.assertTrue(p.exists(), f"missing {name}")
            self.assertGreater(p.stat().st_size, 1000, f"{name} looks empty")

    def test_agg_backend_is_selected(self):
        # `only` keeps this to a single render: the assertion is about the
        # backend, not about the charts.
        F.make_all(self.res, self.out, use_mpl=True, only="energy")
        import matplotlib

        self.assertEqual(matplotlib.get_backend().lower(), "agg",
                         "figures must render headless")

    def test_only_flag_renders_one_chart(self):
        rep = F.make_all(self.res, self.out, use_mpl=True, only="energy")
        self.assertTrue((self.out / "energy_vs_poi.png").exists())
        self.assertFalse((self.out / "ablation.png").exists())

    def test_output_is_deterministic(self):
        """Fixed seeds: a figure regenerated for the deck must not shift."""
        F.make_all(self.res, self.out, use_mpl=True, only="energy")
        first = (self.out / "energy_vs_poi.csv").read_bytes()
        F.make_all(self.res, self.out, use_mpl=True, only="energy")
        self.assertEqual(first, (self.out / "energy_vs_poi.csv").read_bytes())
        self.assertIsInstance(F.FIG_SEED, int)


class TestMissingInputsAreReportedNotFatal(unittest.TestCase):

    def test_empty_results_dir(self):
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "results"
            res.mkdir()
            rep = F.make_all(res, res / "figures", use_mpl=False)
        text = str(rep)
        self.assertIn("SKIPPED", text)
        self.assertIn("eval.runner", text, "must name the command that makes the data")
        # A report is still written, so the pipeline downstream has an artefact.
        self.assertTrue(any(p.name == "figures_report.txt" for p in rep.written))

    def test_cli_returns_zero_with_no_data(self):
        with tempfile.TemporaryDirectory() as d:
            rc = F.main(["--results-dir", str(Path(d) / "nope"),
                         "--out", str(Path(d) / "figs"), "--no-mpl"])
        self.assertEqual(rc, 0, "missing data is a message, not a failure")


# ---------------------------------------------------------------------------
# Aggregation correctness -- a chart that plots the wrong number is worse than
# no chart, so the arithmetic is checked directly.
# ---------------------------------------------------------------------------
class TestAggregation(_Fixture):

    def test_energy_table_matches_the_csv(self):
        rep = F.Report()
        F.fig_energy_vs_poi(self.runs, self.out, None, rep)
        table = F.read_csv(self.out / "energy_vs_poi.csv")
        row = next(r for r in table
                   if r["scenario"] == "sparse" and r["policy"] == "index")
        mine = [r for r in self.runs
                if r["scenario"] == "sparse" and r["policy"] == "index"]
        # MEDIAN, not mean: the oracle has one 2.999 J/det seed against ~0.09
        # elsewhere, and a mean would let it define the reference ceiling.
        self.assertAlmostEqual(
            row["epd_median_j"],
            F._median([r["energy_per_detection_j"] for r in mine]), places=9)
        self.assertAlmostEqual(
            row["energy_median_j"],
            F._median([r["energy_total_j"] for r in mine]), places=9)
        self.assertEqual(row["n_seeds"], len(SEEDS))

    def test_headline_axis_is_energy_per_detection_not_total(self):
        """The budget is binding, so every policy spends ~6 J on the real data.

        A total-energy x-axis collapses the headline chart onto a single
        vertical line and hides the entire claim.  The chart must key on
        `energy_per_detection_j`.
        """
        rep = F.Report()
        F.fig_energy_vs_poi(self.runs, self.out, None, rep)
        cols = F.read_csv(self.out / "energy_vs_poi.csv")[0].keys()
        self.assertIn("epd_median_j", cols)
        self.assertIn("epd_p10_j", cols)
        self.assertIn("epd_p90_j", cols)
        self.assertIn("ENERGY PER DETECTION", str(rep))

    def test_outlier_seed_does_not_move_the_centre(self):
        """One catastrophic seed must not redefine a policy's headline number."""
        rows = [
            {"scenario": "sparse", "policy": "oracle", "seed": s,
             "energy_total_j": 6.0, "poi_60": 0.85,
             "energy_per_detection_j": v, "n_unique_detections": 60.0}
            # The real shape from DESIGN.md section 11: nine ~0.09 seeds and one
            # 2.999 outlier.  The mean is 0.38, over 4x the typical episode.
            for s, v in enumerate([0.09] * 9 + [2.999])
        ]
        rep = F.Report()
        F.fig_energy_vs_poi(rows, self.out, None, rep)
        row = F.read_csv(self.out / "energy_vs_poi.csv")[0]
        self.assertAlmostEqual(row["epd_median_j"], 0.09, places=9)
        self.assertGreater(F._mean([r["energy_per_detection_j"] for r in rows]), 0.3)
        # ...but the outlier is still SHOWN, via the upper spread bound.
        self.assertGreater(row["epd_p90_j"], 0.09)

    def test_a_policy_with_no_detections_is_kept_not_dropped(self):
        """`random` currently scores zero unique detections, so its
        energy-per-detection is NaN.  Silently dropping it would flatter the
        field; it must survive into the table and be labelled."""
        rows = [
            {"scenario": "sparse", "policy": "random", "seed": s,
             "energy_total_j": 5.93, "poi_60": 0.0,
             "energy_per_detection_j": float("nan"), "n_unique_detections": 0.0}
            for s in range(4)
        ] + [
            {"scenario": "sparse", "policy": "index", "seed": s,
             "energy_total_j": 6.0, "poi_60": 0.33,
             "energy_per_detection_j": 0.617, "n_unique_detections": 11.0}
            for s in range(4)
        ]
        rep = F.Report()
        F.fig_energy_vs_poi(rows, self.out, None, rep)
        table = F.read_csv(self.out / "energy_vs_poi.csv")
        pols = {r["policy"] for r in table}
        self.assertIn("random", pols, "a NaN metric must not delete the policy")
        rnd = next(r for r in table if r["policy"] == "random")
        self.assertNotEqual(rnd["epd_median_j"], rnd["epd_median_j"])  # NaN
        self.assertEqual(rnd["epd_n_finite"], 0)
        self.assertIn("random", str(rep))

    def test_headline_reports_a_poi_loss_honestly(self):
        """The real numbers have `index` BELOW `round_robin` on POI@60
        (DESIGN.md section 11.8).  The summary must say so rather than assert a
        win it did not measure."""
        rows = [
            {"scenario": "sparse", "policy": "round_robin", "seed": 0,
             "energy_total_j": 6.0, "poi_60": 0.417,
             "energy_per_detection_j": 1.650, "n_unique_detections": 3.6},
            {"scenario": "sparse", "policy": "index", "seed": 0,
             "energy_total_j": 6.0, "poi_60": 0.333,
             "energy_per_detection_j": 0.617, "n_unique_detections": 11.0},
        ]
        rep = F.Report()
        F.fig_energy_vs_poi(rows, self.out, None, rep)
        text = str(rep)
        self.assertIn("62.6% lower", text, "the energy win, stated as measured")
        self.assertIn("BELOW baseline", text, "the POI loss, stated not hidden")
        self.assertIn("REFERENCE CEILING", text)
        # And the figure title must not claim an outright win.
        table = F.read_csv(self.out / "energy_vs_poi.csv")
        title = F._headline_title(table)
        self.assertIn("below the sweep", title)
        self.assertNotIn("Same interception performance", title)

    def test_headline_says_parity_when_parity_is_real(self):
        """...and the same code still reads correctly when we DO win both."""
        rows = [
            {"scenario": "sparse", "policy": "round_robin", "seed": 0,
             "energy_total_j": 6.0, "poi_60": 0.40,
             "energy_per_detection_j": 1.65, "n_unique_detections": 3.6},
            {"scenario": "sparse", "policy": "index", "seed": 0,
             "energy_total_j": 6.0, "poi_60": 0.45,
             "energy_per_detection_j": 0.62, "n_unique_detections": 11.0},
        ]
        rep = F.Report()
        F.fig_energy_vs_poi(rows, self.out, None, rep)
        self.assertIn("parity or better", str(rep))
        title = F._headline_title(F.read_csv(self.out / "energy_vs_poi.csv"))
        self.assertIn("Same interception performance", title)

    def test_ablation_reads_the_runners_aggregated_schema(self):
        """`eval/runner.py --ablate` writes ONE PRE-AGGREGATED row per
        (scenario, policy) with `_mean`/`_std`/`_count` suffixes -- not one row
        per seed.  Read as if it were runs.csv, every bar came out NaN and the
        chart was silently empty."""
        rows = [
            {"scenario": "sparse", "policy": "index",
             "energy_per_detection_j_mean": 0.6173, "poi_60_mean": 0.275,
             "energy_total_j_mean": 5.999, "energy_per_detection_j_count": 5},
            {"scenario": "sparse", "policy": "round_robin",
             "energy_per_detection_j_mean": 1.7398, "poi_60_mean": 0.400,
             "energy_total_j_mean": 5.999, "energy_per_detection_j_count": 5},
            {"scenario": "dense", "policy": "index",
             "energy_per_detection_j_mean": 0.1915, "poi_60_mean": 0.400,
             "energy_total_j_mean": 5.991, "energy_per_detection_j_count": 5},
        ]
        rep = F.Report()
        F.fig_ablation(rows, self.out, None, rep)
        table = F.read_csv(self.out / "ablation.csv")
        got = {(r["scenario"], r["variant"]): r for r in table}
        self.assertEqual(len(got), 3, "scenario must be part of the group key")
        self.assertAlmostEqual(
            got[("sparse", "index")]["energy_per_detection_j"], 0.6173, places=6)
        self.assertAlmostEqual(got[("sparse", "index")]["poi_60"], 0.275, places=6)
        self.assertEqual(got[("sparse", "index")]["n_seeds"], 5)
        # A dense and a sparse `index` must not be averaged into one bar.
        self.assertAlmostEqual(
            got[("dense", "index")]["energy_per_detection_j"], 0.1915, places=6)

    def test_ablation_still_reads_per_seed_rows(self):
        """The same function must keep working on an un-aggregated table."""
        rep = F.Report()
        F.fig_ablation(F.read_csv(self.res / "ablation.csv"), self.out, None, rep)
        table = F.read_csv(self.out / "ablation.csv")
        names = {r["variant"] for r in table}
        self.assertEqual(names, {"index", "score_raw", "no_deadline"})
        idx = next(r for r in table if r["variant"] == "index")
        self.assertAlmostEqual(idx["energy_per_detection_j"], 0.865, places=6)

    def test_policy_order_is_stable(self):
        got = list(F.group_runs(self.runs, "sparse").keys())
        self.assertEqual(got, [p for p in F.POLICY_ORDER if p in got])

    def test_agile_sorts_last(self):
        rows = self.runs + [{"scenario": "agile", "policy": "index", "poi_60": 0.8}]
        self.assertEqual(F.scenarios_in(rows)[-1], "agile",
                         "the held-out scenario reads as the punchline")

    def test_ttfi_reports_intercepted_fraction(self):
        rep = F.Report()
        F.fig_ttfi(self.runs, self.out, None, rep)
        table = F.read_csv(self.out / "ttfi.csv")
        self.assertTrue(all("n_intercepted" in r and "n_total" in r for r in table),
                        "TTFI is censored; the fraction must always travel with it")

    def test_staleness_trace_is_a_bounded_sawtooth(self):
        steps = F.read_csv(self.res / "steps" / "sparse_index_s0.csv")
        ts, st = F.staleness_trace(steps, horizon_s=12.0)
        self.assertEqual(len(ts), len(st))
        self.assertTrue(all(v >= 0.0 for v in st))
        # Scans land every 0.2 s in the fixture (sleeps excluded), so the gap
        # cannot exceed roughly one sleep-and-scan cycle.
        self.assertLess(max(st), 2.0)

    def test_staleness_of_an_empty_trace(self):
        ts, st = F.staleness_trace([], horizon_s=60.0)
        self.assertEqual((ts, st), ([], []))

    def test_sparkline_and_bar_degrade_on_empty_input(self):
        self.assertEqual(F.sparkline([]), "")
        self.assertEqual(F.sparkline([float("nan")]), "")
        self.assertEqual(len(F.sparkline([1, 2, 3, 4])), 4)
        # Compared against the active ramp, not a literal: the ramp degrades to
        # ASCII on a console that cannot encode U+2581 (see below).
        self.assertEqual(F.sparkline([5, 5, 5]), F._SPARK[0] * 3)
        self.assertEqual(F.bar(float("nan"), 1.0), "")
        self.assertEqual(F.bar(1.0, 0.0), "")

    def test_the_fallback_never_dies_on_the_console_encoding(self):
        """The block-drawing ramp is not encodable in cp1252, the DEFAULT
        Windows console encoding -- so printing the report used to raise
        `UnicodeEncodeError` and take the command down.

        This is the FALLBACK path, whose whole justification is that it works
        when nothing else does (DESIGN.md section 10); a fallback that crashes on
        the most common developer console is worse than no fallback.  The ramp
        degrades to ASCII instead.
        """
        self.assertEqual(len(F._SPARK), 8)
        self.assertEqual(len(F._SPARK_ASCII), 8)
        for ch in F._SPARK_ASCII + F._BAR_ASCII:
            ch.encode("cp1252")  # must not raise
        # And the report printer survives a stream that cannot take the glyphs.
        F._print_safe("blocks: " + F._SPARK_UNICODE + F._BAR_UNICODE)

    def test_missing_columns_do_not_crash(self):
        """Agent C owns runs.csv; a renamed column must degrade, not explode."""
        stripped = [{"scenario": r["scenario"], "policy": r["policy"]}
                    for r in self.runs]
        rep = F.Report()
        F.fig_energy_vs_poi(stripped, self.out, None, rep)
        F.fig_ttfi(stripped, self.out, None, rep)
        self.assertIn("ENERGY PER DETECTION vs POI", str(rep))


# ---------------------------------------------------------------------------
# The dashboard reads the same CSVs, so it is tested alongside the figures.
# ---------------------------------------------------------------------------
class TestDashboard(_Fixture):

    def setUp(self):
        super().setUp()
        from app import dashboard as D

        self.D = D
        self.tr = D.load_trace("sparse_index_s0", self.res / "steps")

    def test_trace_loads_the_metrics_schema(self):
        self.assertGreater(self.tr.n, 0)
        self.assertEqual(self.tr.policy, "index")
        self.assertEqual(self.tr.scenario, "sparse")
        # Sleeps paint nothing, so their span is the -1 sentinel.
        self.assertTrue((self.tr.k_lo[self.tr.kind == "sleep"] < 0).all())
        self.assertTrue((self.tr.k_lo[self.tr.kind == "scan"] >= 0).all())
        self.assertTrue((self.tr.k_hi[self.tr.kind == "scan"]
                         <= self.tr.grid.n_channels).all())

    def test_reasons_are_carried_through(self):
        """The `chosen_reason` string is the whole explainability story."""
        self.assertEqual(len(self.tr.reason), self.tr.n)
        self.assertTrue(any("deadline" in r for r in self.tr.reason),
                        "the fixture must exercise a non-index reason")

    # -- the 2000-channel downsample ---------------------------------------
    def test_downsample_preserves_a_single_channel_detection(self):
        """THE rendering bug on the wide grid.

        The grid is 2000 channels tall and the axes are a few hundred pixels, so
        the raster is resampled no matter what.  `interpolation="nearest"` keeps
        one source row in five -- and the index policy's favourite action is
        1 MHz x 1 ms, i.e. a ONE-channel detection, so most detections simply
        vanished.  The block-max reduction keeps them at full strength.
        """
        import numpy as np

        n_ch, bins = 2000, 40
        codes = np.full((n_ch, bins), self.D.CELL_BG, dtype=np.uint8)
        codes[1000:1005, 10] = self.D.CELL_SCAN      # a 5-channel dwell
        codes[1002, 10] = self.D.CELL_DET            # ...one of which detected

        small = self.D._reduce_rows(codes, 400)
        self.assertEqual(small.shape, (400, bins))
        col = small[:, 10]
        self.assertIn(self.D.CELL_DET, col.tolist(),
                      "the single detected channel must survive the downsample")
        self.assertEqual(int(col.max()), self.D.CELL_DET)
        # ...and it must not smear: exactly the blocks it touched are marked.
        self.assertLessEqual(int(np.count_nonzero(col)), 3)

    def test_downsample_is_a_priority_not_an_average(self):
        """A detection outranks a dwell outranks background, always."""
        import numpy as np

        codes = np.array([[self.D.CELL_BG], [self.D.CELL_SCAN],
                          [self.D.CELL_DET], [self.D.CELL_BG]], dtype=np.uint8)
        self.assertEqual(int(self.D._reduce_rows(codes, 1)[0, 0]),
                         self.D.CELL_DET)
        codes[2, 0] = self.D.CELL_BG
        self.assertEqual(int(self.D._reduce_rows(codes, 1)[0, 0]),
                         self.D.CELL_SCAN)

    def test_downsample_is_a_no_op_when_it_already_fits(self):
        import numpy as np

        codes = np.zeros((50, 4), dtype=np.uint8)
        self.assertIs(self.D._reduce_rows(codes, 400), codes)

    def test_downsample_pads_a_ragged_tail_rather_than_wrapping(self):
        """2000/400 is exact, but the guard must hold for any grid size."""
        import numpy as np

        codes = np.zeros((1001, 3), dtype=np.uint8)
        codes[1000, 0] = self.D.CELL_DET          # the very last channel
        small = self.D._reduce_rows(codes, 400)
        self.assertEqual(int(small[:, 0].max()), self.D.CELL_DET)
        self.assertEqual(int(small[:-1, 0].max()), self.D.CELL_BG,
                         "the tail must not wrap onto the first block")

    def test_painting_never_erases_an_earlier_detection(self):
        """Two actions can land in the same time column; a later dwell must not
        overwrite a detection already painted there."""
        import numpy as np

        codes, col0, col1 = self.D.build_waterfall(self.tr, time_bins=8)
        self.D.paint_upto(self.tr, codes, col0, col1, self.tr.n)
        n_det = sum(int(d.size) for d in self.tr.detections)
        if n_det:
            self.assertIn(self.D.CELL_DET, np.unique(codes).tolist())

    def test_rgb_conversion_shape_and_palette(self):
        import numpy as np

        codes, _, _ = self.D.build_waterfall(self.tr, time_bins=12)
        rgb = self.D.codes_to_rgb(codes, 400)
        self.assertEqual(rgb.shape[1:], (12, 3))
        self.assertLessEqual(rgb.shape[0], 400)
        self.assertTrue(np.allclose(rgb[0, 0], self.D.C_BG))

    def test_misaligned_frequency_still_draws(self):
        """`ChannelGrid.channels_for` raises on misalignment -- a viewer must not."""
        import numpy as np

        lo, hi = self.D._spans(np.array([2.0e9 + 5.5e6 + 1.0]), np.array([5e6]),
                               self.tr.grid)
        self.assertTrue(0 <= lo[0] < hi[0] <= self.tr.grid.n_channels)

    def test_counters_are_monotone_and_honest(self):
        early = self.D.counter_text(self.tr, 5)
        late = self.D.counter_text(self.tr, self.tr.n)
        self.assertIn("energy", early)
        self.assertLessEqual(self.tr.unique_detected_channels(5),
                             self.tr.unique_detected_channels(self.tr.n))
        # POI is an evaluator quantity; the live counter must not claim to be it.
        self.assertNotIn("POI@60", late.split("POI@60 (final)")[0])

    def test_summary_is_read_from_runs_csv(self):
        s = self.D.load_summary("sparse_index_s0", self.res)
        self.assertIn("poi_60", s)
        self.assertEqual(self.D.load_summary("nope", self.res), {})

    def test_ascii_waterfall_needs_no_plotting(self):
        text = self.D.ascii_waterfall(self.tr)
        self.assertIn("waterfall", text)
        self.assertIn("why it moved", text)
        self.assertGreater(len(text.splitlines()), 20)

    def test_missing_trace_names_the_command(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self.D.load_trace("no_such_run", self.res / "steps")
        self.assertIn("eval.runner", str(ctx.exception))

    def test_mp4_without_ffmpeg_falls_back_to_a_gif_EXTENSION_too(self):
        """`--format mp4` with no ffmpeg fell back to PillowWriter but kept the
        `.mp4` path, and Pillow then raised `unknown file extension: .mp4`.

        The fallback failed louder than having no writer at all, so the file
        extension must follow the writer that was actually selected.
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            self.skipTest("matplotlib not installed")

        writer, actual = self.D._writer(animation, "mp4", 10)
        self.assertIsNotNone(writer)
        self.assertIn(actual, ("mp4", "gif"))
        if not animation.FFMpegWriter.isAvailable():
            self.assertEqual(actual, "gif",
                             "no ffmpeg means the output must be named .gif")
        gif_writer, gif_fmt = self.D._writer(animation, "gif", 10)
        self.assertEqual(gif_fmt, "gif")
        self.assertIsNotNone(gif_writer)

    def test_mp4_request_actually_writes_a_readable_file(self):
        """End to end through `render`, which is where the extension is applied."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        out = self.D.render([self.tr], self.out / "anim.mp4", fmt="mp4",
                            frames=2, fps=4, dpi=40)
        self.assertIsNotNone(out)
        self.assertTrue(Path(out).exists(), f"{out} was not written")
        self.assertGreater(Path(out).stat().st_size, 0)
        self.assertIn(Path(out).suffix, (".mp4", ".gif", ".png"))

    def test_cli_returns_two_when_the_run_is_absent(self):
        rc = self.D.main(["--run", "no_such_run", "--steps-dir", str(self.res / "steps"),
                          "--results-dir", str(self.res)])
        self.assertEqual(rc, 2)

    def test_cli_list(self):
        rc = self.D.main(["--list", "--steps-dir", str(self.res / "steps")])
        self.assertEqual(rc, 0)

    def test_png_export_of_a_comparison(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("matplotlib not installed")
        out = self.out / "compare.png"
        got = self.D.main(["--compare", "sparse_index_s0", "sparse_round_robin_s0",
                           "--format", "png", "--out", str(out),
                           "--steps-dir", str(self.res / "steps"),
                           "--results-dir", str(self.res)])
        self.assertEqual(got, 0)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 5000)

    def test_firewall_imports(self):
        """`app/**` may import ONLY `sim.contract` and `sim.config`.

        Another agent's AST scan enforces this project-wide; duplicating it here
        means a breach fails in the file that caused it.
        """
        import ast

        src = Path(self.D.__file__).read_text(encoding="utf-8")
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        sim_mods = {m for m in mods if m == "sim" or m.startswith("sim.")}
        self.assertLessEqual(sim_mods, {"sim.contract", "sim.config"},
                             f"firewall breach: {sim_mods}")
        for banned in ("streamlit", "plotly"):
            self.assertNotIn(banned, mods)

    def test_truth_is_never_referenced(self):
        import ast

        src = Path(self.D.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = {"truth", "truth_bursts", "truth_power", "_world", "emitters"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, banned,
                                 f"firewall: touched .{node.attr}")


if __name__ == "__main__":
    unittest.main()
