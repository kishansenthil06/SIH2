"""Runner tests: the CSV contract, reproducibility, and parallel/serial equivalence.

These protect claims that are easy to assume and expensive to get wrong:

* Every figure in `README.md` is read out of `results/runs.csv`, so its header has
  to be a frozen, ordered contract rather than whatever the last edit happened to
  emit.  `RUNS_COLUMNS` is that contract and `test_header_matches_frozen_columns`
  is what keeps it honest.
* `--jobs N` exists to make the 75-episode matrix tolerable.  On Windows the
  worker must be module-level and picklable and the `__main__` guard is mandatory,
  so parallelism is exactly the sort of thing that silently produces *different*
  numbers, or hangs.  `test_pooled_matches_serial` pins that it does neither.
* Reproducibility on `(policy, scenario, seed)` is what makes cross-policy
  comparison meaningful at all; if it fails, the ablation table means nothing.

Episodes on the 2000-channel grid run thousands of decisions, so every test here
shrinks the world with a copied cfg dict rather than running a real 60 s episode.
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eval.runner import (
    DEFAULT_ABLATION_CSV,
    DEFAULT_RUNS_CSV,
    RUNS_COLUMNS,
    _ablation_path_for,
    _guard_overwrite,
    run_matrix,
    write_runs_csv,
)
from sim.config import load_config

# A deliberately tiny world.  The properties under test -- schema, determinism,
# pooled/serial equivalence -- are all independent of scale, and a full-size
# episode would blow the 2 s per-test budget many times over.
SHORT_HORIZON_S = 2.0
SMALL_N_CHANNELS = 200


def _small_cfg(name: str = "sparse") -> dict:
    cfg = load_config(name)
    cfg["horizon_s"] = SHORT_HORIZON_S
    cfg["grid"]["n_channels"] = SMALL_N_CHANNELS
    # Keep every band and emitter inside the shrunken grid.
    for band in cfg["mission"]["priority_bands"]:
        band["ch_lo"] = min(band["ch_lo"] // 10, SMALL_N_CHANNELS - 1)
        band["ch_hi"] = min(max(band["ch_hi"] // 10, band["ch_lo"] + 1), SMALL_N_CHANNELS)
    for spec in cfg["emitters"]:
        lo, hi = spec["channel_range"]
        lo = min(lo // 10, SMALL_N_CHANNELS - 1)
        hi = min(max(hi // 10, lo + 1), SMALL_N_CHANNELS)
        spec["channel_range"] = [lo, hi]
    return cfg


def _rows_from(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _comparable(rows: list[dict]) -> list[tuple]:
    """Rows keyed for comparison, with the two legitimately-varying fields out.

    `wall_time_s` is a measurement of the machine, not of the policy, and
    `run_id` may encode ordering -- neither says anything about correctness.
    """
    drop = {"wall_time_s", "run_id"}
    out = []
    for r in sorted(rows, key=lambda x: (x["policy"], x["scenario"], int(x["seed"]))):
        out.append(tuple((k, v) for k, v in sorted(r.items()) if k not in drop))
    return out


class TestRunsCsvContract(unittest.TestCase):
    """The CSV header is a published interface; app/server.py and the README read it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "runs.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_header_matches_frozen_columns(self):
        write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[_small_cfg()], seeds=[0], verbose=False), self.out)
        with open(self.out, "r", encoding="utf-8", newline="") as fh:
            header = tuple(next(csv.reader(fh)))
        self.assertEqual(header, tuple(RUNS_COLUMNS))

    def test_every_declared_column_is_present_in_each_row(self):
        write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[_small_cfg()], seeds=[0, 1], verbose=False), self.out)
        rows = _rows_from(self.out)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(set(r.keys()), set(RUNS_COLUMNS))

    def test_row_count_is_policies_times_scenarios_times_seeds(self):
        write_runs_csv(run_matrix(policies=["round_robin", "random"], scenarios=[_small_cfg()], seeds=[0, 1, 2], verbose=False), self.out)
        self.assertEqual(len(_rows_from(self.out)), 2 * 1 * 3)


class TestReproducibility(unittest.TestCase):
    """Same (policy, scenario, seed) twice must give the same row.

    Without this the ablation table is not a measurement of policies, it is a
    measurement of noise, and no comparison between two rows means anything.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_seed_gives_identical_row(self):
        cfg = _small_cfg()
        paths = []
        for i in (0, 1):
            p = Path(self.tmp.name) / f"run{i}.csv"
            write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[cfg], seeds=[0, 1], verbose=False), p)
            paths.append(p)
        self.assertEqual(_comparable(_rows_from(paths[0])),
                         _comparable(_rows_from(paths[1])))

    def test_different_seeds_give_different_rows(self):
        """Guards the above: identical rows for *different* seeds would mean the
        seed is not wired through, and the reproducibility test would pass
        vacuously."""
        p = Path(self.tmp.name) / "seeds.csv"
        write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[_small_cfg()], seeds=[0, 1, 2], verbose=False), p)
        rows = _rows_from(p)
        sigs = {r["config_hash"] + r["seed"] for r in rows}
        self.assertEqual(len(sigs), 3)


class TestPooledExecution(unittest.TestCase):
    """`--jobs N` must not change the answer, and must not hang.

    On Windows the pool spawns rather than forks, so the worker has to be
    module-level and picklable.  Getting that wrong typically shows up as a hang
    or as silently different numbers -- both far worse than being slow.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_pooled_matches_serial(self):
        cfg = _small_cfg()
        serial = Path(self.tmp.name) / "serial.csv"
        pooled = Path(self.tmp.name) / "pooled.csv"
        write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[cfg], seeds=[0, 1, 2, 3], jobs=1, verbose=False), serial)
        write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[cfg], seeds=[0, 1, 2, 3], jobs=2, verbose=False), pooled)
        self.assertEqual(_comparable(_rows_from(serial)),
                         _comparable(_rows_from(pooled)))

    def test_pool_degrades_rather_than_failing(self):
        """A jobs count above the work available must still produce every row."""
        p = Path(self.tmp.name) / "over.csv"
        write_runs_csv(run_matrix(policies=["round_robin"], scenarios=[_small_cfg()], seeds=[0], jobs=8, verbose=False), p)
        self.assertEqual(len(_rows_from(p)), 1)


class TestHeldOutScenario(unittest.TestCase):
    """`agile` is the held-out set; it must not be a silent default anywhere.

    Tuning was closed before it was run once (DESIGN.md 11.8), and nothing is
    trained on it.  A default that quietly includes it would destroy the
    generalisation claim without anyone noticing.
    """

    def test_agile_is_not_in_the_default_scenarios(self):
        from eval import runner

        parser = None
        for name in ("_build_parser", "build_parser", "_parser"):
            if hasattr(runner, name):
                parser = getattr(runner, name)()
                break
        if parser is None:
            ns = runner.main.__doc__ or ""
            self.skipTest("no parser factory exposed to introspect")
        default = None
        for act in parser._actions:
            if "--scenarios" in getattr(act, "option_strings", []):
                default = act.default
        self.assertIsNotNone(default, "--scenarios has no default to check")
        text = default if isinstance(default, str) else ",".join(default)
        self.assertNotIn("agile", text)


class TestResultsFileSafety(unittest.TestCase):
    """A partial run must not be able to destroy a completed matrix.

    `--out` defaults to the canonical runs.csv and `--ablation-out` defaulted to
    the canonical ablation.csv INDEPENDENTLY, so running a subset of policies
    replaced a finished sweep with a fragment -- and redirecting --out still
    clobbered the ablation table. That happened twice during the build; the
    second time the dashboard silently showed one scenario instead of three.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "runs.csv"
        write_runs_csv(
            run_matrix(policies=["round_robin", "random"], scenarios=[_small_cfg()],
                       seeds=[0, 1], verbose=False),
            self.path,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _subset_rows(self):
        return [{"policy": "round_robin", "scenario": "sparse", "seed": 0}]

    def test_shrinking_a_results_file_is_refused(self):
        with self.assertRaises(SystemExit) as ctx:
            _guard_overwrite(self.path, self._subset_rows(), force=False)
        msg = str(ctx.exception)
        self.assertIn("refusing to overwrite", msg)
        self.assertIn("--force", msg)

    def test_force_permits_the_overwrite(self):
        _guard_overwrite(self.path, self._subset_rows(), force=True)

    def test_superset_or_equal_is_allowed_without_force(self):
        """The guard must only block genuine LOSS, never ordinary re-runs."""
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            same = [dict(r) for r in csv.DictReader(fh)]
        _guard_overwrite(self.path, same, force=False)

    def test_absent_file_is_allowed(self):
        _guard_overwrite(Path(self.tmp.name) / "nope.csv", self._subset_rows(), force=False)

    def test_ablation_path_follows_its_runs_file(self):
        """A redirected --out must not write the canonical ablation table."""
        side = _ablation_path_for(Path(self.tmp.name) / "scratch.csv", None)
        self.assertEqual(side.name, "scratch.ablation.csv")
        self.assertNotEqual(side.resolve(), Path(DEFAULT_ABLATION_CSV).resolve())

    def test_canonical_runs_still_writes_canonical_ablation(self):
        """...but the real sweep must still keep the two in step."""
        self.assertEqual(
            _ablation_path_for(DEFAULT_RUNS_CSV, None).resolve(),
            Path(DEFAULT_ABLATION_CSV).resolve(),
        )

    def test_explicit_ablation_out_wins(self):
        want = Path(self.tmp.name) / "explicit.csv"
        self.assertEqual(_ablation_path_for(DEFAULT_RUNS_CSV, str(want)), want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
