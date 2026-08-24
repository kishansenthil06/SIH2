"""Firewall enforcement.  DESIGN.md section 2, all three mechanisms.

    The simulator knows the answer, the evaluator may see it, the agent never does.

A firewall breach does not crash anything -- it silently inflates every number
in the write-up, and the first person to notice is the judge.  So it is enforced
three independent ways and each one is tested here:

1. **Statically** -- an AST scan over every `.py` in `agent/` and `app/`, which
   catches a forbidden import or a `.truth_power` attribute *without executing
   the module*, so a path that never runs in a demo is still checked.
2. **At runtime** -- `sim.env._forbid_agent_callers` walks the stack and raises
   `FirewallViolation` if anything in `agent/` or `app/` is anywhere on it.
3. **Structurally** -- `AgentEnv.__slots__` holds bound methods and plain data;
   there is no `__dict__` to attach a `World` to and no slot that holds one.

This file is also the ONE place permitted to import both `sim.receiver` and
`agent.belief`, because it exists to prove the deliberate duplication of
`pd_curve` has not drifted (DESIGN.md section 2: duplication is cheaper than a
firewall breach, but only if it is checked).
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# `agent/` and `app/` may import these two modules from `sim` and nothing else:
# pure types, grid arithmetic, the config loader.  Neither holds ground truth
# and neither has a reference path to a World.
PERMITTED_SIM_MODULES: frozenset[str] = frozenset({"sim.contract", "sim.config"})

FORBIDDEN_SIM_MODULES: frozenset[str] = frozenset(
    {"sim.env", "sim.emitters", "sim.channel", "sim.receiver"}
)

# Attribute names that only exist on the truth side.  `_world` and `emitters`
# are here because they are the two *reference paths* to a burst table -- the
# attribute itself is harmless, walking to it is not.
FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {"truth", "truth_bursts", "truth_power", "_world", "emitters"}
)

# `sim.stub_env` holds no truth, but agent-side SOURCE importing it would mean
# the agent knows which environment it is in.  Test files may import it -- that
# is what it was frozen for (see its module docstring).
STUB_ENV: str = "sim.stub_env"

SCANNED_PACKAGES: tuple[str, ...] = ("agent", "app")


def _is_test_file(path: Path) -> bool:
    return path.name.startswith("test_") or "tests" in path.parts


def _iter_sources():
    for pkg in SCANNED_PACKAGES:
        d = ROOT / pkg
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


def _imported_modules(tree: ast.AST):
    """Every dotted module name an import statement brings into scope.

    `from sim import env` is reported as `sim.env`, which is the same breach as
    `import sim.env` and must not be able to hide behind a different syntax.
    `from sim.contract import Scan`, by contrast, reports only `sim.contract` --
    the alias there names an object, not a module, and `sim` has no
    sub-subpackages, so "the module part has no dot" separates the two cases
    exactly.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # relative import inside agent/
                continue
            mod = node.module or ""
            out.append((mod, node.lineno))
            if "." not in mod:                  # `from <package> import <module>`
                for a in node.names:
                    out.append((f"{mod}.{a.name}", node.lineno))
    return out


class TestStaticFirewall(unittest.TestCase):
    """Mechanism 1: the AST scan."""

    def test_sources_exist(self):
        # A scan that silently finds nothing to scan is worse than no scan.
        self.assertTrue(list(_iter_sources()), "no agent/ or app/ sources found")

    def test_no_forbidden_sim_imports(self):
        bad = []
        for p in _iter_sources():
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            for mod, line in _imported_modules(tree):
                root = mod.split(".")[0]
                if root != "sim":
                    continue
                if mod in PERMITTED_SIM_MODULES or mod == "sim":
                    continue
                if mod == STUB_ENV and _is_test_file(p):
                    continue
                if mod in FORBIDDEN_SIM_MODULES or mod == STUB_ENV:
                    bad.append(f"{p.relative_to(ROOT)}:{line}: imports {mod}")
                else:
                    bad.append(
                        f"{p.relative_to(ROOT)}:{line}: imports {mod}; only "
                        f"{sorted(PERMITTED_SIM_MODULES)} are permitted"
                    )
        self.assertEqual(
            bad, [],
            "FIREWALL BREACH -- agent/ and app/ may import only sim.contract and "
            "sim.config:\n  " + "\n  ".join(bad),
        )

    def test_no_truth_attribute_access(self):
        bad = []
        for p in _iter_sources():
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                    bad.append(
                        f"{p.relative_to(ROOT)}:{node.lineno}: touches .{node.attr}"
                    )
                # `getattr(env, "truth_power")` is the same breach spelled
                # differently, so the string form is caught too.
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "getattr" and len(node.args) >= 2 \
                        and isinstance(node.args[1], ast.Constant) \
                        and node.args[1].value in FORBIDDEN_ATTRS:
                    bad.append(
                        f"{p.relative_to(ROOT)}:{node.lineno}: "
                        f"getattr(..., {node.args[1].value!r})"
                    )
        self.assertEqual(
            bad, [],
            "FIREWALL BREACH -- agent-side code reached for ground truth:\n  "
            + "\n  ".join(bad),
        )

    def test_scan_catches_a_planted_breach(self):
        """The scan is only worth anything if it fails on a real breach."""
        breach = ast.parse("from sim.env import World\nx = env.truth_power()\n")
        mods = [m for m, _ in _imported_modules(breach)]
        self.assertIn("sim.env", mods)
        attrs = {n.attr for n in ast.walk(breach) if isinstance(n, ast.Attribute)}
        self.assertTrue(attrs & FORBIDDEN_ATTRS)


class TestRuntimeFirewall(unittest.TestCase):
    """Mechanism 2: the stack walk in `sim.env._forbid_agent_callers`."""

    def _agent_module_call(self, method: str, world):
        """Call `world.<method>()` from a frame whose module is named `agent.*`.

        `_forbid_agent_callers` identifies the caller by `frame.f_globals["__name__"]`,
        so executing the call in a globals dict carrying an `agent.` name is
        exactly the situation it is meant to catch -- and it does not require
        planting a breach in a real file.
        """
        ns = {"__name__": "agent.firewall_probe", "world": world}
        exec(compile(f"def probe():\n    return world.{method}()\n",
                     "<agent.firewall_probe>", "exec"), ns)
        return ns["probe"]()

    def test_truth_raises_for_agent_caller(self):
        from sim.contract import FirewallViolation
        from sim.env import make_world

        world = make_world("sparse", 0)
        for method in ("truth", "truth_bursts", "truth_power"):
            with self.subTest(method=method):
                with self.assertRaises(FirewallViolation):
                    self._agent_module_call(method, world)

    def test_truth_is_allowed_from_eval(self):
        # This module is `eval.tests.test_firewall`, i.e. the evaluator side.
        from sim.env import make_world

        world = make_world("sparse", 0)
        self.assertGreater(world.truth_bursts().size, 0)

    def test_truth_bursts_is_read_only(self):
        from sim.env import make_world

        b = make_world("sparse", 0).truth_bursts()
        with self.assertRaises(ValueError):
            b["t_on"][0] = -1.0


class TestStructuralFirewall(unittest.TestCase):
    """Mechanism 3: `AgentEnv` owns nothing that leads to a `World`."""

    def setUp(self):
        from sim.env import make_world

        self.world = make_world("sparse", 0)
        self.env = self.world.agent_view()

    def test_agent_env_has_no_dict(self):
        self.assertFalse(hasattr(self.env, "__dict__"))
        with self.assertRaises(AttributeError):
            self.env.smuggled = self.world           # nothing can be attached

    def test_no_slot_holds_a_world(self):
        from sim.env import World

        for slot in type(self.env).__slots__:
            with self.subTest(slot=slot):
                self.assertNotIsInstance(getattr(self.env, slot), World)

    def test_no_slot_is_named_for_truth(self):
        self.assertFalse(set(type(self.env).__slots__) & FORBIDDEN_ATTRS)

    def test_agent_env_exposes_no_truth_methods(self):
        for name in ("truth", "truth_bursts", "truth_power", "_bursts", "emitters"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.env, name))

    def test_agent_env_satisfies_scan_env(self):
        from sim.contract import ScanEnv

        self.assertIsInstance(self.env, ScanEnv)


class TestPdCurveCrossCheck(unittest.TestCase):
    """`agent/belief.py` reimplements `pd_curve`.  Prove it has not drifted.

    The two have deliberately different signatures -- `sim.receiver.pd_curve`
    takes `(snr_eff_db, dwell_s, bw_hz_per_channel, pfa)` positionally, and
    `agent.belief.pd_curve` takes `(snr_eff_db, dwell_s, pfa=..., channel_bw_hz=...)`
    -- because they were written independently on either side of the firewall.
    Different argument order is not drift; a different number is.
    """

    GRID_N = 100
    TOL = 1e-9

    def _grids(self):
        # 10 x 10 = 100 points spanning the whole regime of DESIGN.md section 1's
        # verified table, from "one decade of dwell below threshold" to
        # saturation.
        snr = np.linspace(-25.0, -5.0, 10)
        dwell = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) * 1e-3
        return snr, dwell

    def test_agrees_to_1e9_over_a_100_point_grid(self):
        from agent.belief import pd_curve as pd_agent
        from sim.receiver import pd_curve as pd_sim

        snr, dwell = self._grids()
        self.assertEqual(snr.size * dwell.size, self.GRID_N)
        pfa, cbw = 1.0e-3, 1.0e6

        worst = 0.0
        for s in snr:
            a = pd_sim(s, dwell, cbw, pfa)
            b = pd_agent(s, dwell, pfa=pfa, channel_bw_hz=cbw)
            worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
        self.assertLessEqual(
            worst, self.TOL,
            f"sim.receiver.pd_curve and agent.belief.pd_curve disagree by "
            f"{worst:.3e}; the duplication in DESIGN.md section 2 has drifted",
        )

    def test_they_currently_agree_exactly(self):
        """Stronger than the 1e-9 contract, and true today -- keep it true.

        If this fails but the 1e-9 test passes, someone has changed one of the
        two implementations in a numerically equivalent way.  That is allowed,
        but it should be a deliberate act, not a surprise.
        """
        from agent.belief import pd_curve as pd_agent
        from sim.receiver import pd_curve as pd_sim

        snr, dwell = self._grids()
        for s in snr:
            np.testing.assert_array_equal(
                np.asarray(pd_sim(s, dwell, 1.0e6, 1.0e-3)),
                np.asarray(pd_agent(s, dwell, pfa=1.0e-3, channel_bw_hz=1.0e6)),
            )

    def test_zero_snr_gives_pfa_on_both_sides(self):
        """`s = 0` must give `P_d = P_fa` -- the identity that removes a whole
        false-alarm branch from the project (DESIGN.md section 1)."""
        from agent.belief import pd_curve as pd_agent
        from sim.receiver import pd_from_linear

        for pfa in (1e-2, 1e-3, 1e-4):
            with self.subTest(pfa=pfa):
                self.assertAlmostEqual(
                    float(pd_from_linear(0.0, 0.010, 1.0e6, pfa)), pfa, places=12)
                self.assertAlmostEqual(
                    float(pd_agent(-np.inf, 0.010, pfa=pfa)), pfa, places=12)

    def test_matches_the_verified_design_table(self):
        """Spot-check the bolded cells of the DESIGN.md section 1 table.

        These were produced by running the formula and are the numbers the whole
        project's SNR/dwell trade-off is argued from, so both implementations
        have to reproduce them, not merely agree with each other.
        """
        from agent.belief import pd_curve as pd_agent
        from sim.receiver import pd_curve as pd_sim

        cases = ((-15.0, 0.010, 0.528), (-18.0, 0.050, 0.672), (-20.0, 0.100, 0.528))
        for snr, dwell, expect in cases:
            with self.subTest(snr=snr, dwell=dwell):
                self.assertAlmostEqual(
                    float(pd_sim(snr, dwell, 1.0e6, 1.0e-3)), expect, places=3)
                self.assertAlmostEqual(
                    float(pd_agent(snr, dwell)), expect, places=3)


class TestEvaluatorSideIsNotScanned(unittest.TestCase):
    """`eval/` is on the truth side by design -- assert that stays deliberate."""

    def test_eval_is_not_in_the_scanned_packages(self):
        self.assertNotIn("eval", SCANNED_PACKAGES)

    def test_oracle_reads_truth_and_says_so(self):
        from eval.baselines import CLAIRVOYANT_LABEL, ClairvoyantGreedy

        src = (ROOT / "eval" / "baselines.py").read_text(encoding="utf-8")
        self.assertIn("truth_bursts", src)
        # It is a CEILING.  Calling it optimal is the claim we cannot support.
        self.assertIn("reference ceiling", CLAIRVOYANT_LABEL)
        self.assertNotIn("optimal", CLAIRVOYANT_LABEL.lower())
        self.assertEqual(ClairvoyantGreedy.label, CLAIRVOYANT_LABEL)

    def test_oracle_refuses_to_run_without_a_world(self):
        """It needs the World, not the AgentEnv -- and says so rather than
        silently scoring against an empty burst table."""
        from sim.config import build_grid, build_mission, load_config

        from eval.baselines import ClairvoyantGreedy

        cfg = load_config("sparse")
        pol = ClairvoyantGreedy()
        with self.assertRaises(RuntimeError):
            pol.reset(build_grid(cfg), build_mission(cfg), 60.0, 0, cfg)


if __name__ == "__main__":
    unittest.main()
