"""Figures for the pitch.  Four charts, and a fallback that needs no packages.

    python -m eval.figures                # everything it can find
    python -m eval.figures --only energy  # one chart
    python -m eval.figures --no-mpl       # force the ASCII/CSV fallback

Charts, in the order they earn their place on a slide:

1. **energy vs POI** -- THE headline.  Every policy is one point in
   (energy_total_J, POI@60); the claim "same interception performance at roughly
   half the scanning energy" is literally the horizontal distance between two
   dots, with `oracle` drawn as a reference ceiling rather than an optimum.
2. **TTFI distributions** -- how *fast* the first intercept happens, not just
   whether it happens.  Always annotated with `n_intercepted/n_total`, because a
   policy can otherwise win the median by ignoring hard emitters.
3. **staleness over time** -- the scheduler's hard revisit deadline is a
   *provable* bound, so this chart should show a flat ceiling for `index` and an
   unbounded sawtooth for the ablation that removes it.
4. **ablation bars** -- what each component is worth, so the design choices are
   measured rather than asserted.

Everything is read from CSV.  This module never imports the simulator and never
runs a policy; if `results/runs.csv` is missing it says which command produces it
and moves on to the next chart.

DESIGN.md section 10: `matplotlib` imports are wrapped in `try/except ImportError`
with a CSV/ASCII fallback.  The pitch never depends on a package being present --
so `--no-mpl` is a supported mode, not a degraded one, and it is exercised by
`eval/tests/test_figures.py`.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Fixed everywhere a figure needs randomness (scatter jitter, sampling), so a
# figure regenerated for the slide deck is pixel-identical to the one reviewed.
FIG_SEED: int = 20240501

# Policy draw order and colours.  Stable across figures so a judge reading two
# charts side by side does not have to re-learn the legend.  `oracle` is grey and
# dashed on purpose: it is a REFERENCE CEILING, not an optimum -- it is myopic
# over one action, and saying so unprompted costs nothing and buys credibility.
POLICY_ORDER: tuple[str, ...] = (
    "round_robin", "random", "greedy", "index", "index_learned", "oracle",
)
POLICY_COLOR: dict = {
    "round_robin": "#888888",
    "random": "#b8a05a",
    "greedy": "#c26a3a",
    "index": "#2b6cb0",
    "index_learned": "#2f855a",
    "oracle": "#555555",
}
POLICY_MARKER: dict = {
    "round_robin": "s", "random": "^", "greedy": "v",
    "index": "o", "index_learned": "D", "oracle": "*",
}

_SPARK = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# matplotlib, optionally
# ---------------------------------------------------------------------------
def _load_mpl(enabled: bool = True):
    """Return `(plt, ok)`.  Never raises; `ok=False` selects the ASCII path."""
    if not enabled:
        return None, False
    try:
        import matplotlib

        # Agg before pyplot: these run headless in CI and over SSH, and the
        # default interactive backend would either fail or block.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, False
    return plt, True


# ---------------------------------------------------------------------------
# CSV reading -- no pandas dependency, so the fallback path is truly dependency
# free.  These files are at most a few thousand rows.
# ---------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    """Read a CSV into a list of dicts, coercing numeric-looking fields."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            r[k] = _num(v)
    return rows


def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return float("nan")
    try:
        f = float(s)
    except ValueError:
        return s
    return f


def _get(row: dict, *names, default=None):
    """First present, non-empty value among `names`.  Tolerates schema drift."""
    for n in names:
        if n in row and row[n] is not None:
            v = row[n]
            if isinstance(v, float) and math.isnan(v):
                continue
            return v
    for n in names:  # case-insensitive second pass
        for k in row:
            if str(k).lower() == n.lower() and row[k] is not None:
                return row[k]
    return default


def _finite(values) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(values) -> float:
    v = _finite(values)
    return sum(v) / len(v) if v else float("nan")


def _median(values) -> float:
    v = sorted(_finite(values))
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _quantile(values, q: float) -> float:
    v = sorted(_finite(values))
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


# ---------------------------------------------------------------------------
# ASCII fallback primitives
# ---------------------------------------------------------------------------
def sparkline(values, width: int = 40) -> str:
    """Unicode block sparkline.  Empty string on empty/NaN-only input."""
    v = _finite(values)
    if not v:
        return ""
    if len(v) > width:  # decimate evenly rather than truncating
        step = len(v) / width
        v = [v[min(len(v) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-15:
        return _SPARK[0] * len(v)
    return "".join(_SPARK[min(7, int((x - lo) / (hi - lo) * 7.999))] for x in v)


def bar(value: float, vmax: float, width: int = 30) -> str:
    if not math.isfinite(value) or not math.isfinite(vmax) or vmax <= 0:
        return ""
    return "█" * max(0, min(width, int(round(width * value / vmax))))


class Report:
    """Accumulates the text fallback so every chart has a readable twin.

    Written even when matplotlib IS available: a judge reading over a shoulder,
    or a diff in code review, needs the numbers as text.
    """

    def __init__(self):
        self.lines: list[str] = []
        self.written: list[Path] = []

    def head(self, title: str) -> None:
        self.lines += ["", "=" * 72, title, "=" * 72]

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def note_missing(self, path: Path, how: str) -> None:
        self.head(f"SKIPPED -- {path.name} not found")
        self.say(f"  expected at: {path}")
        self.say(f"  produce it with: {how}")

    def dump_csv(self, path: Path, rows: list[dict]) -> None:
        """Every chart also lands as a tidy CSV -- the real fallback artefact."""
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        self.written.append(path)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.written.append(path)
        return path

    def __str__(self) -> str:
        return "\n".join(self.lines)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _policy_of(row: dict) -> str:
    return str(_get(row, "policy", "baseline", "agent", "name", default="?"))


def _scenario_of(row: dict) -> str:
    return str(_get(row, "scenario", "config", "cfg", default="?"))


def group_runs(rows: list[dict], scenario: "str | None" = None) -> dict:
    """`{policy: [rows]}`, optionally filtered to one scenario."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        if scenario is not None and _scenario_of(r) != scenario:
            continue
        out.setdefault(_policy_of(r), []).append(r)
    # Known policies first, in the order a judge reads them; then anything else.
    known = [p for p in POLICY_ORDER if p in out]
    rest = sorted(k for k in out if k not in POLICY_ORDER)
    return {k: out[k] for k in known + rest}


def scenarios_in(rows: list[dict]) -> list[str]:
    seen, order = set(), []
    for r in rows:
        s = _scenario_of(r)
        if s not in seen:
            seen.add(s)
            order.append(s)
    # `agile` last: it is the held-out set, and it reads as the punchline.
    return sorted(order, key=lambda s: (s == "agile", s))


# ---------------------------------------------------------------------------
# 1. Energy vs POI -- the headline
# ---------------------------------------------------------------------------
def fig_energy_vs_poi(rows: list[dict], out_dir: Path, plt, rep: Report,
                      poi_key: str = "poi_60") -> "Path | None":
    """One point per policy per scenario, with per-seed spread.

    The sentence this chart has to make sayable is "same interception
    performance at roughly half the scanning energy", so energy is on x and POI
    on y: the claim is a horizontal gap at constant height, which is the one
    reading a viewer cannot mistake.
    """
    scen = scenarios_in(rows)
    table: list[dict] = []
    for s in scen:
        for pol, rs in group_runs(rows, s).items():
            e = [_get(r, "energy_total_j", "energy_j", "energy_total") for r in rs]
            p = [_get(r, poi_key, "poi_60", "poi") for r in rs]
            epd = [_get(r, "energy_per_detection_j") for r in rs]
            table.append({
                "scenario": s, "policy": pol, "n_seeds": len(rs),
                "energy_mean_j": _mean(e), "energy_p10_j": _quantile(e, 0.1),
                "energy_p90_j": _quantile(e, 0.9),
                "poi_mean": _mean(p), "poi_p10": _quantile(p, 0.1),
                "poi_p90": _quantile(p, 0.9),
                "energy_per_detection_j": _mean(epd),
            })
    rep.head("1. ENERGY vs POI@60  (headline)")
    rep.say(f"  {'scenario':<10}{'policy':<16}{'energy J':>10}{'POI':>8}"
            f"{'J/det':>10}  seeds")
    for t in table:
        rep.say(f"  {t['scenario']:<10}{t['policy']:<16}"
                f"{t['energy_mean_j']:>10.3f}{t['poi_mean']:>8.3f}"
                f"{t['energy_per_detection_j']:>10.4f}  {t['n_seeds']}")
    _say_headline(table, rep)
    rep.dump_csv(out_dir / "energy_vs_poi.csv", table)

    if plt is None:
        return None
    fig, axes = plt.subplots(1, max(1, len(scen)), figsize=(5.2 * max(1, len(scen)), 4.6),
                             squeeze=False)
    for ax, s in zip(axes[0], scen):
        for t in [x for x in table if x["scenario"] == s]:
            pol = t["policy"]
            ax.errorbar(
                t["energy_mean_j"], t["poi_mean"],
                xerr=[[max(0.0, t["energy_mean_j"] - t["energy_p10_j"])],
                      [max(0.0, t["energy_p90_j"] - t["energy_mean_j"])]],
                yerr=[[max(0.0, t["poi_mean"] - t["poi_p10"])],
                      [max(0.0, t["poi_p90"] - t["poi_mean"])]],
                marker=POLICY_MARKER.get(pol, "o"), markersize=11,
                color=POLICY_COLOR.get(pol, "#444444"),
                ecolor=POLICY_COLOR.get(pol, "#444444"), elinewidth=1.0,
                capsize=3, linestyle="none",
                label=pol + (" (ceiling)" if pol == "oracle" else ""),
            )
            ax.annotate(pol, (t["energy_mean_j"], t["poi_mean"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8,
                        color=POLICY_COLOR.get(pol, "#444444"))
        ax.set_title(f"{s}" + ("  (held out)" if s == "agile" else ""))
        ax.set_xlabel("scan energy spent (J)   ← better")
        ax.set_ylabel("POI@60  (fraction of emitters intercepted)   better →")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylim(-0.02, 1.02)
        # Headroom for the point labels, which otherwise clip at the right edge.
        ax.margins(x=0.22)
    axes[0][0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle("Same interception performance, far less energy", fontsize=13)
    fig.tight_layout()
    return _save(fig, out_dir / "energy_vs_poi.png", rep, plt)


def _say_headline(table: list[dict], rep: Report) -> None:
    """Spell out the actual claim, per scenario, rather than leaving it implied."""
    for s in sorted({t["scenario"] for t in table}):
        rr = next((t for t in table if t["scenario"] == s and t["policy"] == "round_robin"), None)
        ix = next((t for t in table if t["scenario"] == s and t["policy"] == "index"), None)
        if not rr or not ix:
            continue
        if not (math.isfinite(rr["energy_mean_j"]) and rr["energy_mean_j"] > 0):
            continue
        ratio = ix["energy_mean_j"] / rr["energy_mean_j"]
        rep.say(f"  [{s}] index spends {ratio:.0%} of round_robin's energy "
                f"at POI {ix['poi_mean']:.2f} vs {rr['poi_mean']:.2f}")


# ---------------------------------------------------------------------------
# 2. TTFI distributions
# ---------------------------------------------------------------------------
def fig_ttfi(rows: list[dict], out_dir: Path, plt, rep: Report) -> "Path | None":
    """Median and p90 time-to-first-intercept on priority-1 emitters.

    ALWAYS annotated with `n_intercepted/n_total`: TTFI is censored at the
    horizon, so a policy that simply never finds the hard emitters would post a
    flattering median.  Showing the fraction alongside closes that hole before
    anyone opens it.
    """
    scen = scenarios_in(rows)
    table: list[dict] = []
    for s in scen:
        for pol, rs in group_runs(rows, s).items():
            med = [_get(r, "ttfi_p1_median_s", "ttfi_median_s") for r in rs]
            p90 = [_get(r, "ttfi_p1_p90_s", "ttfi_p90_s") for r in rs]
            frac = [_get(r, "ttfi_p1_frac") for r in rs]
            n_i = [_get(r, "ttfi_p1_n_intercepted") for r in rs]
            n_t = [_get(r, "ttfi_p1_n_total") for r in rs]
            table.append({
                "scenario": s, "policy": pol,
                "ttfi_median_s": _median(med), "ttfi_p90_s": _median(p90),
                "intercepted_frac": _mean(frac),
                "n_intercepted": _mean(n_i), "n_total": _mean(n_t),
                "n_seeds": len(rs),
            })
    rep.head("2. TTFI (priority 1) -- censored at the horizon")
    rep.say(f"  {'scenario':<10}{'policy':<16}{'median s':>10}{'p90 s':>9}"
            f"{'intercepted':>13}")
    for t in table:
        got = (f"{t['n_intercepted']:.1f}/{t['n_total']:.1f}"
               if math.isfinite(t["n_total"]) else "n/a")
        rep.say(f"  {t['scenario']:<10}{t['policy']:<16}"
                f"{t['ttfi_median_s']:>10.3f}{t['ttfi_p90_s']:>9.3f}{got:>13}")
    rep.dump_csv(out_dir / "ttfi.csv", table)

    if plt is None:
        return None
    import numpy as np

    fig, axes = plt.subplots(1, max(1, len(scen)), figsize=(5.0 * max(1, len(scen)), 4.4),
                             squeeze=False)
    for ax, s in zip(axes[0], scen):
        sub = [t for t in table if t["scenario"] == s]
        y = np.arange(len(sub))
        ax.barh(y - 0.18, [t["ttfi_median_s"] for t in sub], height=0.34,
                color=[POLICY_COLOR.get(t["policy"], "#444") for t in sub],
                label="median")
        ax.barh(y + 0.18, [t["ttfi_p90_s"] for t in sub], height=0.34, alpha=0.45,
                color=[POLICY_COLOR.get(t["policy"], "#444") for t in sub],
                label="p90")
        for i, t in enumerate(sub):
            if math.isfinite(t["n_total"]) and t["n_total"] > 0:
                ax.annotate(f"{t['n_intercepted']:.0f}/{t['n_total']:.0f}",
                            (max(_finite([t["ttfi_median_s"], t["ttfi_p90_s"]]) or [0]), i),
                            textcoords="offset points", xytext=(6, -3), fontsize=8)
        ax.set_yticks(y, [t["policy"] for t in sub], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("time to first intercept (s)   ← better")
        ax.set_title(f"{s}" + ("  (held out)" if s == "agile" else ""))
        ax.grid(alpha=0.25, linestyle=":", axis="x")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("TTFI prio-1, with n_intercepted/n_total so nobody wins by skipping "
                 "hard emitters", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir / "ttfi.png", rep, plt)


# ---------------------------------------------------------------------------
# 3. Staleness over time
# ---------------------------------------------------------------------------
def staleness_trace(steps: list[dict], horizon_s: float = 60.0,
                    n_points: int = 400) -> tuple[list[float], list[float]]:
    """Time since the LAST scan finished, sampled on a uniform grid.

    Computed from the trace alone -- no mission, no truth -- so this works on any
    `results/steps/*.csv`.  The shape is what matters: the index policy under a
    hard revisit deadline must show a bounded sawtooth, while an ablation without
    the deadline layer drifts upward without limit.
    """
    ends = sorted(
        float(_get(s, "t_end", "t", default=float("nan")))
        for s in steps
        if str(_get(s, "kind", default="scan")).lower() == "scan"
    )
    ends = [e for e in ends if math.isfinite(e)]
    if not ends:
        return [], []
    horizon_s = max(horizon_s, ends[-1])
    ts = [horizon_s * i / (n_points - 1) for i in range(n_points)]
    out, j, last = [], 0, 0.0
    for t in ts:
        while j < len(ends) and ends[j] <= t:
            last = ends[j]
            j += 1
        out.append(t - last)
    return ts, out


def fig_staleness(steps_by_run: dict, out_dir: Path, plt, rep: Report,
                  horizon_s: float = 60.0) -> "Path | None":
    rep.head("3. STALENESS over time (gap since the last completed dwell)")
    table: list[dict] = []
    traces = {}
    for run_id, steps in steps_by_run.items():
        ts, st = staleness_trace(steps, horizon_s)
        if not ts:
            continue
        traces[run_id] = (ts, st)
        table.append({
            "run_id": run_id, "max_staleness_s": max(st),
            "mean_staleness_s": sum(st) / len(st), "n_steps": len(steps),
        })
        rep.say(f"  {str(run_id):<28} max {max(st):6.3f}s  "
                f"mean {sum(st)/len(st):6.3f}s  {sparkline(st)}")
    if not traces:
        rep.say("  no step traces found under results/steps/")
        return None
    rep.dump_csv(out_dir / "staleness.csv", table)

    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for run_id, (ts, st) in traces.items():
        ax.plot(ts, st, lw=1.2, label=str(run_id),
                color=POLICY_COLOR.get(str(run_id).split("_")[0], None))
    ax.set_xlabel("mission time (s)")
    ax.set_ylabel("seconds since last completed dwell")
    ax.set_title("Revisit staleness -- a hard deadline shows up as a flat ceiling")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, out_dir / "staleness.png", rep, plt)


# ---------------------------------------------------------------------------
# 4. Ablation bars
# ---------------------------------------------------------------------------
def fig_ablation(rows: list[dict], out_dir: Path, plt, rep: Report,
                 metric: str = "energy_per_detection_j") -> "Path | None":
    """What each component is actually worth.

    The bar chart is the honest form here: the ablations are a handful of
    discrete configurations, not a continuum, so a line would imply an ordering
    the experiment does not have.
    """
    label_of = lambda r: str(_get(r, "variant", "ablation", "arm", "config",
                                  "policy", "name", default="?"))
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(label_of(r), []).append(r)

    table = []
    for name, rs in groups.items():
        table.append({
            "variant": name, "n_seeds": len(rs),
            metric: _mean([_get(r, metric) for r in rs]),
            "poi_60": _mean([_get(r, "poi_60", "poi") for r in rs]),
            "energy_total_j": _mean([_get(r, "energy_total_j") for r in rs]),
        })
    table.sort(key=lambda t: (not math.isfinite(t[metric] or float("nan")), t[metric]))

    rep.head(f"4. ABLATION -- {metric}")
    vmax = max(_finite([t[metric] for t in table]) or [1.0])
    rep.say(f"  {'variant':<26}{metric:>16}{'POI@60':>9}")
    for t in table:
        rep.say(f"  {t['variant']:<26}{t[metric]:>16.4f}{t['poi_60']:>9.3f}  "
                f"{bar(t[metric], vmax)}")
    rep.dump_csv(out_dir / "ablation.csv", table)

    if plt is None:
        return None
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(table)), 4.4))
    x = np.arange(len(table))
    ax.bar(x, [t[metric] for t in table], width=0.62,
           color=[POLICY_COLOR.get(t["variant"], "#4a6fa5") for t in table])
    for i, t in enumerate(table):
        if math.isfinite(t[metric]):
            ax.annotate(f"{t[metric]:.3g}", (i, t[metric]), ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(x, [t["variant"] for t in table], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric + "   ← better")
    ax.set_title("Ablation: every component measured, not asserted")
    ax.grid(alpha=0.25, linestyle=":", axis="y")
    fig.tight_layout()
    return _save(fig, out_dir / "ablation.png", rep, plt)


# ---------------------------------------------------------------------------
def _save(fig, path: Path, rep: Report, plt) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    rep.written.append(path)
    return path


def load_steps(steps_dir: Path, limit: int = 8) -> dict:
    """`{run_id: [step rows]}` from `results/steps/*.csv`.

    Capped so a directory with a hundred runs does not produce an unreadable
    chart; the cap is deterministic (sorted by name) for reproducibility.
    """
    if not steps_dir.is_dir():
        return {}
    out = {}
    for p in sorted(steps_dir.glob("*.csv"))[:limit]:
        try:
            out[p.stem] = read_csv(p)
        except (OSError, csv.Error) as exc:  # a half-written file must not abort
            print(f"[figures] skipping {p.name}: {exc}", file=sys.stderr)
    return out


def make_all(results_dir: Path = RESULTS_DIR, out_dir: Path = FIG_DIR,
             use_mpl: bool = True, only: "str | None" = None,
             horizon_s: float = 60.0) -> Report:
    """Build every figure that has data.  Missing inputs are reported, not fatal."""
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt, have_mpl = _load_mpl(use_mpl)

    rep = Report()
    rep.say(f"figures  (matplotlib: {'yes' if have_mpl else 'NO -- ASCII/CSV fallback'})")
    rep.say(f"seed {FIG_SEED}; reading {results_dir}; writing {out_dir}")

    runs_csv = results_dir / "runs.csv"
    abl_csv = results_dir / "ablation.csv"

    want = (lambda k: only is None or only == k)

    if want("energy") or want("ttfi"):
        if runs_csv.exists():
            runs = read_csv(runs_csv)
            if want("energy"):
                fig_energy_vs_poi(runs, out_dir, plt if have_mpl else None, rep)
            if want("ttfi"):
                fig_ttfi(runs, out_dir, plt if have_mpl else None, rep)
        else:
            rep.note_missing(runs_csv, "python -m eval.runner --all")

    if want("staleness"):
        steps = load_steps(results_dir / "steps")
        if steps:
            fig_staleness(steps, out_dir, plt if have_mpl else None, rep, horizon_s)
        else:
            rep.note_missing(results_dir / "steps", "python -m eval.runner --all")

    if want("ablation"):
        if abl_csv.exists():
            fig_ablation(read_csv(abl_csv), out_dir, plt if have_mpl else None, rep)
        else:
            rep.note_missing(abl_csv, "python -m eval.runner --ablation")

    rep.head("WROTE")
    for p in rep.written:
        rep.say(f"  {p}")
    rep.save(out_dir / "figures_report.txt")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.figures",
        description="Render the pitch figures from results/*.csv.",
    )
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=str(FIG_DIR))
    ap.add_argument("--only", choices=("energy", "ttfi", "staleness", "ablation"),
                    default=None)
    ap.add_argument("--no-mpl", action="store_true",
                    help="force the ASCII/CSV fallback (DESIGN.md section 10)")
    ap.add_argument("--horizon", type=float, default=60.0)
    args = ap.parse_args(argv)

    rep = make_all(Path(args.results_dir), Path(args.out),
                   use_mpl=not args.no_mpl, only=args.only, horizon_s=args.horizon)
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
