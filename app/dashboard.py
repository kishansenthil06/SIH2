"""Waterfall dashboard -- the demo a judge watches.

    python -m app.dashboard --run sparse_index_s0
    python -m app.dashboard --compare sparse_index_s0 sparse_round_robin_s0
    python -m app.dashboard --run sparse_index_s0 --format png     # no animation

A channel x time waterfall with the scan trace drawn over it, live energy and
intercept counters, and -- the point of the whole thing -- the `chosen_reason`
string rendered beside the current decision.  A judge can point at any moment on
the screen and ask "why did it go there?", and the answer is already on the
screen: `index`, `deadline:ch=41`, `watchlist:ch=88`, `budget-pace`, `sleep`.
That explainability is why the scheduler's hard constraints live outside the
score (DESIGN.md section 7).

`--compare` is the money shot: the same seed, the adaptive policy above and the
round-robin sweep below, with the energy counters running side by side.

**No streamlit, no plotly** -- neither is installed and both were ruled out.
This is matplotlib only, and matplotlib itself is wrapped in `try/except
ImportError`: with no plotting available at all the dashboard prints an ASCII
waterfall to the terminal, which still demos.  A GIF also demos just as well as
a live dashboard and renders in a fraction of the time, so it is the default.

FIREWALL (DESIGN.md section 2).  This module reads CSVs and imports only
`sim.contract` and `sim.config` from `sim` -- and those only to learn the channel
grid geometry, which is public configuration, not truth.  It never imports
`sim.env`, `sim.emitters`, `sim.channel` or `sim.receiver`, never constructs a
`World`, and never touches `.truth*`, `._world` or `.emitters`.  The dashboard
cannot see the answer; it can only see what the receiver reported.

Input schema is `eval.metrics.TRACE_COLUMNS`:
    step, t_start, t_end, kind, f_center_hz, bw_hz, dwell_s, energy_j,
    n_det, det_channels, best_score, chosen_reason, energy_spent_total
`k_lo`/`k_hi` are not in that schema, so the scanned span is derived from
`f_center_hz` and `bw_hz` against the grid.

The frequency axis is **2000 channels** tall (DESIGN.md section 11.1) against an
axes only a few hundred pixels high, so the raster is reduced by an explicit
priority block-max (`_reduce_rows`) before it reaches `imshow`.  Letting
matplotlib resample instead deletes most single-channel detections, which are
the majority -- see the comment on `RENDER_ROWS`.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.config import build_grid, load_config
from sim.contract import ChannelGrid

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
STEPS_DIR = RESULTS_DIR / "steps"
FIG_DIR = RESULTS_DIR / "figures"

# Frames in the animation.  120 at 15 fps is an 8 s loop -- long enough to read
# the reasons scrolling past, short enough that nobody reaches for the spacebar.
# Each frame is a full Agg render of a dense figure (~0.5 s), so this is also the
# knob that decides whether the export takes one minute or ten.
DEFAULT_FRAMES: int = 120
DEFAULT_FPS: int = 15

# Animation renders at a lower DPI than the still: a GIF is watched, not zoomed
# into, and the frame count multiplies every pixel.
DPI_STILL: int = 140
DPI_ANIM: int = 80

# Waterfall time resolution.  240 columns over a 60 s horizon = 250 ms per
# column, which is coarse enough to stay legible and fine enough that a 10 ms
# dwell still paints a visible mark.
TIME_BINS: int = 240

# Waterfall colours, dark-field like a real spectrum display.
C_BG = (0.05, 0.06, 0.09)
C_SCAN = (0.16, 0.42, 0.68)     # a channel sat inside a dwell
C_DET = (1.00, 0.78, 0.20)      # ...and the detector fired on it
C_TRACE = "#7fd1ff"             # the tuned centre, hopping over time

# Rendered height of the waterfall in image rows.  The grid is 2000 channels
# tall (DESIGN.md section 11.1) and the axes are only ~400-500 screen pixels, so
# the raster MUST be reduced before it reaches imshow.  Doing it here, with a
# priority reduction, rather than letting matplotlib resample:
#
#   * `interpolation="nearest"` samples ONE source row in every ~5, so a
#     single-channel detection -- which is most of them, since the policy's
#     favourite action is 1 MHz x 1 ms -- disappears about 80% of the time.
#   * `interpolation="antialiased"` averages instead, which turns one bright
#     yellow row into a 20%-strength smear that reads as noise.
#
# Both silently delete the one thing the demo exists to show.  `_reduce_rows`
# takes the MAX over each block with detection > scan > background priority, so
# one detected channel in a block of five still paints a full-strength mark.
RENDER_ROWS: int = 400

_ASCII_RAMP = " .:-=+*#%@"


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def _num(v, default=float("nan")):
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parse_channels(v) -> np.ndarray:
    """`det_channels` is space-separated so the CSV field never needs quoting."""
    if v is None:
        return np.empty(0, dtype=np.int32)
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return np.empty(0, dtype=np.int32)
    out = []
    for tok in s.replace(",", " ").split():
        try:
            out.append(int(float(tok)))
        except ValueError:
            continue
    return np.asarray(out, dtype=np.int32)


@dataclass
class Trace:
    """One run's executed action log, plus the geometry needed to draw it."""

    run_id: str
    grid: ChannelGrid
    step: np.ndarray
    kind: np.ndarray
    t_start: np.ndarray
    t_end: np.ndarray
    dwell_s: np.ndarray
    k_lo: np.ndarray
    k_hi: np.ndarray
    energy_j: np.ndarray
    energy_total: np.ndarray
    reason: list[str]
    detections: list[np.ndarray]
    scenario: str = "?"
    policy: str = "?"
    horizon_s: float = 60.0
    summary: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.step.size)

    def unique_detected_channels(self, upto: int) -> int:
        """Distinct channels the receiver has reported by step `upto`.

        Named honestly: this is NOT POI.  POI counts distinct *emitters* and can
        only be computed by the evaluator, which holds the burst table; the
        dashboard is on the far side of the firewall and must not pretend
        otherwise.  The true POI@60 is read from `results/runs.csv` and shown
        beside this as the final figure.
        """
        seen: set[int] = set()
        for d in self.detections[: max(0, upto)]:
            seen.update(int(c) for c in d)
        return len(seen)


# Run-id conventions differ between the runner and hand-written traces
# (`index__sparse__s0__h60` vs `sparse_index_s0`), so both the scenario and the
# policy are recovered by matching known names anywhere in the id rather than by
# position.  Longest-first so `index_learned` is not swallowed by `index`.
_SCENARIOS: tuple[str, ...] = ("sparse", "dense", "agile")
_POLICIES: tuple[str, ...] = (
    "round_robin", "index_learned", "greedy", "random", "index", "oracle",
)


def _parse_run_id(run_id: str) -> tuple[str, str]:
    """`(scenario, policy)`.

    Only the LABEL depends on the policy, and only the GRID GEOMETRY depends on
    the scenario -- and all three scenarios share the same 2000-channel / 2 GHz
    1 MHz grid (DESIGN.md section 11.1) -- so a miss here costs a caption, never
    a wrong picture.
    """
    low = run_id.lower()
    scenario = next((s for s in _SCENARIOS if s in low), "sparse")
    policy = next(
        (p for p in sorted(_POLICIES, key=len, reverse=True) if p in low), ""
    )
    if not policy:
        toks = [t for t in re.split(r"_+", low)
                if t and t != scenario and not re.fullmatch(r"[sh]\d+", t)]
        policy = "_".join(toks) if toks else run_id
    return scenario, policy


def load_trace(run_id: str, steps_dir: Path = STEPS_DIR,
               grid: "ChannelGrid | None" = None,
               horizon_s: float = 60.0) -> Trace:
    path = Path(steps_dir) / f"{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no step trace at {path}\n"
            "Produce one with:\n"
            "    python -m eval.runner --all\n"
            f"Available: {', '.join(p.stem for p in sorted(Path(steps_dir).glob('*.csv'))[:12]) or '(none)'}"
        )
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")

    scenario, policy = _parse_run_id(run_id)
    if grid is None:
        try:
            grid = build_grid(load_config(scenario))
        except (OSError, KeyError, ValueError):
            # A trace must still render if the config is unreadable; the default
            # grid is the frozen one from configs/*.yaml.
            grid = ChannelGrid()

    n = len(rows)
    step = np.arange(n, dtype=np.int64)
    kind = np.array([str(r.get("kind", "scan")).lower() for r in rows])
    t_start = np.array([_num(r.get("t_start"), 0.0) for r in rows])
    t_end = np.array([_num(r.get("t_end"), 0.0) for r in rows])
    dwell = np.array([_num(r.get("dwell_s"), 0.0) for r in rows])
    f_c = np.array([_num(r.get("f_center_hz"), float("nan")) for r in rows])
    bw = np.array([_num(r.get("bw_hz"), 0.0) for r in rows])
    e_j = np.array([_num(r.get("energy_j"), 0.0) for r in rows])
    e_tot = np.array([_num(r.get("energy_spent_total"), float("nan")) for r in rows])
    reason = [str(r.get("chosen_reason", "") or "") for r in rows]
    dets = [_parse_channels(r.get("det_channels")) for r in rows]

    k_lo, k_hi = _spans(f_c, bw, grid)
    is_scan = kind == "scan"
    k_lo = np.where(is_scan, k_lo, -1)
    k_hi = np.where(is_scan, k_hi, -1)

    if not np.isfinite(e_tot).all():  # older traces may omit the running total
        e_tot = np.cumsum(np.nan_to_num(e_j))

    return Trace(
        run_id=run_id, grid=grid, step=step, kind=kind, t_start=t_start,
        t_end=t_end, dwell_s=dwell, k_lo=k_lo, k_hi=k_hi, energy_j=e_j,
        energy_total=e_tot, reason=reason, detections=dets,
        scenario=scenario, policy=policy,
        horizon_s=max(horizon_s, float(np.nanmax(t_end)) if n else horizon_s),
    )


def _spans(f_center_hz, bw_hz, grid: ChannelGrid):
    """`(k_lo, k_hi)` for each row, clipped to the grid.

    `ChannelGrid.channels_for` raises on the slightest misalignment, which is
    correct for the simulator and wrong for a viewer: a trace that is 1 Hz off
    should still draw.  So the arithmetic is repeated here with rounding and
    clipping instead of validation.
    """
    cbw = float(grid.channel_bw_hz)
    n_ch = int(grid.n_channels)
    width = np.maximum(1, np.round(np.nan_to_num(bw_hz) / cbw).astype(np.int64))
    lo = np.round(
        (np.nan_to_num(f_center_hz) - np.nan_to_num(bw_hz) / 2.0 - grid.f_start_hz) / cbw
    ).astype(np.int64)
    lo = np.clip(lo, 0, max(0, n_ch - 1))
    hi = np.clip(lo + width, 1, n_ch)
    return lo, hi


def load_summary(run_id: str, results_dir: Path = RESULTS_DIR) -> dict:
    """The evaluator's metrics for this run, if `results/runs.csv` exists.

    POI is an evaluator quantity -- it needs the burst table -- so the dashboard
    reads the finished number rather than trying to reconstruct it.
    """
    path = Path(results_dir) / "runs.csv"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("run_id", "")) == run_id:
                    return {k: _num(v, v) for k, v in r.items()}
    except (OSError, csv.Error):
        return {}
    return {}


# ---------------------------------------------------------------------------
# Waterfall raster
# ---------------------------------------------------------------------------
# Raster cell codes.  Deliberately ORDERED so that a plain `maximum` reduction
# over a block of channels is exactly the priority rule we want: a block
# containing one detection renders as a detection, a block containing only
# dwells renders as a dwell, and only a wholly untouched block stays background.
CELL_BG: int = 0
CELL_SCAN: int = 1
CELL_DET: int = 2
_PALETTE = np.array([C_BG, C_SCAN, C_DET], dtype=np.float32)


def build_waterfall(tr: Trace, time_bins: int = TIME_BINS):
    """`(codes, col0, col1)` -- an `(n_channels, time_bins)` uint8 code raster and
    the column span each step paints into.

    Painted cumulatively so a frame at step `k` is exactly the first `k` rows of
    the log: the animation is a prefix of the finished picture, never a
    recomputation, which is what keeps it honest and fast.

    The raster holds CELL_* codes rather than RGB.  At 2000 channels an RGB
    buffer is 3x the memory for no benefit, and -- the actual reason -- colour
    cannot be reduced by priority, whereas an ordered integer code can.  See
    `_reduce_rows`.
    """
    n_ch = int(tr.grid.n_channels)
    horizon = max(tr.horizon_s, 1e-6)
    codes = np.full((n_ch, time_bins), CELL_BG, dtype=np.uint8)
    # Column index each step paints into, and the per-step column span of the
    # dwell (a 200 ms dwell is wider than one column).
    col0 = np.clip((tr.t_end - tr.dwell_s) / horizon * time_bins, 0, time_bins - 1).astype(int)
    col1 = np.clip(np.ceil(tr.t_end / horizon * time_bins), 1, time_bins).astype(int)
    col1 = np.maximum(col1, col0 + 1)
    return codes, col0, col1


def paint_range(tr: Trace, codes: np.ndarray, col0, col1,
                start: int, stop: int) -> np.ndarray:
    """Paint steps `[start, stop)` into `codes` in place.  Returns `codes`."""
    for i in range(max(0, start), max(0, min(stop, tr.n))):
        if tr.k_lo[i] < 0:
            continue  # Sleep paints nothing: no dwell, no coverage
        block = codes[tr.k_lo[i]:tr.k_hi[i], col0[i]:col1[i]]
        # `maximum` not assignment: a later dwell must not erase an earlier
        # detection that happened to fall in the same time column.
        np.maximum(block, CELL_SCAN, out=block)
        for c in tr.detections[i]:
            if 0 <= c < codes.shape[0]:
                det = codes[int(c), col0[i]:col1[i]]
                np.maximum(det, CELL_DET, out=det)
    return codes


def paint_upto(tr: Trace, codes: np.ndarray, col0, col1, upto: int) -> np.ndarray:
    """Paint steps `[0, upto)` into `codes` in place.  Returns `codes`."""
    return paint_range(tr, codes, col0, col1, 0, upto)


def _reduce_rows(codes: np.ndarray, target_rows: int = RENDER_ROWS) -> np.ndarray:
    """Downsample the channel axis by BLOCK MAX, preserving det > scan > bg.

    A 2000-row raster shown in a ~450 px axes is resampled by matplotlib no
    matter what we do; the only question is whether we choose the rule or it
    does.  Its two options both lose the signal: `nearest` throws away four rows
    in five (so most single-channel detections vanish outright) and
    `antialiased` averages them into a faint smear.  Taking the max over each
    block instead means one detected channel among five still renders at full
    strength -- which is the honest reduction here, because the question the
    picture answers is "did anything happen in this band?", not "what was the
    average of this band?".

    The block size is `ceil(n_ch / target_rows)`; the last block is short when
    the division is not exact, so the tail is padded with background rather than
    wrapping.  Returns `codes` unchanged when it is already short enough.
    """
    n_ch = codes.shape[0]
    if n_ch <= target_rows:
        return codes
    block = int(math.ceil(n_ch / target_rows))
    n_blocks = int(math.ceil(n_ch / block))
    pad = n_blocks * block - n_ch
    if pad:
        codes = np.vstack([codes, np.full((pad, codes.shape[1]), CELL_BG, np.uint8)])
    return codes.reshape(n_blocks, block, codes.shape[1]).max(axis=1)


def codes_to_rgb(codes: np.ndarray, target_rows: int = RENDER_ROWS) -> np.ndarray:
    """Reduce, then colour.  `(rows, time_bins, 3)` float32 ready for imshow."""
    return _PALETTE[_reduce_rows(codes, target_rows)]


# ---------------------------------------------------------------------------
# ASCII fallback -- works with no plotting library at all
# ---------------------------------------------------------------------------
def ascii_waterfall(tr: Trace, rows: int = 24, cols: int = 76) -> str:
    """A coarse density map plus the decision log.  Demos over SSH."""
    n_ch = int(tr.grid.n_channels)
    grid_counts = np.zeros((rows, cols), dtype=np.float64)
    dets = np.zeros((rows, cols), dtype=np.float64)
    horizon = max(tr.horizon_s, 1e-6)
    for i in range(tr.n):
        if tr.k_lo[i] < 0:
            continue
        c = min(cols - 1, int(tr.t_end[i] / horizon * cols))
        r0 = min(rows - 1, int(tr.k_lo[i] / n_ch * rows))
        r1 = max(r0 + 1, min(rows, int(math.ceil(tr.k_hi[i] / n_ch * rows))))
        grid_counts[r0:r1, c] += 1.0
        for ch in tr.detections[i]:
            dets[min(rows - 1, int(int(ch) / n_ch * rows)), c] += 1.0

    vmax = max(grid_counts.max(), 1.0)
    out = [f"waterfall  {tr.run_id}   ({n_ch} channels x {tr.horizon_s:.0f} s)",
           "   " + "-" * cols]
    for r in range(rows):
        line = []
        for c in range(cols):
            if dets[r, c] > 0:
                line.append("O")           # a detection, unmissable
            elif grid_counts[r, c] <= 0.0:
                line.append(_ASCII_RAMP[0])
            else:
                # Log scale: an adaptive policy camps on a few channels, so a
                # linear ramp against the busiest cell washes out everything
                # else -- which is exactly the structure worth seeing.
                v = math.log1p(grid_counts[r, c]) / math.log1p(vmax)
                line.append(_ASCII_RAMP[max(1, min(len(_ASCII_RAMP) - 1,
                                                   int(v * (len(_ASCII_RAMP) - 1))))])
        ch0 = int(r / rows * n_ch)
        out.append(f"{ch0:3d}" + "".join(line))
    out.append("   " + "-" * cols)
    out.append(f"   t=0{' ' * (cols - 12)}t={tr.horizon_s:.0f}s")
    out.append("")
    out.append(counter_text(tr, tr.n).replace("\n", "   "))
    out.append("")
    out.append("last decisions (why it moved):")
    for i in range(max(0, tr.n - 12), tr.n):
        span = ("sleep" if tr.k_lo[i] < 0
                else f"ch {tr.k_lo[i]:>3d}-{tr.k_hi[i] - 1:<3d}")
        out.append(f"  t={tr.t_end[i]:7.3f}s  {span:<12} "
                   f"{tr.dwell_s[i] * 1e3:6.1f} ms  {tr.reason[i]}")
    return "\n".join(out)


def counter_text(tr: Trace, upto: int) -> str:
    """The live counters.  Energy is exact; POI comes from the evaluator."""
    upto = max(0, min(upto, tr.n))
    e = float(tr.energy_total[upto - 1]) if upto else 0.0
    t = float(tr.t_end[upto - 1]) if upto else 0.0
    n_scan = int(np.sum(tr.k_lo[:upto] >= 0))
    n_det = sum(int(d.size) for d in tr.detections[:upto])
    lines = [
        f"t = {t:6.2f} s / {tr.horizon_s:.0f} s",
        f"energy = {e * 1e3:8.1f} mJ",
        f"scans = {n_scan:4d}   sleeps = {upto - n_scan:4d}",
        f"detections = {n_det:4d}  on {tr.unique_detected_channels(upto):3d} channels",
    ]
    poi = tr.summary.get("poi_60")
    if isinstance(poi, float) and math.isfinite(poi):
        lines.append(f"POI@60 (final) = {poi:.3f}")
    epd = tr.summary.get("energy_per_detection_j")
    if isinstance(epd, float) and math.isfinite(epd):
        lines.append(f"J / detection  = {epd:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# matplotlib rendering
# ---------------------------------------------------------------------------
def _load_mpl():
    """`(plt, animation)` or `(None, None)`.  Agg: these render headless."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    return plt, animation


class _Panel:
    """One run's axes: waterfall, scan trace, reason label, counter box."""

    def __init__(self, ax, tr: Trace, time_bins: int = TIME_BINS,
                 render_rows: int = RENDER_ROWS):
        self.ax, self.tr = ax, tr
        self.codes, self.col0, self.col1 = build_waterfall(tr, time_bins)
        self.time_bins = time_bins
        self.render_rows = render_rows
        n_ch = int(tr.grid.n_channels)

        # `extent` still spans the full 2000 channels, so the y-axis reads in
        # real channel numbers regardless of how many rows the raster was
        # reduced to -- the downsample is a rendering detail, not a relabelling.
        self.im = ax.imshow(
            codes_to_rgb(self.codes, render_rows), origin="lower", aspect="auto",
            interpolation="nearest", extent=(0.0, tr.horizon_s, 0.0, n_ch),
        )
        # The scan trace: the tuned centre hopping over time, drawn as steps so
        # the retunes read as jumps rather than as smooth motion they are not.
        centre = np.where(tr.k_lo >= 0, (tr.k_lo + tr.k_hi) / 2.0, np.nan)
        # Thin and translucent on purpose: at ~900 steps the retune jumps are a
        # dense picket fence, and a heavier line buries the waterfall under it.
        self.trace_line, = ax.plot([], [], color=C_TRACE, lw=0.6, alpha=0.35,
                                   drawstyle="steps-post", zorder=3)
        self._trace_x, self._trace_y = tr.t_end, centre
        self.cursor = ax.axvline(0.0, color="#ff5f6d", lw=1.2, zorder=4)

        self.reason_txt = ax.text(
            0.012, 0.965, "", transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1c2230", ec="#ff5f6d", alpha=0.92),
            zorder=5,
        )
        self.counter_txt = ax.text(
            0.988, 0.03, "", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, family="monospace", color="#d8e2ef",
            bbox=dict(boxstyle="round,pad=0.35", fc="#11161f", ec="#3c4657", alpha=0.9),
            zorder=5,
        )
        ax.set_ylabel("channel")
        ax.set_xlabel("mission time (s)")
        ax.set_title(f"{tr.run_id}   [{tr.policy}]", fontsize=10, loc="left")
        self.painted = 0

    def artists(self):
        return (self.im, self.trace_line, self.cursor,
                self.reason_txt, self.counter_txt)

    def draw_upto(self, upto: int):
        upto = max(0, min(upto, self.tr.n))
        if upto < self.painted:      # a loop restart: repaint from scratch
            self.codes[...] = CELL_BG
            self.painted = 0
        # Only the NEW steps are painted; the raster is cumulative, so a frame
        # costs O(new steps) rather than O(all steps so far).
        paint_range(self.tr, self.codes, self.col0, self.col1,
                    self.painted, upto)
        self.painted = upto
        self.im.set_data(codes_to_rgb(self.codes, self.render_rows))

        self.trace_line.set_data(self._trace_x[:upto], self._trace_y[:upto])
        t = float(self.tr.t_end[upto - 1]) if upto else 0.0
        self.cursor.set_xdata([t, t])
        # THE explainability line: why this action, at this instant.
        why = self.tr.reason[upto - 1] if upto else ""
        span = ("sleep" if not upto or self.tr.k_lo[upto - 1] < 0
                else f"ch {self.tr.k_lo[upto-1]}-{self.tr.k_hi[upto-1]-1}")
        self.reason_txt.set_text(
            f"t={t:6.2f}s  {span}\nwhy: {why or 'n/a'}"
        )
        self.counter_txt.set_text(counter_text(self.tr, upto))
        return self.artists()


def _style(fig, axes):
    fig.patch.set_facecolor("#0b0e14")
    for ax in axes:
        ax.set_facecolor(C_BG)
        for s in ax.spines.values():
            s.set_color("#39404f")
        ax.tick_params(colors="#9fb0c6", labelsize=8)
        ax.xaxis.label.set_color("#9fb0c6")
        ax.yaxis.label.set_color("#9fb0c6")
        ax.title.set_color("#e6edf6")


def render(traces: list[Trace], out: Path, fmt: str = "gif",
           frames: int = DEFAULT_FRAMES, fps: int = DEFAULT_FPS,
           time_bins: int = TIME_BINS, dpi: "int | None" = None) -> "Path | None":
    """Render one or two traces.  Returns the written path, or None if it fell
    back to ASCII.

    `fmt`: `gif` (PillowWriter -- always available with matplotlib), `mp4`
    (FFMpegWriter, falls back to gif when ffmpeg is absent), or `png` (the final
    frame only, which is the fastest artefact and demos fine on a slide).
    """
    plt, animation = _load_mpl()
    if plt is None:
        print("[dashboard] matplotlib unavailable -- ASCII waterfall:\n",
              file=sys.stderr)
        for tr in traces:
            print(ascii_waterfall(tr))
        return None

    n = len(traces)
    fig, axs = plt.subplots(n, 1, figsize=(12.0, 4.0 * n + 0.6), squeeze=False,
                            sharex=True)
    axes = [axs[i][0] for i in range(n)]
    _style(fig, axes)
    panels = [_Panel(ax, tr, time_bins) for ax, tr in zip(axes, traces)]

    title = ("scan waterfall -- " + traces[0].run_id if n == 1
             else "same seed, side by side:  " + "   vs   ".join(t.policy for t in traces))
    fig.suptitle(title, color="#e6edf6", fontsize=12)
    _legend(fig, plt)
    # Leave a strip top and bottom for the title and the legend respectively.
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "png":
        for p in panels:
            p.draw_upto(p.tr.n)
        fig.savefig(out, dpi=dpi or DPI_STILL, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out

    n_steps = max(t.n for t in traces)
    frames = max(2, min(frames, n_steps))
    marks = np.unique(np.linspace(1, n_steps, frames).astype(int))

    def update(k):
        arts = []
        for p in panels:
            arts.extend(p.draw_upto(int(k)))
        return arts

    fig.set_dpi(dpi or DPI_ANIM)
    anim = animation.FuncAnimation(fig, update, frames=marks, interval=1000 // fps,
                                   blit=False, repeat=False)
    writer, actual = _writer(animation, fmt, fps)
    if writer is None:  # no encoder at all -> a PNG still demos
        for p in panels:
            p.draw_upto(p.tr.n)
        png = out.with_suffix(".png")
        fig.savefig(png, dpi=dpi or DPI_STILL, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[dashboard] no animation writer available; wrote {png}", file=sys.stderr)
        return png
    # The EXTENSION must follow the writer that was actually selected, not the
    # format that was asked for.  `--format mp4` on a machine with no ffmpeg
    # falls back to PillowWriter, and Pillow raises `unknown file extension:
    # .mp4` -- so the fallback used to fail louder than having no writer at all.
    out = out.with_suffix("." + actual)
    anim.save(str(out), writer=writer, savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    return out


def _writer(animation, fmt: str, fps: int):
    """`(writer, actual_format)`.  `actual_format` may differ from `fmt`.

    Returning the format alongside the writer is what keeps the file extension
    honest when the requested encoder is unavailable -- see `render`.
    """
    if fmt == "mp4":
        try:
            if animation.FFMpegWriter.isAvailable():
                return animation.FFMpegWriter(fps=fps, bitrate=2400), "mp4"
        except Exception:
            pass
        print("[dashboard] ffmpeg not found; falling back to GIF", file=sys.stderr)
    try:
        return animation.PillowWriter(fps=fps), "gif"
    except Exception:
        return None, fmt


def _legend(fig, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=C_SCAN, label="channel inside a dwell"),
        Patch(facecolor=C_DET, label="detection reported"),
        Line2D([], [], color=C_TRACE, label="tuned centre (scan trace)"),
        Line2D([], [], color="#ff5f6d", label="now"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", fontsize=8, ncol=4,
                     frameon=False, bbox_to_anchor=(0.5, 0.002))
    for t in leg.get_texts():
        t.set_color("#c8d4e3")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _prepare(run_id: str, args) -> Trace:
    tr = load_trace(run_id, Path(args.steps_dir), horizon_s=args.horizon)
    tr.summary = load_summary(run_id, Path(args.results_dir))
    return tr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m app.dashboard",
        description="Waterfall + scan trace + why-it-moved, from results/steps/*.csv.",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", metavar="RUN_ID", help="render one run")
    g.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                   help="two runs stacked -- use the SAME seed")
    g.add_argument("--list", action="store_true", help="list available run ids")
    ap.add_argument("--format", choices=("gif", "mp4", "png"), default="gif")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--dpi", type=int, default=None,
                    help=f"default {DPI_STILL} for png, {DPI_ANIM} for gif/mp4")
    ap.add_argument("--steps-dir", default=str(STEPS_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--ascii", action="store_true",
                    help="print the terminal waterfall instead of rendering")
    args = ap.parse_args(argv)

    if args.list:
        found = sorted(p.stem for p in Path(args.steps_dir).glob("*.csv"))
        if not found:
            print(f"no traces in {args.steps_dir}; run: python -m eval.runner --all",
                  file=sys.stderr)
            return 2
        print("\n".join(found))
        return 0

    try:
        ids = [args.run] if args.run else list(args.compare)
        traces = [_prepare(r, args) for r in ids]
    except (FileNotFoundError, ValueError) as exc:
        print(f"[dashboard] {exc}", file=sys.stderr)
        return 2

    if args.ascii:
        for tr in traces:
            print(ascii_waterfall(tr))
        return 0

    stem = "_vs_".join(t.run_id for t in traces) if len(traces) > 1 else traces[0].run_id
    out = Path(args.out) if args.out else FIG_DIR / f"waterfall_{stem}.{args.format}"
    written = render(traces, out, fmt=args.format, frames=args.frames, fps=args.fps,
                     dpi=args.dpi)
    if written is not None:
        print(f"[dashboard] wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
