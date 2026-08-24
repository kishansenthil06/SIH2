"""Burst table -> per-channel signal power.  Pure functions, no state, no RNG.

The one idea in this module is **time-averaging**:

    rho[c] = ( sum over bursts covering c of  overlap_seconds * linear_snr ) / (t1 - t0)

Averaging (rather than "was it on at the midpoint?") is what makes two behaviours
the demo depends on *emergent* instead of hand-coded:

* a short dwell that lands in an OFF gap sees `rho = 0`, hence `P_d = P_fa` --
  a genuine miss, produced by the physics rather than by a rule;
* a long dwell over an intermittent emitter loses SNR linearly in duty cycle but
  gains samples linearly in dwell, and `P_d` depends on `sqrt(N)*s`, so
  `sqrt(N)` outruns the loss.  Long dwells therefore catch intermittent emitters
  -- the exact trade-off the index policy has to discover.

Everything here is vectorised over the burst table.  The table is small
(hundreds of rows) but `window_rho` runs on every single step.
"""
from __future__ import annotations

import numpy as np


def window_overlap_s(bursts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Seconds each burst overlaps `[t0, t1)`.  Zero for non-overlapping bursts."""
    if bursts.size == 0:
        return np.empty(0, dtype=np.float64)
    lo = np.maximum(bursts["t_on"], t0)
    hi = np.minimum(bursts["t_off"], t1)
    return np.maximum(hi - lo, 0.0)


def window_rho(
    bursts: np.ndarray, t0: float, t1: float, n_channels: int
) -> np.ndarray:
    """Time-averaged LINEAR SNR per channel over `[t0, t1)`.

    Returned in linear units, never dB, because a silent channel is exactly 0.0
    here and `-inf` dB would poison every downstream expression.  The receiver
    converts to dB only for the *reported* estimate, where 0 never appears.
    """
    rho = np.zeros(n_channels, dtype=np.float64)
    dur = t1 - t0
    if bursts.size == 0 or dur <= 0.0:
        return rho

    ov = window_overlap_s(bursts, t0, t1)
    sel = ov > 0.0
    if not np.any(sel):
        return rho

    lin = 10.0 ** (bursts["snr_db"][sel] / 10.0)
    contrib = ov[sel] * lin
    lo = np.clip(bursts["ch_lo"][sel].astype(np.int64), 0, n_channels)
    hi = np.clip(bursts["ch_hi"][sel].astype(np.int64), 0, n_channels)

    # Difference-array trick: an O(1) update per burst for a contiguous channel
    # block, then one cumsum.  Keeps this O(n_bursts + n_channels) with no
    # Python loop over the 200 channels (DESIGN.md section 10).
    diff = np.zeros(n_channels + 1, dtype=np.float64)
    np.add.at(diff, lo, contrib)
    np.add.at(diff, hi, -contrib)
    np.cumsum(diff[:-1], out=rho)
    # cumsum of exactly-cancelling +v/-v pairs can leave -1e-19 residue.
    np.maximum(rho, 0.0, out=rho)
    return rho / dur


def _bin_edges(bursts: np.ndarray, dt_s: float, n_bins: int):
    """Half-open bin index range `[i_lo, i_hi)` touched by each burst."""
    i_lo = np.floor(bursts["t_on"] / dt_s).astype(np.int64)
    i_hi = np.ceil(bursts["t_off"] / dt_s).astype(np.int64)
    np.clip(i_lo, 0, n_bins, out=i_lo)
    np.clip(i_hi, 0, n_bins, out=i_hi)
    return i_lo, i_hi


def _raster(bursts: np.ndarray, dt_s: float, n_bins: int, n_channels: int,
            values: np.ndarray, dtype) -> np.ndarray:
    """2-D difference-array rasteriser: add `values[k]` over each burst's rectangle."""
    out = np.zeros((n_bins, n_channels), dtype=dtype)
    if bursts.size == 0 or n_bins <= 0:
        return out
    i_lo, i_hi = _bin_edges(bursts, dt_s, n_bins)
    c_lo = np.clip(bursts["ch_lo"].astype(np.int64), 0, n_channels)
    c_hi = np.clip(bursts["ch_hi"].astype(np.int64), 0, n_channels)
    sel = (i_hi > i_lo) & (c_hi > c_lo)
    if not np.any(sel):
        return out
    i_lo, i_hi, c_lo, c_hi, v = i_lo[sel], i_hi[sel], c_lo[sel], c_hi[sel], values[sel]

    diff = np.zeros((n_bins + 1, n_channels + 1), dtype=np.float64)
    np.add.at(diff, (i_lo, c_lo), v)
    np.add.at(diff, (i_lo, c_hi), -v)
    np.add.at(diff, (i_hi, c_lo), -v)
    np.add.at(diff, (i_hi, c_hi), v)
    acc = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:n_bins, :n_channels]
    out[:] = acc
    return out


def rasterize_occupancy(
    bursts: np.ndarray, dt_s: float, n_bins: int, n_channels: int
) -> np.ndarray:
    """(n_bins, n_channels) bool: is ANY emitter radiating in this cell?

    Counts are accumulated as integers-in-float and compared with a 0.5
    threshold, so exactly-cancelling overlaps can never produce a phantom True.
    """
    ones = np.ones(bursts.size, dtype=np.float64)
    acc = _raster(bursts, dt_s, n_bins, n_channels, ones, np.float64)
    return acc > 0.5


def rasterize_power(
    bursts: np.ndarray, dt_s: float, n_bins: int, n_channels: int
) -> np.ndarray:
    """(n_bins, n_channels) float32: summed LINEAR SNR.  Used by the oracle only."""
    if bursts.size == 0:
        return np.zeros((max(n_bins, 0), n_channels), dtype=np.float32)
    lin = 10.0 ** (bursts["snr_db"] / 10.0)
    acc = _raster(bursts, dt_s, n_bins, n_channels, lin, np.float64)
    np.maximum(acc, 0.0, out=acc)
    return acc.astype(np.float32)


def bursts_overlapping(bursts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Sub-table of bursts overlapping `[t0, t1)`.  Used by `eval/metrics.py`."""
    if bursts.size == 0:
        return bursts
    return bursts[window_overlap_s(bursts, t0, t1) > 0.0]
