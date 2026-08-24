"""FROZEN AT PHASE 0 -- YAML loading, validation, and config hashing.

Validation is deliberately strict and noisy: a scenario that silently loads with a
wrong constant is far more expensive than one that refuses to load at all.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from sim.contract import ChannelGrid, Mission

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

SCENARIOS: tuple[str, ...] = ("sparse", "dense", "agile")

# Candidate action sets.  Frozen because the evaluator, the policy and the
# marginal-Pd lookup table must all agree on them.
BW_CANDIDATES_MHZ: tuple[int, ...] = (1, 2, 5, 10, 20)
DWELL_CANDIDATES_MS: tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100, 200)
SLEEP_CANDIDATES_MS: tuple[float, ...] = (10, 50, 200)


class ConfigError(ValueError):
    """A config that would silently produce meaningless results."""


def _coerce_numeric(node):
    """Recursively turn numeric-looking strings into floats.

    YAML 1.1 (which PyYAML implements) only recognises an exponent as a float if
    it carries a sign and a decimal point -- so `2.000e9` and `1.0e6` load as
    *strings*.  Writing `2.0e+9` everywhere would work but makes the configs
    unpleasant to read and easy to get subtly wrong, so we coerce here instead.
    """
    if isinstance(node, dict):
        return {k: _coerce_numeric(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_numeric(v) for v in node]
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            return node
    return node


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"missing required key {key!r} in {where}")
    return d[key]


def load_config(scenario: str | dict | Path) -> dict:
    """Load and validate a scenario config.  Accepts a name, a path, or a dict."""
    if isinstance(scenario, dict):
        cfg = json.loads(json.dumps(scenario))  # deep copy, plain types only
    else:
        p = Path(scenario)
        if not p.suffix:
            p = CONFIG_DIR / f"{scenario}.yaml"
        if not p.exists():
            raise ConfigError(f"no such config: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    cfg = _coerce_numeric(cfg)
    # `name` and `score_mode` are legitimately strings; _coerce_numeric leaves
    # them alone because they do not parse as floats.
    validate_config(cfg)
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail loudly on anything that would make results meaningless."""
    name = cfg.get("name", "<unnamed>")
    for key in ("horizon_s", "grid", "receiver", "energy", "mission", "emitters", "agent"):
        _require(cfg, key, name)

    g = cfg["grid"]
    if g["channel_bw_hz"] <= 0 or g["n_channels"] < 1:
        raise ConfigError(f"{name}: degenerate grid")

    rx, en = cfg["receiver"], cfg["energy"]
    if not (0.0 < rx["pfa"] < 1.0):
        raise ConfigError(f"{name}: pfa must be in (0,1), got {rx['pfa']}")

    # The energy model and the timing model are derived from the same physical
    # constants.  If they ever drift apart, the headline number is a lie.
    expected_lf = en["L_d_w"] / rx["f_slew_hz_per_s"]
    if abs(en["L_f_j_per_hz"] - expected_lf) > 1e-15:
        raise ConfigError(
            f"{name}: L_f_j_per_hz={en['L_f_j_per_hz']!r} is inconsistent with "
            f"L_d_w/f_slew_hz_per_s={expected_lf!r}.  The energy model and the "
            f"timing model must not drift apart."
        )
    if en["L_0_j"] < en["L_d_w"] * rx["t_settle_s"] - 1e-15:
        raise ConfigError(
            f"{name}: L_0_j is smaller than the settling energy L_d*t_settle"
        )

    m = cfg["mission"]
    n = g["n_channels"]
    for band in m["priority_bands"]:
        if not (0 <= band["ch_lo"] < band["ch_hi"] <= n):
            raise ConfigError(f"{name}: priority band {band} outside [0,{n})")
    for ch in m.get("watch_list", []) or []:
        if not (0 <= ch < n):
            raise ConfigError(f"{name}: watch_list channel {ch} outside [0,{n})")
    if set(m["weights"]) != set(m["deadlines_s"]):
        raise ConfigError(f"{name}: mission weights and deadlines cover different priorities")

    for spec in cfg["emitters"]:
        lo, hi = spec["channel_range"]
        if not (0 <= lo < hi <= n):
            raise ConfigError(f"{name}: emitter channel_range {spec['channel_range']} outside grid")
        if spec["mean_on_s"] <= 0 or spec["mean_off_s"] <= 0:
            raise ConfigError(f"{name}: emitter sojourn times must be positive")
        if spec["snr_db"][0] > spec["snr_db"][1]:
            raise ConfigError(f"{name}: emitter snr_db range is inverted")

    a = cfg["agent"]
    for bw in a["bw_candidates_mhz"]:
        if bw * 1e6 % g["channel_bw_hz"] != 0:
            raise ConfigError(f"{name}: bw candidate {bw} MHz is not a whole number of channels")
    if not (0.0 < a["prior_pi_on"] < 1.0):
        raise ConfigError(f"{name}: prior_pi_on must be in (0,1)")


def config_hash(cfg: dict) -> str:
    """Stable hash of the canonicalised config, written into every results row."""
    d = {k: v for k, v in cfg.items() if k != "config_hash"}
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_grid(cfg: dict) -> ChannelGrid:
    g = cfg["grid"]
    return ChannelGrid(
        f_start_hz=float(g["f_start_hz"]),
        n_channels=int(g["n_channels"]),
        channel_bw_hz=float(g["channel_bw_hz"]),
    )


def build_mission(cfg: dict) -> Mission:
    """Expand priority bands into per-channel arrays.

    Later bands override earlier ones, so the catch-all priority-3 band is listed
    LAST in the YAML and is applied first here (reversed), letting specific bands
    win.  Untasked channels keep priority 0 and weight 0.
    """
    n = int(cfg["grid"]["n_channels"])
    m = cfg["mission"]
    weights = {int(k): float(v) for k, v in m["weights"].items()}
    deadlines = {int(k): float(v) for k, v in m["deadlines_s"].items()}

    priority = np.zeros(n, dtype=np.int32)
    # Apply broadest (highest priority number = least specific) first.
    for band in sorted(m["priority_bands"], key=lambda b: -int(b["priority"])):
        priority[band["ch_lo"]: band["ch_hi"]] = int(band["priority"])

    w = np.zeros(n, dtype=np.float64)
    for prio, wp in weights.items():
        w[priority == prio] = wp

    return Mission(
        priority=priority,
        w=w,
        deadlines_s=deadlines,
        watch_list=np.asarray(m.get("watch_list") or [], dtype=np.int32),
        watch_deadline_s=float(m.get("watch_deadline_s", 0.3)),
    )
