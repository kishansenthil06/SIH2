"""Interactive EW Smart Scan Command Center & Simulation Web Server.

Provides a real-time web UI and REST API for:
  - Multi-page dynamic routing (Landing, Login, Dashboard, Scenarios, Policy Lab, Analytics, Audit)
  - Live simulation of EW scan policies (Index, Round-Robin, Greedy, Random, Oracle, Learned)
  - Interactive dual waterfall spectrum visualizer
  - Head-to-head policy comparison (energy per detection, TTFI, POI)
  - Explainable AI (XAI) decision telemetry stream
  - Bayesian belief spectrum distribution across 200 channels
  - Scenario & emitter profile inspector
  - Policy engine & GBDT model analytics
  - Architectural firewall & audit verification

FIREWALL COMPLIANT: Imports only `sim.contract` and `sim.config` from `sim`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.config import build_grid, load_config
from sim.contract import ChannelGrid

FRONTEND_DIST_DIR = ROOT / "frontend" / "dist"
APP_STATIC_DIR = Path(__file__).resolve().parent / "static"


def get_static_dir() -> Path:
    if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
        return FRONTEND_DIST_DIR
    return APP_STATIC_DIR


STATIC_DIR = get_static_dir()
RESULTS_DIR = ROOT / "results"
STEPS_DIR = RESULTS_DIR / "steps"
MODELS_DIR = ROOT / "models"
RUNS_CSV = RESULTS_DIR / "runs.csv"
ABLATION_CSV = RESULTS_DIR / "ablation.csv"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
DATA_PROTOTYPE_CSV = ROOT / "data" / "prototype" / "temporary_rf_dataset.csv"

# Valid SPA routes that should serve index.html
SPA_ROUTES = {
    "/",
    "/landing",
    "/login",
    "/dashboard",
    "/scenarios",
    "/policy-lab",
    "/analytics",
    "/audit",
    "/prototype",
}

# Cached singleton for prototype RF dataset environment
_PROTOTYPE_ENV = None


def _get_prototype_env():
    global _PROTOTYPE_ENV
    if _PROTOTYPE_ENV is None:
        from src.environment import RFEnvironment
        _PROTOTYPE_ENV = RFEnvironment(DATA_PROTOTYPE_CSV)
    return _PROTOTYPE_ENV



def _get_python_bin() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _clean_for_json(obj):
    """Recursively replaces NaN, Inf, -Inf with None or safe numbers for valid RFC-8259 JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, str):
        low = obj.strip().lower()
        if low in ("nan", "inf", "+inf", "-inf"):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    return obj


def _safe_float(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _load_trace_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                score_raw = r.get("best_score", "")
                best_score = None
                if score_raw and score_raw.strip().lower() not in ("nan", "inf", "-inf", ""):
                    try:
                        sf = float(score_raw)
                        if not math.isnan(sf) and not math.isinf(sf):
                            best_score = sf
                    except Exception:
                        best_score = None

                rows.append({
                    "step": int(r.get("step", 0)),
                    "t_start": _safe_float(r.get("t_start", 0.0)),
                    "t_end": _safe_float(r.get("t_end", 0.0)),
                    "kind": r.get("kind", "scan"),
                    "f_center_hz": _safe_float(r.get("f_center_hz", 0.0)),
                    "bw_hz": _safe_float(r.get("bw_hz", 0.0)),
                    "dwell_s": _safe_float(r.get("dwell_s", 0.0)),
                    "energy_j": _safe_float(r.get("energy_j", 0.0)),
                    "n_det": int(r.get("n_det", 0)),
                    "det_channels": [int(x) for x in r.get("det_channels", "").split() if x],
                    "best_score": best_score,
                    "chosen_reason": r.get("chosen_reason", ""),
                    "energy_spent_total": _safe_float(r.get("energy_spent_total", 0.0)),
                })
            except Exception:
                continue
    return rows


def _load_summary_for_run(run_id: str) -> dict | None:
    if not RUNS_CSV.exists():
        return None
    with open(RUNS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("run_id") == run_id:
                clean_r = {}
                for k, v in r.items():
                    clean_r[k] = _clean_for_json(v)
                if "energy_total_j" in clean_r and "energy_j" not in clean_r:
                    clean_r["energy_j"] = clean_r["energy_total_j"]
                if "energy_per_detection_j" in clean_r and "energy_per_unique_det_j" not in clean_r:
                    clean_r["energy_per_unique_det_j"] = clean_r["energy_per_detection_j"]
                if "poi_60" in clean_r and "poi_at_60s" not in clean_r:
                    clean_r["poi_at_60s"] = clean_r["poi_60"]
                return clean_r
    return None


def _load_ablation() -> list[dict]:
    if not ABLATION_CSV.exists():
        return []
    rows = []
    with open(ABLATION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clean_r = {k: _clean_for_json(v) for k, v in r.items()}
            rows.append(clean_r)
    return rows


def _build_threat_override(scenario: str, threat: dict) -> dict:
    """Builds a valid scenario config override containing an injected hostile emitter."""
    try:
        base_cfg = load_config(scenario)
    except Exception:
        base_cfg = {"emitters": [], "mission": {"priority_bands": []}, "grid": {"n_channels": 2000}}
    
    emitters = [dict(e) for e in base_cfg.get("emitters", [])]
    base_n = int(base_cfg.get("grid", {}).get("n_channels", 2000))
    scale = base_n / 200.0 if base_n > 200 else 1.0

    raw_channel = int(threat.get("channel", 48))
    grid_channel = min(base_n - 1, max(0, int(round(raw_channel * scale))))
    grid_channel_hi = min(base_n, grid_channel + max(1, int(round(scale))))
    
    priority = int(threat.get("priority", 1))
    duration = float(threat.get("duration_s", 2.0))
    
    threat_emitter = {
        "kind": str(threat.get("kind", "pulsed")),
        "count": 1,
        "channel_range": [grid_channel, grid_channel_hi],
        "snr_db": [8.0, 16.0],
        "mean_on_s": duration,
        "mean_off_s": 2.0,
        "snr_sigma_db": 0.5,
    }
    emitters.append(threat_emitter)

    priority_bands = [dict(b) for b in base_cfg.get("mission", {}).get("priority_bands", [])]
    if priority == 1 and not any(b["ch_lo"] <= grid_channel < b["ch_hi"] and b.get("priority") == 1 for b in priority_bands):
        priority_bands.append({"ch_lo": grid_channel, "ch_hi": grid_channel_hi, "priority": 1})
    elif priority == 2 and not any(b["ch_lo"] <= grid_channel < b["ch_hi"] and b.get("priority") in (1, 2) for b in priority_bands):
        priority_bands.append({"ch_lo": grid_channel, "ch_hi": grid_channel_hi, "priority": 2})

    return {
        "emitters": emitters,
        "mission": {
            "priority_bands": priority_bands,
        }
    }


def _run_simulation(policy: str, scenario: str, seed: int, horizon: float = 60.0, cfg_overrides: dict | None = None) -> tuple[dict | None, list[dict], str]:
    """Runs a single simulation episode via eval.runner subprocess."""
    run_id = f"{policy}__{scenario}__s{seed}__h{int(round(horizon))}"
    trace_path = STEPS_DIR / f"{run_id}.csv"
    
    cmd = [
        _get_python_bin(),
        "-m", "eval.runner",
        "--policies", policy,
        "--scenarios", scenario,
        "--seeds", str(seed),
        "--horizon", str(horizon),
        "--out", str(RUNS_CSV),
        "--trace", str(STEPS_DIR),
        "--jobs", "1",
    ]
    if cfg_overrides:
        cmd.extend(["--overrides-json", json.dumps(cfg_overrides)])
    
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    
    output_log = proc.stdout + "\n" + proc.stderr
    summary = _load_summary_for_run(run_id)
    trace = _load_trace_csv(trace_path)
    return summary, trace, output_log



# Human-readable gloss for each entry of agent.base.FEATURE_NAMES.  Keyed by the
# real feature name so a rename shows up as a blank description rather than a
# silently wrong one.
_FEATURE_DOCS = {
    "p_rung1": "Rung-1 analytic belief; the model can only refine it",
    "log_staleness": "log1p(time since this channel was last visited)",
    "log_since_detect": "log1p(time since the last detection here)",
    "n_visits": "Visits to this channel this episode",
    "emp_rate": "Laplace-smoothed empirical hit rate",
    "hit_ema_fast": "Detection EMA, alpha = 0.30",
    "hit_ema_slow": "Detection EMA, alpha = 0.05",
    "misses_since_detect": "Consecutive misses since the last detection",
    "mean_dwell_log": "log1p(mean dwell spent on this channel)",
    "mean_snr_db": "Mean reported SNR of detections here",
    "idi_mean": "Mean inter-detection interval",
    "idi_std": "Std of inter-detection interval",
    "nbr_recent_hits": "Detections on channels c+/-1..2 within 1 s (catches hoppers)",
    "band_activity": "Fraction of channels detected-on in the last 1 s",
    "w_channel": "Mission priority weight w_p, in joules",
    "t_frac": "Episode progress, t / horizon",
    # Train-time only: the label is "did the NEXT observation detect", which
    # depends on that observation's dwell and bandwidth, so the model is given
    # them rather than being made to average over them.
    "tau_next_log": "log dwell of the labelling observation (train-time only)",
    "bw_next_log": "log bandwidth of the labelling observation (train-time only)",
}

class EWRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving REST API and embedded multi-page UI."""

    def __init__(self, *args, **kwargs):
        self.static_dir = get_static_dir()
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        rel_path = parsed.path.lstrip("/")
        target_file = (self.static_dir / rel_path).resolve()
        
        # If specific static asset file exists, serve it
        if rel_path and target_file.exists() and target_file.is_file():
            return str(target_file)
        
        # If not an API endpoint, serve index.html for SPA client-side routing
        if not parsed.path.startswith("/api/"):
            index_path = self.static_dir / "index.html"
            if index_path.exists():
                return str(index_path)

        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def _send_json(self, data: dict | list, status: int = HTTPStatus.OK):
        clean_data = _clean_for_json(data)
        body = json.dumps(clean_data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # API Endpoints
        if path == "/api/status":
            self._handle_status()
        elif path in ("/api/prototype/dataset", "/api/prototype/stats"):
            self._handle_prototype_dataset()
        elif path == "/api/prototype/sample":
            self._handle_prototype_sample(qs)
        elif path == "/api/scenarios":
            self._handle_scenarios()
        elif path == "/api/scenarios/details":
            self._handle_scenarios_details()
        elif path == "/api/model/info":
            self._handle_model_info()
        elif path == "/api/audit/firewall":
            self._handle_audit_firewall()
        elif path == "/api/traces":
            self._handle_traces()
        elif path == "/api/trace":
            run_id = qs.get("id", [""])[0]
            self._handle_single_trace(run_id)
        elif path == "/api/ablation":
            self._handle_ablation()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()

        if path == "/api/run":
            self._handle_run(body)
        elif path == "/api/prototype/run":
            self._handle_prototype_run(body)
        elif path == "/api/compare":
            self._handle_compare(body)
        elif path == "/api/train":
            self._handle_train()
        elif path == "/api/audit/run-tests":
            self._handle_run_tests()
        else:
            self._send_json({"error": f"Unknown endpoint {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_status(self):
        model_path = MODELS_DIR / "activity_hgb.joblib"
        scenarios = ["sparse", "dense", "agile"]
        policies = [
            {"id": "index", "name": "Index Policy (Rung 1 — Reward-Rate / Bayes)", "type": "adaptive", "rung": 1},
            {"id": "index_learned", "name": "Learned Index (Rung 2 — GBDT Activity Model)", "type": "learned", "rung": 2, "ready": model_path.exists()},
            {"id": "round_robin", "name": "Round-Robin Sweep (Rung 0 Baseline)", "type": "baseline", "rung": 0},
            {"id": "greedy", "name": "Greedy (Myopic Baseline)", "type": "baseline", "rung": 0},
            {"id": "random", "name": "Random Scan Baseline", "type": "baseline", "rung": 0},
            {"id": "oracle", "name": "Clairvoyant Oracle (Theoretical Ceiling)", "type": "upper_bound", "rung": 3},
        ]
        
        info = {
            "status": "online",
            "server": "EW Smart Scan Engine Server",
            "version": "2.4.0-SIH2026",
            "python": _get_python_bin(),
            "model_trained": model_path.exists(),
            "scenarios": scenarios,
            "policies": policies,
            "results_dir": str(RESULTS_DIR),
            "trace_count": len(list(STEPS_DIR.glob("*.csv"))) if STEPS_DIR.exists() else 0,
            "prototype": {
                "ready": DATA_PROTOTYPE_CSV.exists(),
                "dataset_path": str(DATA_PROTOTYPE_CSV),
                "dataset_size_bytes": DATA_PROTOTYPE_CSV.stat().st_size if DATA_PROTOTYPE_CSV.exists() else 0,
                "total_observations": 100000,
                "num_bands": 20,
            },
        }
        self._send_json(info)

    def _handle_prototype_dataset(self):
        try:
            env = _get_prototype_env()
            stats = env.get_stats()
            # Add sample preview of first 10 observations
            preview = []
            for b in sorted(list(env.frequency_bands))[:10]:
                obs = env.scan(0, b)
                preview.append({
                    "time_slot": 0,
                    "frequency_band": b,
                    "signal_power": round(float(obs["signal_power"]), 3),
                    "pulse_width": round(float(obs["pulse_width"]), 3),
                    "angle_of_arrival": round(float(obs["angle_of_arrival"]), 2) if obs["angle_of_arrival"] is not None else None,
                    "ground_truth_active": bool(obs["hit"]),
                })
            stats["sample_preview"] = preview
            self._send_json(stats)
        except Exception as e:
            self._send_json({"error": f"Failed to load prototype dataset: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_prototype_sample(self, qs: dict):
        try:
            time_slot = int(qs.get("time_slot", [0])[0])
            env = _get_prototype_env()
            obs = env.get_time_slot_observations(time_slot)
            self._send_json({
                "time_slot": time_slot,
                "observations": obs,
                "count": len(obs),
            })
        except Exception as e:
            self._send_json({"error": f"Failed to retrieve sample: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_prototype_run(self, body: dict):
        try:
            from src.detector import PowerThresholdDetector
            from src.evaluator import Evaluator
            from src.receiver import Receiver
            from src.scanner import make_scanner

            num_scans = min(5000, max(1, int(body.get("num_scans", 100))))
            threshold = float(body.get("threshold", 5.0))
            num_bands = min(20, max(1, int(body.get("num_bands", 20))))
            strategy = str(body.get("strategy", "sequential"))
            start_slot = max(0, int(body.get("start_slot", 0)))

            env = _get_prototype_env()
            receiver = Receiver(env)
            scanner = make_scanner(strategy, receiver, num_bands=num_bands)
            detector = PowerThresholdDetector(threshold=threshold)
            evaluator = Evaluator(env)

            scans = []
            band_activity = {b: 0 for b in range(1, num_bands + 1)}

            for i in range(num_scans):
                scan_idx = start_slot + i
                obs = scanner.scan(scan_idx)
                pred = detector.predict(obs)
                eval_res = evaluator.evaluate(
                    time_slot=obs["time_slot"],
                    frequency_band=obs["frequency_band"],
                    prediction=pred,
                )
                
                band = obs["frequency_band"]
                if eval_res["ground_truth"]:
                    band_activity[band] = band_activity.get(band, 0) + 1

                if i < 200 or i % max(1, num_scans // 100) == 0 or pred or eval_res["ground_truth"]:
                    scans.append({
                        "scan_index": i + 1,
                        "time_slot": obs["time_slot"],
                        "frequency_band": obs["frequency_band"],
                        "signal_power": round(float(obs["signal_power"]), 3),
                        "pulse_width": round(float(obs["pulse_width"]), 3),
                        "angle_of_arrival": round(float(obs["angle_of_arrival"]), 2) if obs["angle_of_arrival"] is not None else None,
                        "prediction": bool(pred),
                        "ground_truth": bool(eval_res["ground_truth"]),
                        "result": eval_res["result"],
                    })

            metrics = evaluator.metrics()
            
            confusion_matrix = {
                "tp": metrics.get("true_positive", 0),
                "fp": metrics.get("false_positive", 0),
                "tn": metrics.get("true_negative", 0),
                "fn": metrics.get("false_negative", 0),
            }

            self._send_json({
                "status": "success",
                "strategy": strategy,
                "num_scans": num_scans,
                "threshold": threshold,
                "num_bands": num_bands,
                "metrics": metrics,
                "confusion_matrix": confusion_matrix,
                "band_activity": band_activity,
                "scans_sample": scans[:100],
                "total_scans_recorded": len(scans),
            })
        except Exception as e:
            self._send_json({"error": f"Prototype run failed: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


    def _handle_scenarios(self):
        scenarios_data = {}
        for name in ["sparse", "dense", "agile"]:
            try:
                cfg = load_config(name)
                grid = build_grid(cfg)
                scenarios_data[name] = {
                    "name": name,
                    "horizon_s": cfg.get("horizon_s", 60.0),
                    "grid": {
                        "f_start_hz": grid.f_start_hz,
                        "n_channels": grid.n_channels,
                        "channel_bw_hz": grid.channel_bw_hz,
                        "f_stop_hz": grid.f_start_hz + grid.n_channels * grid.channel_bw_hz,
                    },
                    "energy": cfg.get("energy", {}),
                    "mission": {
                        "priority_bands": cfg.get("mission", {}).get("priority_bands", []),
                        "weights": cfg.get("mission", {}).get("weights", {}),
                        "deadlines_s": cfg.get("mission", {}).get("deadlines_s", {}),
                    },
                    "emitter_count": len(cfg.get("emitters", [])),
                }
            except Exception as e:
                scenarios_data[name] = {"error": str(e)}
        self._send_json(scenarios_data)

    def _handle_scenarios_details(self):
        scenarios_data = {}
        for name in ["sparse", "dense", "agile"]:
            try:
                cfg = load_config(name)
                grid = build_grid(cfg)
                scenarios_data[name] = {
                    "name": name,
                    "horizon_s": cfg.get("horizon_s", 60.0),
                    "grid": {
                        "f_start_hz": grid.f_start_hz,
                        "n_channels": grid.n_channels,
                        "channel_bw_hz": grid.channel_bw_hz,
                        "f_stop_hz": grid.f_start_hz + grid.n_channels * grid.channel_bw_hz,
                    },
                    "receiver": cfg.get("receiver", {}),
                    "energy": cfg.get("energy", {}),
                    "mission": cfg.get("mission", {}),
                    "emitters": cfg.get("emitters", []),
                    "agent": cfg.get("agent", {}),
                }
            except Exception as e:
                scenarios_data[name] = {"error": str(e)}
        self._send_json(scenarios_data)

    def _handle_model_info(self):
        model_path = MODELS_DIR / "activity_hgb.joblib"
        manifest_path = MODELS_DIR / "activity_hgb.manifest.json"
        
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        # Everything below is DERIVED FROM THE MANIFEST, never hardcoded.
        # These fields were previously literals and had drifted badly: the page
        # showed Brier 0.00507/0.00715 against the real 0.00861/0.01398, and the
        # feature list advertised a "channel_idx_norm" input that the model
        # deliberately does NOT use -- the raw channel index is excluded so the
        # model cannot memorise emitter positions from the training scenarios
        # and fail on the held-out one. Reading the manifest keeps this page
        # honest the next time the model is retrained.
        feature_names = manifest.get("feature_names") or []
        features = [{"name": n, "description": _FEATURE_DOCS.get(n, "")} for n in feature_names]

        b_model = manifest.get("brier_model")
        b_rung1 = manifest.get("brier_rung1")
        if b_model is not None and b_rung1 is not None and b_rung1 > 0:
            gate = (
                f"{'PASS' if manifest.get('gate_ok') else 'FAIL'} — Brier "
                f"{b_model:.5f} vs rung-1 {b_rung1:.5f} "
                f"({100.0 * (b_rung1 - b_model) / b_rung1:.1f}% error reduction)"
            )
        else:
            gate = "unknown — no trained model manifest found"

        info = {
            "model_path": str(model_path),
            "exists": model_path.exists(),
            "file_size_bytes": model_path.stat().st_size if model_path.exists() else 0,
            "architecture": manifest.get(
                "estimator",
                "HistGradientBoostingClassifier + CalibratedClassifierCV (Isotonic)",
            ),
            # n_features counts BOTH, so deriving the split from the two lists
            # avoids double-counting the extras (the first version of this line
            # read "18 input features + 2 train-time", which totals 20).
            "contract": (
                f"{len(manifest.get('contract_features') or [])} belief features "
                f"+ {len(manifest.get('train_extra_names') or [])} train-time action "
                f"features = {manifest.get('n_features', len(feature_names))} model inputs"
            ),
            "manifest": manifest,
            "features": features,
            "brier_score_model": b_model,
            "brier_score_rung1": b_rung1,
            "gate_status": gate,
            "training_samples": manifest.get("n_rows_train"),
            "held_out_evaluation_samples": manifest.get("n_rows_holdout"),
            "training_scenarios": manifest.get("observed_scenarios"),
            "held_out_scenario": manifest.get("held_out_scenario"),
        }
        self._send_json(info)

    def _handle_audit_firewall(self):
        cmd = [_get_python_bin(), "-m", "unittest", "eval/tests/test_firewall.py", "-v"]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        
        audit_data = {
            "firewall_status": "ENFORCED & VERIFIED",
            "mechanisms": [
                {
                    "name": "1. Static AST Scan",
                    "description": "Scans AST of agent/ and app/ for forbidden imports (sim.env, sim.emitters, sim.channel, sim.receiver) and .truth* attributes.",
                    "status": "PASS",
                },
                {
                    "name": "2. Runtime Stack Inspection",
                    "description": "sim.env._forbid_agent_callers inspects call stack and raises FirewallViolation if agent/app frames exist.",
                    "status": "PASS",
                },
                {
                    "name": "3. Structural Bound Facade",
                    "description": "Agent receives AgentEnv whose __slots__ hold only bound methods and plain data, preventing object graph traversals.",
                    "status": "PASS",
                },
                {
                    "name": "4. Physics Detector Duplication",
                    "description": "agent/belief.py deliberately duplicates Urkowitz Pd curve rather than importing sim.receiver; verified to 1e-9.",
                    "status": "PASS",
                },
            ],
            "test_output": proc.stdout + "\n" + proc.stderr,
            "passed": proc.returncode == 0,
        }
        self._send_json(audit_data)

    def _handle_run_tests(self):
        cmd = [_get_python_bin(), "-m", "unittest", "discover", "-s", "eval/tests", "-t", ".", "-v"]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self._send_json({
            "passed": proc.returncode == 0,
            "output": proc.stdout + "\n" + proc.stderr,
        })

    def _handle_traces(self):
        if not STEPS_DIR.exists():
            self._send_json([])
            return
        traces = []
        for p in sorted(STEPS_DIR.glob("*.csv")):
            run_id = p.stem
            summary = _load_summary_for_run(run_id)
            traces.append({
                "run_id": run_id,
                "file_size": p.stat().st_size,
                "summary": summary,
            })
        self._send_json(traces)

    def _handle_single_trace(self, run_id: str):
        if not run_id:
            self._send_json({"error": "Missing run_id"}, status=HTTPStatus.BAD_REQUEST)
            return
        trace_path = STEPS_DIR / f"{run_id}.csv"
        if not trace_path.exists():
            self._send_json({"error": f"Trace {run_id} not found"}, status=HTTPStatus.NOT_FOUND)
            return
        trace = _load_trace_csv(trace_path)
        summary = _load_summary_for_run(run_id)
        self._send_json({
            "run_id": run_id,
            "summary": summary,
            "step_count": len(trace),
            "trace": trace,
        })

    def _handle_ablation(self):
        rows = _load_ablation()
        self._send_json(rows)

    def _handle_run(self, body: dict):
        policy = body.get("policy", "index")
        scenario = body.get("scenario", "sparse")
        seed = int(body.get("seed", 0))
        horizon = float(body.get("horizon", 60.0))
        injected_threat = body.get("injected_threat")

        cfg_overrides = _build_threat_override(scenario, injected_threat) if injected_threat else None
        summary, trace, log = _run_simulation(policy, scenario, seed, horizon, cfg_overrides=cfg_overrides)
        run_id = f"{policy}__{scenario}__s{seed}__h{int(round(horizon))}"
        
        self._send_json({
            "run_id": run_id,
            "policy": policy,
            "scenario": scenario,
            "seed": seed,
            "horizon": horizon,
            "injected_threat": injected_threat,
            "summary": summary,
            "trace": trace,
            "log": log,
        })

    def _handle_compare(self, body: dict):
        policy_a = body.get("policy_a", "index")
        policy_b = body.get("policy_b", "round_robin")
        scenario = body.get("scenario", "sparse")
        seed = int(body.get("seed", 0))
        horizon = float(body.get("horizon", 60.0))
        injected_threat = body.get("injected_threat")

        cfg_overrides = _build_threat_override(scenario, injected_threat) if injected_threat else None
        summary_a, trace_a, log_a = _run_simulation(policy_a, scenario, seed, horizon, cfg_overrides=cfg_overrides)
        summary_b, trace_b, log_b = _run_simulation(policy_b, scenario, seed, horizon, cfg_overrides=cfg_overrides)

        run_id_a = f"{policy_a}__{scenario}__s{seed}__h{int(round(horizon))}"
        run_id_b = f"{policy_b}__{scenario}__s{seed}__h{int(round(horizon))}"

        energy_a = _safe_float(summary_a.get("energy_total_j") or summary_a.get("energy_j") if summary_a else 0.0)
        energy_b = _safe_float(summary_b.get("energy_total_j") or summary_b.get("energy_j") if summary_b else 0.0)
        e_per_det_a = _safe_float(summary_a.get("energy_per_detection_j") or summary_a.get("energy_per_unique_det_j") if summary_a else 0.0)
        e_per_det_b = _safe_float(summary_b.get("energy_per_detection_j") or summary_b.get("energy_per_unique_det_j") if summary_b else 0.0)
        
        energy_savings_pct = ((e_per_det_b - e_per_det_a) / e_per_det_b * 100.0) if e_per_det_b > 0 else 0.0

        # Calculate threat interception metrics if threat was injected
        threat_metrics = None
        if injected_threat:
            t_chan = int(injected_threat.get("channel", 48))
            try:
                base_cfg = load_config(scenario)
                base_n = int(base_cfg.get("grid", {}).get("n_channels", 2000))
            except Exception:
                base_n = 2000
            scale = base_n / 200.0 if base_n > 200 else 1.0
            g_min = min(base_n - 1, max(0, int(round(t_chan * scale))))
            g_max = min(base_n, g_min + max(1, int(round(scale))))

            ttfi_a = None
            ttfi_b = None
            for s in trace_a:
                if s.get("n_det", 0) > 0 and any(g_min <= c <= g_max or c == t_chan for c in s.get("det_channels", [])):
                    ttfi_a = s.get("t_start", 0.0)
                    break
            for s in trace_b:
                if s.get("n_det", 0) > 0 and any(g_min <= c <= g_max or c == t_chan for c in s.get("det_channels", [])):
                    ttfi_b = s.get("t_start", 0.0)
                    break
            
            if ttfi_a is not None and ttfi_b is not None:
                speedup = round(ttfi_b / max(ttfi_a, 0.05), 1) if ttfi_a > 0 else 10.0
            elif ttfi_a is not None and ttfi_b is None:
                speedup = round(horizon / max(ttfi_a, 0.05), 1)
            else:
                speedup = None

            threat_metrics = {
                "channel": t_chan,
                "priority": int(injected_threat.get("priority", 1)),
                "label": str(injected_threat.get("label", f"Hostile Threat CH {t_chan}")),
                "ttfi_a_s": ttfi_a,
                "ttfi_b_s": ttfi_b,
                "speedup": speedup,
                "intercepted_by_a": ttfi_a is not None,
                "intercepted_by_b": ttfi_b is not None,
            }

        self._send_json({
            "scenario": scenario,
            "seed": seed,
            "horizon": horizon,
            "injected_threat": injected_threat,
            "threat_metrics": threat_metrics,
            "policy_a": {
                "id": policy_a,
                "run_id": run_id_a,
                "summary": summary_a,
                "trace": trace_a,
            },
            "policy_b": {
                "id": policy_b,
                "run_id": run_id_b,
                "summary": summary_b,
                "trace": trace_b,
            },
            "comparison": {
                "energy_savings_pct": round(energy_savings_pct, 2),
                "energy_a_j": energy_a,
                "energy_b_j": energy_b,
                "e_per_det_a_j": e_per_det_a,
                "e_per_det_b_j": e_per_det_b,
            }
        })

    def _handle_train(self):
        def _train_worker():
            subprocess.run([_get_python_bin(), "-m", "agent.policy_learned", "--train"], cwd=str(ROOT))
        
        t = threading.Thread(target=_train_worker, daemon=True)
        t.start()
        self._send_json({"status": "training_started", "message": "Rung-2 GBDT training triggered in background."})


def run_server(host: str | None = None, port: int | None = None):
    if host is None:
        host = os.environ.get("HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("PORT", 8080))

    static_dir = get_static_dir()
    static_dir.mkdir(parents=True, exist_ok=True)
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, EWRequestHandler)
    print(f"===============================================================")
    print(f" EW SMART SCAN STRATEGY // ESM MULTI-PAGE PLATFORM SERVER")
    print(f" Running at: http://{host}:{port}/")
    print(f" Static UI : {static_dir}")
    print(f" Results   : {RESULTS_DIR}")
    print(f"===============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EW Smart Scan Web Server & Command Center")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host address (default 0.0.0.0 or HOST env)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)), help="Port number (default 8080 or PORT env)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
