"""The evaluation harness.  One command regenerates every number in the pitch.

    python -m eval.runner --policies round_robin,random,greedy,index,oracle \
                          --scenarios sparse,dense --seeds 0-9 --horizon 60 \
                          --out results/runs.csv --trace results/steps/ --jobs 4

    python -m eval.runner --tune                    # fair-tune the round robin
    python -m eval.runner --collect --seeds 100-119 # rung-2 logs, index only
    python -m eval.runner --ablate                  # rebuild results/ablation.csv

Three things this module is careful about, each of which has bitten a project
like this before:

* **Seeds never overlap.**  Evaluation uses 0-9, rung-2 collection uses 100-119
  (DESIGN.md section 10).  `--collect` refuses a seed below `COLLECT_SEED_MIN`
  and normal runs warn if you hand them a collection seed, so training and test
  cannot silently merge.
* **`agile` is HELD OUT.**  It is absent from `DEFAULT_SCENARIOS` and running it
  requires naming it explicitly, which prints a warning.  Training on it, or
  reporting it before CP3, destroys the generalisation claim.
* **A missing optional dependency is a skip, not a crash.**  Policies are
  resolved through a name -> factory dict whose imports happen *inside* the
  factory, so an absent `models/activity_hgb.joblib` drops `index_learned` with
  a message instead of taking the whole sweep down with an ImportError.

Windows note: `multiprocessing` uses `spawn`, so the pool worker must be a
module-level picklable function (`_worker`) and `if __name__ == "__main__":` is
mandatory.  `--jobs` degrades to serial with a printed warning rather than
hanging if a pool cannot be created.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from eval.metrics import METRIC_KEYS, TRACE_COLUMNS, EpisodeLog, compute_metrics
from sim.config import load_config
from sim.contract import null_obs
from sim.env import make_world

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TRACE_DIR = RESULTS_DIR / "steps"
LOG_DIR = ROOT / "data" / "logs"

DEFAULT_RUNS_CSV = RESULTS_DIR / "runs.csv"
DEFAULT_ABLATION_CSV = RESULTS_DIR / "ablation.csv"

# DESIGN.md section 10.  Disjoint by construction, asserted at parse time.
EVAL_SEEDS: tuple[int, ...] = tuple(range(0, 10))
COLLECT_SEEDS: tuple[int, ...] = tuple(range(100, 120))
COLLECT_SEED_MIN: int = 100

# `agile` is the HELD-OUT set and is deliberately not in this tuple.
DEFAULT_SCENARIOS: tuple[str, ...] = ("sparse", "dense")
HELD_OUT_SCENARIO: str = "agile"

DEFAULT_POLICIES: tuple[str, ...] = (
    "round_robin", "random", "greedy", "index", "index_learned", "oracle",
)

# Runaway guard.  The index policy takes ~4600 steps over a 60 s horizon; a
# policy that manages 200k has a bug that would otherwise hang the sweep.
MAX_STEPS: int = 400_000


# ---------------------------------------------------------------------------
# results schema -- FROZEN.  `test_eval_runner.py` asserts the header matches.
# ---------------------------------------------------------------------------
_HEAD: tuple[str, ...] = (
    "run_id", "policy", "policy_label", "scenario", "seed", "config_hash",
    "horizon_s", "budget_j",
)
# Fairness audit: did the policy actually survive the horizon, or did it run
# itself out of energy and coast?  `scan_span_frac` is the last scan's end over
# the horizon -- the number that rejects a baseline that dies at t = 5 s.
_AUDIT: tuple[str, ...] = ("scan_span_frac", "reached_horizon", "budget_exhausted")
_TAIL: tuple[str, ...] = (
    "alpha", "beta_learned", "score_mode", "model_id", "wall_time_s",
)
# METRIC_KEYS already carries the full action/energy breakdown (n_scans,
# n_sleeps, dwell_time_s, retune_time_s, sleep_time_s, energy_scan_j,
# energy_retune_j, energy_fixed_j, energy_sleep_j) as well as every metric, so
# the results schema tracks `eval/metrics.py` automatically: renaming a metric
# there fails the header test here rather than silently dropping a column.
RUNS_COLUMNS: tuple[str, ...] = _HEAD + METRIC_KEYS + _AUDIT + _TAIL


# ---------------------------------------------------------------------------
# policy registry -- name -> factory, with the import INSIDE the factory
# ---------------------------------------------------------------------------
class PolicyUnavailable(RuntimeError):
    """This policy cannot be constructed here (missing model, missing module).

    Raised by a factory and caught by the runner, which prints a skip line and
    carries on.  A sweep must not die because rung 2 has not been trained yet.
    """


def _f_round_robin(cfg: dict, **kw):
    from eval.baselines import RoundRobinPolicy, load_tuned_round_robin

    params = dict(cfg.get("baselines", {}).get("round_robin", {}))
    tuned = load_tuned_round_robin()
    if tuned:
        params.update(tuned)
    # Only the three tuned knobs; `collect_logs` and friends are not ours.
    params.update({k: v for k, v in kw.items()
                   if k in ("bw_mhz", "dwell_ms", "sweep_period_s") and v is not None})
    return RoundRobinPolicy(
        bw_mhz=float(params.get("bw_mhz", 5.0)),
        dwell_ms=float(params.get("dwell_ms", 10.0)),
        sweep_period_s=(None if params.get("sweep_period_s") is None
                        else float(params["sweep_period_s"])),
    )


def _f_random(cfg: dict, **kw):
    from eval.baselines import RandomPolicy

    keep = ("bw_candidates_mhz", "dwell_candidates_ms")
    return RandomPolicy(**{k: v for k, v in kw.items() if k in keep})


def _f_greedy(cfg: dict, collect_logs: bool = False, **kw):
    from agent.policy_index import GreedyPolicy

    return GreedyPolicy(collect_logs=collect_logs)


def _f_index(cfg: dict, collect_logs: bool = False, **kw):
    from agent.policy_index import IndexPolicy

    return IndexPolicy(collect_logs=collect_logs)


def _f_index_learned(cfg: dict, collect_logs: bool = False, **kw):
    """`IndexPolicy` with the rung-2 model attached via `belief.attach_model`.

    The model has to be attached AFTER `reset()`, because `reset()` is what
    builds the `Belief`.  `_LearnedIndexPolicy` is the three-line subclass that
    does that and nothing else -- the learned path is otherwise bit-identical to
    rung 1, which is guarantee 1 of DESIGN.md section 8.
    """
    from agent.policy_index import IndexPolicy      # noqa: F401  (import check)

    try:
        from agent.policy_learned import ActivityModel
    except ImportError as exc:                      # sklearn/joblib absent
        raise PolicyUnavailable(f"agent.policy_learned unavailable: {exc}") from exc

    # NOTE: `learned.enabled` is deliberately NOT consulted here.  Policy
    # selection in the runner is explicit -- asking for `index_learned` by name
    # means "the learned variant".  Honouring `enabled: false` would make this
    # row silently identical to `index`, i.e. an inert ablation row that looks
    # like a result.  The "off by default" guarantee of DESIGN.md s.8 is about
    # the plain `index` policy, which never attaches a model at all (see
    # `_f_index` above) -- not about this explicitly-named variant.
    learned = cfg.get("agent", {}).get("learned", {}) or {}
    path = ROOT / str(learned.get("model_path", "models/activity_hgb.joblib"))
    beta = float(learned.get("beta", 0.0))
    if not path.exists():
        raise PolicyUnavailable(
            f"no trained rung-2 model at {path}; train one with "
            f"`python -m agent.policy_learned --train`"
        )
    try:
        model = ActivityModel.load(path, beta=beta)
    except Exception as exc:                        # corrupt / stale contract
        raise PolicyUnavailable(f"could not load {path}: {exc}") from exc
    return _LearnedIndexPolicy(model, collect_logs=collect_logs)


def _f_oracle(cfg: dict, **kw):
    from eval.baselines import ClairvoyantGreedy

    return ClairvoyantGreedy()


POLICY_FACTORIES: dict = {
    "round_robin": _f_round_robin,
    "random": _f_random,
    "greedy": _f_greedy,
    "index": _f_index,
    "index_learned": _f_index_learned,
    "oracle": _f_oracle,
}


def _lazy_learned_base():
    from agent.policy_index import IndexPolicy
    return IndexPolicy


class _LearnedIndexPolicy:
    """Composition wrapper so importing this module never imports sklearn.

    Delegates the whole `Policy` protocol to an `IndexPolicy`, attaching the
    rung-2 model to its belief immediately after `reset()`.
    """

    name = "index_learned"

    def __init__(self, model, collect_logs: bool = False):
        self._model = model
        self._inner = _lazy_learned_base()(collect_logs=collect_logs)
        self._inner.name = "index_learned"
        self.beta = float(getattr(model, "beta", 0.0))

    def reset(self, grid, mission, horizon_s, seed, cfg) -> None:
        self._inner.reset(grid, mission, horizon_s, seed, cfg)
        self._inner.belief.attach_model(self._model, self.beta)

    def act(self, obs):
        return self._inner.act(obs)

    def log_rows(self):
        return self._inner.log_rows()

    @property
    def last_score(self):
        return self._inner.last_score

    @property
    def last_reason(self):
        return self._inner.last_reason


def make_policy(name: str, cfg: dict, **kw):
    """Resolve a policy name.  Raises `PolicyUnavailable`, never `ImportError`."""
    try:
        factory = POLICY_FACTORIES[name]
    except KeyError:
        raise PolicyUnavailable(
            f"unknown policy {name!r}; known: {sorted(POLICY_FACTORIES)}"
        ) from None
    try:
        return factory(cfg, **kw)
    except PolicyUnavailable:
        raise
    except ImportError as exc:
        raise PolicyUnavailable(f"{name}: {exc}") from exc


def available_policies(names, cfg: dict, verbose: bool = True) -> list[str]:
    """Drop policies that cannot be constructed, with one message each."""
    ok = []
    for n in names:
        try:
            make_policy(n, cfg)
        except PolicyUnavailable as exc:
            if verbose:
                print(f"[skip] {n}: {exc}", file=sys.stderr)
            continue
        ok.append(n)
    return ok


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------
def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _build_cfg(scenario: str, horizon_s: float | None,
               overrides: dict | None = None) -> dict:
    """Load a scenario, apply overrides, then RE-VALIDATE and re-hash.

    Going back through `load_config` rather than mutating in place is what keeps
    `config_hash` honest: an ablation run (`score_mode: raw`, a different
    horizon) gets its own hash, so two rows in `runs.csv` that disagree can
    always be told apart by the hash rather than by hoping the CLI was recorded.
    """
    cfg = load_config(scenario)
    changed = False
    if horizon_s is not None and float(horizon_s) != float(cfg["horizon_s"]):
        cfg["horizon_s"] = float(horizon_s)
        changed = True
    if overrides:
        _deep_update(cfg, overrides)
        changed = True
    if changed:
        cfg.pop("config_hash", None)
        cfg = load_config(cfg)          # revalidate and re-hash
    return cfg


def run_id_for(policy: str, scenario: str, seed: int, horizon_s: float) -> str:
    """Deterministic, filesystem-safe, and stable across runs.

    Stability matters: the trace file name is the join key the dashboard uses,
    and the reproducibility test compares two runs row by row.
    """
    return f"{policy}__{scenario}__s{int(seed)}__h{int(round(horizon_s))}"


def run_episode(
    policy_name: str,
    scenario: str,
    seed: int,
    horizon_s: float | None = None,
    trace_dir=None,
    collect_logs: bool = False,
    log_dir=None,
    policy_kwargs: dict | None = None,
    cfg_overrides: dict | None = None,
) -> dict:
    """Run one episode end to end and return a `RUNS_COLUMNS`-shaped row.

    The world is constructed first and the policy is handed only
    `world.agent_view()` -- the `AgentEnv` facade -- so the firewall holds by
    construction here as well as by test.  The one exception is the oracle,
    which exposes `set_world`; that method exists nowhere in `agent/`.
    """
    t_wall = time.perf_counter()
    cfg = _build_cfg(scenario, horizon_s, cfg_overrides)
    world = make_world(cfg, int(seed))
    env = world.agent_view()

    policy = make_policy(policy_name, cfg, collect_logs=collect_logs,
                         **(policy_kwargs or {}))
    if hasattr(policy, "set_world"):
        policy.set_world(world)                 # oracle only; see docstring
    policy.reset(env.grid, env.mission, world.horizon_s, int(seed), cfg)

    log = EpisodeLog(energy=cfg["energy"], horizon_s=world.horizon_s,
                     n_channels=env.grid.n_channels)

    obs = null_obs()
    steps = 0
    while not obs.done:
        action = policy.act(obs)
        obs = env.step(action)
        log.record_obs(
            obs,
            best_score=float(getattr(policy, "last_score", math.nan)),
            chosen_reason=str(getattr(policy, "last_reason", "")),
        )
        steps += 1
        if steps >= MAX_STEPS:
            raise RuntimeError(
                f"{policy_name}/{scenario}/seed {seed}: exceeded {MAX_STEPS} "
                f"steps at t={obs.t:.3f}s -- the policy is not advancing the clock"
            )

    metrics = compute_metrics(
        world.truth_bursts(), log, env.mission,
        horizon_s=world.horizon_s, n_channels=env.grid.n_channels,
    )

    rid = run_id_for(policy_name, scenario, seed, world.horizon_s)
    scans = [s for s in log.steps if s.kind == "scan" and s.dwell_s > 0.0]
    t_last_scan = max((s.t_end for s in scans), default=0.0)

    acfg = cfg.get("agent", {})
    row = {
        "run_id": rid,
        "policy": policy_name,
        "policy_label": policy_label(policy_name),
        "scenario": scenario,
        "seed": int(seed),
        "config_hash": cfg["config_hash"],
        "horizon_s": float(world.horizon_s),
        "budget_j": float(cfg["energy"]["budget_j"]),
        "scan_span_frac": t_last_scan / max(world.horizon_s, 1e-12),
        "reached_horizon": int(obs.t >= world.horizon_s - 1e-9),
        "budget_exhausted": int(obs.energy_total >= float(cfg["energy"]["budget_j"]) - 1e-9),
        "alpha": float(acfg.get("alpha_staleness", 0.0)) if policy_name != "greedy" else 0.0,
        "beta_learned": float(getattr(policy, "beta", 0.0)),
        "score_mode": str(acfg.get("score_mode", "rate")),
        "model_id": _model_id(policy),
        "wall_time_s": time.perf_counter() - t_wall,
    }
    row.update(metrics)

    if trace_dir:
        write_trace(rid, log, trace_dir)
    if collect_logs:
        write_agent_log(rid, policy, scenario, seed, log_dir or LOG_DIR)
    return row


def policy_label(name: str) -> str:
    """Human-facing label.  The oracle is a CEILING; it is never called optimal."""
    from eval.baselines import CLAIRVOYANT_LABEL

    return {
        "round_robin": "round robin (fair-tuned)",
        "random": "random",
        "greedy": "greedy ablation (no decay, no scheduler)",
        "index": "index (rung 1)",
        "index_learned": "index + learned model (rung 2)",
        "oracle": CLAIRVOYANT_LABEL,
    }.get(name, name)


def _model_id(policy) -> str:
    model = getattr(policy, "_model", None)
    if model is None:
        return ""
    man = getattr(model, "manifest", {}) or {}
    return str(man.get("model_id") or man.get("trained_at") or "activity_hgb")


def write_trace(run_id: str, log: EpisodeLog, trace_dir) -> Path:
    """`results/steps/{run_id}.csv`, schema fixed by `eval.metrics.TRACE_COLUMNS`."""
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{run_id}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(TRACE_COLUMNS), extrasaction="ignore")
        wr.writeheader()
        wr.writerows(log.trace_rows())
    return path


def write_agent_log(run_id: str, policy, scenario: str, seed: int, log_dir) -> Path | None:
    """Per-decision agent-side rows for rung-2 training (`data/logs/`).

    `agent/policy_learned.py` filters on the `scenario` and `seed` columns, which
    is where the "collection 100-119, evaluation 0-9" disjointness is actually
    enforced -- so both columns are written on every row.
    """
    rows = policy.log_rows()
    if not rows:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.csv"
    cols = ["run_id", "scenario", "seed"] + list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            r = dict(r)
            r.update(run_id=run_id, scenario=scenario, seed=int(seed))
            wr.writerow(r)
    return path


# ---------------------------------------------------------------------------
# parallel driver
# ---------------------------------------------------------------------------
def _worker(job: dict) -> dict:
    """Module-level and picklable -- required by `spawn` on Windows.

    Returns the row on success and a marked-up row carrying the traceback on
    failure, so one broken (policy, scenario, seed) cannot abort the sweep.
    """
    try:
        return run_episode(**job)
    except Exception as exc:                                    # noqa: BLE001
        return {
            "__error__": f"{type(exc).__name__}: {exc}",
            "__traceback__": traceback.format_exc(limit=6),
            "policy": job.get("policy_name"), "scenario": job.get("scenario"),
            "seed": job.get("seed"),
        }


def run_matrix(policies, scenarios, seeds, horizon_s=None, trace_dir=None,
               collect_logs=False, log_dir=None, jobs: int = 1,
               verbose: bool = True) -> list[dict]:
    """Run the full (policy x scenario x seed) matrix.  Serial or pooled."""
    jobs_list = [
        dict(policy_name=p, scenario=sc, seed=int(sd), horizon_s=horizon_s,
             trace_dir=(str(trace_dir) if trace_dir else None),
             collect_logs=collect_logs,
             log_dir=(str(log_dir) if log_dir else None))
        for sc in scenarios for p in policies for sd in seeds
    ]
    n = len(jobs_list)
    if verbose:
        print(f"[runner] {n} episodes: {len(policies)} policies x "
              f"{len(scenarios)} scenarios x {len(seeds)} seeds "
              f"(jobs={jobs})")

    rows: list[dict] = []
    if jobs and jobs > 1 and n > 1:
        rows = _run_pooled(jobs_list, jobs, verbose)
        if rows is None:
            rows = []
            jobs = 1
    if not rows:
        t0 = time.perf_counter()
        for i, job in enumerate(jobs_list, 1):
            rows.append(_worker(job))
            if verbose:
                _progress(i, n, t0)
        if verbose:
            print()

    errs = [r for r in rows if "__error__" in r]
    for e in errs:
        print(f"[error] {e['policy']}/{e['scenario']}/seed {e['seed']}: "
              f"{e['__error__']}\n{e.get('__traceback__', '')}", file=sys.stderr)
    return [r for r in rows if "__error__" not in r]


def _run_pooled(jobs_list, jobs: int, verbose: bool):
    """Pooled execution.  Returns None (and warns) if a pool cannot be used.

    Falling back is deliberate: a harness that hangs on a machine where `spawn`
    misbehaves is worse than one that quietly takes four times as long.
    """
    import multiprocessing as mp

    try:
        ctx = mp.get_context("spawn")
        rows: list[dict] = []
        t0 = time.perf_counter()
        with ctx.Pool(processes=int(jobs)) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs_list), 1):
                rows.append(r)
                if verbose:
                    _progress(i, len(jobs_list), t0)
        if verbose:
            print()
        return rows
    except Exception as exc:                                    # noqa: BLE001
        print(f"[warn] multiprocessing pool unavailable ({type(exc).__name__}: "
              f"{exc}); falling back to serial", file=sys.stderr)
        return None


def _progress(i: int, n: int, t0: float) -> None:
    el = time.perf_counter() - t0
    eta = el / max(i, 1) * (n - i)
    print(f"\r  {i}/{n} episodes  elapsed {el:6.1f}s  eta {eta:6.1f}s",
          end="", flush=True)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------
def write_runs_csv(rows, path=DEFAULT_RUNS_CSV) -> Path:
    """One row per episode.  Header is `RUNS_COLUMNS`, exactly and in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {p: i for i, p in enumerate(DEFAULT_POLICIES)}
    rows = sorted(rows, key=lambda r: (str(r.get("scenario")),
                                       order.get(str(r.get("policy")), 99),
                                       int(r.get("seed", 0))))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(RUNS_COLUMNS), extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in RUNS_COLUMNS})
    return path


# The columns the ablation table reports.  Headline first: DESIGN.md section 6
# names energy per detection as THE number.
ABLATION_METRICS: tuple[str, ...] = (
    "energy_per_detection_j", "poi_60", "poi_p1_60", "n_unique_detections",
    "energy_total_j", "coverage_frac", "ttfi_p1_median_s", "max_staleness_p1_s",
    "false_alarm_rate_per_dwell",
)


def build_ablation(runs_csv=DEFAULT_RUNS_CSV, out=DEFAULT_ABLATION_CSV,
                   metrics=ABLATION_METRICS, verbose: bool = True):
    """`results/ablation.csv` -- mean and std by (scenario, policy).

    THIS TABLE IS THE RESULT.  One command regenerates it from `runs.csv`, so
    nothing in the write-up is a number somebody typed in by hand.

    `energy_per_detection_j` is `inf` for a policy that detected nothing, which
    would poison the mean; those rows are counted in `n_no_detection` and
    excluded from that column's statistics rather than silently dropped.
    """
    import pandas as pd

    df = pd.read_csv(runs_csv)
    if df.empty:
        raise ValueError(f"{runs_csv} has no rows")
    df = df.replace([np.inf, -np.inf], np.nan)
    cols = [m for m in metrics if m in df.columns]

    tab = pd.pivot_table(
        df, index=["scenario", "policy"], values=cols,
        aggfunc=["mean", "std", "count"], dropna=False,
    )
    tab.columns = [f"{m}_{a}" for a, m in tab.columns]
    tab = tab.reset_index()

    order = {p: i for i, p in enumerate(DEFAULT_POLICIES)}
    tab["_o"] = tab["policy"].map(lambda p: order.get(p, 99))
    tab = tab.sort_values(["scenario", "_o"]).drop(columns="_o")

    lab = df.drop_duplicates(["policy"]).set_index("policy")["policy_label"]
    tab.insert(2, "policy_label", tab["policy"].map(lab))
    n_no_det = (df.groupby(["scenario", "policy"])["n_unique_detections"]
                .apply(lambda s: int((s.fillna(0) == 0).sum())))
    tab["n_no_detection"] = [
        int(n_no_det.get((s, p), 0)) for s, p in zip(tab["scenario"], tab["policy"])
    ]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    if verbose:
        print_ablation(tab, metrics=cols)
        print(f"\n[runner] ablation table -> {out}")
    return tab


def print_ablation(tab, metrics=("energy_per_detection_j", "poi_60",
                                 "n_unique_detections", "energy_total_j")) -> None:
    """ASCII rendering -- the pitch never depends on matplotlib being present."""
    metrics = [m for m in metrics if f"{m}_mean" in tab.columns]
    head = f"{'scenario':9s} {'policy':16s}" + "".join(f"{m[:20]:>22s}" for m in metrics)
    print("\n" + head)
    print("-" * len(head))
    for _, r in tab.iterrows():
        line = f"{str(r['scenario']):9s} {str(r['policy']):16s}"
        for m in metrics:
            mu, sd = r.get(f"{m}_mean"), r.get(f"{m}_std")
            mu = float(mu) if mu == mu else float("nan")
            sd = 0.0 if (sd is None or sd != sd) else float(sd)
            line += f"{mu:>13.4g} +-{sd:<7.3g}"
        print(line)


# ---------------------------------------------------------------------------
# CP2 head-to-head
# ---------------------------------------------------------------------------
def head_to_head(scenario: str = "sparse", seeds=(0, 1, 2), horizon_s: float = 60.0,
                 a: str = "index", b: str = "round_robin", verbose: bool = True) -> dict:
    """The CP2 number: energy per detection, `index` vs fair-tuned `round_robin`.

    Reported as a ratio because that is the claim: "same interception
    performance at roughly half the scanning energy".  Both terms are the
    headline metric from DESIGN.md section 6 -- `energy_total_J / distinct
    (emitter_id, activation_id)`.
    """
    out: dict = {"scenario": scenario, "seeds": list(seeds), "horizon_s": horizon_s}
    for name in (a, b):
        rows = [run_episode(name, scenario, s, horizon_s=horizon_s) for s in seeds]
        out[name] = {
            "energy_per_detection_j": float(np.mean(
                [r["energy_per_detection_j"] for r in rows])),
            "energy_total_j": float(np.mean([r["energy_total_j"] for r in rows])),
            "n_unique_detections": float(np.mean(
                [r["n_unique_detections"] for r in rows])),
            "poi_60": float(np.mean([r["poi_60"] for r in rows])),
            "poi_p1_60": float(np.mean([r["poi_p1_60"] for r in rows])),
            "coverage_frac": float(np.mean([r["coverage_frac"] for r in rows])),
            "scan_span_frac": float(np.mean([r["scan_span_frac"] for r in rows])),
            "rows": rows,
        }
    out["ratio_energy_per_detection"] = (
        out[b]["energy_per_detection_j"] / out[a]["energy_per_detection_j"]
    )
    if verbose:
        _print_head_to_head(out, a, b)
    return out


def _print_head_to_head(h, a, b) -> None:
    print()
    print("=" * 72)
    print(f"CP2 HEAD-TO-HEAD  --  {h['scenario']}, seeds {h['seeds']}, "
          f"horizon {h['horizon_s']:.0f} s")
    print("=" * 72)
    print(f"{'':22s}{'energy/detection':>18s}{'energy J':>12s}"
          f"{'uniq det':>10s}{'POI@60':>9s}")
    for name in (a, b):
        d = h[name]
        print(f"{policy_label(name)[:21]:22s}"
              f"{d['energy_per_detection_j']:>18.5f}"
              f"{d['energy_total_j']:>12.3f}"
              f"{d['n_unique_detections']:>10.1f}"
              f"{d['poi_60']:>9.3f}")
    r = h["ratio_energy_per_detection"]
    print("-" * 72)
    print(f"  {b} costs {r:.2f}x the energy per detection of {a}")
    print(f"  equivalently, {a} needs {1.0 / r:.1%} of the energy per detection")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_seeds(spec) -> tuple[int, ...]:
    """Parse `0-9`, `0,3,7`, `0-4,9`, or several such arguments.

    Ranges are INCLUSIVE at both ends -- `0-9` is ten seeds, matching how
    DESIGN.md section 10 writes them.
    """
    if isinstance(spec, (list, tuple)):
        parts: list[str] = []
        for s in spec:
            parts.extend(str(s).split(","))
    else:
        parts = str(spec).split(",")
    out: list[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p.lstrip("-"):
            lo, _, hi = p.partition("-")
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(f"seed range {p!r} is inverted")
            out.extend(range(a, b + 1))
        else:
            out.append(int(p))
    if not out:
        raise ValueError(f"no seeds parsed from {spec!r}")
    # Stable de-duplication: the same seed twice would double-weight one world.
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return tuple(uniq)


def _split_list(spec) -> tuple[str, ...]:
    if isinstance(spec, (list, tuple)):
        parts: list[str] = []
        for s in spec:
            parts.extend(str(s).split(","))
    else:
        parts = str(spec).split(",")
    return tuple(p.strip() for p in parts if p.strip())


def _check_seeds(seeds, collect: bool) -> None:
    """Enforce the disjointness in DESIGN.md section 10 loudly, not silently."""
    if collect:
        bad = [s for s in seeds if s < COLLECT_SEED_MIN]
        if bad:
            raise SystemExit(
                f"--collect seeds must be >= {COLLECT_SEED_MIN} (rung-2 collection "
                f"uses {COLLECT_SEEDS[0]}-{COLLECT_SEEDS[-1]}); got {bad}. "
                f"Training on evaluation seeds 0-9 would invalidate every rung-2 "
                f"number in the write-up."
            )
    else:
        bad = [s for s in seeds if s >= COLLECT_SEED_MIN]
        if bad:
            print(f"[warn] evaluating on collection seeds {bad}: rung-2 was "
                  f"TRAINED on {COLLECT_SEEDS[0]}-{COLLECT_SEEDS[-1]}, so these "
                  f"are not held out", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.runner",
        description="Evaluation harness: runs the policy x scenario x seed matrix "
                    "and exports runs.csv, per-step traces and the ablation table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES),
                   help="comma- or space-separated names (default: all)")
    p.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS),
                   help=f"default {DEFAULT_SCENARIOS}; {HELD_OUT_SCENARIO!r} is "
                        f"HELD OUT and must be named explicitly")
    p.add_argument("--seeds", nargs="+", default=["0-9"],
                   help="ranges like 0-9 (inclusive) or lists like 0,3,7")
    p.add_argument("--horizon", type=float, default=None,
                   help="override horizon_s (default: the scenario's own)")
    p.add_argument("--out", default=str(DEFAULT_RUNS_CSV))
    p.add_argument("--trace", default=None,
                   help="directory for per-step traces, e.g. results/steps/")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--collect", action="store_true",
                   help=f"rung-2 log collection: index policy only, seeds "
                        f">= {COLLECT_SEED_MIN}")
    p.add_argument("--log-dir", default=str(LOG_DIR))
    p.add_argument("--tune", action="store_true",
                   help="fair-tune the round robin on sparse, then exit")
    p.add_argument("--head-to-head", action="store_true",
                   help="print the CP2 index vs round_robin energy/detection ratio")
    p.add_argument("--ablate", action="store_true",
                   help="rebuild results/ablation.csv from an existing runs.csv "
                        "and exit")
    p.add_argument("--ablation-out", default=str(DEFAULT_ABLATION_CSV))
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    if args.ablate:
        build_ablation(args.out, args.ablation_out, verbose=verbose)
        return 0

    if args.tune:
        from eval.baselines import fair_tune_round_robin

        seeds = parse_seeds(args.seeds) if args.seeds != ["0-9"] else (0, 1, 2)
        fair_tune_round_robin(scenario="sparse", seeds=seeds,
                              horizon_s=float(args.horizon or 60.0), verbose=verbose)
        return 0

    if args.head_to_head:
        seeds = parse_seeds(args.seeds) if args.seeds != ["0-9"] else (0, 1, 2)
        head_to_head(scenario=_split_list(args.scenarios)[0], seeds=seeds,
                     horizon_s=float(args.horizon or 60.0), verbose=verbose)
        return 0

    scenarios = _split_list(args.scenarios)
    if HELD_OUT_SCENARIO in scenarios:
        print(f"[warn] {HELD_OUT_SCENARIO!r} is the HELD-OUT scenario. It must "
              f"never be trained on and should not be reported before CP3.",
              file=sys.stderr)

    if args.collect:
        seeds = parse_seeds(args.seeds if args.seeds != ["0-9"] else ["100-119"])
        _check_seeds(seeds, collect=True)
        policies = ("index",)
        if verbose:
            print(f"[runner] rung-2 collection: index policy, "
                  f"{len(scenarios)} scenarios x {len(seeds)} seeds -> "
                  f"{args.log_dir}")
    else:
        seeds = parse_seeds(args.seeds)
        _check_seeds(seeds, collect=False)
        cfg0 = _build_cfg(scenarios[0], args.horizon)
        policies = tuple(available_policies(_split_list(args.policies), cfg0, verbose))
        if not policies:
            print("[error] no runnable policies", file=sys.stderr)
            return 2

    rows = run_matrix(
        policies, scenarios, seeds, horizon_s=args.horizon,
        trace_dir=args.trace, collect_logs=bool(args.collect),
        log_dir=args.log_dir, jobs=int(args.jobs), verbose=verbose,
    )
    if not rows:
        print("[error] every episode failed", file=sys.stderr)
        return 3

    out = write_runs_csv(rows, args.out)
    if verbose:
        print(f"[runner] {len(rows)} rows -> {out}")

    if not args.collect and len({r["policy"] for r in rows}) > 1:
        try:
            build_ablation(out, args.ablation_out, verbose=verbose)
        except Exception as exc:                                # noqa: BLE001
            print(f"[warn] ablation table not built: {exc}", file=sys.stderr)
    return 0


__all__ = [
    "RUNS_COLUMNS", "POLICY_FACTORIES", "PolicyUnavailable",
    "make_policy", "available_policies", "run_episode", "run_matrix",
    "write_runs_csv", "write_trace", "write_agent_log", "build_ablation",
    "head_to_head", "parse_seeds", "run_id_for", "policy_label",
    "EVAL_SEEDS", "COLLECT_SEEDS", "DEFAULT_SCENARIOS", "DEFAULT_POLICIES",
    "ABLATION_METRICS", "main",
]


if __name__ == "__main__":       # MANDATORY on Windows: `spawn` re-imports this
    sys.exit(main())
