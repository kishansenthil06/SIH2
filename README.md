# Smart Scan Strategy for Electronic Warfare

**SIH 2026 — energy-aware adaptive spectrum scanning for Electronic Support Measures (ESM).**

An EW receiver can listen carefully to only a small slice of the spectrum at a time.
The traditional answer is a round-robin sweep: scan band 1, 2, 3, … and repeat. But
radar and EW environments are **sparse in time and frequency** — only a few emitters
are usually radiating — so a sweep spends most of its energy listening to empty air,
and takes a long time to come back round to the band that matters.

This project replaces the sweep with a policy that decides **where to point next, how
long to dwell, and when to do nothing at all.**

> We did not invent adaptive scanning. We made the scan **policy** energy-aware while
> holding detection performance flat.

---

## The idea in one paragraph

The receiver keeps a **belief** — for every channel, `P(active now)` — updated by Bayes
on each observation and decayed back toward the prior as time since the last visit
grows. Bands turn on and off whether or not we are watching, which makes this a
**restless multi-armed bandit** rather than a standard one, and that decay is the entire
difficulty of the problem. A scheduler then picks the action maximising expected
priority-weighted detections **per joule per second**, under hard constraints: revisit
deadlines per priority class, an energy budget, and a mandatory watch list.

Everything resolves to one equation:

```
r(t) = Σ_p [ w_p · detected(p,t) ]  −  L_d·t_dwell  −  L_f·|Δf|  −  L_0
```

Because `L_f` is non-zero the scheduler learns not to thrash across the band — nearby
frequencies get visited together. Because `L_0` exists, **doing nothing is sometimes
optimal**, which is the only way an energy claim can ever be true.

---

## Architecture

```
   ===================== SIMULATED WORLD (holds ground truth) =====================
   |   EMITTER SCENARIO  ->  CHANNEL + RECEIVER  ->  ENERGY DETECTOR              |
   |   on/off Markov         path loss, noise        P_d(SNR, dwell), P_fa        |
   |   priority class        tuner BW, retune                                     |
   ================|=====================|====================|====================
        ground     |              action |                    | detections + SNR
        truth      |   tune(f,BW,dwell)  |                    | EVERY STEP
     (evaluator    |     or sleep(dt)    |                    |
       only)       |                     v                    v
                   |     ===== SCAN AGENT — SEES OBSERVATIONS ONLY =====
                   |     |  SCHEDULER  <-  POLICY / VALUE  <-  BELIEF STATE  |
                   |     |  deadlines      reward-rate         P(active|band)|
                   |     |  energy budget  index               decays unseen |
                   |     |  watch list                                       |
                   |     |        refit activity model on own logs (SLOW)    |
                   |     =====================================================
                   v
        EVALUATION — baselines: round-robin | random | greedy | oracle
                     metrics:   POI | TTFI | coverage | ENERGY PER DETECTION
                     identical scenarios, identical seeds
```

- **Fast loop:** detector → belief, every single scan step.
- **Slow loop:** the activity model refits on the **agent's own logs**, never on truth.
- **Firewall:** the simulator knows the answer, the evaluator may see it, the agent never
  does. Enforced in code three ways — see below.

---

## The firewall

An agent that can peek at ground truth proves nothing. `agent/` and `app/` may import
only `sim.contract` and `sim.config`, never `sim.env`/`emitters`/`channel`/`receiver`.
Enforcement is not by convention:

1. **Structural** — policies receive an `AgentEnv` facade whose `__slots__` hold only
   *bound methods*, so there is no attribute path from the agent to a `World`.
2. **Static** — an AST scan over `agent/**` and `app/**` fails the build on any
   forbidden import or any `.truth*` attribute access.
3. **Runtime** — `World.truth()` inspects the call stack and raises `FirewallViolation`
   if an `agent.*` or `app.*` frame is anywhere below it.

`agent/belief.py` therefore *deliberately reimplements* the detector curve rather than
importing it; a cross-check test asserts the two agree to 1e-9. Duplication is cheaper
than a firewall breach.

---

## Layout

```
sim/     contract.py  config.py  stub_env.py     <- frozen interface
         emitters.py  channel.py  receiver.py  env.py
agent/   base.py                                 <- frozen protocols + feature contract
         belief.py  policy_index.py  scheduler.py  policy_learned.py
eval/    baselines.py  metrics.py  runner.py  figures.py  tests/
app/     dashboard.py
configs/ sparse.yaml  dense.yaml  agile.yaml
```

`DESIGN.md` is the authoritative technical specification — verified formulas, constants,
and the reasoning behind each.

---

## Quick start

```bash
pip install numpy scipy pandas scikit-learn pyyaml matplotlib

# tests (stdlib unittest; pytest not required)
python -m unittest discover -s eval/tests -t . -v

# head-to-head: does the index policy actually beat the sweep?
python -m eval.runner --policies round_robin,index --scenarios sparse --seeds 0-2

# collect logs and train the rung-2 activity model (seeds disjoint from evaluation)
python -m eval.runner --collect --seeds 100-119
python -m agent.policy_learned --train

# the full ablation — this table is the result
python -m eval.runner --policies round_robin,random,greedy,index,index_learned,oracle \
                      --scenarios sparse,dense,agile --seeds 0-9 --jobs 4 --trace

python -m eval.figures
python -m app.dashboard --run <run_id>
```

---

## Results

Full matrix: **5 policies x 3 scenarios x 5 seeds = 75 episodes**, on a 2 GHz /
2000-channel grid, 60 s horizon, 6 J budget (0.1 W average). Regenerate with
`python -m eval.runner ...`; raw rows in `results/runs.csv`.

Energy per detection is reported as a **median**. That is deliberate, not a
flattering choice: a policy that detects nothing on some seed has an *infinite*
energy per detection on that seed, so the mean is undefined or dominated by one
outlier for `random`, `greedy` and (on `agile`) the oracle. Both statistics are
given for the two policies that matter so the difference is visible.

| scenario | policy | J/detection (median) | POI@60 |
|---|---|---|---|
| **sparse** (8 emitters) | round_robin | 2.000 | **0.400** |
| | random | no detections | 0.000 |
| | greedy | 5.999 | 0.150 |
| | index | **0.667** | 0.275 |
| | oracle | 0.109 | 0.825 |
| **dense** (24 emitters) | round_robin | 0.500 | 0.350 |
| | random | 1.747 | 0.092 |
| | greedy | 0.544 | 0.325 |
| | index | **0.181** | **0.400** |
| | oracle | 0.333 | 0.525 |
| **agile** (hoppers, *held out*) | round_robin | 3.000 | 0.171 |
| | random | 2.952 | 0.086 |
| | greedy | 2.996 | 0.186 |
| | index | **0.600** | **0.386** |
| | oracle | 1.198 | 0.286 |

Index vs the fair-tuned sweep, both statistics:

| scenario | median | mean | POI@60 |
|---|---|---|---|
| sparse | −66.7% | −64.5% | 0.275 vs 0.400 (**below**) |
| dense | −63.8% | −67.5% | 0.400 vs 0.350 (above) |
| agile | −80.0% | −72.9% | 0.386 vs 0.171 (**2.3x**) |

**The claim, stated honestly and conditionally:**

> Adaptive scanning cuts energy per detection by 64-80% and matches or beats a
> tuned sweep on interception **once the environment has enough structure to learn
> from** - most of all against frequency-agile emitters, on scenarios it never saw.
> In a very sparse band a blind sweep remains hard to beat on pure discovery.

That conditional is the useful engineering result. Reporting only `dense` and
`agile` would be cherry-picking; the `sparse` POI deficit is analysed in
`DESIGN.md` sections 11.7-11.8 and left open rather than papered over.

### Three caveats that must travel with these numbers

- **Energy per detection can be gamed** by harvesting cheap detections, so it is
  never quoted without POI beside it. On `dense`, `index` beats the *oracle* on
  energy per detection precisely this way - the oracle takes the higher POI.
- **The oracle is not a valid upper bound on `agile`.** It is clairvoyant *greedy*
  over one action, so against emitters hopping every 50 ms it dwells on a channel
  the emitter has already left. Clairvoyance about the present does not help when
  the target moves inside your own action.
- **`random` detects nothing on `sparse`.** A uniformly random 1 MHz scan almost
  never lands on one of 8 emitters in 2000 channels, so that baseline is
  degenerate at this grid width rather than merely weak.

### Ablation: what the belief and scheduler actually contribute

`greedy` is `index` with Markov propagation removed, the staleness bonus zeroed and
the scheduler disabled. It collapses to **5.999 J/detection at POI 0.150** on
`sparse` and **2.996 at POI 0.186** on `agile`, against `index`'s 0.667 / 0.275 and
0.600 / 0.386 - a direct measurement of what the belief decay and the constrained
scheduler are worth, rather than an assertion that they matter.

## Metrics

| metric | definition |
|---|---|
| POI @ T | fraction of distinct emitters intercepted within the horizon |
| TTFI, priority-1 | median time to first intercept, censored at horizon |
| Emitter-time coverage | % of emitter-active seconds the receiver was pointed at |
| **Energy per detection** | joules / distinct `(emitter, activation)` pairs |
| Max staleness, priority-1 | worst revisit gap on a high-priority channel |
| False alarm rate | per channel-dwell, comparable to the configured `P_fa` |

## Policy rungs

Each rung runs and demos independently, so there is always something that works.

- **Rung 0 — round-robin sweep.** The baseline, and the integration smoke test.
- **Rung 1 — index policy.** Reward-rate index over the belief. Beats the sweep, and
  every decision it makes is explainable by pointing at the screen.
- **Rung 2 — learned activity model.** A calibrated gradient-boosted classifier
  predicting `P(active | band, recent history)`, trained on the agent's **own logs**.
  It drops into the belief layer behind a flag; the index policy above it is untouched,
  and three independent gates mean it cannot regress rung 1.
- **Rung 3 — deep RL.** Deliberately **out of scope**. In a build this short a tuned
  heuristic beats an undertrained agent.

---

## Honest notes

- The source architecture's index formula subtracts joules from a probability. Taken
  literally it makes retune cost negligible, so the policy thrashes and **loses to the
  sweep** (measured: 79.7 J at POI 0.50, versus round-robin's 61.1 J at POI 0.88). We
  express priority weights in joules and use a reward *rate*; `score_mode: raw` keeps the
  original form available so the difference can be ablated rather than asserted.
- The oracle is **clairvoyant greedy — a reference ceiling, not the optimum.** It is
  myopic over one action.
- Round-robin is **fair-tuned**: its bandwidth, dwell and sweep period are grid-searched
  to maximise POI subject to matching the index policy's energy. The search is exported
  to `results/roundrobin_tuning.csv`.
- The energy budget is genuinely binding (0.1 W average, ~10% duty cycle), so **every**
  policy — baselines included — must pace itself with sleep to survive the horizon.
- `agile` is a **held-out** scenario. Nothing trains on it and it is not run until the
  final ablation.

---

## Out of scope

Real SDR hardware in the loop · signal classification / modulation recognition ·
direction finding or angle of arrival · multi-receiver and networked sensors ·
pulse-level PRI deinterleaving · deep RL · anything requiring a real RF dataset.

Simulation only, with ground truth available to the evaluator alone.
