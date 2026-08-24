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

## Metrics

| metric | target vs sweep |
|---|---|
| POI @ T — fraction of emitters intercepted within horizon | parity or better |
| TTFI, priority-1 — median time to first intercept | 2–4× faster |
| Emitter-time coverage — % of emitter-active seconds observed | parity or better |
| **Energy per detection** ← the headline | **40–60% lower** |
| Max staleness, priority-1 | hard-bounded |
| False alarm rate | no worse |

The sentence all of this is built toward:

> *"Same interception performance at roughly half the scanning energy, measured against
> a clairvoyant upper bound."*

If a feature does not help say that sentence, it is not in scope.

---

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
