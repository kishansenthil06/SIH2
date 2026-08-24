# Smart Scan Strategy for EW — Design Reference

Energy-aware adaptive spectrum scanning for Electronic Support Measures (ESM),
simulation only. The receiver maintains a **belief** over channel occupancy and
schedules `Scan`/`Sleep` actions to intercept high-priority emitters quickly while
spending far less sensing energy than a round-robin sweep.

This is a **restless multi-armed bandit**: bands turn on and off whether or not we
are watching, so belief decays toward the prior between visits. That decay is the
entire difficulty of the problem, and it is why an adaptive policy *beats* a sweep
rather than merely matching it.

**The objective — every design argument resolves to a term in this:**

```
r(t) = Σ_p [ w_p · detected(p,t) ]  −  L_d·t_dwell  −  L_f·|Δf|  −  L_0
```

**The sentence we are building toward:**

> "Same interception performance at roughly half the scanning energy, measured
> against a clairvoyant upper bound."

If a feature does not help say that sentence, it is out of scope.

---

## 0. Status — Phase 0 is DONE and FROZEN

| file | status |
|---|---|
| `sim/contract.py` | **frozen** — types, `ChannelGrid`, `Mission`, `ScanEnv` |
| `agent/base.py` | **frozen** — `Policy`, `BeliefLike`, `FEATURE_NAMES` (F=16), `EnergyState` |
| `sim/config.py` | **frozen** — loader, validation, `config_hash` |
| `sim/stub_env.py` | **frozen** — `StubEnv`, satisfies `ScanEnv`, no truth |
| `configs/{sparse,dense,agile}.yaml` | **frozen** |
| `eval/tests/test_contract.py` | 21 tests, all passing |

Do not edit any of the above. If you believe the contract is wrong, say so in your
report — do not change it.

---

## 1. Verified numbers — use these exactly

All checked by running them; do not re-derive or substitute.

### Belief decay (closed form)
For a 2-state CTMC with rates `λ_on` (off→on), `λ_off` (on→off):

```
π = λ_on/(λ_on+λ_off)          Λ = λ_on+λ_off
p(t+Δt) = π + (p(t) − π)·exp(−Λ·Δt)
```

Verified identical to `scipy.linalg.expm(Q·Δt)` to 1e-12. **No matrix exponential
at runtime.** Note it also decays *upward* from below-prior: a channel confidently
observed empty drifts back up toward π.

### Detector (Urkowitz energy detector, Gaussian approximation)
`N = dwell_s · channel_bw_hz` complex samples, `s` = linear SNR:

```python
N   = dwell_s * 1e6
s   = 10**(snr_eff_db/10)
P_d = norm.sf((norm.isf(pfa) - np.sqrt(N)*s) / (1.0 + s))
```

`s = 0` gives `P_d = Q(Q⁻¹(P_fa)) = P_fa` automatically — **no separate
false-alarm branch is needed anywhere.**

Verified at `P_fa = 1e-3`, `B = 1 MHz` — **5 dB of SNR ≈ one decade of dwell**:

| SNR dB | 1 ms | 2 ms | 5 ms | 10 ms | 20 ms | 50 ms | 100 ms | 200 ms |
|---|---|---|---|---|---|---|---|---|
| −10 | 0.526 | 0.895 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| −15 | 0.021 | 0.052 | 0.204 | **0.528** | 0.910 | 1.000 | 1.000 | 1.000 |
| −18 | 0.005 | 0.010 | 0.026 | 0.069 | 0.202 | **0.672** | 0.971 | 1.000 |
| −20 | 0.003 | 0.004 | 0.009 | 0.019 | 0.049 | 0.199 | **0.528** | 0.914 |
| −22 | 0.002 | 0.003 | 0.004 | 0.007 | 0.014 | 0.048 | 0.138 | 0.395 |

### Energy — derived from timing so the two models cannot drift
`sim/config.py` asserts `L_f == L_d / f_slew`.

| const | value | check |
|---|---|---|
| `L_d` | 1.0 J/s | 10 ms dwell = 10 mJ |
| `L_f` | 2.0e-11 J/Hz | 200 MHz hop = 4.0 mJ, 4.5 ms |
| `L_0` | 2.0e-3 J | any scan ≥ 2 mJ |
| `L_sleep` | 0.01 J/s | 100 ms sleep = 1 mJ = 1/12 of one 10 ms scan |
| `w_p` | {1:0.100, 2:0.030, 3:0.010} **joules** | prio-1 intercept ≈ 8 short scans |

```
E(Scan)  = L_0 + L_d·dwell_s + L_f·|f − f_last|
E(Sleep) = L_sleep · dt_s
t_retune = 0 if f == f_last else t_settle_s + |Δf|/f_slew
```

Full 200-channel sweep @ 5 MHz/10 ms = 40 scans = **484 mJ in 0.424 s** (exact).

### The budget is binding — this matters for baselines
`budget_j = 6.0` over `horizon_s = 60` is an average of **0.1 W**, i.e. at most a
~10% duty cycle at `L_d = 1 W`. Continuous sweeping at 5 MHz/10 ms exhausts the
budget at **t = 5.26 s**, only 8.8% of the horizon.

**Consequence: every policy, baselines included, must pace itself with `Sleep` to
survive the horizon.** A baseline that runs out of energy at t=5 s is not a fair
comparison and would rig the headline in our favour. Sweep *period* is therefore
part of the round-robin fair-tuning search (see §6).

---

## 2. Firewall — non-negotiable

**The simulator knows the answer, the evaluator may see it, the agent never does.**

- `agent/**` and `app/**` may import **only** `sim.contract` from `sim`. Never
  `sim.env`, `sim.emitters`, `sim.channel`, `sim.receiver`.
- Never call or reference `.truth()`, `.truth_bursts()`, `.truth_power()`,
  `._world`, or `.emitters` from agent-side code.
- Policies receive `World.agent_view()` (an `AgentEnv` facade), never a `World`.
- `agent/belief.py` **deliberately reimplements** `pd_curve` rather than importing
  `sim.receiver`. A cross-check test asserts they agree to 1e-9. Duplication is
  cheaper than a firewall breach.

Enforced three ways: structurally (`AgentEnv.__slots__` holds only bound methods),
statically (AST scan in `eval/tests/test_firewall.py`), and at runtime
(`truth()` inspects the call stack and raises `FirewallViolation`).

---

## 3. The rung-1 index — corrected from the source doc

The source document's §5 writes the index as:

```
score(b) = P_hat(active_b)·w_b·(1 + α·staleness_b) − cost(a)      # WRONG
```

A prototype implemented this literally and **it loses to the sweep**: 79.7 J at
POI 0.50, versus round-robin's 61.1 J at POI 0.88. Two defects:

1. **Incommensurate units** — a dimensionless probability minus joules makes the
   cost term effectively zero, so the policy thrashes across the band paying
   `L_f·|Δf|` every step and burns *more* energy than a sweep.
2. **Unbounded staleness** — `(1+α·staleness)` grows without limit, so the sleep
   branch never fires; sleep is the only action that saves energy.

**The correction, used throughout this project:** `w_p` is expressed **in joules**
(see the config) so `gain − cost` is dimensionally consistent, and the index is a
**reward rate** because actions have variable duration:

```
stale_norm  = min(staleness / t_ref, staleness_cap)          # t_ref=1.0, cap=20
value[c]    = p_eff[c] · w[c] · (1 + α·stale_norm[c])        # α = 0.5
gain(a)     = pd_bar[bw, dwell] · Σ_{c ∈ a} value[c]
cost(a)     = L_0 + L_d·dwell + L_f·|f_a − f_last|
duration(a) = t_retune(f_a) + dwell
score_rate(a) = (gain(a) − cost(a)) / duration(a)            # DEFAULT
score_raw(a)  =  gain(a) − cost(a)                           # doc's literal form
```

Without the rate form, a 200 ms dwell gaining 60 mJ beats a 5 ms dwell gaining
20 mJ, despite the latter earning 4× the reward per second of mission time.
Keep `score_mode: raw` working — ablating it is a good slide ("we tried the naive
index; here is why the rate form wins").

**Sleep needs no threshold.** `Sleep(dt)` has `gain=0`, `cost=L_sleep·dt`,
`duration=dt`, so `score_rate(Sleep) = −L_sleep = −0.010 W` **independent of dt**
(verified), while a hopeless scan scores ≈ −1.1 W. Sleep is therefore selected
exactly when every scan candidate has negative reward rate — the mathematically
correct answer to "is doing nothing optimal right now?"

---

## 4. Sensing model — a miss is not an absence

A miss means one of four things: nothing was there; something below sensitivity;
something silent during the dwell; or something outside the tuned bandwidth.
Collapsing these into "empty" teaches the agent the wrong lesson. So observations
enter the belief as **likelihoods, not facts**, and there are **two** `P_d`s:

- **On a detection** the agent has the reported SNR, so it evaluates the curve
  there. A −20 dB detection at 2 ms has `P_d ≈ 0.004` vs `P_fa = 1e-3` →
  likelihood ratio 4, so belief rises but does **not** saturate. A −10 dB / 5 ms
  detection has LR ≈ 1000 → belief → 1. *The agent correctly distrusts marginal
  detections, straight out of the maths.*
- **On a miss** there is no SNR to condition on, so use a **marginal**
  `pd_bar[bw, dwell]`, precomputed once by integrating the curve over the agent's
  assumed `snr ~ N(−15, 5) dB` — its spec-sheet belief, **not** truth. A 1 ms miss
  barely moves the belief; a 100 ms miss crushes it.

```
p_post = p·L1 / (p·L1 + (1−p)·L0)      clipped to [1e-6, 1−1e-6]
  detection: L1 = pd_at_reported_snr,  L0 = P_fa
  miss:      L1 = 1 − pd_bar[bw,dwell], L0 = 1 − P_fa
```

**Bandwidth penalty is load-bearing, not cosmetic.** Without it the widest scan
strictly dominates (same time, same energy, 20× the channels) and the bandwidth
knob is degenerate:

```
snr_eff_db = snr_db − 1.0·log2(bw_hz/1e6) + (6.0 if gain else 0.0)
```

At 1 dB/octave a 20 MHz scan is 4.3 dB less sensitive than a 1 MHz one, needing
~7× the dwell to reach the same weak emitter. That single constant creates the
"wide-and-fast to explore, narrow-and-long to confirm" behaviour the demo needs.
False alarms are drawn per channel, so wide scans also generate more junk — a
second reason not to max out bandwidth.

---

## 5. Scenarios

| scenario | emitters | occupancy | role |
|---|---|---|---|
| `sparse` | 8 | 4% | tuning set |
| `dense` | 24 | 15% | stress; same budget, so it bites harder |
| `agile` | 14 (4 hoppers) | 7% | **HELD OUT — do not run before CP3** |

SNR placement is what makes the story true:
- **priority 3** (routine): −8…−12 dB → 1–2 ms suffices → cheap
- **priority 2**: −14…−17 dB → ~10 ms
- **priority 1** (rare threat): −18…−21 dB → 50–200 ms

*A sweep cannot afford a 100 ms dwell on 200 channels; an agent that has narrowed
to three candidates can.* That is the whole argument.

`agile` is the held-out set. Running it early, or training on it, destroys the
generalisation claim. Nobody executes it before CP3.

---

## 6. Metrics (`eval/metrics.py`)

The evaluator holds the burst table and replays the recorded action log against it.
A detection over window `[t0,t1)` on channel `c` is a **true positive** for emitter
`e` iff some burst of `e` covers `c` and overlaps `[t0,t1)`.

| metric | definition |
|---|---|
| **POI@T** | fraction of emitters with ≥1 true positive in `[0,T]`; T ∈ {10,30,60} s |
| **TTFI prio-1** | `t_first_true_detection − t_first_activation`; median and p90, **censored at horizon**, always reported with `n_intercepted/n_total` so a policy cannot win by ignoring hard emitters |
| **Emitter-time coverage** | fraction of emitter-active seconds during which that channel was inside an in-progress dwell. **Geometric** — independent of whether the detector fired. Retune time does **not** count. |
| **ENERGY PER DETECTION** ← headline | `energy_total_J / n_unique_detections`, where unique = distinct `(emitter_id, activation_id)`. Using `activation_id` means a hopper's 20 hops count once and re-detecting one burst 50× does not inflate the denominator. |
| **Max staleness prio-1** | worst gap between consecutive coverings of a prio-1 mission channel, including `0→first` and `last→T` |
| **False alarm rate** | detections with no overlapping burst / total channel-dwells (comparable to `P_fa`, so it doubles as a calibration check), and also per second |

### Baselines
- `round_robin` — **fair-tuned**: grid-search `(bw, dwell, sweep_period)` on
  `sparse` to maximise POI@60 subject to `energy ≤ index policy's energy` **and
  surviving the full horizon**. Document the search; it defeats "you rigged the
  baseline" before it is raised.
- `random` — uniform `k_lo`, bw and dwell from the same candidate sets, paced the
  same way.
- `greedy` — index with `α=0`, **no Markov propagation** (belief = raw Laplace hit
  rate, never decays), **scheduler disabled**. Isolates exactly what the belief and
  the scheduler contribute.
- `index` — rung 1.
- `index_learned` — rung 2.
- `oracle` — clairvoyant greedy reading `truth_power`. Label it **"reference
  ceiling", not "optimal"** — it is myopic over one action. Saying so unprompted
  buys credibility and costs nothing.

---

## 7. Scheduler — hard constraints only, never scores

```python
Scheduler.select(cands, scores, belief, t, energy) -> (action, reason)
```

It **receives scores it did not compute and cannot recompute** — the learner
proposes value, the scheduler picks under hard constraints. This separation is
what makes the behaviour explainable live. Layers, in order:

1. **Feasibility** — drop candidates overrunning the horizon or the remaining budget.
2. **Revisit deadlines (hard)** — `{1: 0.5s, 2: 2.0s, 3: 10.0s}` on mission
   channels. If any channel is overdue, restrict candidates to those covering the
   *most* overdue one and take the best-scoring. This makes "max staleness
   hard-bounded" a **provable** property.
3. **Watch list** — own deadline (0.3 s), treated as priority 1.
4. **Budget pacing** — `allowed(t) = budget·(t/horizon + 0.05)`; if overspent,
   suppress scans and force `Sleep`, *except* under an active deadline override.
5. **Sleep clamp** — `dt = min(dt, next_deadline − t)`, floored at 1 ms.

Every decision returns a `reason` ∈ {`index`, `deadline:ch=k`, `watchlist:ch=k`,
`budget-pace`, `sleep`, `fallback`}, rendered beside the waterfall. That is the
"explain any decision to a judge pointing at the screen" property, and it is why
deadlines live here rather than inside the score.

---

## 8. Rung 2 — learned activity model

`FEATURE_NAMES` (F=16) is frozen in `agent/base.py`. **The raw channel index is
deliberately excluded** — it would let the model memorise emitter positions from
`sparse` and destroy generalisation to `agile`. Only *relative* spectral context is
allowed, via `nbr_recent_hits` (detections on `c±1..2` within 1 s), which is the
feature that catches the hopper.

**Label:** `y = 1` iff the **next** observation of that channel reports a
detection. So the model predicts `P(detect next)`, **not** `P(active)` — feeding
that straight into Bayes would double-count the detector. Invert it:

```
p_active_hat = clip((p_det_hat − P_fa) / (pd_bar[bw_next, τ_next] − P_fa), 1e-4, 1−1e-4)
```

Three lines, and it is the answer to *"how do you know your ML isn't just
relearning your detector?"* `τ_next`/`bw_next` are extra model inputs at train time
(see `TRAIN_EXTRA_NAMES`) so it learns the dwell dependence instead of averaging
over it.

Model: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, early_stopping=True)`
wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`. **Calibration is
mandatory** — the output feeds Bayes, so it must be a probability, not a ranking.

Trained on rung-1 logs over `{sparse, dense}` × seeds **100–119**, disjoint from
evaluation seeds 0–9. **`agile` is never trained on.**

**Three independent guarantees it cannot regress rung 1:**
1. `beta` defaults to **0.0**; the learned path is off unless explicitly enabled.
2. Gated out below `min_visits_for_model = 3`, the cold-start regime where rung 1
   is provably right (it is the prior).
3. **Automatic Brier gate at load** — if the model does not beat rung-1's Brier on
   held-out log rows, force `beta = 0`, making the demo path bit-identical to rung 1.

---

## 9. Scope

**IN:** simulator (≤30 emitters, 1 MHz grid) · receiver with `P_d`, `P_fa`, retune
latency · explicit energy accounting · belief with decay · index scheduler with
revisit deadlines · learned activity model · three scenarios · evaluation harness
with four baselines · waterfall dashboard with scan trace.

**OUT — say this out loud in the write-up; scoping discipline is scored:** real SDR
hardware · signal classification / modulation recognition · direction finding /
AoA · multi-receiver or networked sensors · pulse-level PRI deinterleaving · deep
RL (rung 3) · anything needing a real RF dataset.

---

## 10. Conventions

- Python 3.14. numpy 2.5, scipy 1.18, pandas 3.0, sklearn 1.9, pyyaml, matplotlib 3.11.
- **`pytest` is NOT installed.** Tests use stdlib `unittest.TestCase`, run with
  `python -m unittest discover -s eval/tests -t .`. Keep any single test under 2 s
  and the whole suite under 30 s so it can run at every merge.
- Vectorise with numpy; no Python loop over 200 channels in a hot path.
- Every result carries its `config_hash` and seed.
- Seeds: evaluation **0–9**, rung-2 collection **100–119**. Never overlap them.
- `matplotlib` imports must be wrapped in `try/except ImportError` with a CSV/ASCII
  fallback. The pitch never depends on a package being present.
- Comment the *why*, not the *what*. Every magic constant gets a one-line
  justification.

---

## 11. Findings from the tuning loop (measured, not assumed)

Five defects were found by measurement while getting the index policy to beat the
sweep. Each is recorded because the *reasoning* is the deliverable, not just the
constant.

### 11.1 The grid was too narrow, and it deleted the premise

**This was the big one.** At 200 channels a 5 MHz/10 ms sweep has a period of
0.42 s, so it revisits every channel **4.7x during a typical 2 s emitter burst**.
Blind coverage already sees essentially every activation, so there is nothing for
a belief to exploit and adaptivity can only add overhead. No amount of tuning can
beat a sweep in that regime — and four successive tuning attempts confirmed it.

The source problem statement describes a **0-5 GHz** span: *"if your receiver has
just scanned 1 GHz, it may take several more scanning steps before reaching 4 GHz
-- that creates intercept/detection delay."* Sweep latency **is** the problem.
Choosing a 200 MHz grid quietly removed it.

Widened to **2000 channels / 2 GHz**. Sweep period becomes 4.2 s against ~2 s
bursts, so a sweep now misses more than it catches, and the result inverts:

| policy | energy per detection | POI@60 | unique detections |
|---|---|---|---|
| round_robin | 2.000 J | 0.333 | 3 |
| index | **0.597 J** | 0.333 | **11** |
| oracle (ceiling) | 0.093 J | 1.000 | 66.3 |

**70% lower energy per detection at identical POI.** If a future change makes the
sweep competitive again, check the sweep period against the emitter on-time before
touching the policy.

### 11.2 Priority weights must exceed the dwell they require

At `w = {1: 0.1, 2: 0.03, 3: 0.01}` a priority-1 detection was worth 0.100 J while
the 100 ms dwell a -20 dB emitter *needs* costs 0.102 J. Long dwells could never
repay their own energy **even at certainty**, so the policy correctly refused to
ever look. Raised 10x to `{1: 1.0, 2: 0.3, 3: 0.1}`, which puts break-even at
p ~ 0.12: speculative long dwells stay negative at the prior, confident ones pay.

### 11.3 The prior is P(channel active), not emitter duty cycle

First attempt at per-class priors used `{3: 0.40}` — the routine emitters' *duty
cycle*. But the belief needs *P(this channel is occupied AND radiating)*: 4
emitters over 150 routine channels at 41% duty is ~1.1% of channels live, not 41%.
The 36x overestimate made wide scans look enormously profitable (a 20 MHz window
appeared to contain 8 expected active emitters instead of 0.2).

### 11.4 A static prior throws away the only durable thing the agent learns

`p` answers "is this channel radiating right now" — transient, and it should decay.
But a detection also proves an emitter *lives* there, which is permanent. Decaying
to a band-wide constant discards that, so the agent re-derived the spectrum's
layout from scratch every couple of seconds.

Fixed with one Beta-Bernoulli posterior per channel (`prior_pseudocount`): belief
decays toward that channel's **own** observed hit rate. After 10 visits a channel
with 4 detections holds a 0.254 floor while a barren one falls to 0.0041 — a 62x
separation, and the asymmetry is the mechanism by which adaptivity beats a sweep.

### 11.5 Coverage without adequate dwell is not coverage

A deadline names *which* channel but the reward-rate score still chose the dwell —
and rate always prefers the shortest action, because duration is its denominator.
The observed result was a 2 ms look at a -20 dB threat emitter: `P_d = 0.004`. The
deadline was satisfied, staleness looked healthy, and priority-1 POI was exactly
zero. `deadline_min_pd` now requires a deadline visit to be long enough to actually
see the class it is checking (50 ms on threat, 10 ms on prio-2, 1 ms on routine).

### 11.6 Priority-1 was infeasible by 20x, and that is a real result

`prio_1 POI = 0` for every non-clairvoyant policy is **not a metric bug**. On the
original 200-channel grid: `D_1 = 0.5 s` over 20 threat channels demands 40
channel-visits/s, and at the 50 ms needed for `P_d ~ 0.5` that is **2.0 s of dwell
per second against a budget of 0.100 s/s — over by 20x**. Even spending the entire
6 J in the threat band yields 0.012 s of expected overlap with 2.34 s of ON time.

The oracle succeeds only because it knows *when* to look: ~0.2 s of
perfectly-placed dwell per burst. Deadlines were reset to values whose total
mandated dwell actually fits the budget (`{1: 30, 2: 8, 3: 20}` s). Quote a
staleness guarantee only after checking it against the budget arithmetic.

### 11.7 Raising the exploration bonus does NOT fix the POI deficit (falsified)

The index policy wins on energy (0.617 vs 1.650 J/detection, 62.6% lower) but
loses on POI@60 (0.229-0.333 vs 0.417). It gets *more* unique detections but
covers *fewer distinct emitters* — it harvests repeat activations from known
channels instead of finding new ones.

The obvious hypothesis was that the adaptive prior (§11.4) over-weights known
channels: a productive channel holds a ~0.254 floor against 0.011 for an unvisited
one (23x), while the staleness bonus maxes at 11x, so exploitation wins by ~3x.
Flipping that needs `0.011*(1+20*alpha) > 0.254*1.5`, i.e. `alpha > 1.68`.

**Measured on sparse, seeds 0-2 — the hypothesis is false:**

| config | J/detection | POI@60 |
|---|---|---|
| round_robin | 1.650 | **0.417** |
| alpha=0.5 (current) | **0.597** | 0.333 |
| alpha=2.0 | 1.952 | 0.167 |
| alpha=4.0 | 1.438 | 0.250 |
| alpha=8.0 | 1.333 | 0.167 |
| alpha=4.0, pseudocount=2.0 | 2.332 | 0.292 |

Every raised `alpha` is worse on **both** axes. The balance does flip as the
arithmetic predicted, but chasing staleness across 2000 mostly-empty channels burns
energy faster than it finds emitters — with 8 emitters in 2000 channels, an
untargeted revisit is almost certainly wasted, so a stronger exploration bonus buys
nothing and costs budget.

**Implication:** the POI gap is not an exploration-weight tuning problem. Either it
needs a mechanism that distinguishes "this channel is worth a first look" from
"this channel is stale" (an explicit information-gain term, not a staleness
multiplier), or POI parity is simply not achievable for a belief-driven policy at
this emitter density and the honest claim is the energy one alone. Do not re-run
the alpha sweep; it is recorded here as closed.

### 11.8 Exploration is not the binding constraint (three hypotheses rejected)

Continuing §11.7. The POI deficit (index 0.333 vs round_robin 0.417) was attacked
three ways. **All three failed**, and the failures are informative enough to record
so nobody repeats them.

Diagnostic that framed it — index on `sparse` seed 0, 2000-channel grid:
- visits only **494 of 2000 channels (24.7%)**
- picks **1 MHz x 1 ms for 1217 of 1455 scans** (single-channel exploit scans)
- a full-band 20 MHz x 2 ms discovery sweep costs **0.40 J — 15 fit in the budget**,
  while round_robin's 5 MHz x 10 ms sweep costs 4.80 J and manages only 1.25

| hypothesis | result | verdict |
|---|---|---|
| raise `alpha_staleness` 0.5 -> 2/4/8 | POI 0.333 -> 0.167/0.250/0.167 | rejected (§11.7) |
| additive `info_gain_weight` 0.0005-0.03 | POI 0.333 -> 0.167-0.292, J/det all worse | rejected |
| `score_mode: raw` (doc's literal form) | POI 0.083, J/det inf, 16% coverage | rejected |

**Why round-robin wins POI:** it covers all 2000 channels once at 5 MHz x **10 ms** —
an adequate dwell (`P_d ~ 0.72-0.85` for routine emitters). The index covers 25% of
channels at 1 MHz x **1 ms**, a dwell too short to detect reliably. The gap is not
how much it explores but that **every action it picks is too short to detect**.

**Why that happens, and why it is hard:** the reward-RATE index divides by duration,
so it systematically prefers the shortest action. `score_mode: raw` removes that
bias and immediately over-corrects — it picks 20 MHz x 100 ms, burns the budget, and
covers 16% of the band. Rate under-dwells; raw over-dwells. Neither form selects a
*mid* dwell, because neither has any term that values detection reliability per se.

**Conclusion:** the fix is not a weight. It needs the score to represent the
probability of *actually detecting*, not expected value per unit time — e.g.
optimising expected distinct-emitters-found rather than expected weighted
detections, which is what POI actually measures. That is a redesign of the
objective, not a tuning pass, and it is out of scope for the current build. The
honest headline is therefore the energy result alone (§11.1), with the POI deficit
reported rather than papered over.

`info_gain_weight` is left implemented but defaulted to 0.0 — the machinery is
sound and cheap, it simply does not help at this emitter density.

### 11.9 The claim holds on `dense`, and the reason is the interesting part

Full ablation, 5 policies x 2 scenarios x 5 seeds (`results/runs.csv`), median
energy per detection and mean POI@60:

| scenario | policy | J/detection | POI@60 | unique det |
|---|---|---|---|---|
| **dense** | round_robin | 0.500 | 0.350 | 10.8 |
| | random | 1.747 | 0.092 | 3.0 |
| | greedy | 0.544 | 0.325 | 11.0 |
| | **index** | **0.181** | **0.400** | 32.0 |
| | oracle | 0.333 | 0.525 | 18.6 |
| **sparse** | round_robin | 1.999 | 0.400 | 3.6 |
| | random | inf (zero detections) | 0.000 | 0.0 |
| | greedy | 5.999 | 0.150 | 1.2 |
| | **index** | **0.667** | 0.275 | 10.4 |
| | oracle | 0.109 | 0.825 | 47.2 |

**On `dense` the index policy wins BOTH axes**: 63.8% lower energy per detection
*and* higher POI (0.400 vs 0.350). That is the full claim, on the harder scenario.

**On `sparse` it wins energy only** (66.7% lower) and trails on POI (0.275 vs
0.400) — the deficit analysed and left open in §11.8.

**Why the two differ, and it is not a tuning artefact.** Adaptive scanning pays
when the environment has enough structure to learn. `dense` has 24 emitters, so a
belief built from the agent's own observations is genuinely informative and
revisiting pays. `sparse` has 8 emitters spread over 2000 channels, so belief
cannot help with *discovery* — there is almost nothing to infer from a miss on an
empty channel — and a blind sweep remains competitive on breadth. State this
directly rather than reporting only the favourable scenario: the honest claim is
"adaptive scanning pays once the environment is dense enough to learn from", which
is a more useful engineering result than an unqualified win.

**`greedy` earns its place in the table.** On `sparse` it collapses to 5.999
J/detection at POI 0.150 — far worse than both `index` and `round_robin`. Since
`greedy` is `index` with Markov propagation removed, the staleness bonus zeroed and
the scheduler disabled, that gap is a direct measurement of what the belief decay
and the constrained scheduler contribute.

**Caveat that must travel with the headline number.** On `dense`, `index` (0.181)
beats the *oracle* (0.333) on energy per detection. The oracle is not broken: it
optimises priority-*weighted* gain and takes the higher POI (0.525 vs 0.400), while
`index` accumulates more cheap routine detections (32 vs 18.6). But it does show
that **energy per detection alone can be gamed by harvesting cheap detections**, so
it must never be quoted without POI beside it.

### 11.10 Held-out `agile`: the result generalises, and the oracle stops being a ceiling

`agile` was never tuned on and never trained on (tuning was closed at §11.8 before
it was run once). 3 policies x 5 seeds:

| policy | J/detection | POI@60 | unique det |
|---|---|---|---|
| round_robin | 2.999 | 0.171 | 2.8 |
| **index** | **0.600** | **0.386** | 9.6 |
| oracle | 1.198 | 0.286 | 4.8 |

**80% lower energy per detection AND 2.25x the POI**, on unseen scenarios. This is
the strongest evidence in the project, and it is the least surprising: `agile`'s
emitters hop channel every 50 ms, which is precisely the regime where a fixed sweep
pattern is worst and where relative spectral context is worth most.

Read together with §11.9 the trend is monotone in how much structure there is to
exploit:

| scenario | emitters | energy result | POI result |
|---|---|---|---|
| sparse | 8 / 2000 ch | 66.7% lower | **loses** 0.275 vs 0.400 |
| dense | 24 / 2000 ch | 63.8% lower | wins 0.400 vs 0.350 |
| agile | 14, hopping | 80.0% lower | wins 0.386 vs 0.171 |

The honest claim is therefore conditional and more useful than an unqualified one:
**adaptive scanning pays once there is enough structure to learn, and pays most
against agile emitters — but a blind sweep is hard to beat on pure discovery in a
very sparse band.**

**The oracle is NOT a valid upper bound on `agile`.** `index` beats it on both axes
(0.600 vs 1.198 J/detection, POI 0.386 vs 0.286). This is myopia, not a defect: the
oracle is clairvoyant *greedy* over a single action, so against emitters that hop
every 50 ms it commits to a dwell on a channel the emitter has already left.
Clairvoyance about the present does not help when the target moves within your
action. This is strong evidence for keeping the "reference ceiling, not optimal"
label — but on `agile` it must not be presented as a bound at all, only as a
comparison point, and the reason must be stated.

### 11.11 Known non-invariant: `Obs.t` can exceed `horizon_s` by up to one retune

Found by `eval/tests/test_sim_env.py` and independently reproduced.

`World._step_scan` truncates the **dwell** against the horizon but not the
**retune** that precedes it:

```python
t0 = self.t + t_retune                                   # unbounded
dwell = min(action.dwell_s, max(0.0, self.horizon_s - t0))
self.t = t0 + dwell
```

So a scan issued at `t == horizon_s` still charges its full retune. Measured on
the shipped 2 GHz grid with a full-span hop: horizon 0.100 s, resulting
`Obs.t = 0.1403` — an overrun of **40.3 ms**, matching `t_settle + 2e9/50e9`.

**Why it is documented rather than fixed.** The executed dwell truncates to
exactly 0.0, so no sensing happens in the overrun; `done` is already True; and
every metric censors at `T`, so no reported number moves. Charging the retune
energy is also defensible physics — the receiver really did spend that time
slewing. Clamping the clock instead would make it lie about elapsed time.

**What to watch.** `Obs.t <= horizon_s` is NOT an invariant and must not be
assumed — notably by anything plotting a time axis from a trace, which will see
a slightly over-long final step. A passing test pins the current behaviour so a
future change to it is a deliberate decision rather than an accident.

### 11.12 The deadline bound does NOT hold — the guarantee in §7 is false as written

§7 claims the hard revisit deadline makes "max staleness hard-bounded" a
*provable* property. **It is not.** Found by `eval/tests/test_agent_scheduler.py`,
which ships an intentionally failing test, and confirmed independently against
the shipped `results/runs.csv`:

Bound = `D_1 + max_dwell + max_retune` = 30 + 0.2 + 0.0045 = **30.2045 s**.
**11 of 15 index episodes exceed it**, and on `sparse` max staleness reaches the
full 60 s horizon — i.e. some priority-1 channels are never visited at all.

| scenario | seeds over bound | worst max staleness |
|---|---|---|
| sparse | 5 / 5 | **60.00 s** (horizon) |
| dense | 4 / 5 | 31.93 s |
| agile | 3 / 5 | 30.70 s |

**Mechanism 1 — overdue-ness is ranked in absolute seconds.**
`agent/scheduler.py` computes `over = stale - deadline_prio` and takes
`argmax(over)`. Deadlines are `{1: 30, 2: 8, 3: 20}` s, so the *highest* priority
class has the *longest* deadline: at t = 30 s a routine channel is already +10 s
overdue while a threat channel is only at 0. Priority-1 is therefore structurally
last in the very mechanism meant to protect it. A prio-2 channel 32 s overdue
outranks a prio-1 channel 10 s overdue even when the prio-1 candidate scores 9x
higher.

**Mechanism 2 — the candidate set is deadline-blind.**
`IndexPolicy._enumerate` shortlists the top `windows_per_bw` windows *by value*.
When nothing in that shortlist covers the most-overdue channel,
`_deadline_override` returns `None` and the "hard" constraint is silently dropped
with no record. Instrumented: the override wanted to fire on **641 of 986
decisions and was dropped for want of a covering candidate on 635 — 99.1%**.
A prio-1 deadline override fired **zero times in every run measured**.

**Consequence for the write-up.** Do not claim a bounded revisit guarantee. The
deadline layer is *best-effort*, and on `sparse` it is close to inoperative for
priority 1. The §11.5 min-dwell rule is sound where it fires but is end-to-end
vacuous for prio-1 for the same reason — no override fires to apply it to.

**The likely fix** (not applied; it changes scheduling behaviour and would
invalidate the current results table): rank overdue-ness *relative* to the
deadline (`stale / deadline`) rather than in absolute seconds, and have the
enumerator inject a covering candidate for the most-overdue channel so the
override always has something to select.
