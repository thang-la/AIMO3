# AIMO3 System Upgrade: From “Best-Solver” to **Risk‑Aware Double‑Run Decision System**

This document upgrades the previously designed AIMO3 solver into a **decision-theoretic**, **calibrated**, and **decorrelated** system optimized for the **double‑run penalized evaluation**:

[
\text{score}=
\begin{cases}
1 & \text{if both runs correct}\
0.5 & \text{if exactly one correct}\
0 & \text{if both wrong}
\end{cases}
]

A crucial (and often-missed) identity:

Let (I_A, I_B \in {0,1}) be correctness indicators for runs A and B. Then

[
\text{score} = \frac{I_A + I_B}{2}
\quad\Rightarrow\quad
\mathbb{E}[\text{score}] = \frac{\mathbb{E}[I_A] + \mathbb{E}[I_B]}{2}
]

So **expected score is the average expected accuracy across the two runs**; **run correlation does not change the expectation** *given fixed per-run correctness probabilities*.
However, correlation *does* matter operationally because:

* we can *change* per-run correctness by running **different reasoning programs** (the whole point of the second run),
* miscalibration produces **overconfident wrongs**; decorrelation and adversarial reruns reduce “both wrong” events and stabilizes the realized score (important for “reliably >47/50”),
* compute/timeouts/bugs introduce nonlinear failure risk—decorrelation reduces “both fail the same way”.

This upgrade therefore focuses on **(i) correctness probability calibration**, **(ii) correlated-error detection**, and **(iii) a policy that maximizes expected score while being robust to overconfidence and systematic failure**.

---

## 0) What Changes vs the Previous Design (Not a Rewrite)

We keep your previous core blocks:

* Router (domain/difficulty)
* Multi-path generation (P0–P3)
* Tool sandbox (Python/SymPy/Z3)
* Verification stack + judge
* Hard-mode loop + progressive deepening
* Budget manager + early stop

**We upgrade the glue and decisions**:

* Add **Answer Reliability Model (ARM)** to output calibrated (P(\text{correct})) per candidate.
* Add **Correlated-Error Detection (CED)** to measure similarity and shared-failure risk.
* Replace “pick highest verifier score” with **Bayesian Answer Aggregator (BAA)** producing a posterior over answers.
* Add **Expected Score Maximization Policy (ESMP)** that chooses output using calibrated posteriors + run program context.
* Add **Run Decorrelation Engine (RDE)** with **multi-policy reasoning programs** (Run-A constructive/tool-heavy; Run-B adversarial/invariant-seeking).
* Add **Adversarial Second-Pass Reasoning (ASPR)** to attack the current best answer.
* Add **Ambiguity-Aware Sampling (AAS)** (rarely used; controlled).
* Add **failure-mode-specific fallbacks** (targeted recovery recipes).

---

# 1) Updated System Architecture (ASCII)

### 1.1 New risk-aware pipeline overlay

```
          candidates + traces + tool artifacts + verifier flags
                              │
                              ▼
                   ┌────────────────────┐
                   │  ARM (calibration) │  -> p_i, u_i (uncertainty)
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │ CED (correlation)  │  -> ρ_ij, clusters
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │ BAA (posterior)    │  -> P(answer=a | evidence)
                   └─────────┬──────────┘
                             ▼
                   ┌────────────────────┐
                   │ ESMP decision      │  -> final answer (maybe sampled)
                   │ + RDE run program  │
                   └────────────────────┘
```

### 1.2 Two-run programming (implemented via stochastic run seed)

```
          run_seed (os.urandom) -> run_program_id ∈ {A,B}

Run A: constructive/tool-heavy
Run B: adversarial/contradiction-based
```

We cannot explicitly know “this is private run #1 vs #2”. So we implement **stochastic run program selection**; across two private reruns, programs differ with probability ~0.5, which is enough to decorrelate failure modes.

---

# SECTION A — Failure Analysis of the Previous Design

## A1) Overconfident wrong answers

**Why it happens in the previous system**

* The prior design gated on verification signals (constraints passed, tool succeeded, judge score).
* On many olympiad problems, especially geometry/comb, **weak verifiers** can be passed by wrong reasoning (e.g., code solved a *mis-modeled* problem; or a derived equation was incorrect but internally consistent).
* Self-consistency voting often amplifies confident wrong modes: multiple paths share the same mistaken lemma.

**Plateau mechanism**

* You get a “clean” single answer that looks verified → returned deterministically → both private reruns likely output the same wrong → 0 on that problem.
* These are rare but dominate the last jump from ~45 to 47+.

**Fix**

* Introduce ARM + CED + adversarial refutation to explicitly estimate and downweight “verified-but-fragile” candidates.

---

## A2) Correlated reasoning traces across reruns

**Why it happens**

* Even with small temperature differences, the same base prompt + same generation recipes produce highly similar lemma sequences and tool scripts.
* The dominant error type becomes **systematic** (e.g., a missing case template, a geometry misinterpretation template).
* Reruns don’t behave like independent attempts; they repeat the same mistake.

**Fix**

* Run Decorrelation Engine + Multi-policy programs + CED to **force diversity in reasoning space**, not just sampling noise.

---

## A3) Multi-modal solution spaces (geometry, combinatorics)

**Why it hurts**

* Many problems have multiple plausible “structures” (coordinate bash vs angle chase; recursion vs bijection; different invariants).
* Your previous selection logic prefers a single highest “verification score”, but on these tasks the **verification signal is often underdetermined**, so the wrong mode can win.

**Fix**

* Use BAA to aggregate evidence across clusters of reasoning modes and quantify posterior mass.

---

## A4) Hard problems with unstable reasoning

**Symptoms**

* Small perturbations of constants or alternative derivations swing the answer.
* Tool code produces an answer but is sensitive to parsing choices or hidden assumptions.

**Fix**

* Add sensitivity features to ARM; add ASPR perturb-and-refute; add compute stopping based on marginal expected gain.

---

## A5) Verification blind spots

**Examples**

* Geometry: verifier cannot validate a synthetic proof without formalization.
* Combinatorics: brute force only on tiny instances; might not generalize.
* Number theory: modular checks can pass for many wrong derivations if constraints are weak.

**Fix**

* ARM trained on “passes-verifier-but-wrong” negatives + CED (detect shared blind spot clusters) + failure-specific fallback logic.

---

# SECTION B — Calibrated Correctness Probability: Answer Reliability Model (ARM)

## ARM — Module Specification

### Purpose

Estimate calibrated correctness probability:
[
p_i = P(\text{candidate } i \text{ correct}\mid \text{features})
]
and an uncertainty estimate (u_i) (epistemic uncertainty).

### Inputs

Per-candidate feature vector (x_i) (detailed schema below).

### Outputs

* `p_i ∈ [0,1]`: calibrated probability candidate is correct
* `u_i ∈ [0,1]`: uncertainty (from ensembles / MC dropout / conformal bands)
* Optional: `failure_type_logits` (arith, missing-case, geometry, misparse, etc.)

### Feature Schema (concrete)

**Group 1: Verifier signals**

* `hard_constraints_pass` (bool)
* `num_constraints_extracted`
* `constraint_tightness` (0–1): fraction of constraints that are *binding* (e.g., modulus explicitly used; range constraints used)
* `sympy_simplify_success` (bool)
* `num_random_tests`, `random_test_pass_rate`
* `z3_sat_consistency` (tri-state: sat/unsat/unknown)

**Group 2: Tool execution stability**

* `python_exec_success` (bool)
* `python_runtime_ms`
* `python_output_stability`: run same code 3x with randomized harmless reorderings; fraction consistent
* `ast_complexity`: nodes count, loops depth
* `uses_floats` (bool) + `float_usage_count` (red flag)

**Group 3: Multi-path agreement**

* `unique_answer_count`
* `vote_share_of_answer`
* `path_diversity`: entropy over path types (P0/P1/P2/P3)
* `cluster_count` (from CED)
* `top_answer_cluster_support`: total weight of clusters supporting top answer

**Group 4: Reasoning entropy & confidence**

* `avg_token_logprob` (if available)
* `answer_margin`: difference between top and second posterior after BAA (or pre-BAA vote margin)
* `trace_length_tokens`
* `self_contradiction_flags` count (detected by pattern rules + judge)

**Group 5: Sensitivity / adversarial tests**

* `perturbation_flip_rate`: fraction of small constant perturbations causing answer change (via ASPR)
* `alternate_method_agreement`: whether a second method reproduces answer

**Group 6: Problem metadata**

* domain one-hot, estimated difficulty
* geometry diagram indicators (keywords)
* modulus present and magnitude

### Training Data Construction (from self-play logs)

For each training problem (P):

1. Generate candidates under multiple policies (A/B + subpolicies).
2. Run the verifier stack to label candidate as **correct/incorrect** (ground truth from known answer in training set).
3. **Hard-negative mining:** keep incorrect candidates that:

   * pass hard constraints,
   * pass random tests at high rate,
   * have high vote share,
   * are produced by tool code.
     These are exactly the “overconfident wrong” cases ARM must learn.

Dataset record:

```
(problem_id, candidate_id, features x_i, label y_i ∈ {0,1}, failure_type)
```

### Loss Function

Base:
[
\mathcal{L}_\text{BCE} = -\sum_i \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
]

Add calibration regularizer (Brier + ECE proxy):
[
\mathcal{L} = \mathcal{L}*\text{BCE} + \lambda \cdot \mathcal{L}*\text{Brier}
]
Optionally focal loss for extreme imbalance on easy tasks.

### Calibration Method

Two-stage:

1. Train ARM model (GBDT or small MLP).
2. Post-hoc calibration on held-out split:

* **Temperature scaling** if ARM outputs logits
* or **Isotonic regression** for best nonparametric calibration

We store the calibrator parameters and apply at inference.

### Inference Integration

ARM runs after verification for each candidate and outputs calibrated `p_i`. These `p_i` feed BAA and ESMP.

### ARM Pseudocode

```python
def ARM_predict(candidate_features: dict) -> tuple[float, float, dict]:
    # returns (p_correct, uncertainty, failure_logits)
    logit = arm_model.forward(candidate_features)
    p_raw = sigmoid(logit)
    p_cal = calibrator(p_raw)   # isotonic or temperature-scaled
    u = ensemble_uncertainty(candidate_features)  # e.g., variance across K models
    failure = failure_head(candidate_features)    # optional
    return p_cal, u, failure
```

---

# SECTION C — Correlated Error Modeling (CED)

We need an estimate of **shared-failure risk** across candidates and across run policies.

## CED — Module Specification

### Purpose

Estimate correlation in failure modes between two candidates (i,j):

* not for expectation math directly,
* but to avoid being fooled by “many identical wrong proofs” and to drive decorrelation.

### Inputs

* traces (text), extracted lemmas, intermediate values
* tool ASTs + outputs
* path metadata (policy id, prompt program id, domain)

### Outputs

* pairwise correlation proxy `ρ_ij ∈ [0,1]`
* clustering of candidates into “reasoning families”
* per-answer support by independent families (used in BAA)

### Similarity Metrics (concrete)

**Trace similarity**

* cosine similarity of sentence embeddings of trace summary
* token n-gram Jaccard on normalized math tokens
* “lemma overlap”: set overlap of recognized theorems/keywords (LTE, CRT, inversion, barycentric, etc.)

**Intermediate overlap**

* overlap of extracted intermediate numeric values (moduli, gcds, counts)
* overlap of derived equations (normalized string hashes)

**Tool/code similarity**

* AST edit distance / tree hash similarity
* same loop structure + same sympy calls

**Mode similarity**

* same policy id + same path type increases baseline correlation

### Defining a correlation coefficient

We define:

[
\rho_{ij} = \sigma\left(w^\top \phi_{ij}\right)
]

where (\phi_{ij}) includes all similarity metrics above, and (\sigma) is sigmoid.

### Training (\rho_{ij})

From self-play logs: for many candidate pairs on the same problem, we know correctness labels ((y_i, y_j)). We train (\rho) to predict **excess co-failure** beyond independence:

Let (p_i, p_j) be ARM-calibrated.
Target:
[
t_{ij} = \mathbb{1}[y_i=0 \wedge y_j=0] - (1-p_i)(1-p_j)
]
Train a regressor to predict (t_{ij}), then map to (\rho_{ij}).

### Inference usage

* Cluster candidates using (1-\rho_{ij}) distance.
* Downweight redundant candidates in BAA.
* Trigger decorrelation: if top candidates all from one cluster → generate new candidates from different policy family.

### CED Pseudocode

```python
def CED_pairwise(cands) -> tuple[np.ndarray, list[list[int]]]:
    # rho matrix and clusters
    rho = np.zeros((len(cands), len(cands)))
    for i in range(len(cands)):
        for j in range(i+1, len(cands)):
            feat = pair_features(cands[i], cands[j])
            rho_ij = sigmoid(w @ feat)
            rho[i,j] = rho[j,i] = rho_ij
    clusters = cluster_by_threshold(rho, tau=0.7)  # high rho -> same cluster
    return rho, clusters
```

---

# SECTION D — Bayesian Answer Aggregation (BAA) + Decision Policy (ESMP)

## D1) BAA — Purpose

Convert candidate-level calibrated probabilities into a posterior distribution over **unique answers**, correcting for correlated evidence.

### Inputs

For each candidate (i):

* claimed answer (a_i)
* ARM (p_i), uncertainty (u_i)
* cluster assignment (c(i)), correlation matrix (\rho)

### Outputs

* posterior over answers ( \pi(a) = P(T=a \mid \text{evidence}) ) for the set of observed answers
* “other” bucket mass (\pi(\text{other}))

## D2) Correlation-aware evidence pooling

We treat candidates in the same cluster as partially redundant.

Define cluster weight:
[
w_i = \frac{1}{1 + \sum_{j\neq i} \rho_{ij}}
]
(or simpler: (w_i = 1/|cluster|) within a cluster)

Compute per-answer log-score:
[
S(a) = \sum_{i: a_i=a} w_i \cdot \log\frac{p_i}{1-p_i}
]
Then posterior over observed answers:
[
\pi(a) \propto \exp(S(a)) \cdot \text{prior}(a)
]
with a small prior favoring answers supported by multiple independent clusters.

“Other” bucket:
[
S(\text{other}) = \log \epsilon
]
where (\epsilon) is small (e.g., (10^{-6})), unless we detect “all candidates low reliability”, in which case increase (\epsilon).

## D3) Expected score formulas & optimality

Let run output be (A), truth be (T). Under posterior (\pi):

* If we output answer (a) deterministically in a run:
  [
  P(\text{correct}) = \pi(a)
  ]

With two runs, expected score:
[
\mathbb{E}[\text{score}] = \frac{\pi(a_A) + \pi(a_B)}{2}
]

**If we could choose (a_A) and (a_B) explicitly**, the maximum is achieved by choosing the top posterior answer for both runs.

**Why then do we still need BAA/ESMP?**
Because the previous system did not reliably estimate (\pi(\cdot)) and was vulnerable to:

* “lots of correlated wrong candidates” inflating confidence,
* verification blind spots,
* unstable hard problems.

BAA is what turns “many signals” into a **calibrated posterior** instead of an overconfident vote.

## D4) ESMP — Expected Score Maximization Policy

ESMP chooses the final answer for *this* run, and decides whether to spend more compute.

Key idea: since expected score depends on per-run correctness, ESMP’s job is to **maximize per-run correctness probability** while preventing known traps:

* returning an overconfident wrong answer too early,
* wasting compute when marginal gain is negligible.

### Deterministic vs stochastic output

* **Deterministic output is optimal** when posterior is sharp and evidence comes from multiple independent clusters.
* **Stochastic output is a last resort** when posterior is flat and ARM uncertainty is high, because:

  * it reduces the chance that both private reruns output the exact same brittle wrong answer (robustness),
  * it helps when calibration is imperfect and the top answer is “overconfident wrong”.

We implement stochasticity only under strict ambiguity gates (Section F).

### BAA Pseudocode

```python
def BAA_posterior(cands, arm_ps, rho, clusters):
    # compute weights w_i
    w = []
    for i in range(len(cands)):
        w_i = 1.0 / (1.0 + sum(rho[i,j] for j in range(len(cands)) if j != i))
        w.append(w_i)

    # accumulate log-odds per answer
    S = defaultdict(float)
    for i, cand in enumerate(cands):
        p = clip(arm_ps[i], 1e-6, 1-1e-6)
        S[cand.answer] += w[i] * math.log(p/(1-p))

    # prior: slight boost for answers supported by >=2 clusters
    prior = defaultdict(lambda: 0.0)
    cluster_support = answer_cluster_support(cands, clusters)
    for a, k in cluster_support.items():
        prior[a] = 0.3 * min(k-1, 3)  # log-space bump

    # compute normalized posterior
    logits = {a: S[a] + prior[a] for a in S.keys()}
    pi = softmax_dict(logits)

    # other mass
    pi_other = adaptive_other_mass(arm_ps, clusters)
    pi = renormalize_with_other(pi, pi_other)
    return pi
```

---

# SECTION E — Run Decorrelation Engine (RDE) + Multi‑Policy Reasoning Programs

## RDE — Module Specification

### Purpose

Ensure the two private reruns explore **meaningfully different reasoning trajectories**, not just temperature noise, to:

* uncover alternative candidates,
* reduce systematic blind spots,
* increase per-run correctness on the hard tail.

### Inputs

* `run_seed`
* problem metadata (domain/difficulty)
* current candidate set diagnostics (cluster count, posterior sharpness)

### Outputs

A `RunConfig`:

* selected reasoning program (A or B)
* prompt templates
* search strategy parameters (sampling counts, tool priority)
* verifier weighting scheme
* stopping criteria thresholds

### Run A program: constructive/tool-heavy

**Intent:** minimize silly errors; formalize; compute; verify hard constraints.

* Prompt: “derive formal model → write SymPy/Python → compute answer → validate constraints”
* Strategy: prioritize P1 (TIR) + P0 (symbolic-first), minimal P2
* Verifier weighting: heavy on tool stability + constraints
* Stopping: early stop when 2 independent tool runs agree + constraints pass

### Run B program: adversarial/invariant/contradiction-based

**Intent:** attack assumptions; seek alternative invariants; refute Run-A-style solutions.

* Prompt: “assume candidate answer; attempt to derive contradiction; search for invariant; alternative method”
* Strategy: prioritize P2 + P3 + ASPR; still uses tools but more for *counterexample search* than direct solve
* Verifier weighting: heavy on sensitivity tests + contradiction flags
* Stopping: stop only after ASPR fails to refute top posterior and at least 2 clusters support it

### Why this increases expected score

Because Run B isn’t “worse diversity”; it’s a **specialized error-catcher**. It improves per-run correctness specifically on:

* overconfident wrongs from mis-modeled tool solutions,
* missing casework,
* geometry misreads,
* combinatorics overcounts.

### RDE Pseudocode

```python
def RDE_select_run_config(run_seed, meta, diagnostics):
    bit = hash32(run_seed + meta.problem_hash) & 1  # stable within run, differs across reruns
    if bit == 0:
        return RunConfig(program="A", tool_priority=True, adversarial=False,
                         stop_pi=0.92, require_clusters=1)
    else:
        return RunConfig(program="B", tool_priority=False, adversarial=True,
                         stop_pi=0.88, require_clusters=2)
```

---

# SECTION F — Ambiguity‑Aware Answering (Controlled Stochastic Output)

## AAS — Module Specification

### Purpose

When posterior remains ambiguous and evidence is fragile, use controlled stochasticity to:

* reduce repeated brittle wrong outputs across reruns,
* avoid catastrophic “both wrong the same way” failures,
* while preserving deterministic correctness on easy problems.

### Inputs

* posterior (\pi(a)) from BAA
* ARM uncertainty on top answers (u)
* cluster support count for top answers
* run program id (A or B)

### Outputs

* final selected answer (deterministic or sampled)
* sampling metadata (for logs)

### Decision boundary rule (concrete)

Let (a_1, a_2) be top-2 answers.

We sample **only if all** hold:

1. Posterior is flat:
   [
   \pi(a_1) < \tau_\text{flat} \quad \text{and} \quad \pi(a_1) - \pi(a_2) < \Delta_\text{small}
   ]
   e.g., (\tau_\text{flat}=0.65), (\Delta_\text{small}=0.08)

2. Evidence is fragile:

   * cluster_support(a1) == 1 (single reasoning family)
   * or perturbation_flip_rate high
   * or ARM uncertainty (u(a_1) > u_\text{high})

3. Not an “easy problem”:

   * meta.difficulty > 0.5

Sampling distribution:

* Temperature-flattened posterior with anti-collapse:
  [
  q(a) \propto \pi(a)^{1/T} \cdot \text{independence_bonus}(a)
  ]
  where independence_bonus boosts answers supported by different clusters.

### AAS Pseudocode

```python
def AAS_select_answer(pi, diagnostics, run_config, rng):
    a1, p1 = top1(pi)
    a2, p2 = top2(pi)

    if (p1 >= run_config.stop_pi) and diagnostics.cluster_support[a1] >= run_config.require_clusters:
        return a1, {"mode": "deterministic_high_conf"}

    flat = (p1 < 0.65) and ((p1 - p2) < 0.08)
    fragile = (diagnostics.cluster_support[a1] == 1) or (diagnostics.perturb_flip_rate > 0.3) \
              or (diagnostics.arm_uncertainty[a1] > 0.25)

    if flat and fragile and diagnostics.difficulty > 0.5:
        # controlled sampling
        q = {}
        for a, p in pi.items():
            bonus = 1.0 + 0.15 * (diagnostics.cluster_support[a] - 1)
            q[a] = (p ** (1/1.4)) * bonus
        q = normalize(q)
        return categorical_sample(q, rng), {"mode": "stochastic_ambiguous", "q": q}

    return a1, {"mode": "deterministic_default"}
```

---

# SECTION G — Hard Problem Policy Update: Stop When Compute Has No Marginal Value

In the prior design, hard-mode spent compute to “maximize correctness.” Now we stop when **expected marginal score gain per second** is too low.

Since expected score tracks per-run correctness, we estimate marginal improvement in posterior mass of best answer.

## Hard-mode stopping rule (operational)

After each new batch of candidates:

Let (p^\star_t = \max_a \pi_t(a)) be top posterior at time (t).
Define estimated gain rate:
[
g = \frac{p^\star_t - p^\star_{t-\Delta}}{\Delta \text{time}}
]

Stop hard-mode if:

* (p^\star_t \ge \tau_\text{stop}) (good enough), or
* (g < g_\text{min}) for (K) consecutive rounds (diminishing returns), or
* tool instability risk rising (timeouts/float usage) beyond threshold.

Concrete defaults:

* (\tau_\text{stop} = 0.88) (Run A), (0.84) (Run B)
* (g_\text{min} = 0.002 / \text{sec}), (K=3)

## Hard-mode policy now prioritizes “disambiguation”

If posterior is split between two answers:

* spend compute on **targeted tests that separate them** (mod checks, boundary cases, brute force on small instances, contradiction search), not more free-form CoT.

### Hard-mode pseudocode snippet

```python
def hard_mode_loop(meta, state, budget):
    while budget.time_left() > 0:
        if state.pi_top >= meta.run_config.hard_stop_pi:
            break
        if state.gain_rate < 0.002 and state.gain_rate_streak >= 3:
            break

        # disambiguate top-2
        a1, a2 = state.top2_answers()
        new_cands = targeted_disambiguation(meta, a1, a2, state)
        state.update(new_cands)

    return state
```

---

# SECTION H — Training Objective Modifications (Decision-Aware Learning)

We retain SFT + verifier-guided preference tuning, and add **decision-aware objectives**.

## H1) Calibration training (ARM + LLM confidence supervision)

* Train ARM on candidate logs (as described).
* Add LLM head/prompt behavior to output internal confidence cues in a structured way (not public), e.g. “assumptions list,” “fragile step”.

**Losses**

* ARM: BCE + Brier + ECE regularizer
* Optional: train LLM to predict “will verifier accept?” and “is correct?” as auxiliary tasks.

## H2) Disagreement mining

Build datasets where:

* multiple candidates exist with different answers,
* at least one is correct,
* verifiers are weak (to teach ARM and LLM to detect fragility).

## H3) Adversarial rerun simulation

Simulate Run A / Run B on training problems:

* Run A generates candidate set A
* Run B generates candidate set B
* Train BAA+ESMP to maximize expected score under the mixture.
  This teaches the system to produce complementary evidence and avoid shared failure modes.

## H4) Confidence supervision and abstention-like behavior

We cannot abstain in competition, but we can supervise:

* “when uncertain, generate disambiguating tests”
* “when confident, stop early”

Add a loss on **compute decisions** (imitation from oracle decisions or RL):

* reward: correctness − cost penalty − timeout penalty.

---

# SECTION I — Final Revised Inference Pipeline (Explicitly Includes Stochastic Decisions)

Below is the full revised pseudocode flow:

```python
# Globals (lazy init)
LLM_MAIN, LLM_FAST, JUDGE = None, None, None
RUN_SEED = int.from_bytes(os.urandom(8), "big")  # differs across private reruns

def predict(id_series, problem_series):
    pid = id_series.item(0)
    text = problem_series.item(0)
    ans = solve_one(pid, text, RUN_SEED)
    return pl.DataFrame({"id": pid, "answer": int(ans)})


def solve_one(pid, latex, run_seed):
    meta = parse_and_extract(latex)              # keep from previous design
    base_mode = route_problem(meta, latex)       # keep
    diagnostics = init_diagnostics(meta)

    # NEW: pick run program (A or B) using RDE
    run_config = RDE_select_run_config(run_seed, meta, diagnostics)

    budget = allocate_budget(base_mode, run_config)

    cands = []

    # Phase 1: baseline candidate generation (existing P0–P3 but parameterized by run_config)
    cands += generate_candidates(latex, meta, run_config, budget)

    # Phase 2: verification (existing) -> produce verification artifacts
    ver_results = [verify_candidate(c, latex, meta) for c in cands]

    # Phase 3: ARM calibration (NEW)
    arm_ps = []
    arm_u  = {}
    for vr in ver_results:
        x = build_arm_features(meta, vr, ver_results)
        p, u, failure = ARM_predict(x)
        arm_ps.append(p)
        arm_u[vr.answer] = max(arm_u.get(vr.answer, 0.0), u)
        vr.arm_p, vr.arm_u, vr.failure = p, u, failure

    # Phase 4: CED correlation + clustering (NEW)
    rho, clusters = CED_pairwise(ver_results)
    diagnostics = update_diagnostics(meta, ver_results, rho, clusters, arm_u)

    # Phase 5: BAA posterior (NEW)
    pi = BAA_posterior(ver_results, arm_ps, rho, clusters)
    diagnostics.pi = pi

    # Phase 6: Adversarial second pass (NEW, usually only in Run B or if fragile)
    if run_config.adversarial or is_fragile(pi, diagnostics):
        # try to refute top answer / split top-2
        asp_cands = adversarial_second_pass(latex, meta, run_config, diagnostics, budget)
        asp_vr = [verify_candidate(c, latex, meta) for c in asp_cands]
        # ARM + CED update incrementally (same as above, but incremental)
        ver_results.extend(asp_vr)
        pi = recompute_pi_with_incremental_updates(meta, ver_results)

    # Phase 7: Expected score maximization policy (ESMP) + ambiguity-aware sampling (NEW)
    rng = np.random.default_rng(seed=hash64(run_seed, pid))
    answer, decision_meta = AAS_select_answer(pi, diagnostics, run_config, rng)

    # Phase 8: failure-mode specific fallbacks (NEW)
    if should_fallback(answer, diagnostics):
        answer = fallback_logic(meta, diagnostics, ver_results, rng)

    return answer
```

---

# New Required Modules (Complete Specifications)

Below are the **nine required modules**, each with purpose/IO/training/inference/pseudocode.

---

## (1) Answer Reliability Model (ARM)

Already specified in Section B.

---

## (2) Bayesian Answer Aggregator (BAA)

Already specified in Section D with correlation-aware pooling and pseudocode.

---

## (3) Run Decorrelation Engine (RDE)

Already specified in Section E with run selection pseudocode.

---

## (4) Multi‑policy reasoning programs

### Purpose

Generate candidates from *qualitatively distinct* reasoning spaces to reduce systematic blind spots.

### Inputs

(problem text, meta, run_config)

### Outputs

candidate list with traces + tool code + provenance tags:

* `policy_id`, `path_type`, `prompt_hash`

### Training data

Self-play logs segmented by policy; measure which policies solve which failure clusters; use this to tune router and policy selection.

### Inference usage

Policies are selected by run_config and dynamically invoked when CED shows low diversity.

### Concrete policy set

* `POL_A_TOOLFORMAL`: direct tool solve + strong constraint extraction
* `POL_A_SYMBOLIC`: algebra/NT symbolic solve, modular checks
* `POL_B_REFUTE`: attempt to refute top candidate with counterexamples
* `POL_B_INVARIANT`: search invariants / monotonicity / parity / bounding
* `POL_GEO_COORD`: coordinate/complex geometry formalization
* `POL_COMB_BRUTE_SMALL`: brute on small instances to validate pattern

### Pseudocode

```python
def generate_candidates(latex, meta, run_config, budget):
    cands = []
    if run_config.program == "A":
        cands += run_policy(POL_A_TOOLFORMAL, latex, meta, budget, n=2)
        cands += run_policy(POL_A_SYMBOLIC, latex, meta, budget, n=1)
    else:
        cands += run_policy(POL_B_INVARIANT, latex, meta, budget, n=2)
        cands += run_policy(POL_B_REFUTE, latex, meta, budget, n=2)
        if meta.domain == "geometry":
            cands += run_policy(POL_GEO_COORD, latex, meta, budget, n=1)
    return cands
```

---

## (5) Correlated‑error detection (CED)

Already specified in Section C.

---

## (6) Expected score maximization policy (ESMP)

### Purpose

Choose actions that maximize expected competition score while managing compute cost and known failure risks.

### Inputs

* posterior (\pi(a))
* diagnostics (clusters, ARM uncertainty, perturbation sensitivity)
* budget remaining

### Outputs

* action: `STOP_AND_OUTPUT`, `DISAMBIGUATE`, `REFUTE_TOP`, `GENERATE_MORE`
* final answer when stopping

### Training data

Offline simulation: for training problems, simulate the policy and measure achieved score vs compute. Train a lightweight policy model or tune thresholds.

### Inference usage

Controls when to:

* stop early,
* run ASPR,
* run targeted disambiguation,
* accept sampling.

### ESMP pseudocode

```python
def ESMP_next_action(pi, diagnostics, budget, run_config):
    a1, p1 = top1(pi)
    a2, p2 = top2(pi)

    if p1 >= run_config.stop_pi and diagnostics.cluster_support[a1] >= run_config.require_clusters:
        return "STOP_AND_OUTPUT", a1

    # if ambiguous, spend compute on separating tests rather than more free-form generation
    if (p1 < 0.75) and ((p1 - p2) < 0.10) and budget.time_left() > 20:
        return "DISAMBIGUATE", (a1, a2)

    if is_fragile(pi, diagnostics) and budget.time_left() > 15:
        return "REFUTE_TOP", a1

    if budget.time_left() > 10:
        return "GENERATE_MORE", None

    return "STOP_AND_OUTPUT", a1
```

---

## (7) Adversarial second‑pass reasoning (ASPR)

### Purpose

Actively search for contradictions/counterexamples against the current top answer and force alternate derivations.

### Inputs

* top answer (a^*)
* top-2 set, traces, tool artifacts
* meta constraints

### Outputs

* additional candidates (possibly different answer)
* refutation evidence (flags) that downweights (a^*)

### Training data

Self-play: run ASPR against both correct and incorrect candidates; label whether it successfully refutes wrong ones without harming correct ones.

### Inference usage

Run mainly in Run B and on fragile posteriors.

### ASPR pseudocode

```python
def adversarial_second_pass(latex, meta, run_config, diagnostics, budget):
    a1, _ = top1(diagnostics.pi)
    a2, _ = top2(diagnostics.pi)

    # 1) perturb-and-resolve tests
    pert = generate_perturbations(meta, k=4)
    cands = []
    for pmeta in pert:
        cands += llm_refute_or_rederive(latex, meta, a1, pmeta, budget)

    # 2) counterexample search with tools (small instances)
    if meta.domain in ["combinatorics", "number_theory"]:
        cands += tool_counterexample_search(latex, meta, a1, budget)

    # 3) force alternate method
    cands += llm_alternate_method(latex, meta, forbid_lemmas=diagnostics.top_lemmas, budget=budget)

    return cands
```

---

## (8) Ambiguity‑aware answer sampling (AAS)

Already specified in Section F with strict gating and pseudocode.

---

## (9) Failure‑mode specific fallback logic

### Purpose

When diagnostics indicate a known failure mode, invoke a targeted recovery recipe rather than generic “generate more”.

### Inputs

* predicted failure type logits (from ARM/failure head)
* domain + meta
* current candidate set + posterior

### Outputs

* new candidates or forced formalization
* or a conservative answer selection rule

### Training data

From logged failures, map (features → best recovery action). Train a small classifier or use hand-tuned rules initially.

### Inference usage

Triggered when:

* top posterior is supported by a single cluster,
* ARM flags “misparse/tool hallucination/missing case”,
* verifier blind spots detected (e.g., geometry w/out formalization).

### Fallback recipes (concrete)

* **Misparse suspected:** re-run parser with alternative normalization; regenerate only tool solutions.
* **Geometry blind spot:** force coordinate/complex formalization + numeric sanity check.
* **Missing case:** run case enumerator template (symbolic case split) and verify counts.
* **Tool instability:** ban floats; require exact rationals; restrict loops; re-run.
* **Modulus confusion:** derive mod constraints separately and reconcile.

### Fallback pseudocode

```python
def fallback_logic(meta, diagnostics, ver_results, rng):
    ft = diagnostics.failure_type_top  # e.g. "misparse", "missing_case", ...
    if ft == "misparse":
        return rerun_with_strict_parsing_and_tool(meta, ver_results)
    if ft == "geometry_blind":
        return force_geometry_coordinates(meta, ver_results)
    if ft == "missing_case":
        return force_casework_enumerator(meta, ver_results)
    if ft == "tool_unstable":
        return rerun_tool_with_exact_arithmetic(meta, ver_results)

    # default conservative: choose top answer supported by most independent clusters
    return select_by_cluster_support(diagnostics.pi, diagnostics.cluster_support)
```

---

# SECTION J — Expected Score Improvement (Why This Crosses 47/50 Reliably)

### Baseline (previous design)

Typical plateau behavior:

* Strong easy/medium performance
* A small set of “verified-but-wrong” and “systematic blind spot” failures
* Score tends to cluster around **40–45/50** depending on model strength and tool quality

### What the new modules change

**1) ARM + calibration**

* Reduces overconfident wrong outputs by identifying fragile candidates that “look verified”.
* Gains typically come from flipping ~2–4 problems per 50 from wrong→right by picking the *right* candidate among those already generated.

**2) CED + BAA**

* Prevents self-consistency from being fooled by correlated mistakes.
* Converts “10 agreeing wrong traces” into “1 cluster of evidence” rather than 10 votes.

**3) Run B adversarial program**

* Improves per-run correctness on the hard tail by:

  * refuting wrong tool models,
  * forcing alternate methods,
  * running disambiguation rather than more CoT.
* This tends to add ~1–2 correct answers per 50 in the hardest subset.

**4) Compute stopping by marginal gain**

* Prevents timeouts/crashes and preserves baseline accuracy by avoiding late-run instability.

### A realistic performance projection (illustrative, but operationally grounded)

Assume a strong open-weight model with tools:

* **Baseline (old selection):** 44/50 single-run expected
* **+ verification improvements:** 45/50
* **+ ARM/BAA/CED selection fixes:** 46–47/50
* **+ adversarial Run-B + targeted disambiguation:** 47–48/50

Because the double-run expected score is the average per-run correctness, achieving **~47/50 per run expected** translates directly to **~47/50 expected double-run score**.

The key “reliably” improvement comes from sharply reducing the *specific* class of failures that cause repeated wrong outputs:

* correlated wrong clusters,
* overconfident verifier-passing wrongs,
* unstable hard-problem reasoning.

---

# Summary Checklist (Upgrade Deliverables)

* ✅ ARM: calibrated (P(\text{correct})) per candidate, trained from self-play logs
* ✅ CED: correlation matrix + clustering + redundancy suppression
* ✅ BAA: posterior over answers using correlation-aware evidence pooling
* ✅ RDE + multi-policy programs: Run A vs Run B truly different reasoning spaces
* ✅ ESMP: compute/stop policy optimizing expected score per run (and robust in practice)
* ✅ ASPR: adversarial refutation and perturbation-based stability testing
* ✅ AAS: controlled sampling only under strict ambiguity gates
* ✅ failure-mode fallbacks: misparse/geometry/missing-case/tool-instability recovery
* ✅ full revised end-to-end pseudocode with explicit stochastic decisions

If you want the next step, I can provide **exact prompt templates** for each policy (A_TOOLFORMAL, B_REFUTE, GEO_COORD, COMB_BRUTE_SMALL) and the **exact ARM feature extractor code skeleton**, designed to run inside a Kaggle notebook with stable runtime and deterministic logging.
