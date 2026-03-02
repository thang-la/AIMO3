## 1) Unified system overview

### 1.1 Core principle: the solver is a belief-state controller

The system is not “generate → verify → pick”. It is a **controller** that maintains a live posterior over answers and chooses the next reasoning action by **expected value of information (EVI)** under a compute budget.

**State variables (maintained online, updated after every action):**

* `S[a]`: unnormalized log-score for answer `a` (log posterior up to a constant); includes a special bucket `OTHER`.
* `π[a] = softmax(S)[a]`: current posterior belief over observed answers + `OTHER`.
* `U[a]`: uncertainty for `a` (epistemic; from ARM ensemble/variance + cluster diversity).
* `Clusters`: evidence clusters of candidates (CED output), each cluster has:

  * `cluster_id`, `member_ids`, `cluster_mode`, `cluster_weight`, `top_lemmas`, `tool_signature`.
* `Top`: `(a1, a2, …)` top answers by `π`.
* `Diagnostics`:

  * `cluster_support[a]` = number of clusters supporting `a`
  * `fragility[a]` = sensitivity + refutation risk summary
  * `blindspot_flags` = geometry-no-formalization, casework-missing, misparse-risk, tool-model-risk, etc.
* `Budget`: remaining time + token budgets + sandbox budget.
* `RunConfig`: program identity (A vs B) + allowed policies + verifier weighting + stop thresholds.
* `ActionHistory`: executed actions + outcomes for incremental correlation and marginal-gain estimation.

**Actuators (actions the controller can take):**

* `GEN(policy_id, n, temp, max_tokens)` → produce candidates (answer+trace+optional code)
* `VERIFY(candidate_id, level)` → shallow→deep verification
* `REFUTE(answer=a, mode)` → adversarial refutation attempt with tools
* `DISAMBIGUATE(a, b)` → targeted separating tests
* `REPARSE(strictness)` → re-extract constraints / rebuild tool model
* `HYP_START(k)` / `HYP_STEP(hyp_id, op)` → hypothesis search transitions (hard-mode)
* `STOP(mode)` → output deterministic or controlled stochastic selection

### 1.2 Native modules and their contract (not add-ons)

All modules are invoked inside the loop and expose deterministic APIs:

**Parser / Constraint Extractor**

* Input: raw LaTeX
* Output: `Meta` (domain priors, modulus/range constraints, entities), `ConstraintSet`
* Must support reparse with different strictness levels: `parse(latex, strictness)`.

**Router (fast model + rules)**

* Input: `Meta`, short normalized text
* Output: `domain`, `difficulty`, `tool_need`, `risk_profile`
* Used to set priors and action set.

**Multi-policy Reasoning Programs (native)**
Each policy defines:

* prompt template + decoding params
* expected candidate base-rate `r(policy, meta)` for ARM/BAA
* typical failure modes (used for action selection)

Policy families:

* `A_TOOLFORMAL`: tool-first formalization, exact arithmetic, deep constraint validation
* `A_SYMBOLIC`: sympy-driven algebra/NT transforms, CRT/LTE modules
* `B_INVARIANT`: invariant/bounds reasoning, lemma mining, minimal computation
* `B_REFUTE`: contradiction search targeting the current top answer; counterexample mining
* `GEO_COORD`: coordinate/complex geometry formalization + numeric sanity suite
* `COMB_SMALLBRUTE`: brute for small instances to validate pattern + extrapolation guardrails

**Tool Sandbox**

* Executes candidate code in restricted environment.
* Provides `ToolArtifacts`: stdout, derived equations, intermediate values, runtime, determinism score.

**Verifier (tiered, invoked incrementally)**

* `verify_shallow`: format/range/modulus, trivial constraints
* `verify_deep`: sympy checks, random tests, Z3 (if applicable), small brute checks

**ARM: Answer Reliability Model (native)**

* Input: candidate features (verifier signals + entropy + stability + sensitivity + cluster context)
* Output: `logBF_support` (support for `T=a_i` vs `T≠a_i`), `u_i` (uncertainty), `failure_logits`

**CED: Correlated-error detection (native)**

* Input: new candidate + existing candidates
* Output: `cluster_id`, pairwise `ρ` to neighbors, `redundancy_weight w_i`

**Belief Updater (native)**

* Converts ARM/Verifier/Refutation outputs to incremental updates of `S[a]`, `U[a]`, `Clusters`.

**Action Value Model (AVM) + Expected-Score Policy (native)**

* AVM predicts outcome distribution for each action given state features:

  * `P(improve_top)` / `E[ΔV]` / `P(refute_if_wrong)` / expected cost
* Controller chooses action maximizing:
  [
  \text{EVI(action)} = \mathbb{E}[V(s') - V(s)] - \lambda \cdot \text{cost} - \eta \cdot \text{timeout_risk}
  ]
  where (V(s)=\max_a \pi[a]) (expected correctness if stop now, deterministic).

**Run Decorrelation Engine (native)**

* RunConfig is selected at initialization and can be adapted if CED shows low diversity:

  * Program A: constructive/tool-heavy
  * Program B: adversarial/invariant/refutation-heavy
* Selection is a deterministic function of `(run_seed, problem_hash)` so two private reruns differ with high probability.

**Ambiguity-Aware Output (native)**

* Output decision is part of `STOP(mode)` action, chosen by the controller using posterior/uncertainty/fragility gates.

**Failure-mode fallback logic (native)**

* Triggered as actions (not post-hoc): e.g., `REPARSE(strict)` when misparse-risk crosses threshold; `GEO_COORD` when geometry-blindspot triggers.

### 1.3 Inconsistencies between the legacy solver and expected-score-optimal behavior (and why they break it)

These are conflicts that must be eliminated for a unified decision engine:

1. **Heuristic candidate scoring (`score(c)=…`) vs calibrated posterior**

* Conflict: heuristic mixing treats signals as commensurate and independent; it is not calibrated.
* Breakage: stop decisions based on this score cause **overconfident wrong stops** (hard tail), reducing per-run correctness and thus expected double-run score.

2. **Sequential “generate many → verify → select” vs interleaved belief control**

* Conflict: sequential blocks spend compute without knowing whether it increases correctness probability.
* Breakage: wastes budget on redundant evidence; under-allocates compute to ambiguous problems; increases timeout risk on hard problems → lowers per-run accuracy.

3. **Self-consistency voting vs correlation-aware evidence**

* Conflict: voting implicitly assumes samples are i.i.d.; they are not.
* Breakage: correlated wrong traces dominate votes; increases probability of selecting wrong answer even when alternative cluster exists.

4. **Early stopping triggered by “two paths agree + judge>0.9” vs posterior thresholds with fragility checks**

* Conflict: agreement can be inside one reasoning family; judge can be fooled by consistent-but-wrong derivations.
* Breakage: premature stop on brittle wrong; also fails to request refutation when it has high EVI.

5. **Hard-mode as “more attempts” vs hypothesis search / disambiguation**

* Conflict: more attempts often means more correlated candidates from same failure mode.
* Breakage: marginal value of compute collapses; increases runtime and instability without increasing correctness probability.

6. **Run decorrelation as “temperature/prompt variation” vs programmatic decorrelation guided by correlation diagnostics**

* Conflict: temperature changes do not guarantee lemma-space diversity.
* Breakage: two private reruns can still explore the same failure basin; loses the main advantage of reruns (finding a different correct structure).

7. **Training optimized for solution correctness only vs decision-making competence**

* Conflict: solver learns to produce answers, not to (i) estimate uncertainty, (ii) decide when to refute, (iii) decide when to diversify, (iv) trust tools conditionally.
* Breakage: inference policy remains heuristic and brittle; calibration errors dominate the last few problems.

8. **Compute allocation by fixed difficulty buckets vs EVI-based compute**

* Conflict: difficulty estimates are noisy; value of compute depends on current posterior shape and evidence diversity.
* Breakage: undercompute when posterior is split; overcompute when already solved; increases timeouts and missed improvements.

---

## 2) Belief-state inference algorithm pseudocode

### 2.1 Data structures

```python
class Candidate:
    id: int
    answer: int
    policy_id: str
    path_type: str  # P0/P1/P2/P3
    trace: str
    trace_summary: str
    tool_code: str | None
    tool_artifacts: dict | None
    verifier: dict  # shallow/deep results
    arm_logBF: float
    arm_u: float
    failure_logits: dict
    cluster_id: int
    redundancy_w: float

class BeliefState:
    S: dict[int|str, float]       # log-scores; includes "OTHER"
    pi: dict[int|str, float]      # posterior = softmax(S)
    U: dict[int|str, float]       # uncertainty by answer
    clusters: dict[int, dict]     # cluster metadata
    candidates: list[Candidate]
    meta: dict
    constraints: dict
    run_config: dict
    budget: Budget
    diagnostics: dict
    action_history: list[dict]
```

### 2.2 Initialization

```python
def init_state(latex: str, run_seed: int) -> BeliefState:
    meta, constraints = parse(latex, strictness="normal")
    base_mode = route(meta, latex)

    run_config = select_run_config(run_seed, problem_hash(latex), meta, base_mode)
    budget = allocate_budget(meta, base_mode, run_config)

    st = BeliefState()
    st.meta, st.constraints, st.run_config, st.budget = meta, constraints, run_config, budget
    st.candidates = []
    st.clusters = {}

    st.S = {"OTHER": 0.0}     # log prior mass for unseen answers
    st.pi = {"OTHER": 1.0}
    st.U = {"OTHER": 1.0}

    st.diagnostics = init_diagnostics(meta, constraints, run_config)
    st.action_history = []
    return st
```

### 2.3 Main loop: Belief-State Reasoning Solver

```python
def solve_one(pid: str, latex: str, run_seed: int) -> int:
    st = init_state(latex, run_seed)

    # Always take a small diverse seed batch to populate hypotheses/answers
    seed_actions = seed_action_set(st)  # depends on domain + run program
    for act in seed_actions:
        execute_action_and_update(st, act)
        if should_stop(st): 
            return output_answer(st)

    while st.budget.time_left() > 0:
        st.pi = softmax(st.S)
        st.diagnostics = update_diagnostics(st)

        if should_stop(st):
            return output_answer(st)

        actions = enumerate_actions(st)
        best_act = argmax(actions, key=lambda a: EVI(st, a))

        if EVI(st, best_act) <= st.run_config["evi_stop_floor"]:
            return output_answer(st)

        execute_action_and_update(st, best_act)

    return output_answer(st)
```

### 2.4 Action enumeration (adaptive)

```python
def enumerate_actions(st: BeliefState) -> list[dict]:
    acts = []

    # Generate actions across allowed policies
    for pol in st.run_config["allowed_policies"]:
        if policy_applicable(pol, st.meta, st.diagnostics):
            acts.append({"type":"GEN", "policy":pol, "n":1})

    # Verification deepening on promising candidates / clusters
    for cid in top_candidate_ids(st, k=3):
        if st.candidates[cid].verifier.get("deep_done") is False:
            acts.append({"type":"VERIFY", "cand_id":cid, "level":"deep"})

    # Refutation if top answer fragile or single-cluster supported
    a1 = top1_answer(st.pi)
    if needs_refute(st, a1):
        acts.append({"type":"REFUTE", "answer":a1, "mode":st.run_config["refute_mode"]})

    # Disambiguation if posterior split
    a1, a2 = top2_answers(st.pi)
    if is_split(st, a1, a2):
        acts.append({"type":"DISAMBIGUATE", "a":a1, "b":a2})

    # Reparse if misparse risk or constraint instability
    if st.diagnostics["misparse_risk"] > 0.6:
        acts.append({"type":"REPARSE", "strictness":"strict"})

    # Hypothesis search (hard-mode) as structured actions
    if st.diagnostics["hardness"] > 0.75 and st.diagnostics["cluster_diversity"] < 2:
        if not st.diagnostics.get("hyp_started"):
            acts.append({"type":"HYP_START", "k":st.run_config["hyp_k"]})
        else:
            for hyp_id in select_active_hypotheses(st, k=2):
                for op in ["LEMMA", "MODEL", "BRUTE_SMALL", "SWITCH_REP", "CONTRADICTION"]:
                    acts.append({"type":"HYP_STEP", "hyp_id":hyp_id, "op":op})

    return acts
```

### 2.5 EVI computation (operational approximation)

```python
def V(st):  # expected correctness if stop now (deterministic)
    return max(st.pi.values())

def EVI(st: BeliefState, act: dict) -> float:
    feats = state_action_features(st, act)

    # AVM predicts:
    # - expected delta in V
    # - probability action refutes top if top is wrong
    # - probability action discovers a better answer cluster
    # - expected cost + timeout risk
    pred = AVM_predict(feats)

    return pred["E_deltaV"] - st.run_config["lambda_cost"] * pred["cost"] - st.run_config["eta_timeout"] * pred["timeout_risk"]
```

### 2.6 Execute action and update belief (interleaved)

```python
def execute_action_and_update(st: BeliefState, act: dict):
    if act["type"] == "GEN":
        new_cands = run_policy_generate(st, act["policy"], n=act["n"])
        for cand in new_cands:
            integrate_candidate_online(st, cand)

    elif act["type"] == "VERIFY":
        cand = st.candidates[act["cand_id"]]
        cand.verifier = run_verifier(st, cand, level=act["level"])
        # Recompute ARM + belief update incrementally as new evidence about same answer
        integrate_candidate_evidence_update(st, cand)

    elif act["type"] == "REFUTE":
        ev = run_refutation(st, act["answer"], mode=act["mode"])
        integrate_refutation_online(st, ev)

    elif act["type"] == "DISAMBIGUATE":
        ev = run_disambiguation_tests(st, act["a"], act["b"])
        integrate_disambiguation_online(st, ev)

    elif act["type"] == "REPARSE":
        meta2, constraints2 = parse(st.meta["raw_latex"], strictness=act["strictness"])
        integrate_reparse_online(st, meta2, constraints2)

    elif act["type"] == "HYP_START":
        st.diagnostics["hyp_started"] = True
        st.diagnostics["hypotheses"] = init_hypotheses(st, k=act["k"])

    elif act["type"] == "HYP_STEP":
        hyp = st.diagnostics["hypotheses"][act["hyp_id"]]
        evs = hypothesis_step(st, hyp, op=act["op"])
        for ev in evs:
            integrate_hypothesis_evidence_online(st, ev)

    st.action_history.append({"act": act, "pi_top": top1_prob(st.pi), "time_left": st.budget.time_left()})
```

### 2.7 Stop rule and output action

```python
def should_stop(st: BeliefState) -> bool:
    a1, p1 = top1(st.pi)
    # calibrated stop requires: high posterior + independent cluster support + low fragility
    if p1 >= st.run_config["stop_pi"] \
       and st.diagnostics["cluster_support"].get(a1, 0) >= st.run_config["min_clusters"] \
       and st.diagnostics["fragility"].get(a1, 1.0) <= st.run_config["max_fragility"]:
        return True
    return False

def output_answer(st: BeliefState) -> int:
    st.pi = softmax(st.S)
    a1, p1 = top1(st.pi)
    a2, p2 = top2(st.pi)

    # ambiguity-aware output is a STOP(mode) decision, not a fallback
    if should_sample(st, a1, a2):
        q = sampling_distribution(st)  # posterior tempered + cluster bonus
        return categorical_sample(q, rng=st.diagnostics["rng"])
    return a1
```

---

## 3) Candidate belief update equations

### 3.1 Posterior representation

Maintain unnormalized log-scores over the set of observed answers `A_obs` plus `OTHER`:

* Initialize:
  [
  S(\text{OTHER}) = 0,\quad S(a)=\log \varepsilon \text{ when } a \text{ first appears (e.g., } \varepsilon=10^{-6}\text{)}
  ]
* Posterior:
  [
  \pi(a) = \frac{e^{S(a)}}{e^{S(\text{OTHER})} + \sum_{a'\in A_{\text{obs}}} e^{S(a')}}
  ]

### 3.2 Evidence objects (what updates beliefs)

Every action produces one or more evidence objects `e` with fields:

* target answer(s): `a` or `(a,b)` or `OTHER`
* `type ∈ {support, refute, disambiguate, constraint, reparse}`
* `logBF(e)`: log Bayes factor contributed to the target hypothesis vs its complement
* `w(e)`: redundancy/correlation weight (0–1)

Belief update rule is additive in log-space:
[
S(a) \leftarrow S(a) + w(e)\cdot \log BF(e)
]

### 3.3 Computing `logBF` for a candidate (support evidence)

A generated candidate `i` proposes answer `a_i` and produces features `x_i`.

**ARM outputs a Bayes factor directly**:

* `ARM_logBF(x_i)` is trained to approximate:
  [
  \log BF_i \approx \log \frac{P(x_i \mid T=a_i)}{P(x_i \mid T\neq a_i)}
  ]
  This avoids dependence on arbitrary priors and prevents double counting verifier signals.

**Hard constraint override** (high-reliability refutation):

* If a candidate violates a hard constraint extracted from the statement (range/modulus/divisibility explicitly specified), treat as near-certain refutation:
  [
  \log BF_i = -M,\quad M\in[15,30]
  ]
  (or set (S(a_i)=-\infty) if constraint extraction reliability is high).

**Correlation-aware weight**:
When integrating candidate `i`, compute similarity to existing candidates and assign to a cluster. Let `ρ_ij` be the CED correlation proxy to each existing member `j` of the cluster. Define redundancy weight:
[
w_i = \frac{1}{1 + \sum_{j\in \text{cluster}(i)} \rho_{ij}}
]
Then support update:
[
S(a_i)\leftarrow S(a_i) + w_i \cdot \log BF_i
]

### 3.4 Incremental updates from deeper verification (same candidate, more evidence)

Verification is tiered. A candidate integrated after shallow verify may later get deep verification results. Treat the delta as a new evidence object `e_deep` with its own Bayes factor.

To prevent double counting, `ARM_logBF` is computed on the **current evidence set**; when deep verification runs, recompute:

* `logBF_new = ARM_logBF(x_i_with_deep)`
* `logBF_old = ARM_logBF(x_i_shallow)`
  and apply only the increment:
  [
  \Delta \log BF = \log BF_{\text{new}} - \log BF_{\text{old}}
  ]
  [
  S(a_i) \leftarrow S(a_i) + w_i \cdot \Delta \log BF
  ]

### 3.5 Adversarial refutation updates (explicit negative evidence)

Refutation action targets a specific answer `a`.

Refutation returns one of:

* `FOUND_COUNTEREXAMPLE` (strong refute)
* `FOUND_CONTRADICTION` (strong refute)
* `INCONCLUSIVE`
* `FAILED_TO_REFUTE` (weak support)

Map to Bayes factor:
[
\log BF_{\text{refute}}(a)=
\begin{cases}
-M & \text{counterexample/contradiction found}\
-\mu & \text{inconclusive with mild concern flags (e.g., unstable tool model)}\
+\nu & \text{failed to refute under strong refutation protocol}
\end{cases}
]
with typical values:

* (M\in[15,30]) (nearly eliminate posterior mass)
* (\mu\in[0.5,2])
* (\nu\in[0.2,1]) (bounded: “no refutation found” is weak)

Correlation weight for refutation depends on whether refutation reuses the same cluster’s lemmas/tool signature:
[
w_{\text{refute}} = 1 - \max_{j\in\text{supporters of }a} \rho(\text{refute}, j)
]
Update:
[
S(a) \leftarrow S(a) + w_{\text{refute}}\cdot \log BF_{\text{refute}}(a)
]

### 3.6 Disambiguation tests between two answers

A disambiguation action returns a test outcome `y` designed to have different likelihood under `T=a` vs `T=b`:
[
\log BF_{\text{dis}}(y; a,b)=\log\frac{P(y\mid T=a)}{P(y\mid T=b)}
]
Update both:
[
S(a)\leftarrow S(a)+w\cdot \log BF_{\text{dis}},\quad
S(b)\leftarrow S(b)-w\cdot \log BF_{\text{dis}}
]
The likelihood model (P(y\mid T=a)) is learned from self-play disambiguation logs (Section 4).

### 3.7 Uncertainty update (epistemic, used for action selection and sampling gates)

ARM provides per-candidate uncertainty `u_i` (e.g., ensemble variance). For each answer `a`, aggregate:
[
U(a) = \frac{\sum_{i: a_i=a} w_i \cdot u_i}{\sum_{i: a_i=a} w_i + \delta}
]
Then fragility combines:

* `U(a)` (epistemic uncertainty)
* sensitivity flip rate from perturbation tests
* cluster support count (more clusters ⇒ less fragile)

---

## 4) Training objective changes

### 4.1 Training data artifacts (produced by self-play + tool verification)

The training pipeline must log **step-level decision traces**, not just final solutions.

For each training problem:

* Run the full belief-state solver under both Run A and Run B programs.
* Record at each step (t):

  * `state_features φ(s_t)` (posterior shape, cluster diversity, fragility, budget, domain)
  * `action a_t`
  * `outcome o_t` (new candidates, verifier results, refutation result)
  * `ΔV_t = V(s_{t+1}) - V(s_t)` where (V(s)=\max_a \pi[a])
  * `cost_t` (time/tokens/sandbox runtime)
* Store candidate-level records:

  * features `x_i`, truth correctness label, failure type, tool model correctness
* Store pairwise records for correlation:

  * `(i, j, pair_features, cofail_label)`.

### 4.2 Models trained and their losses

#### (A) Solver LLM fine-tuning targets (decision-support behavior)

LLM outputs are structured into:

* `candidate_answer`
* `tool_code` (optional)
* `assumptions` list
* `lemma_list` (for hypothesis search)
* `refutation_plan` (for Run B)
* `disambiguation_tests` (for split posterior)

**Losses**

1. **SFT for correct candidate generation**
   [
   \mathcal{L}*{\text{SFT}} = -\log P*\theta(\text{correct structured output}\mid \text{prompt})
   ]
2. **Tool robustness loss** (exact arithmetic, no floats, deterministic loops)

* binary penalties supervised from tool execution logs:
  [
  \mathcal{L}_{\text{tool}} = \alpha \cdot \text{CE}(\text{float_use}) + \beta \cdot \text{CE}(\text{timeout_risk})
  ]

3. **Refutation quality SFT** (generate counterexample search programs / contradiction sketches)
   [
   \mathcal{L}*{\text{refute-SFT}} = -\log P*\theta(\text{valid refutation artifact}\mid \text{refute prompt})
   ]

#### (B) ARM: Bayes-factor correctness estimator

Train ARM to output `logBF` directly.

Let label (y_i=1) if candidate answer equals truth else 0.
Let ARM output `logBF_i`. Convert to probability using a learned base-rate prior `r(policy, meta)`:
[
\hat{p}*i = \sigma(\logit(r) + \logBF_i)
]
Loss:
[
\mathcal{L}*{\text{ARM}} = \text{BCE}(y_i, \hat{p}_i) + \lambda \cdot \text{Brier}(y_i, \hat{p}_i)
]
Calibration:

* fit isotonic/temperature scaling on held-out to minimize ECE; store calibrator applied to (\hat{p}_i) (and correspondingly to `logBF_i`).

Uncertainty head:

* predict Beta parameters ((\alpha_i,\beta_i)) or variance; optimize NLL:
  [
  \mathcal{L}_{U} = -\log \text{BetaBinomial}(y_i;\alpha_i,\beta_i)
  ]

Failure-type head:
[
\mathcal{L}_{\text{fail}} = \text{CE}(\text{failure_type}, \hat{q}_i)
]

#### (C) CED: correlation / redundancy estimator

Train pairwise correlation proxy (\rho_{ij}) to predict **excess co-failure** beyond independence.
Given ARM-calibrated (p_i,p_j), define target:
[
t_{ij} = \mathbf{1}[y_i=0 \wedge y_j=0] - (1-p_i)(1-p_j)
]
Loss:
[
\mathcal{L}*{\text{CED}} = \text{Huber}(t*{ij}, \hat{t}*{ij}) \quad \Rightarrow \quad \rho*{ij}=\sigma(\hat{t}_{ij})
]

#### (D) AVM: Action Value Model (EVI predictor)

Regression targets from logs:

* `E_deltaV` (expected improvement in (V(s)))
* `timeout_risk`
* `expected_cost`
  Loss:
  [
  \mathcal{L}_{\text{AVM}} = \text{MSE}(\Delta V, \widehat{\Delta V}) + \gamma\cdot \text{MSE}(\text{cost},\widehat{\text{cost}}) + \eta\cdot \text{BCE}(\text{timeout},\widehat{\text{timeout}})
  ]

#### (E) Decision policy training (when to search/stop/refute/diversify/trust tools)

Two compatible training routes:

1. **Imitation from oracle rollouts**
   For each logged state, define an oracle action by counterfactual evaluation (simulate candidate outcomes from the dataset or from a learned dynamics model) that maximizes:
   [
   R = \mathbf{1}[\text{correct}] - \lambda \cdot \sum_t \text{cost}*t - \eta \cdot \mathbf{1}[\text{timeout}]
   ]
   Train a small policy model (π*\psi(a\mid s)) with cross-entropy:
   [
   \mathcal{L}*{\text{policy}} = -\log π*\psi(a^* \mid s)
   ]

2. **Offline RL fine-tuning** (GRPO/DPO-style on action sequences)
   Reward is the same (R). Optimize the policy over action sequences to maximize expected reward.

**Tool trust learning target (explicit)**
Label whether the tool formalization matches the statement semantics (`tool_model_correct ∈ {0,1}`) from training truth. Train a classifier integrated into ARM failure head; include penalty in decision reward if tool model is likely wrong.

---

## 5) Hard-problem hypothesis search algorithm

Hard problems are handled by a **hypothesis state machine** integrated into the belief-state solver (via `HYP_START` and `HYP_STEP` actions). The goal is not “more samples”; it is to search over **solution structures** and attach evidence to them.

### 5.1 Hypothesis state definition

A hypothesis (h) is a structured object:

* `h.tag`: structural template (examples: `NT_LTE`, `NT_CRT`, `COMB_IE`, `COMB_INVARIANT`, `GEO_COORD`, `GEO_POWERPOINT`, `ALG_SYMMETRY`, `ALG_FUNC_EQ`)
* `h.assumptions`: explicit assumptions introduced by the reasoning plan
* `h.artifacts`: partial derivations:

  * derived equations, invariants, constraints, parameterizations
  * code skeletons
* `h.tests`: executed tests + outcomes
* `h.weight`: log-score for hypothesis plausibility (separate from answer beliefs)
* `h.proposed_answers`: list of `(a, local_logBF)` derived under this hypothesis

Maintain hypothesis posterior:
[
\Pi(h) \propto \exp(W(h))
]

### 5.2 Hypothesis transitions (actions)

Each transition produces **evidence** that updates:

* hypothesis weight (W(h))
* global answer scores (S(a)) via candidate evidence

Allowed transitions:

1. **LEMMA(h)**
   Prompt LLM to produce a lemma + minimal proof sketch + conditions.

* Tool-check lemma on random consistent instantiations (or symbolic check if possible).
* Evidence:

  * if lemma holds on tests: `logBF_lemma > 0`
  * if falsified: `logBF_lemma = -M` and prune hypothesis

2. **MODEL(h)**
   Build a formal model consistent with statement under hypothesis:

* coordinate placement / parameterization / recurrence / CRT system
* produce executable exact code that outputs candidate answer
* deep verify: constraints + determinism + small-instance sanity

3. **BRUTE_SMALL(h)**
   If problem admits small-instance brute variants:

* generate reduced instances, brute compute, detect pattern
* attach evidence only if pattern survives multiple sizes and matches hypothesis constraints

4. **CONTRADICTION(h)**
   Target either:

* the hypothesis assumptions, or
* the current top global answer under this hypothesis
  Attempt to derive inconsistency or produce counterexample.

5. **SWITCH_REP(h)**
   Change representation inside same hypothesis family:

* geometry: synthetic ↔ coord ↔ complex
* combinatorics: bijection ↔ recursion ↔ generating function
* number theory: valuation ↔ order ↔ lifting ↔ CRT
  Transition produces new artifacts and resets some tests.

### 5.3 Hypothesis search controller (VOI-driven)

```python
def init_hypotheses(st: BeliefState, k: int) -> dict[int, dict]:
    tags = propose_hypothesis_tags(st.meta, k=k)  # fast model + rules
    hyps = {}
    for hid, tag in enumerate(tags):
        hyps[hid] = {"id": hid, "tag": tag, "assumptions": [], "artifacts": {}, "tests": [],
                    "W": 0.0, "proposed_answers": []}
    return hyps

def hypothesis_step(st: BeliefState, hyp: dict, op: str) -> list[dict]:
    # returns evidence objects; some evidence objects correspond to new Candidates
    if op == "LEMMA":
        lemma = llm_generate_lemma(st, hyp)
        ok, strength = tool_check_lemma(st, hyp, lemma)
        ev = {"type":"hyp_lemma", "hyp_id": hyp["id"], "logBF": (+strength if ok else -20.0)}
        return [ev]

    if op == "MODEL":
        cand = llm_generate_model_and_code(st, hyp)  # Candidate with tool_code
        integrate_candidate_online(st, cand)         # updates S(a) immediately
        ev = {"type":"hyp_model", "hyp_id": hyp["id"], "logBF": cand.arm_logBF * cand.redundancy_w}
        return [ev]

    if op == "BRUTE_SMALL":
        ok, strength, maybe_answer = brute_small_and_infer(st, hyp)
        evs = [{"type":"hyp_brute", "hyp_id": hyp["id"], "logBF": (+strength if ok else -5.0)}]
        if maybe_answer is not None:
            cand = Candidate(answer=maybe_answer, policy_id="COMB_SMALLBRUTE", path_type="HYP", ...)
            integrate_candidate_online(st, cand)
        return evs

    if op == "CONTRADICTION":
        a1 = top1_answer(st.pi)
        outcome = run_hypothesis_contradiction(st, hyp, target_answer=a1)
        # contradiction found refutes either hypothesis or answer depending on outcome
        return map_contradiction_to_evidence(st, hyp, outcome)

    if op == "SWITCH_REP":
        hyp2 = switch_representation(hyp)
        ev = {"type":"hyp_switch", "hyp_id": hyp["id"], "logBF": +0.2}  # mild support for exploration
        return [ev]

    return []
```

### 5.4 Integrating hypothesis evidence into the global belief state

Hypothesis evidence updates hypothesis weights:
[
W(h) \leftarrow W(h) + w_h \cdot \log BF_h
]
and may also generate candidates that update `S(a)` through the standard candidate update equations (Section 3).

Hypothesis selection for subsequent steps uses:
[
\text{VOI}(h,op)=\mathbb{E}[\Delta V \mid h,op] - \lambda\cdot \text{cost}
]
where the expectation is predicted by AVM using features `(Π(h), tag, op, current π(a), cluster_diversity, fragility)`.

---

## 6) Why this architecture specifically improves >47 reliability

1. **Stops are tied to calibrated posterior mass, not heuristics**

* The stop condition requires:

  * `π[a*] ≥ stop_pi`,
  * `cluster_support[a*] ≥ min_clusters`,
  * `fragility[a*] ≤ max_fragility`.
* Because `π` is produced by correlation-weighted Bayes-factor updates (not votes), `stop_pi` is tuned so that:

  * among problems where the solver stops early, empirical correctness ≈ `stop_pi` (after calibration).
* This removes the dominant failure mode that blocks 47+: **premature overconfident wrong outputs**.

2. **Evidence is discounted by redundancy in real time**

* Correlated candidate families are detected immediately (CED) and assigned small `w_i`.
* Ten near-identical wrong derivations contribute roughly the evidence of ~one derivation, preventing “consensus hallucinations”.
* This directly increases correctness on geometry/comb where reasoning families are narrow and mistakes are systematic.

3. **Compute is spent where it increases posterior, not where it “feels hard”**

* The controller chooses actions by predicted `E_deltaV`, so it:

  * deep-verifies only candidates likely to move belief,
  * refutes only when the top answer is fragile (high expected gain),
  * disambiguates when posterior is split (maximizing information efficiency),
  * avoids additional generation when it is redundant (low EVI).
* This converts time into posterior mass efficiently and reduces timeouts/crashes on the hard tail.

4. **Adversarial refutation is a first-class belief operator**

* A refutation success applies a large negative Bayes factor to the targeted answer, collapsing its posterior mass quickly.
* This is the mechanism that “rescues” problems where a tool model or a dominant lemma is wrong but self-consistent.
* Run B is not “diversity”; it is a belief-correcting operator that increases per-run correctness specifically on the hardest problems.

5. **Hard problems become hypothesis search with testable intermediate claims**

* Instead of sampling more full solutions, the solver searches over structured hypotheses and attaches evidence to them via:

  * lemma tests,
  * small brute invariants,
  * alternate representations,
  * contradiction/counterexample attempts.
* This reduces unstable reasoning: hypotheses that do not survive tests lose weight early, and only those producing consistent testable artifacts contribute to answer belief.

6. **Reruns improve realized reliability by changing the action distribution, not the temperature**

* The run program (A vs B) changes:

  * which policies are available,
  * which actions have high prior EVI (refute vs tool-formalize),
  * the verifier weighting and stop thresholds,
  * the disambiguation/refutation emphasis.
* Across the two private reruns, the solver is likely to execute materially different action sequences and therefore discover different evidence clusters—raising per-run correctness on the hard tail rather than repeating the same failure basin.

7. **Ambiguity-aware output is bounded and state-gated**

* Stochastic output is only allowed when:

  * posterior is flat (`π[a1]` low and margin small),
  * evidence is single-cluster or fragile,
  * difficulty is nontrivial.
* This prevents harming easy-problem accuracy while reducing the chance of identical brittle wrong outputs under calibration errors.

The combined effect is not “more clever selection”; it is a solver whose **entire control loop is optimized to increase the probability that the returned integer is correct under limited compute**, with explicit mechanisms to (i) measure reliability, (ii) discount correlated evidence, (iii) refute dominant wrong hypotheses, and (iv) spend compute only when it moves the posterior.
