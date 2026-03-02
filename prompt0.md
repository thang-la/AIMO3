# Technical Design Document: **AIMO3 >47/50** Open-Source Math Reasoning System (Training + Inference)

**Target:** >47/50 on AIMO3 (integer answers 0–99999), robust to **double-run penalized accuracy** on the private set.
**Hard constraints:** open-weight models only, no internet at inference, Kaggle inference-server pattern (1 problem per call), reproducible.

AIMO3 specifics that shape the design:

* **110 original IMO/National-Olympiad style problems**, across algebra/NT/comb/geo, with **5-digit answers**, and **explicit modulus in statements**. ([AIMO Prize][1])
* Kaggle provides **H100 GPUs** for contestants, enabling large open-weight models like **GPT-OSS-120B** and Qwen3-family variants. ([AIMO Prize][1])
* **Inference server must start quickly** (startup limit ~15 minutes); heavy model load should be **lazy/on-first-call**. ([GitHub][2])
* Gateway serves **one row at a time**; total runtime budget is effectively **up to ~9 hours**. ([GitHub][3])
* Private scoring: the notebook is rerun and predictions concatenated; per-problem score is **1 if both correct, 0.5 if one correct, 0 if none**, i.e. *average accuracy over two runs*. ([GitHub][4])

---

## 1. Executive Summary

We build a **full-stack solver** that combines:

1. **A strong open-weight reasoning model** (primary)
2. **A fast auxiliary model** (router + draft/speculative decoding + secondary solver)
3. **Tool-augmented reasoning** (safe Python + SymPy + optional Z3)
4. **Verifier-guided candidate selection** (symbolic checks + neural judge + consistency tests)
5. **Adaptive compute allocation** per problem (progressive deepening)

### Why single-model systems will fail

Single-model, single-pass approaches fail due to:

* **Translation errors** (LaTeX → math objects) and missing constraints
* **Arithmetic slips** and brittle long chain-of-thought
* **Geometry** requiring coordinate/angle-chasing formalization
* **Combinatorics** needing careful casework / inclusion–exclusion
* **Lack of “hard rejection”**: without verification, plausible wrong answers survive

Hence we use a **mixture-of-strategies** with **routing** and **verification-driven arbitration**.

---

## 2. System Architecture Diagram (ASCII)

### 2.1 High-level (Training + Inference)

```
                ┌──────────────────────────────────────────────┐
                │              OFFLINE TRAINING                │
                │                                              │
Open Datasets ──┼─► (Dedup + Contamination Guard) ──┐          │
                │                                   │          │
Synthetic Gen ──┼─► (Programmatic Families + SymPy) ─┼─► SFT    │
                │                                   │          │
Self-Play  ─────┼─► (Generate→Verify→Filter) ───────┘          │
                │                                              │
Verifier Data ──┼─► Train Neural Judge (small) + PRM (opt)      │
                │                                              │
RL Stage   ─────┼─► Verifier-guided RL (GRPO/DPO style)         │
                └──────────────────────────────────────────────┘

                ┌──────────────────────────────────────────────┐
                │             KAGGLE INFERENCE (NO NET)         │
                │  (InferenceServer: 1 problem per predict())   │
                │                                              │
Problem ─► Normalize/Parse ─► Route ─► Candidate Gen ─► Verify  │
                               │            │            │      │
                               │            ▼            ▼      │
                               │     Tool Sandbox     Neural    │
                               │   (Py/SymPy/Z3)      Judge     │
                               ▼                                │
                         Budget Manager  ─► Decision Policy ─► Answer
                └──────────────────────────────────────────────┘
```

### 2.2 Inference micro-architecture

```
┌─────────────────────────────┐
│ AIMO3Solver.solve(problem)  │
└──────────────┬──────────────┘
               ▼
   ┌─────────────────────┐
   │ 1) Parse & Extract   │  modulus, variables, entities,
   │    constraints       │  geometry objects, domains
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ 2) Router (fast)     │  predicts domain+difficulty+mode
   └──────┬───────┬──────┘
          │       │
          ▼       ▼
┌────────────────┐  ┌──────────────────────────┐
│ 3A) Symbolic    │  │ 3B) LLM Multi-Path       │
│     First Pass  │  │     Generation           │
│ (regex+SymPy)   │  │  CoT / TIR / Backsolve   │
└───────┬────────┘  └───────────┬──────────────┘
        ▼                       ▼
      candidates          candidates+traces
        └───────────────┬───────────────┘
                        ▼
             ┌──────────────────────┐
             │ 4) Verification       │
             │ - constraints         │
             │ - plug-in checks      │
             │ - randomized tests    │
             │ - judge scoring       │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ 5) Select & Return    │
             │  - confidence gating  │
             │  - fallback recovery  │
             └──────────────────────┘
```

---

## 3. Model Strategy (Multi-model System)

### 3.1 Primary reasoning LLM (open-weight)

**Option A (preferred on H100):** `openai/gpt-oss-120b`

* Open-weight, MoE-style; designed for tool use; fits on a single H100-class GPU. ([OpenAI Developers][5])
* Requires its expected response formatting (“harmony”-style) for best behavior. ([Hugging Face][6])

**Option B (fallback / smaller GPU):** Qwen3 math-capable instruct model (dense or MoE)

* Qwen3 models are open-weight and support “thinking vs non-thinking” modes. ([GitHub][7])

**Why not rely purely on the base LLM?**
The AIMO3 reference bench shows a large gap between naive open-weight usage and frontier closed models; closing that gap requires **tool use + verification + training adaptation**, not just prompting. ([studylib.net][8])

### 3.2 Auxiliary fast model (router + draft + backup solver)

Use a **7B–20B** open-weight model (on GPU in 4-bit or on CPU) for:

* **Problem routing** (domain, difficulty, recommended strategy)
* **Speculative decoding** draft model (if using vLLM speculative)
* **Cheap “second opinion” solve** when primary is uncertain

Candidates:

* `openai/gpt-oss-20b` (paired with gpt-oss-120b) ([Hugging Face][6])
* `Qwen3-14B` (or similar) ([GitHub][7])

### 3.3 Tooling modules (non-LLM)

* **Symbolic engine:** SymPy (solve, simplify, nt-theory helpers)
* **Arithmetic engine:** Python big integers + `fractions.Fraction`
* **Optional solver:** Z3 (for discrete constraints / Diophantine / case splits)
* **Safe code sandbox:** restricted Python executor with time+memory limits

### 3.4 Neural verifier / reranker (optional but recommended)

A **small judge model** (1B–7B) trained to score candidate solutions:

* Input: (problem, candidate trace, candidate answer, verification artifacts)
* Output: probability-correct + error-type logits (arithmetic vs logic vs constraint)

This judge is **not** the final authority; it is used alongside symbolic checks.

---

## 4. Data Strategy (Avoid Contamination, Still Get Olympiad Competence)

Even though AIMO3 problems are stated to be original (“zero risk of data contamination”), we still enforce **strict provenance** and avoid any AIMO3-derived community datasets. ([AIMO Prize][1])

### 4.1 Hard contamination rules (enforced by code)

**Blocked sources:**

* Any dataset explicitly containing **AIMO3** / “reference.csv solutions” / Kaggle discussion dumps
* Any dataset created after AIMO3 launch and tagged with “AIMO3 solutions”
* Any text that matches reference bench problems (hash-based exact match + near-dup)

**Near-duplicate filter:**

* Normalize LaTeX (strip whitespace, canonicalize commands)
* Compute MinHash shingles over tokens
* Drop samples with Jaccard similarity > 0.85 to any blocked item or to each other

### 4.2 Allowed data categories

**A) Public math QA + contest datasets (answers + solutions)**
Used to teach core transformations and theorem patterns.

**B) Programmatically generated olympiad-style families (the backbone)**
We generate *AI-hard* problems where:

* solution requires multi-step reasoning
* final answer is not guessable (uniform-ish 0–99999)
* constraints are tight and verifiable by code

**C) Self-play rollouts**
Model generates solutions; we verify with sandbox tools; only verified correct traces are kept.

### 4.3 Dataset mixture proportions (SFT stage)

A practical mixture for the **final SFT** (example; tune via ablations):

* 40% **Programmatic synthetic families** (fully verified)
* 25% **Curated contest-style problems** (with solutions)
* 20% **Tool-integrated reasoning traces** (Python/SymPy)
* 10% **Verifier training pairs** (correct vs incorrect, same problem)
* 5% **Format & robustness** (answer extraction, modulus, edge cases)

### 4.4 “AI-hard” synthetic generation: concrete templates

We generate families with **known solvers**:

* **Number theory:**

  * LTE valuations; CRT systems; multiplicative orders
  * “count v_p of expression, then output mod M”
* **Algebra:**

  * integer solutions to polynomial constraints; symmetric sums; functional equations with finite support
* **Combinatorics:**

  * counting with inclusion–exclusion, generating functions, recursion with mod
* **Geometry (tool-verifiable):**

  * coordinate-geometry / complex plane setups with randomized parameters but consistent constraints

Each family has:

* generator `gen(params)->problem_text`
* solver `solve(params)->answer`
* verifier `check(problem_text, answer)->bool` (independent from solver)

---

## 5. Training Pipeline (Reproducible, Verifier-Guided)

### 5.1 Stage 0: Base model selection + format alignment

* If using **gpt-oss**, keep its expected formatting and tool-call conventions (important for stability). ([Hugging Face][6])
* Pin exact versions of:

  * transformers / vLLM
  * gpt-oss python package (if required)
  * sympy

### 5.2 Stage 1: SFT (Supervised Fine-Tuning)

**Objective:**
Teach the model to produce:

1. structured plan
2. tool code when appropriate
3. robust final answer extraction

**Training example schema (JSONL):**

```json
{
  "prompt": "<problem latex>",
  "response": {
    "domain": "number_theory",
    "plan": ["..."],
    "tool_calls": [{"language":"python","code":"..."}],
    "final": {"answer": 57447, "modulus": 99991}
  }
}
```

**Key implementation detail:**
Train the model to output **both**:

* a machine-readable `final.answer` integer
* and a short “reason summary” (not required at inference, but helps judge training)

### 5.3 Stage 2: Verifier dataset construction (for judge + RL)

For each training problem:

* sample K solutions from the model (different prompts/temperatures)
* run tool-verifier to label `(correct/incorrect)` + failure reason
* build preference pairs: (correct trace > incorrect trace)

### 5.4 Stage 3: Verifier-guided post-training (DPO/GRPO-style)

**Preferred (simpler, stable): DPO**

* For each pair (good, bad), optimize preference

**Optional (stronger, harder): GRPO-like**

* Reward =

  * +1 if verified correct answer
  * +0.2 if passes constraint checks
  * −0.5 if code unsafe / times out
  * −0.2 if answer format invalid

### 5.5 Curriculum learning schedule

* Weeks 1–2: mostly easy/medium synthetic + tool traces
* Weeks 3–4: ramp difficulty; add geometry and heavy comb
* Weeks 5+: hard-only batches; mine “model failure” clusters and oversample

### 5.6 Scaling expectations (practical)

* Biggest gains come from:

  1. tool-integrated traces
  2. verified synthetic data at scale
  3. judge-guided selection

---

## 6. Inference-Time Reasoning Engine (Hierarchical, Multi-Path)

### 6.1 Problem classification (fast, robust)

**Inputs:** raw LaTeX string
**Outputs:**

* `domain ∈ {algebra, number_theory, combinatorics, geometry, mixed}`
* `difficulty ∈ [0,1]`
* `modulus` if present (regex)
* `requires_tool ∈ {yes/no}`

**Implementation:**

* Regex-based feature extractor + small router model (7B/20B) for final label
* Cache parsing results by `problem_id`

### 6.2 Multi-path candidate generation (4 primary paths)

We generate candidates in a staged way, stopping early if confidence is high.

**Path P0 — Symbolic-first (no LLM if possible)**

* If problem matches known patterns (explicit equations, gcd/lcm, counting with small params)
* Try SymPy / number theory functions immediately

**Path P1 — Tool-Integrated Reasoning (TIR)**

* Ask LLM to write Python/SymPy code returning integer answer
* Run code in sandbox, collect output + intermediate artifacts

**Path P2 — Pure reasoning (CoT / scratchpad)**

* Ask LLM to derive answer without tools
* Use mainly for geometry insights / theorem selection

**Path P3 — Backward / constraint-driven**

* Ask LLM to propose constraints on answer (mod, parity, bounds)
* Use targeted search (CRT enumeration, factor constraints, small parameter brute force)

### 6.3 Answer extraction rules (deterministic)

* Always parse last occurrence of:

  * `\boxed{...}`
  * `FINAL: <int>`
  * JSON `final.answer`
* Clamp to `[0, 99999]` **only after** applying modulus logic
* If modulus present: reduce `ans % modulus`, then map into `[0,99999]` if required by statement

---

## 7. Self-Consistency, Verification, and Arbitration

### 7.1 Verification stack (ordered)

1. **Format & range**
2. **Explicit constraint checks**

   * modulus, divisibility, parity, positivity
3. **Symbolic re-evaluation** (if we have a formal model)

   * plug answer into derived equations
4. **Randomized property tests**

   * for identities or derived formulas, test on small random instances
5. **Neural judge score**

   * ranks candidates when symbolic checks are weak (e.g., pure geometry)

### 7.2 Confidence score (concrete)

For candidate `c`:

```
score(c) =
  2.0 * passes_hard_constraints
+ 1.5 * symbolic_consistency
+ 1.0 * randomized_tests_pass_rate
+ 1.0 * judge_prob_correct
+ 0.5 * self_consistency_vote_share
- 2.0 * any_contradiction_flag
- 1.0 * sandbox_timeout_or_error
```

Hard constraints are binary gates; failing them usually discards the candidate.

---

## 8. Error Detection & Recovery

### 8.1 Detect common failure modes

**Arithmetic slip detection**

* Recompute with Python bigints from extracted intermediate expressions if available
* Ask model to restate final computation as Python and re-run

**Casework incompleteness**

* Ask model for “missing cases” and produce a checklist
* Run small brute force for small parameter versions if possible

**Geometry misread**

* Force coordinate placement and compute with sympy (complex plane / vectors)
* Verify derived equalities numerically for random instantiations that preserve constraints

**Modular mistakes**

* Run CRT validation for multiple moduli (e.g., check mod 2,3,5,7,11) if consistent with derivation

### 8.2 Recovery policy (progressive deepening)

* If top candidate score < threshold:

  * allocate more attempts in TIR path
  * switch prompt style (theorem-first vs computation-first)
  * activate “lemma mining” loop (hard problems)

---

## 9. Double-Run Penalized Scoring Strategy (Private Evaluation)

Private score effectively averages correctness across two reruns. ([GitHub][4])
So the goal is **high accuracy per run**, not merely diversity.

Still, we exploit reruns safely by:

### 9.1 Run-level stochasticity (controlled, logged)

* Seed = `int.from_bytes(os.urandom(4),'big')`
* Use the seed only to choose between *two strong prompt programs*:

  * Program A: tool-heavy, conservative
  * Program B: more theorem/insight heavy, slightly different decomposition

This is legitimate: it changes internal reasoning paths, while keeping outputs valid.

### 9.2 Temperature scheduling (minimal)

* Router + judge calls: `temperature=0`
* Candidate generation:

  * easy: `temp=0`, 1 attempt
  * medium: `temp=0.2`, 2 attempts
  * hard: `temp=0.4`, 4–6 attempts
* Heavy reliance on verification to avoid “creative wrongness”.

**Important:** Because server startup must be fast, load model lazily on first call, and cache it globally. ([GitHub][2])

---

## 10. Hard Problem Strategy (The “Boss Fight” Pipeline)

### 10.1 Hardness detection (fast heuristics)

Trigger hard-mode if:

* router difficulty > 0.75
* no candidate passes hard constraints after baseline attempts
* domain is geometry+combinatorics mixed
* statement length / nested definitions exceed thresholds

### 10.2 Hard-mode loop (bounded search)

**Core idea:** Treat solving as search over *formalization attempts*, not raw text reasoning.

**Hard-mode components**

* Lemma mining: propose 3–6 lemmas, each with a micro-verification test
* Program synthesis search: generate multiple python programs; keep those that pass sanity checks
* Proof sketch synthesis: compress to a minimal invariant / key identity
* Pruned brute force on small instances: verify pattern, then generalize

### 10.3 Meta-reasoning step

Ask the model:

* “What is the *most fragile* inference in your solution?”
* “What alternate method can independently confirm the answer?”
  Then force an alternate derivation path.

---

## 11. Verification Module Design (Formal-ish, SymPy-centric)

### 11.1 LaTeX parsing strategy

Avoid fragile full LaTeX parsing as a single point of failure.

We do:

* lightweight normalization (remove formatting, standardize `\cdot`, `^`, `\frac`)
* regex extraction of:

  * modulus patterns (“mod 99991”, “remainder when divided by …”)
  * variable symbols
  * equation-like substrings (contains `=`)

For full formalization, use the LLM to output **SymPy-native code**, not raw LaTeX.

### 11.2 Sandbox execution (safe python)

* AST whitelist:

  * allowed imports: `math`, `fractions`, `itertools`, `sympy`, `numpy` (optional)
  * disallow filesystem/network/subprocess
* Time limit per run: 1–5 seconds (configurable)
* Memory cap: e.g. 1–2 GB per subprocess
* Capture stdout/stderr, parse `ANSWER = <int>`

### 11.3 Verification primitives

* `check_modulus(ans, modulus)`
* `check_integer_constraints(ans)`
* `check_equations(ans, sympy_equations)`
* `randomized_identity_tests(fn, trials=30)`
* `small_instance_bruteforce(template_solver)`

---

## 12. Compute Budget Allocation (Kaggle-realistic)

Constraints from the gateway:

* one problem per call ([GitHub][3])
* up to ~9 hours total runtime ([GitHub][3])
* start server within ~15 minutes ([GitHub][2])

### 12.1 Token/time budgets (example policy)

**Per problem default caps**

* Easy: 10–20s, 1 attempt, max 800–1200 output tokens
* Medium: 30–60s, 2–3 attempts, tool run 1–2 times
* Hard: 2–6 min, 4–8 attempts, lemma loop + multiple tool runs

### 12.2 Progressive deepening

Stop early when:

* same answer found by ≥2 independent paths **and**
* passes all extracted constraints **and**
* judge score > 0.9

### 12.3 Memory plan on H100

* Main model in FP8/INT4 (vLLM) where possible
* Keep only one large model resident; auxiliary model quantized or CPU

---

## 13. Expected Performance Model (Targets + Risk)

AIMO3 is harder than AIMO2; reference bench notes many open-weight models struggle on the hardest problems without substantial technique improvements. ([studylib.net][8])

### 13.1 Target accuracy by domain (post-training + verifier)

* Algebra: 96–99%
* Number theory: 94–98%
* Combinatorics: 90–96%
* Geometry: 88–95%

To reach >47/50, geometry+comb failures must be aggressively reduced via tool formalization and search.

### 13.2 Where failures concentrate

* Geometry configurations requiring nontrivial synthetic reasoning
* Combinatorics with deep invariants + tricky overcount
* Problems requiring *invented lemma* not in training distribution
* Parser misses modulus/constraint phrase → wrong normalization

---

## 14. Ablation Studies Plan (Competition-style, concrete)

Evaluate on:

* reference bench (10) **only as validation, not training** ([studylib.net][8])
* multiple out-of-distribution olympiad sets (held out by year/source)
* synthetic hard set (generated, fully verified)

### 14.1 Ablations

1. **No tools** vs tools
2. **No judge** vs judge
3. **Single-path** vs multi-path (P0–P3)
4. **No progressive deepening** (fixed compute) vs adaptive
5. **No hard-mode loop** vs hard-mode
6. **No RL/DPO** vs DPO-only vs DPO+GRPO
7. **No synthetic families** vs 10% vs 40% synthetic

Metrics:

* raw accuracy
* average of two independent reruns (simulate private scoring)
* runtime distribution (p50/p90)
* “catastrophic failure rate” (timeouts, format invalid)

---

## 15. Failure Analysis Framework (Automated, Iterative)

### 15.1 Logging schema (per problem)

* normalized problem text hash
* extracted constraints (modulus, equations, domains)
* candidate list:

  * answer
  * generation path (P0–P3)
  * tool outputs
  * verification flags
  * judge score
* final decision rationale

### 15.2 Error clustering taxonomy

* **Arithmetic slip**
* **Constraint extraction error** (missed modulus, misread “remainder of pq”)
* **Combinatorial overcount/undercount**
* **Geometry misformalization**
* **Casework incompleteness**
* **Tool hallucination** (wrong code, wrong assumptions)
* **Search budget failure** (ran out of attempts)

Use clustering on:

* router features + judge error-type logits + textual fingerprints
  to prioritize new synthetic families and prompt fixes.

---

## 16. Reasoning Strategy Pseudocode (Core Inference)

### 16.1 Main `predict()` entry (Kaggle server)

```python
# Global singletons (lazy-init)
LLM_MAIN = None
LLM_FAST = None
JUDGE = None
RUN_SEED = int.from_bytes(os.urandom(4), "big")

def predict(id_series, problem_series):
    pid = id_series.item(0)
    text = problem_series.item(0)

    ans = solve_one(pid, text, run_seed=RUN_SEED)
    return pl.DataFrame({"id": pid, "answer": int(ans)})
```

### 16.2 Solver orchestration

```python
def solve_one(pid: str, latex: str, run_seed: int) -> int:
    meta = parse_and_extract(latex)     # modulus, domain hints, etc.
    mode = route_problem(meta, latex)   # domain, difficulty, strategy

    budget = allocate_budget(mode)      # time/attempt caps

    candidates = []

    # P0: symbolic-first
    if mode.try_symbolic_first:
        candidates += sympy_first_pass(latex, meta, budget)

    # P1/P2/P3: LLM multi-path with progressive deepening
    while budget.remaining_time() > 0 and not confident_enough(candidates, meta):
        # Generate next batch depending on what failed so far
        batch = generate_candidates_multimodal(latex, meta, mode, budget, run_seed)
        # Verify
        verified = [verify_candidate(c, latex, meta) for c in batch]
        candidates += verified

        # Recovery triggers
        if too_many_invalid(candidates):
            mode = switch_prompt_program(mode, run_seed)
        if stuck(candidates) and mode.allow_hard_mode:
            candidates += hard_mode_loop(latex, meta, budget, run_seed)

    best = select_final_answer(candidates, meta)
    return best
```

### 16.3 Candidate generation (multi-path)

```python
def generate_candidates_multimodal(latex, meta, mode, budget, seed):
    out = []

    # Path P1: Tool-integrated reasoning (code)
    if budget.allow_tool_calls and mode.tool_priority:
        out += llm_generate_python_solutions(latex, meta, n=mode.n_tool_attempts, seed=seed)

    # Path P2: Pure reasoning
    out += llm_generate_reasoned_solutions(latex, meta, n=mode.n_cot_attempts, seed=seed)

    # Path P3: Backsolve / constraint-driven search prompts
    if mode.use_backsolve:
        out += llm_generate_constraints_then_search(latex, meta, n=mode.n_backsolve_attempts, seed=seed)

    return out
```

### 16.4 Verification + scoring

```python
def verify_candidate(cand, latex, meta):
    # cand contains answer + optional python code + trace
    v = VerificationResult(answer=cand.answer)

    v.hard_ok = check_range(v.answer) and check_modulus(v.answer, meta)
    if not v.hard_ok:
        v.score = -999
        return v

    if cand.python_code:
        v.tool_ok, v.tool_artifacts = sandbox_run_and_validate(cand.python_code, meta)

    v.symbolic_ok = symbolic_checks_if_possible(latex, v.answer, meta)
    v.random_ok   = randomized_tests_if_possible(cand, meta)

    v.judge_prob  = judge_score(latex, cand.trace, v.answer, v.tool_artifacts)

    v.score = (
        2.0 * v.hard_ok +
        1.5 * v.symbolic_ok +
        1.0 * v.random_ok +
        1.0 * v.judge_prob
    )
    return v
```

---

## 17. Minimum Viable Version (2 weeks)

### MVP (aim: 40+ / 50)

* One strong open model (e.g., Qwen3-14B / math-tuned)
* Tool sandbox + SymPy-first patterns
* 2–3 attempts per problem + basic voting
* Robust answer extraction + modulus parsing

### To push from ~40 → 47+

* Add:

  * **program-synthesis search** (multiple code candidates, prune by checks)
  * **neural judge** trained on verified positives/negatives
  * **hard-mode lemma mining** for geometry/comb
  * **DPO/RLAIF** on verifier-labeled rollouts
  * **large primary model on H100** (e.g., gpt-oss-120b) with tool-centric prompting ([OpenAI Developers][5])

---

## 18. Roadmap to 47+ (What must be true)

To reliably exceed **47/50**, the system must:

1. Solve nearly all algebra/NT via **tool formalization + verification**
2. Reduce geometry failure by enforcing **coordinate/complex formalizations**
3. Reduce combinatorics failure by:

   * enumerating small instances to validate conjectures
   * using inclusion–exclusion / recursion code templates
4. Have **high operational robustness** (no crashes, no timeouts, no bad formatting)
5. Achieve **high per-run accuracy** (private score is average over reruns) ([GitHub][4])

---

If you want, I can also provide:

* a concrete repository layout (`src/`, `configs/`, `scripts/`) with pinned versions,
* exact prompt templates for each path (P0–P3),
* and a “synthetic family library” spec (20–30 generators) that is fully solver-verifiable and designed to match AIMO3 difficulty without leaking anything.

[1]: https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched "https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched"
[2]: https://raw.githubusercontent.com/LuciferK47/AIMO-3/main/kaggle_evaluation/core/relay.py "https://raw.githubusercontent.com/LuciferK47/AIMO-3/main/kaggle_evaluation/core/relay.py"
[3]: https://raw.githubusercontent.com/LuciferK47/AIMO-3/main/kaggle_evaluation/aimo_3_gateway.py "https://raw.githubusercontent.com/LuciferK47/AIMO-3/main/kaggle_evaluation/aimo_3_gateway.py"
[4]: https://github.com/govher-s/AI-Math_Olympiad "https://github.com/govher-s/AI-Math_Olympiad"
[5]: https://developers.openai.com/api/docs/models/gpt-oss-120b "https://developers.openai.com/api/docs/models/gpt-oss-120b"
[6]: https://huggingface.co/openai/gpt-oss-120b "https://huggingface.co/openai/gpt-oss-120b"
[7]: https://github.com/QwenLM/Qwen3 "https://github.com/QwenLM/Qwen3"
[8]: https://studylib.net/doc/28138782/aimo3-reference-problems "https://studylib.net/doc/28138782/aimo3-reference-problems"
