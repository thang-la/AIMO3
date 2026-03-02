# Technical Documentation: AIMO3 End-to-End System

## 1. Scope

This repository implements the full inference + training-data workflow defined in `prompt0.md`:

- inference architecture with parse -> route -> multi-path generation -> verification -> arbitration
- Kaggle-compatible server entrypoint with lazy singleton initialization
- offline data pipeline for synthetic SFT data, self-play rollouts, and verifier pair generation
- contamination guard and near-duplicate filtering

The default runtime backend is deterministic and local (`HeuristicLLMBackend`) so the system runs immediately without downloading model weights. Real model backends can be plugged in behind the same interface.

## 2. Repository Layout

- `aimo3/config.py`: global solver settings, time/attempt budgets, thresholds.
- `aimo3/models.py`: core dataclasses and enums (`ProblemMetadata`, `RouteDecision`, `Candidate`, `VerificationResult`, `SolveResult`).
- `aimo3/parsing.py`: normalization and extraction (modulus, equations, variables, number tokens, domain hints).
- `aimo3/router.py`: domain/difficulty routing and strategy flags.
- `aimo3/budget.py`: progressive-deepening budget manager.
- `aimo3/llm.py`: generation backend interface, heuristic backend, and judge placeholder.
- `aimo3/sandbox.py`: safe Python executor (AST allowlist + timeout + memory cap).
- `aimo3/symbolic.py`: P0 symbolic-first solver rules (safe arithmetic + optional SymPy linear case).
- `aimo3/generator.py`: multi-path candidate generation (P1/P2/P3).
- `aimo3/verifier.py`: hard checks, tool execution, scoring, vote-share consistency, final selection.
- `aimo3/hard_mode.py`: bounded hard-mode search loop.
- `aimo3/controller.py`: orchestration class `AIMO3Solver`.
- `aimo3/kaggle_server.py`: `predict(id_series, problem_series)` entrypoint.
- `aimo3/cli.py`: local CLI (`solve-one`, `solve-csv`).
- `aimo3/training/*`: synthetic generation, contamination filtering, self-play, verifier pair construction.
- `docs/TECHNICAL_DOCUMENTATION.md`: this document.

## 3. Inference Flow

### 3.1 Step-by-step

1. `parse_problem` normalizes statement and extracts metadata:
   - modulus candidates
   - equation-like substrings
   - numeric tokens and variable hints
   - hash for reproducibility/logging
2. `route_problem` decides:
   - `domain ∈ {algebra, number_theory, combinatorics, geometry, mixed}`
   - difficulty score and tier (`easy/medium/hard`)
   - whether to prioritize tools, symbolic-first, backsolve, and hard-mode
3. `allocate_budget` configures per-problem caps (time, attempts, tool runs).
4. Candidate generation:
   - `P0`: symbolic-first (`symbolic_first_pass`)
   - `P1`: tool-integrated candidates (Python snippets)
   - `P2`: reasoning candidates
   - `P3`: backsolve/constraint candidates
5. Verification for each candidate:
   - sandbox execution if code is present
   - hard constraints (integer/range + modulus normalization)
   - symbolic consistency heuristic
   - randomized check proxy
   - judge score
6. Arbitration:
   - apply self-consistency vote share by normalized answer
   - compute weighted score
   - pick top candidate
7. Fallback:
   - if no valid candidate survives, return deterministic hash-based answer.

### 3.2 Confidence and progressive deepening

The solver loops until one of:

- confidence threshold met
- attempts/time budget exhausted
- candidate cap reached

Hard mode is activated when repeated rounds produce invalid candidates and router allows it.

## 4. Scoring Function

Candidate score follows the weighted policy from prompt design:

```text
score =
  2.0 * hard_ok
+ 1.5 * symbolic_ok
+ 1.0 * random_ok
+ 1.0 * judge_prob
+ 0.5 * vote_share
- 2.0 * contradiction
- 1.0 * sandbox_error
```

`vote_share` is computed from answer agreement among valid candidates.

## 5. Safe Tool Sandbox

`run_python_sandbox` enforces:

- AST allowlist (rejects disallowed nodes/imports)
- allowed imports: `math`, `fractions`, `itertools`, `sympy`, `numpy`, `collections`
- isolated subprocess execution
- timeout cutoff
- memory cap via `resource` (platform-dependent)
- output parsing from `ANSWER` variable or last printed integer

This gives a practical safe-by-default execution path for tool-generated code.

## 6. Kaggle Inference Pattern

`aimo3.kaggle_server`:

- keeps a global lazy singleton solver (`_SOLVER`) to satisfy startup constraints
- keeps one run seed (`_RUN_SEED`) for stable rerun behavior
- exposes `predict(id_series, problem_series)` and returns:
  - `polars.DataFrame` if available
  - else `pandas.DataFrame`
  - else plain dict

## 7. Training Data Pipeline

CLI entrypoint: `aimo3-train` (or `python -m aimo3.training.pipeline`)

### 7.1 Synthetic SFT

`build-synthetic` creates records with schema:

```json
{
  "prompt": "...",
  "response": {
    "domain": "...",
    "plan": ["..."],
    "tool_calls": [{"language": "python", "code": "ANSWER = ..."}],
    "final": {"answer": 12345, "modulus": 99991}
  },
  "meta": {"family": "..."}
}
```

Families included:

- number theory modular arithmetic
- algebra symmetric transformations
- combinatorics binomial counting
- geometry-style numeric transformations

### 7.2 Contamination guard

`training/contamination.py` applies:

- blocklist patterns (`AIMO3`, `reference.csv`, Kaggle discussion markers)
- normalization hash utility
- near-duplicate filter via token shingles + Jaccard threshold (default `0.85`)

### 7.3 Self-play and verifier pairs

- `self-play`: run solver on SFT prompts and collect candidate outcomes
- `verifier-pairs`: produce chosen/rejected pairs from hard-valid vs hard-invalid candidates

### 7.4 End-to-end pipeline

`run-all` executes:

1. synthetic generation
2. self-play rollouts
3. verifier pair construction

Outputs are JSONL artifacts in the configured work directory.

## 8. Operational Logging

Each solved problem writes `runs/<id>.json` containing:

- parsed metadata snapshot
- route decision
- full candidate verification table
- final answer and selection reason

This supports failure analysis and iterative tuning.

## 9. Extension Points

To connect real models:

1. Implement a class extending `BaseLLMBackend` in `aimo3/llm.py`.
2. Wire it into `AIMO3Solver(config, llm_main=..., llm_fast=...)`.
3. Optionally replace `NeuralJudge` with a trained verifier model.

No controller logic changes are required.

## 10. Runbook

Install:

```bash
pip install -e ".[solver,dev]"
```

Solve CSV:

```bash
aimo3 solve-csv --input reference.csv --output submission.csv --evaluate
```

Run training data pipeline:

```bash
aimo3-train run-all --workdir artifacts/training --count 500
```

Run tests:

```bash
pytest -q
```

## 11. Current Limits

- Default backend is heuristic, not a true frontier LLM; it is infrastructure-complete but not benchmark-optimized.
- Symbolic parser intentionally conservative to keep failure-safe behavior.
- Verifier is extensible; advanced formal checks (deep geometry formalization, Z3 integrations) are stub-ready and can be added without changing interfaces.
