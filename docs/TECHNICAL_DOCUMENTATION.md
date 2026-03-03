# Technical Documentation: AIMO3 End-to-End System

## 1. Scope

This repository implements the full inference + training-data workflow defined in `prompt0.md`:

- inference architecture with parse -> route -> multi-path generation -> verification -> arbitration
- Kaggle-compatible server entrypoint with lazy singleton initialization
- offline data pipeline for synthetic SFT data, self-play rollouts, and verifier pair generation
- contamination guard and near-duplicate filtering

The default runtime backend is **strict real-model mode** (`CompetitionLLMBackend`), with lazy runtime loading for `vLLM`/`Transformers`. Heuristic mode exists only as explicit fallback for local smoke tests.

## 2. Repository Layout

- `aimo3/config.py`: global solver settings, time/attempt budgets, thresholds.
- `aimo3/models.py`: core dataclasses and enums (`ProblemMetadata`, `RouteDecision`, `Candidate`, `VerificationResult`, `SolveResult`).
- `aimo3/parsing.py`: normalization and extraction (modulus, equations, variables, number tokens, domain hints).
- `aimo3/router.py`: domain/difficulty routing and strategy flags.
- `aimo3/budget.py`: progressive-deepening budget manager.
- `aimo3/llm.py`: generation backend interface, competition LLM backend (vLLM/Transformers), strict runtime checks, heuristic fallback, and judge placeholder.
- `aimo3/sandbox.py`: safe Python executor (AST allowlist + timeout + memory cap).
- `aimo3/symbolic.py`: P0 symbolic-first solver rules (safe arithmetic + optional SymPy linear case).
- `aimo3/generator.py`: multi-path candidate generation (P1/P2/P3) and repair generation (P4).
- `aimo3/verifier.py`: hard checks, tool/validator execution, scoring with vote + path-diversity, final selection.
- `aimo3/hard_mode.py`: bounded hard-mode search loop.
- `aimo3/controller.py`: orchestration class `AIMO3Solver`.
- `aimo3/memory.py`: optional retrieval over solved references (exact/near matches).
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
   - `P4`: repair candidates generated from top prior attempts
   - `P5` (optional): memory retrieval candidate for repeated/near-seen statements
5. Verification for each candidate:
   - sandbox execution if code is present
   - optional `validator_code` execution for independent acceptance/rejection
   - hard constraints (integer/range + modulus normalization)
   - symbolic consistency heuristic
   - randomized check proxy
   - judge score
6. Arbitration:
   - apply self-consistency vote share by normalized answer
   - compute weighted score
   - pick top candidate
7. Fallback:
   - strict mode (`enforce_real_backend=True`) throws if no valid candidate survives.
   - demo mode can optionally use deterministic fallback.

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
+ 0.6 * path_diversity
+ 0.8 * validator_bonus
- 2.0 * contradiction
- 1.0 * sandbox_error
```

`vote_share` is computed from answer agreement among valid candidates.
`path_diversity` rewards agreement across independent reasoning paths.

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
- reads backend/model settings from environment variables (`AIMO3_BACKEND`, `AIMO3_MODEL_MAIN`, `AIMO3_MODEL_FAST`, etc.)
- auto-resolves repo-like model refs to mounted Kaggle paths under `/kaggle/input` when possible
- exposes `predict(id_series, problem_series)` and returns:
  - `polars.DataFrame` if available
  - else `pandas.DataFrame`
  - else plain dict

## 6.1 Runtime Backends

`CompetitionLLMBackend` supports:

- `vllm` runtime (`vllm.LLM`) with tensor-parallel and memory utilization controls
- `transformers` runtime (`AutoModelForCausalLM` + generation pipeline)
- `auto` mode: tries `vllm` then `transformers`

Path prompting strategy:

- `P1` tool path returns JSON containing `python_code`, `validator_code`, and `final_answer`
- `P2` reasoning path returns JSON `final_answer` with explicit independent-check trace
- `P3` backsolve path returns JSON with constraints-based derivation
- `P4` repair path critiques top prior candidates and proposes corrected answer/code

Output parser extracts:

- JSON payload (`final_answer`, `answer`, nested `final.answer`)
- `\boxed{...}` form
- `FINAL: ...` form

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

Runtime debug tracing can be enabled via CLI/env:

- CLI: `--debug`, `--debug-file`, `--debug-raw-output`, `--debug-max-chars`
- Env: `AIMO3_DEBUG=1`, `AIMO3_DEBUG_FILE=...`, `AIMO3_DEBUG_RAW_OUTPUT=1`, `AIMO3_DEBUG_MAX_CHARS=...`

When enabled, the solver emits JSON-lines events for each stage:
`solve_start`, `parse_done`, `route_done`, candidate generation/verification batches, repair/hard-mode loops, and final selection.

## 9. Extension Points

To connect real models:

1. Implement a class extending `BaseLLMBackend` in `aimo3/llm.py`.
2. Wire it into `AIMO3Solver(config, llm_main=..., llm_fast=...)`.
3. Optionally replace `NeuralJudge` with a trained verifier model.

No controller logic changes are required.

## 10. Runbook

Install:

```bash
pip install -e ".[solver,runtime,dev]"
```

Solve CSV:

```bash
aimo3 solve-csv --input reference.csv --output submission.csv --evaluate
```

Solve with reference-memory bootstrap (useful for repeated public benchmark sets):

```bash
aimo3 solve-csv --input public.csv --output submission.csv \
  --allow-reference-lookup --reference-path reference.csv
```

Competition run (example):

```bash
export AIMO3_BACKEND=vllm
export AIMO3_MODEL_MAIN=/kaggle/input/models/openai/gpt-oss-120b
export AIMO3_MODEL_FAST=/kaggle/input/models/openai/gpt-oss-20b
export AIMO3_TENSOR_PARALLEL_SIZE=1
export AIMO3_ENFORCE_REAL_BACKEND=1
aimo3 solve-csv --input test.csv --output submission.csv
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

- Guaranteed leaderboard threshold (e.g., `>47/50`) cannot be promised without full weight selection, prompt calibration, and validation against held-out distributions.
- Symbolic parser is intentionally conservative to avoid unsafe over-parsing.
- Verifier is extensible; advanced formal checks (deep geometry formalization, Z3 integrations) are stub-ready and can be added without changing interfaces.
