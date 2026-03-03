# AIMO3 End-to-End System

Open-source implementation of the architecture in `prompt0.md`:

- parse + constraint extraction
- adaptive router + compute budget
- multi-path candidate generation (P0/P1/P2/P3)
- iterative repair loop (P4) from top verified candidates
- verification stack + arbitration
- hard-mode recovery loop
- optional memory retrieval for exact/near-seen statements
- Kaggle-style `predict(id_series, problem_series)` entrypoint
- reproducible training-data pipeline (synthetic + self-play + verifier pairs)

Built-in pre-finetune exact pattern solvers (symbolic):

- floor-sum -> divisor-sum valuation transforms
- additive functional-equation counting via constraint enumeration
- integer story-system solver (ages/sweets class)

## Competition Mode (default)

The solver now runs in **strict competition mode** by default:

- requires a real LLM runtime (`vLLM` or `Transformers`)
- no hidden heuristic fallback
- raises an error if model runtime is unavailable

To use demo fallback explicitly (local smoke only), set:

```bash
export AIMO3_BACKEND=heuristic
export AIMO3_ALLOW_DEMO_FALLBACK=1
export AIMO3_ENFORCE_REAL_BACKEND=0
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[solver,runtime,dev]"
```

Solve one problem:

```bash
aimo3 solve-one --problem "What is the remainder when 123456 is divided by 97?"
```

Run with explicit production backend:

```bash
aimo3 solve-one \
  --backend vllm \
  --model-main /kaggle/input/models/openai/gpt-oss-120b \
  --problem "..."
```

If model path is omitted in Kaggle offline, runtime attempts to auto-discover matching model folders under `/kaggle/input`.

Run CSV -> submission:

```bash
aimo3 solve-csv --input reference.csv --output submission.csv
```

Enable memory retrieval against a solved reference file:

```bash
aimo3 solve-csv --input public.csv --output submission.csv \
  --allow-reference-lookup --reference-path reference.csv
```

Enable full runtime debug trace (JSON-lines to stderr):

```bash
aimo3 solve-csv --input public.csv --output submission.csv --debug
```

Write debug trace to file and include raw LLM outputs:

```bash
aimo3 solve-one --problem "..." --debug --debug-raw-output --debug-file runs/debug.jsonl
```

Kaggle-style entrypoint:

```python
from aimo3.kaggle_server import predict
```

Full technical documentation: [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)
