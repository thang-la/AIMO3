# Local Runbook

## 1. Environment

```bash
cd /home/main/Projects/AIMO3
python -m venv .venv
source .venv/bin/activate
pip install -e .[math,train,test,inference]
```

## 2. Required runtime model configuration

```bash
export AIMO3_MODEL_ID="openai/gpt-oss-120b"
export AIMO3_MODEL_PATH="/path/to/local/openai-gpt-oss-120b"
export AIMO3_LLM_BACKEND="vllm"            # or transformers
export AIMO3_TP="1"
export AIMO3_VLLM_GPU_MEMORY_UTILIZATION="0.95"
export AIMO3_VLLM_MAX_MODEL_LEN="2048"     # increase only if VRAM still has room
export AIMO3_VLLM_MAX_NUM_SEQS="1"
export AIMO3_VLLM_MAX_NUM_BATCHED_TOKENS="1024"
export AIMO3_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="1024"
export AIMO3_VLLM_ASYNC_SCHEDULING="1"
export AIMO3_VLLM_ENABLE_PREFIX_CACHING="0"
export AIMO3_VLLM_ENFORCE_EAGER="0"
```

If config is invalid or backend missing, solver fails fast with `StrictModelRequirementError`.

## 3. Quick solver run

```bash
python main.py "Find the remainder when 123456 is divided by 97."
```

## 4. Programmatic run

```bash
python - <<'PY'
from aimo3.controller import AIMO3Solver
solver = AIMO3Solver()
print(solver.solve_one("p1", "Find the remainder when 123456 is divided by 97.", run_seed=42))
PY
```

## 5. Basic quality checks

```bash
python -m compileall -q aimo3 tests main.py submission.py
python -m pytest -q
```

## 6. Local smoke without loading 120B (dev only)

Use explicit test double injection:

```python
from aimo3.controller import AIMO3Solver
from aimo3.models import DeterministicHeuristicModel

stub = DeterministicHeuristicModel()
solver = AIMO3Solver(main_model=stub, fast_model=stub)
```

This bypass is for development/tests only.
