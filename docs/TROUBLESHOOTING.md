# Troubleshooting

## StrictModelRequirementError: vLLM import failed

Symptom:
- `StrictModelRequirementError: Không import được vLLM...`

Fix:
- install inference deps: `pip install -e .[inference]`
- or set `AIMO3_LLM_BACKEND=transformers`

## Wrong model ID configured

Symptom:
- error says model must be `openai/gpt-oss-120b`

Fix:
- `export AIMO3_MODEL_ID="openai/gpt-oss-120b"`

## OOM when loading 120B

Fix options:
- lower memory pressure (`AIMO3_TP`, backend settings)
- ensure hardware matches expected competition GPU profile
- keep one main model resident and avoid extra large model loads

If you hit:
- `ValueError: No available memory for the cache blocks`

Use:

```bash
export AIMO3_VLLM_GPU_MEMORY_UTILIZATION=0.99
export AIMO3_VLLM_MAX_MODEL_LEN=2048
export AIMO3_VLLM_MAX_NUM_SEQS=1
export AIMO3_VLLM_MAX_NUM_BATCHED_TOKENS=2048
export AIMO3_VLLM_ENFORCE_EAGER=1
```

If sampler warmup OOM appears (`warming up sampler with 1024 dummy requests`), lower GPU utilization:

```bash
export AIMO3_VLLM_GPU_MEMORY_UTILIZATION=0.93
```

The loader retries multiple memory profiles automatically, but explicit envs are still recommended for stability.

## Slow startup in Kaggle

Checks:
- ensure lazy initialization is intact (`aimo3/kaggle_predict.py`)
- avoid loading model at import-time

## Missing `polars`

`aimo3/kaggle_predict.py` includes a fallback shim for local debugging, but Kaggle runtime should use real `polars`.

## `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`

This usually indicates protobuf version mismatch from preinstalled packages. It is often noisy but non-fatal.
If it becomes fatal in your environment, pin protobuf to a compatible version (commonly `4.25.x`) in your runtime image.
