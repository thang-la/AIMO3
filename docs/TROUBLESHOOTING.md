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

## Slow startup in Kaggle

Checks:
- ensure lazy initialization is intact (`aimo3/kaggle_predict.py`)
- avoid loading model at import-time

## Missing `polars`

`aimo3/kaggle_predict.py` includes a fallback shim for local debugging, but Kaggle runtime should use real `polars`.
