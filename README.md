# AIMO3 Belief-State Research System

Production-style implementation aligned to `prompt0.md`, `prompt1.md`, `prompt2.md`:
- belief-state controller
- calibrated reliability + correlated-error control
- correlation-aware Bayesian answer aggregation
- EVI-based action policy
- A/B run-program decorrelation
- adversarial refutation and hypothesis search

## Mandatory model policy

Default runtime is enforced to use:
- `openai/gpt-oss-120b`

The solver fails fast if this requirement is not met.

## Quick start

```bash
cd /home/main/Projects/AIMO3
python -m venv .venv
source .venv/bin/activate
pip install -e .[math,train,test,inference]
```

Set runtime env:

```bash
export AIMO3_MODEL_ID="openai/gpt-oss-120b"
export AIMO3_MODEL_PATH="/path/to/local/gpt-oss-120b"
export AIMO3_LLM_BACKEND="vllm"   # or transformers
export AIMO3_TP="1"
```

Run:

```bash
python main.py "Find the remainder when 123456 is divided by 97."
```

## Entrypoints

- Kaggle: `submission.predict`
- Internal: `aimo3.controller.AIMO3Solver.solve_one`

## Documentation

- Architecture: `docs/ARCHITECTURE.md`
- Local runbook: `docs/RUNBOOK_LOCAL.md`
- Training pipeline: `docs/TRAINING_PIPELINE.md`
- Kaggle submission: `docs/KAGGLE_SUBMISSION.md`
- Config reference: `docs/CONFIG_REFERENCE.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Competition strategy: `docs/COMPETITION_STRATEGY.md`

## Utility scripts

- `scripts/prepare_kaggle_bundle.sh`: build source zip for Kaggle Dataset upload
- `scripts/smoke_local.py`: local smoke with injected stub model

## Notes on competition outcome

This repository can be aligned to the required architecture and hardened for competition operation.
Actual leaderboard rank cannot be guaranteed because it depends on hidden test distribution, runtime stability under Kaggle constraints, training data quality, and ablation tuning quality.
