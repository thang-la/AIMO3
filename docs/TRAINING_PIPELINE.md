# Training Pipeline

This repository includes training scaffolding for the decision stack (ARM/CED/AVM/policy), aligned with prompt1-2.

## 1. Artifacts and schemas

- Records schema: `aimo3/training/records.py`
- Logging/export helpers: `aimo3/training/logging.py`

Logged artifacts:
- candidate records
- pairwise correlation records
- step-level decision records

## 2. Build rollouts

Input JSONL format example:

```json
{"id":"p1","problem":"...","answer":12345}
```

Build rollout files:

```bash
python -m aimo3.training.build_rollouts /path/to/train.jsonl /path/to/out_rollouts
```

## 3. Flatten to JSONL for model training

Use helpers in `aimo3/training/logging.py` (or your own pipeline) to flatten rollout objects into:
- ARM candidate records JSONL
- CED pair records JSONL
- AVM/policy decision-step JSONL

## 4. Train ARM

```bash
python -m aimo3.training.train_arm /path/to/arm_records.jsonl /path/to/arm_model.json
```

Outputs:
- feature keys
- model weights
- calibration params

## 5. Train CED

```bash
python -m aimo3.training.train_ced /path/to/ced_records.jsonl /path/to/ced_model.json
```

## 6. Train AVM

```bash
python -m aimo3.training.train_avm /path/to/decision_records.jsonl /path/to/avm_model.json
```

## 7. Train policy model

```bash
python -m aimo3.training.train_policy /path/to/decision_records.jsonl /path/to/policy_model.json
```

## 8. Integrating trained artifacts

Current runtime uses heuristic in-memory models by default for ARM/CED/AVM behavior.
Integrate trained artifacts by adding model-loader adapters in:
- `aimo3/arm.py`
- `aimo3/ced.py`
- `aimo3/avm.py`

The module boundaries are already in place for this swap.
