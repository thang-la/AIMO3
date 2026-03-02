# AIMO3 End-to-End System

Open-source implementation of the architecture in `prompt0.md`:

- parse + constraint extraction
- adaptive router + compute budget
- multi-path candidate generation (P0/P1/P2/P3)
- verification stack + arbitration
- hard-mode recovery loop
- Kaggle-style `predict(id_series, problem_series)` entrypoint
- reproducible training-data pipeline (synthetic + self-play + verifier pairs)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[solver,dev]"
```

Solve one problem:

```bash
aimo3 solve-one --problem "What is the remainder when 123456 is divided by 97?"
```

Run CSV -> submission:

```bash
aimo3 solve-csv --input reference.csv --output submission.csv
```

Kaggle-style entrypoint:

```python
from aimo3.kaggle_server import predict
```

Full technical documentation: [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)
