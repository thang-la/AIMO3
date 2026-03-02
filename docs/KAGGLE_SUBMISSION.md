# Kaggle Submission Runbook (AIMO3)

## 1. Required entrypoint

Competition notebook should import:

```python
from submission import predict
```

`submission.py` exports the required `predict` function.

## 2. Package project into Kaggle Dataset

Upload project source as a Dataset and attach it in notebook.
If model weights are external, attach a second Dataset containing GPT-OSS-120B files.

## 3. Notebook setup

```python
!cp -r /kaggle/input/<source-dataset>/AIMO3 /kaggle/working/AIMO3
%cd /kaggle/working/AIMO3
!pip install -q --no-deps .
```

Set env vars:

```python
import os
os.environ["AIMO3_MODEL_ID"] = "openai/gpt-oss-120b"
os.environ["AIMO3_MODEL_PATH"] = "/kaggle/input/<model-dataset>/gpt-oss-120b"
os.environ["AIMO3_LLM_BACKEND"] = "vllm"
os.environ["AIMO3_TP"] = "1"
os.environ["AIMO3_VLLM_GPU_MEMORY_UTILIZATION"] = "0.99"
os.environ["AIMO3_VLLM_MAX_MODEL_LEN"] = "2048"
os.environ["AIMO3_VLLM_MAX_NUM_SEQS"] = "1"
os.environ["AIMO3_VLLM_MAX_NUM_BATCHED_TOKENS"] = "2048"
os.environ["AIMO3_VLLM_ENFORCE_EAGER"] = "1"
```

## 4. Integrate with Kaggle inference server

Use official gateway pattern and pass `predict` directly.

```python
from submission import predict
import kaggle_evaluation.aimo_3_inference_server as aimo3

server = aimo3.AIMO3InferenceServer(predict)
server.serve()
```

## 5. Startup/runtime constraints

- Model load must be lazy: handled in `aimo3/kaggle_predict.py`
- One-row-per-call predict contract is respected.

## 6. Pre-submit checklist

- `predict` returns columns: `id`, `answer`
- no network calls in inference
- model path accessible in Kaggle runtime
- startup and total runtime within competition budget

## 7. Validation snippet in notebook

```python
import polars as pl
from submission import predict
print(predict(pl.Series(["demo"]), pl.Series(["Find remainder when 100 is divided by 7."])))
```
