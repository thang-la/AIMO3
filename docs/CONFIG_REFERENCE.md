# Configuration Reference

## Environment variables

- `AIMO3_MODEL_ID`
  - Required value: `openai/gpt-oss-120b`
- `AIMO3_MODEL_PATH`
  - Local path or HF identifier for model weights
- `AIMO3_LLM_BACKEND`
  - `vllm` or `transformers`
- `AIMO3_TP`
  - tensor parallel size for vLLM loader
- `AIMO3_VLLM_GPU_MEMORY_UTILIZATION`
  - default `0.98`, commonly `0.99` on Kaggle H100
- `AIMO3_VLLM_MAX_MODEL_LEN`
  - default `4096`, reduce to `2048` if KV cache OOM
- `AIMO3_VLLM_MAX_NUM_SEQS`
  - default `1` for conservative memory profile
- `AIMO3_VLLM_MAX_NUM_BATCHED_TOKENS`
  - default `max(2048, max_model_len)`
- `AIMO3_VLLM_ENFORCE_EAGER`
  - default `1` to reduce torch.compile memory pressure

## Runtime notes

- Main model is lazily loaded on first generation call.
- `AIMO3Solver` default constructor enforces the mandatory model.
- For tests/dev, a stub model can be injected explicitly through constructor arguments.
