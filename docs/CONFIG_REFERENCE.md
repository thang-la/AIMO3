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
  - default `0.95`; tune in range `0.91-0.97` when sampler/KV OOM happens
- `AIMO3_VLLM_MAX_MODEL_LEN`
  - default `2048` for 1x H100 stability profile
- `AIMO3_VLLM_MAX_NUM_SEQS`
  - default `1` for conservative memory profile
- `AIMO3_VLLM_MAX_NUM_BATCHED_TOKENS`
  - default `1024` (recipe-aligned for GPT-OSS on 1x H100 TP1)
- `AIMO3_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE`
  - default `min(2048, max_num_batched_tokens)`
- `AIMO3_VLLM_ASYNC_SCHEDULING`
  - default `1`
- `AIMO3_VLLM_ENABLE_PREFIX_CACHING`
  - default `0` (disabled for memory headroom and stable profiling)
- `AIMO3_VLLM_ENFORCE_EAGER`
  - default `0`; set `1` only as OOM fallback

## Runtime notes

- Main model is lazily loaded on first generation call.
- `AIMO3Solver` default constructor enforces the mandatory model.
- For tests/dev, a stub model can be injected explicitly through constructor arguments.
