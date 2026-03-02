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

## Runtime notes

- Main model is lazily loaded on first generation call.
- `AIMO3Solver` default constructor enforces the mandatory model.
- For tests/dev, a stub model can be injected explicitly through constructor arguments.
