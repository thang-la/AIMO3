# Architecture (Prompt0-1-2 Aligned)

This codebase implements the belief-state solver architecture specified in `prompt0.md`, `prompt1.md`, and `prompt2.md`.

## 1. Controller-first design

Main orchestrator: `AIMO3Solver` in `aimo3/controller.py`.

Core loop:
1. `parse()` + `route()` + `select_run_config()` + `allocate_budget()`
2. seed actions (policy-dependent)
3. loop:
   - `enumerate_actions()`
   - choose action by `EVI` from AVM
   - execute action (`GEN/VERIFY/REFUTE/DISAMBIGUATE/REPARSE/HYP_*`)
   - online belief update
   - stop by posterior + cluster support + fragility
4. output policy (`AAS`) with strict gating

## 2. Native module mapping

- Parser/constraints: `aimo3/parser.py`
- Router: `aimo3/router.py`
- Run decorrelation engine (A/B): `aimo3/rde.py`
- Multi-policy reasoning programs: `aimo3/policies/__init__.py`
- Safe sandbox: `aimo3/sandbox.py`
- Tiered verifier: `aimo3/verifier.py`
- ARM (calibrated support/logBF): `aimo3/arm.py`
- CED (correlated-error + clustering): `aimo3/ced.py`
- BAA (posterior over answers): `aimo3/baa.py`
- Belief updater/evidence ops: `aimo3/belief.py`
- AVM + ESMP (EVI policy): `aimo3/avm.py`, `aimo3/esmp.py`
- ASPR + disambiguation: `aimo3/aspr.py`, `aimo3/disambiguate.py`
- Hypothesis search state machine: `aimo3/hypothesis.py`
- AAS + fallbacks: `aimo3/aas.py`, `aimo3/fallback.py`

## 3. Belief-state representation

`BeliefState` (see `aimo3/types.py`) maintains:
- `s[a]`: log-score
- `pi[a]`: posterior
- `u[a]`: uncertainty
- clusters/diagnostics/budget/action history

Belief updates are additive in log-space via `Evidence`.
Support evidence is correlation-weighted and rebuilt after CED updates.
Non-support evidence (refute/disambiguation/hypothesis) is preserved across support recomputation.

## 4. Two-run strategy

`select_run_config()` sets program A or B from `(run_seed, problem_hash)`:
- Program A: constructive/tool-formalization heavy
- Program B: adversarial/refutation/invariant heavy

This enforces programmatic decorrelation, not temperature-only variation.

## 5. Mandatory model requirement

Default runtime requires:
- `openai/gpt-oss-120b`

Enforced by `load_required_gpt_oss_120b()` in `aimo3/models.py`.
