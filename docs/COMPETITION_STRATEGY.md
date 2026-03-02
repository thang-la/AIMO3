# Competition Strategy (Operational)

This file defines what to optimize to maximize score under AIMO3 constraints.

## 1. Optimize expected per-run correctness

Private score is average over two reruns, so the direct objective is per-run correctness.
Prioritize:
- calibration quality of ARM
- reduction of correlated wrong clusters via CED + BAA
- robust stop policy thresholds

## 2. Required ablation sequence

1. ARM calibration ablation (Brier/ECE)
2. CED redundancy discount ablation
3. BAA vs naive vote ablation
4. ESMP thresholds (`stop_pi`, `fragility`, `min_clusters`)
5. ASPR impact on hard-tail subset
6. Run A/B decorrelation impact on paired rerun simulation

## 3. Runtime tuning targets

- zero format failures
- zero startup failures
- bounded timeout rate
- controlled hard-mode budget by marginal gain

## 4. Non-negotiable checks before submit

- deterministic one-row output schema
- lazy model init works under Kaggle server
- no network call path during inference
- robust handling of parser misses + fallback path

