from __future__ import annotations

import argparse
import json
from pathlib import Path

from aimo3.training.contamination import blocked_source, drop_near_duplicates
from aimo3.training.self_play import run_self_play
from aimo3.training.synthetic import generate_synthetic_dataset
from aimo3.training.verifier_data import build_verifier_pairs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_build_synthetic(args: argparse.Namespace) -> int:
    records = generate_synthetic_dataset(count=args.count, seed=args.seed)
    # Contamination guard.
    records = [r for r in records if not blocked_source(str(r.get("prompt", "")), source="synthetic")]
    records = drop_near_duplicates(records, text_key="prompt", threshold=args.jaccard_threshold)
    _write_jsonl(Path(args.output), records)
    print(f"Wrote {len(records)} synthetic SFT records -> {args.output}")
    return 0


def cmd_self_play(args: argparse.Namespace) -> int:
    records = _read_jsonl(Path(args.input))
    rollouts = run_self_play(records)
    _write_jsonl(Path(args.output), rollouts)
    print(f"Wrote {len(rollouts)} self-play rollouts -> {args.output}")
    return 0


def cmd_verifier_pairs(args: argparse.Namespace) -> int:
    rollouts = _read_jsonl(Path(args.input))
    pairs = build_verifier_pairs(rollouts)
    _write_jsonl(Path(args.output), pairs)
    print(f"Wrote {len(pairs)} verifier preference pairs -> {args.output}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    synthetic_path = workdir / "sft_synthetic.jsonl"
    selfplay_path = workdir / "self_play.jsonl"
    verifier_path = workdir / "verifier_pairs.jsonl"

    synth = argparse.Namespace(
        count=args.count,
        seed=args.seed,
        output=str(synthetic_path),
        jaccard_threshold=args.jaccard_threshold,
    )
    cmd_build_synthetic(synth)

    selfplay = argparse.Namespace(input=str(synthetic_path), output=str(selfplay_path))
    cmd_self_play(selfplay)

    verifier = argparse.Namespace(input=str(selfplay_path), output=str(verifier_path))
    cmd_verifier_pairs(verifier)
    print("Training data pipeline complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIMO3 training data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser("build-synthetic")
    p_syn.add_argument("--count", type=int, default=1000)
    p_syn.add_argument("--seed", type=int, default=0)
    p_syn.add_argument("--jaccard-threshold", type=float, default=0.85)
    p_syn.add_argument("--output", required=True)
    p_syn.set_defaults(func=cmd_build_synthetic)

    p_self = sub.add_parser("self-play")
    p_self.add_argument("--input", required=True)
    p_self.add_argument("--output", required=True)
    p_self.set_defaults(func=cmd_self_play)

    p_pair = sub.add_parser("verifier-pairs")
    p_pair.add_argument("--input", required=True)
    p_pair.add_argument("--output", required=True)
    p_pair.set_defaults(func=cmd_verifier_pairs)

    p_all = sub.add_parser("run-all")
    p_all.add_argument("--workdir", default="artifacts/training")
    p_all.add_argument("--count", type=int, default=1000)
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--jaccard-threshold", type=float, default=0.85)
    p_all.set_defaults(func=cmd_run_all)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
