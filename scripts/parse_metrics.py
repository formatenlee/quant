"""Parse BPC metrics.jsonl for train/val gap and FiLM gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/checkpoints/bpc")
    runs = sorted(root.glob("run_*/metrics.jsonl"))
    if not runs:
        print(f"No metrics.jsonl under {root}")
        return 1
    for p in runs[-5:]:
        lines = [l for l in p.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        epochs = [json.loads(l) for l in lines if json.loads(l).get("phase") == "epoch"]
        print(f"=== {p.parent.name} epochs logged: {len(epochs)}")
        if not epochs:
            continue
        picks = {epochs[0]["epoch"], epochs[-1]["epoch"], 99, 199, 299, 399}
        for ep in epochs:
            if ep["epoch"] not in picks:
                continue
            tr = ep.get("train_loss")
            va = ep.get("val_loss")
            gap = (va - tr) / tr * 100 if tr and va else None
            gap_s = f" gap={gap:+.1f}%" if gap is not None else ""
            print(
                f" ep {ep['epoch']:4d} train={tr} val={va}"
                f" vol={ep.get('train_purity_vol')}/{ep.get('val_purity_vol')}"
                f" gate={ep.get('train_film_codebook_gate')}{gap_s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
