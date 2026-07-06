"""Analyze BPC training metrics.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEY_EPOCHS = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else r"f:\downloads\metrics.csv")
    df = pd.read_csv(path)
    ep = df[df["phase"] == "epoch"].copy()
    ep = ep[ep["val_loss"].notna()]

    print(f"File: {path}")
    print(f"Epoch rows with val: {len(ep)} | epoch {ep['epoch'].min():.0f}-{ep['epoch'].max():.0f}")
    print(f"Train samples: {int(ep['train_samples'].iloc[-1])} | Val: {int(ep['val_samples'].iloc[-1])}")

    print("\n=== Total loss ===")
    print(f"{'epoch':>6} {'train':>8} {'val':>8} {'val/train':>10}")
    for e in KEY_EPOCHS:
        rows = ep[ep["epoch"] == e]
        if rows.empty:
            continue
        r = rows.iloc[0]
        ratio = r["val_loss"] / r["train_loss"]
        print(f"{int(r['epoch']):6d} {r['train_loss']:8.4f} {r['val_loss']:8.4f} {ratio:10.3f}")

    best = ep.loc[ep["val_loss"].idxmin()]
    print(
        f"\nBest val_loss: epoch {int(best['epoch'])} "
        f"train={best['train_loss']:.4f} val={best['val_loss']:.4f}"
    )

    print("\n=== Purity breakdown (train / val) ===")
    agents = [
        ("regime", "purity_regime"),
        ("attack", "purity_attack"),
        ("path", "purity_path"),
        ("vol_struct", "purity_vol_struct"),
        ("momentum", "purity_momentum"),
    ]
    for e in [99, 199, 299, 399, 499]:
        rows = ep[ep["epoch"] == e]
        if rows.empty:
            continue
        r = rows.iloc[0]
        print(f"\n--- epoch {e} ---")
        print(
            f"  purity_total: {r['train_loss_purity_total']:.4f} / {r['val_loss_purity_total']:.4f} "
            f"(ratio {r['val_loss_purity_total']/r['train_loss_purity_total']:.2f}x)"
        )
        for label, key in agents:
            tr = r[f"train_loss_{key}"]
            va = r[f"val_loss_{key}"]
            print(f"  {label:12s}: {tr:.6f} / {va:.6f}  (val/train {va/tr:.2f}x)")

    print("\n=== Val/Train purity ratio (every 100 ep from 99) ===")
    sub = ep[ep["epoch"] >= 99].copy()
    sub["purity_ratio"] = sub["val_loss_purity_total"] / sub["train_loss_purity_total"]
    sub["path_ratio"] = sub["val_loss_purity_path"] / sub["train_loss_purity_path"]
    sub["vol_ratio"] = sub["val_loss_purity_vol_struct"] / sub["train_loss_purity_vol_struct"]
    for e in range(99, int(sub["epoch"].max()) + 1, 100):
        rows = sub[sub["epoch"] == e]
        if rows.empty:
            continue
        r = rows.iloc[0]
        print(
            f"ep {e:4d}  total={r['purity_ratio']:.2f}x  "
            f"path={r['path_ratio']:.2f}x  vol={r['vol_ratio']:.2f}x"
        )

    print("\n=== VQ & FiLM (val) ===")
    for e in [99, 199, 299, 399, 499]:
        rows = ep[ep["epoch"] == e]
        if rows.empty:
            continue
        r = rows.iloc[0]
        print(
            f"ep {e:4d}  usage={r['val_vq_usage_rate']*100:.1f}%  "
            f"perplexity={r['val_vq_perplexity']:.1f}  "
            f"residual={r['val_vq_residual_mean']:.3f}  "
            f"recon_cos={r['val_recon_cosine']:.4f}  "
            f"film_cb_gamma={r.get('val_film_codebook_gamma_abs', float('nan')):.5f}  "
            f"film_cb_gate={r.get('val_film_codebook_gate', float('nan')):.3f}"
        )

    # Detect largest purity jump between consecutive logged val epochs
    ep_sorted = ep.sort_values("epoch")
    ep_sorted["purity_delta"] = ep_sorted["val_loss_purity_total"].diff()
    jump = ep_sorted.loc[ep_sorted["purity_delta"].idxmax()]
    print(
        f"\nLargest val purity_total jump: +{jump['purity_delta']:.4f} "
        f"(epoch {int(jump['epoch'])} vs prior)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
