"""Check minute download / qlib coverage by category."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_cursor.config import load_config


def _has_minute(cfg, asset_type: str, code: str) -> bool:
    code = str(code).zfill(6)
    base = cfg.indices_min_dir if asset_type == "index" else cfg.etf_min_dir
    p = base / f"{code}.parquet"
    return p.exists() and p.stat().st_size > 100


def main() -> int:
    cfg = load_config()
    meta = cfg.meta_dir
    manifest = pd.read_parquet(meta / "qlib_manifest.parquet")
    uni = pd.read_parquet(cfg.universe_path)
    m = manifest.merge(
        uni[["code", "category", "asset_type", "name"]].drop_duplicates("code"),
        on="code",
        how="left",
        suffixes=("", "_u"),
    )
    for col in ("asset_type", "category", "name"):
        ucol = f"{col}_u"
        if ucol in m.columns:
            m[col] = m[col].fillna(m[ucol])
            m = m.drop(columns=[ucol])

    def report(label: str, sub: pd.DataFrame) -> None:
        total = len(sub)
        ok = sum(1 for _, r in sub.iterrows() if _has_minute(cfg, r["asset_type"], r["code"]))
        pct = 100.0 * ok / total if total else 0.0
        print(f"{label:22s} {ok:4d}/{total:4d}  ({pct:5.1f}%)")

    print("=== Minute parquet (data/etf_min, data/indices_min) ===")
    report("broad_etf", m[(m.asset_type == "etf") & (m.category == "broad_etf")])
    report("all_etf", m[m.asset_type == "etf"])
    report("industry_index", m[(m.asset_type == "index") & (m.category == "industry")])
    report("broad+major_index", m[(m.asset_type == "index") & (m.category.isin(["broad", "major"]))])
    report("all_index", m[m.asset_type == "index"])
    report("ALL qlib", m)

    etf_n = len(list(cfg.etf_min_dir.glob("*.parquet")))
    idx_n = len(list(cfg.indices_min_dir.glob("*.parquet")))
    print(f"\nParquet files on disk: etf_min={etf_n}, indices_min={idx_n}")

    reports = sorted(meta.glob("minute_download_report_*.parquet"))
    if reports:
        r = pd.read_parquet(reports[-1])
        print(f"\nLatest report: {reports[-1].name}")
        print(r["status"].value_counts().to_string())

    mp = meta / "qlib_manifest_1min.parquet"
    print(f"\nqlib_manifest_1min: {len(pd.read_parquet(mp)) if mp.exists() else 'NOT YET'}")

    feat = cfg.qlib_data_dir / "features"
    if feat.exists():
        n1 = sum(1 for d in feat.iterdir() if d.is_dir() and (d / "close.1min.bin").exists())
        print(f"qlib close.1min.bin count: {n1}")

    logs = sorted(meta.glob("pipeline_bg_*.log"))
    if logs:
        lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\nLatest pipeline log: {logs[-1].name} ({len(lines)} lines)")
        for ln in lines[-5:]:
            print(" ", ln.encode("ascii", errors="replace").decode("ascii"))

    pid = meta / "pipeline.pid"
    if pid.exists():
        print(f"\nPID file: {pid.read_text(encoding='utf-8').strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
