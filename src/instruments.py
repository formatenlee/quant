from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from quant_cursor.config import Config

FREQ_COVERAGE_NAME = "qlib_freq_coverage.parquet"

PresetGroup = Literal[
    "all_a",
    "all_index_etf",
    "etf",
    "index",
    "industry_index",
    "industry_etf",
    "broad_index",
    "broad_etf",
    "large_cap",
    "major_index",
]

PRESET_FILTERS: dict[str, dict] = {
    "all_a": {"asset_types": ["stock"]},
    "all_index_etf": {"asset_types": ["index", "etf"]},
    "etf": {"asset_types": ["etf"]},
    "index": {"asset_types": ["index"]},
    "industry_index": {"asset_types": ["index"], "categories": ["industry"]},
    "industry_etf": {"asset_types": ["etf"], "categories": ["industry_etf"]},
    "broad_index": {"asset_types": ["index"], "categories": ["major", "broad"]},
    "broad_etf": {"asset_types": ["etf"], "categories": ["broad_etf"]},
    "large_cap": {"asset_types": ["stock"], "categories": ["large_cap"]},
    "major_index": {"asset_types": ["index"], "categories": ["major"]},
}


def load_manifest(config: Config | None = None, manifest_path: Path | None = None) -> pd.DataFrame:
    cfg = config or Config()
    path = manifest_path or (cfg.meta_dir / "qlib_manifest.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Qlib manifest 不存在: {path}，请先执行 pipeline 或 qlib export")
    return pd.read_parquet(path)


def load_universe_table(config: Config | None = None) -> pd.DataFrame:
    cfg = config or Config()
    path = cfg.universe_path
    if not path.exists():
        raise FileNotFoundError(f"标的池不存在: {path}")
    return pd.read_parquet(path)


def load_freq_coverage_table(config: Config | None = None) -> pd.DataFrame | None:
    cfg = config or Config()
    path = cfg.meta_dir / FREQ_COVERAGE_NAME
    if not path.exists():
        return None
    return pd.read_parquet(path)


def query_instruments(
    config: Config | None = None,
    *,
    manifest_path: Path | None = None,
    instruments: list[str] | None = None,
    asset_types: list[str] | None = None,
    categories: list[str] | None = None,
    sw_l2_codes: list[str] | None = None,
    sw_l2_names: list[str] | None = None,
    groups: list[str] | None = None,
    min_rows: int = 1,
    max_instruments: int | None = None,
    prefer_minute: bool = False,
    require_minute: bool = False,
    minute_min_rows: int = 1,
) -> list[str]:
    """
    从 qlib_manifest 筛选 Qlib instrument id。

    groups 预设: all_a, etf, index, industry_index, broad_index, large_cap 等。
    sw_l2_codes / sw_l2_names: 申万二级行业过滤（仅 stock）。
    prefer_minute: 有分钟数据的标的排在前面（配合 max_instruments 优先入选）。
    require_minute: 仅返回有分钟数据的标的。
    """
    if instruments:
        ids = list(dict.fromkeys(instruments))
        return ids[:max_instruments] if max_instruments else ids

    manifest = load_manifest(config, manifest_path)
    universe = None
    if sw_l2_codes or sw_l2_names or groups:
        try:
            universe = load_universe_table(config)
        except FileNotFoundError:
            pass

    merged = manifest.copy()
    if universe is not None and "code" in universe.columns:
        extra_cols = [c for c in ("sw_l2_code", "sw_l2_name", "category") if c in universe.columns]
        if extra_cols:
            merged = merged.merge(
                universe[["code", *extra_cols]].drop_duplicates("code"),
                on="code",
                how="left",
                suffixes=("", "_u"),
            )
            if "category_u" in merged.columns:
                merged["category"] = merged["category"].fillna(merged["category_u"])
                merged = merged.drop(columns=["category_u"])

    if groups:
        group_frames = []
        for g in groups:
            spec = PRESET_FILTERS.get(g)
            if spec is None:
                raise ValueError(f"未知预设组: {g}，可选: {list(PRESET_FILTERS)}")
            part = merged
            if "asset_types" in spec:
                part = part[part["asset_type"].isin(spec["asset_types"])]
            if "categories" in spec:
                part = part[part["category"].isin(spec["categories"])]
            group_frames.append(part)
        merged = pd.concat(group_frames).drop_duplicates("qlib_id")

    if asset_types:
        merged = merged[merged["asset_type"].isin(asset_types)]
    if categories:
        merged = merged[merged["category"].isin(categories)]
    if sw_l2_codes and "sw_l2_code" in merged.columns:
        codes = {c.zfill(6) for c in sw_l2_codes}
        merged = merged[merged["sw_l2_code"].astype(str).isin(codes)]
    if sw_l2_names and "sw_l2_name" in merged.columns:
        merged = merged[merged["sw_l2_name"].isin(sw_l2_names)]
    if "rows" in merged.columns and min_rows > 0:
        merged = merged[merged["rows"] >= min_rows]

    cov = load_freq_coverage_table(config)
    if cov is not None and "qlib_id" in cov.columns:
        cov_cols = [c for c in ("has_1min", "min_rows", "min_start", "min_end") if c in cov.columns]
        if cov_cols:
            merged = merged.merge(
                cov[["qlib_id", *cov_cols]].drop_duplicates("qlib_id"),
                on="qlib_id",
                how="left",
            )
            merged["has_1min"] = merged["has_1min"].fillna(False)
            if require_minute:
                merged = merged[merged["has_1min"]]
            if minute_min_rows > 0 and "min_rows" in merged.columns:
                min_rows_num = pd.to_numeric(merged["min_rows"], errors="coerce").fillna(0)
                has_min = merged["has_1min"] & min_rows_num.ge(minute_min_rows)
                if require_minute:
                    merged = merged[has_min]
                merged.loc[~has_min, "has_1min"] = False

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if prefer_minute and "has_1min" in merged.columns:
        sort_cols.append("has_1min")
        ascending.append(False)
    if "rows" in merged.columns:
        sort_cols.append("rows")
        ascending.append(False)
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=ascending, na_position="last")
    else:
        merged = merged.sort_values(["asset_type", "rows"], ascending=[True, False], na_position="last")
    ids = merged["qlib_id"].astype(str).tolist()
    if max_instruments:
        ids = ids[:max_instruments]
    return ids
