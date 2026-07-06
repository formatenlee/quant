"""BPC-v4 本地环境验证脚本"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    print(f"project root: {root}")

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    venv_py = root / ".venv" / "Scripts" / "python.exe"
    check(".venv python", venv_py.exists(), str(venv_py))

    qlib_dir = root / "data" / "qlib_data"
    check(
        "qlib_data",
        (qlib_dir / "calendars").is_dir() and (qlib_dir / "features").is_dir(),
        str(qlib_dir),
    )

    manifest = root / "data" / "meta" / "qlib_manifest.parquet"
    check("qlib_manifest", manifest.is_file(), str(manifest))

    kronos = root / "models" / "NeoQuasar" / "Kronos-Tokenizer-base" / "config.json"
    check("Kronos tokenizer", kronos.is_file(), str(kronos.parent))

    try:
        import torch

        check("torch", True, f"{torch.__version__} cuda={torch.cuda.is_available()}")
    except ImportError as e:
        check("torch", False, str(e))

    try:
        import qlib

        check("qlib", True, getattr(qlib, "__version__", "unknown"))
    except ImportError as e:
        check("qlib", False, str(e))

    try:
        import matplotlib

        check("matplotlib", True, matplotlib.__version__)
    except ImportError as e:
        check("matplotlib", False, str(e))

    try:
        import huggingface_hub
        import einops
        import safetensors

        check("bpc_v4 extras", True, "huggingface_hub, einops, safetensors")
    except ImportError as e:
        check("bpc_v4 extras", False, str(e))

    try:
        import quant_cursor
        from quant_cursor.config import load_config
        from quant_cursor.bpc_v4.kronos import resolve_kronos_local_path

        cfg = load_config()
        kpath = resolve_kronos_local_path(str(kronos.parent))
        check("quant_cursor import", True, str(Path(quant_cursor.__file__).parent))
        check("qlib_data_dir", cfg.qlib_data_dir.is_dir(), str(cfg.qlib_data_dir))
        check("kronos resolve", kpath is not None, kpath or "not found")
    except Exception as e:
        check("quant_cursor import", False, str(e))

    failed = [c for c in checks if not c[1]]
    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(checks)} checks")
        return 1
    print(f"ALL PASSED: {len(checks)}/{len(checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
