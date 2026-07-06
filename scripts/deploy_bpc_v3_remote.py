#!/usr/bin/env python3
"""
将 BPC-v3 及训练依赖同步到远程 SSH 主机。

用法（在项目根 e:\\quant_cursor 下）:
  python scripts/deploy_bpc_v3_remote.py
  python scripts/deploy_bpc_v3_remote.py --dry-run
  python scripts/deploy_bpc_v3_remote.py --run-preflight

依赖: pip install paramiko
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(__file__).resolve().parent / "remote_bpc_v3.env"

# v3 训练所需的最小同步集（保持 v2 bpc 基座 + v3 模块）
SYNC_REL_PATHS: tuple[str, ...] = (
    "quant_cursor/__init__.py",
    "quant_cursor/__main__.py",
    "quant_cursor/config.py",
    "quant_cursor/bpc",
    "quant_cursor/bpc_v3",
)


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}；请复制 remote_bpc_v3.env.example 为 remote_bpc_v3.env 并填写凭据"
        )
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _iter_local_files() -> list[Path]:
    files: list[Path] = []
    for rel in SYNC_REL_PATHS:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"本地缺失: {src}")
        if src.is_file():
            files.append(src)
            continue
        for p in src.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and not p.suffix.endswith(".pyc"):
                files.append(p)
    return files


def _remote_path(local: Path, remote_pkg_root: str) -> str:
    rel = local.relative_to(ROOT).as_posix()
    return f"{remote_pkg_root}/{rel}"


def _sftp_mkdirs(sftp, remote_dir: str) -> None:
    parts: list[str] = []
    for part in remote_dir.strip("/").split("/"):
        parts.append(part)
        cur = "/" + "/".join(parts)
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def deploy(*, dry_run: bool = False) -> None:
    try:
        import paramiko
    except ImportError:
        print("请先安装: pip install paramiko", file=sys.stderr)
        raise SystemExit(1)

    cfg = _load_env(ENV_FILE)
    host = cfg["REMOTE_HOST"]
    port = int(cfg.get("REMOTE_PORT", "22"))
    user = cfg["REMOTE_USER"]
    password = cfg["REMOTE_PASSWORD"]
    project_root = cfg["REMOTE_PROJECT_ROOT"].rstrip("/")
    # PYTHONPATH={project_root} → 包根为 {project_root}/quant_cursor/...
    remote_pkg_root = project_root

    local_files = _iter_local_files()
    print(f"待同步 {len(local_files)} 个文件 → {user}@{host}:{port}:{remote_pkg_root}")

    if dry_run:
        for f in local_files[:10]:
            print(" ", _remote_path(f, remote_pkg_root))
        if len(local_files) > 10:
            print(f"  ... 共 {len(local_files)} 个")
        return

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    sftp = client.open_sftp()

    try:
        _sftp_mkdirs(sftp, remote_pkg_root)
        for local in local_files:
            remote = _remote_path(local, remote_pkg_root)
            _sftp_mkdirs(sftp, os.path.dirname(remote))
            sftp.put(str(local), remote)
            # 脚本可执行
            if local.suffix == ".sh":
                st = sftp.stat(remote)
                sftp.chmod(remote, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("同步完成.")
    finally:
        sftp.close()
        client.close()


def run_remote_preflight(cfg: dict[str, str]) -> None:
    import paramiko

    host = cfg["REMOTE_HOST"]
    port = int(cfg.get("REMOTE_PORT", "22"))
    user = cfg["REMOTE_USER"]
    password = cfg["REMOTE_PASSWORD"]
    project_root = cfg["REMOTE_PROJECT_ROOT"].rstrip("/")
    venv = cfg.get("REMOTE_VENV_ACTIVATE", "source ~/pdl/venv/bin/activate")

    cmd = (
        f"cd {project_root} && "
        f"{venv} && "
        f"export PYTHONPATH={project_root}:$PYTHONPATH && "
        f"python -m quant_cursor.bpc_v3.preflight --qlib --sample-size 5000 --max-instruments 48"
    )
    print("远程执行:", cmd)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    try:
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=600)
        del stdin
        for line in stdout:
            print(line, end="" if line.endswith("\n") else "\n")
        err = stderr.read().decode()
        if err.strip():
            print(err, file=sys.stderr)
        code = stdout.channel.recv_exit_status()
        if code != 0:
            raise SystemExit(code)
    finally:
        client.close()


def main() -> int:
    p = argparse.ArgumentParser(description="BPC-v3 远程部署")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-preflight", action="store_true", help="部署后远程跑 preflight smoke")
    args = p.parse_args()
    deploy(dry_run=args.dry_run)
    if args.run_preflight and not args.dry_run:
        run_remote_preflight(_load_env(ENV_FILE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
