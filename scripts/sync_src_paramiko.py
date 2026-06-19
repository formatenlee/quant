#!/usr/bin/env python3
"""Sync local src/ to remote quant_cursor/ via SFTP (password auth)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = Path(os.environ.get("QUANT_SYNC_LOCAL", ROOT / "src"))
HOST = os.environ.get("QUANT_SYNC_HOST", "183.232.132.248")
PORT = int(os.environ.get("QUANT_SYNC_PORT", "31407"))
USER = os.environ.get("QUANT_SYNC_USER", "user")
REMOTE = os.environ.get("QUANT_SYNC_REMOTE", "/home/user/pdl/mylab/quant/quant_cursor")

PASSWORD_KEYS = (
    "SYNC_SSH_PASSWORD",
    "QUANT_SYNC_PASSWORD",
    "SSH_PASSWORD",
    "QUANT_SSH_PASSWORD",
    "SERVER_SSH_PASSWORD",
    "SSH_PASS",
    "PASSWORD",
)


def resolve_password() -> str:
    for key in PASSWORD_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    keys = ", ".join(PASSWORD_KEYS)
    raise SystemExit(
        f"未找到 SSH 密码。请在 Cloud Agent Secrets 中添加名为 SYNC_SSH_PASSWORD 的密钥，"
        f"然后重启 agent。已检查: {keys}"
    )


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    parts = remote_path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += f"/{part}"
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def upload_tree(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    count = 0
    for path in sorted(local.rglob("*")):
        rel = path.relative_to(local).as_posix()
        if not rel:
            continue
        if any(
            part in {"__pycache__", ".ipynb_checkpoints", "quant_cursor"}
            for part in path.parts
        ):
            continue
        if path.name.endswith(".pyc") or path.name.endswith("-Copy1.py"):
            continue

        remote_path = f"{remote.rstrip('/')}/{rel}"
        if path.is_dir():
            ensure_remote_dir(sftp, remote_path)
            continue

        ensure_remote_dir(sftp, str(Path(remote_path).parent))
        sftp.put(str(path), remote_path)
        count += 1
    return count


def main() -> int:
    password = resolve_password()
    print(f"[sync] {LOCAL_DIR}/ -> {USER}@{HOST}:{PORT}:{REMOTE}/")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=HOST,
        port=PORT,
        username=USER,
        password=password,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    try:
        sftp = client.open_sftp()
        try:
            ensure_remote_dir(sftp, REMOTE)
            n = upload_tree(sftp, LOCAL_DIR, REMOTE)
        finally:
            sftp.close()

        _, stdout, _ = client.exec_command(
            f"test -f {REMOTE}/bpc_v4/kronos_model/__init__.py && echo OK"
        )
        marker = stdout.read().decode().strip()
        print(f"[sync] uploaded {n} files; remote marker={marker or 'missing'}")
    finally:
        client.close()

    print("[sync] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
