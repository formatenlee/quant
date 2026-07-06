"""SFTP 同步本地 quant_cursor 代码到测试服务器（Windows 无 rsync 时使用）。"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

import paramiko

DEFAULT_HOST = "183.232.132.248"
DEFAULT_PORT = 31407
DEFAULT_USER = "user"
DEFAULT_REMOTE_ROOT = "/home/user/pdl/mylab/quant/quant_cursor"
DEFAULT_LOCAL_SRC = Path(__file__).resolve().parent.parent / "src"

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def _should_skip(path: Path) -> bool:
    if path.name in SKIP_DIRS:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote: str) -> None:
    parts = remote.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = f"{cur}/{p}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def _upload_tree(sftp: paramiko.SFTPClient, local: Path, remote: str) -> tuple[int, int]:
    files = dirs = 0
    _ensure_remote_dir(sftp, remote)
    for root, dirnames, filenames in os.walk(local):
        root_path = Path(root)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = root_path.relative_to(local)
        remote_root = remote if str(rel) == "." else f"{remote}/{rel.as_posix()}"
        _ensure_remote_dir(sftp, remote_root)
        for name in filenames:
            lp = root_path / name
            if _should_skip(lp):
                continue
            rp = f"{remote_root}/{name}"
            sftp.put(str(lp), rp)
            files += 1
        dirs += 1
    return files, dirs


def main() -> int:
    p = argparse.ArgumentParser(description="同步 quant_cursor 代码到测试服务器")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--password", default=os.environ.get("QUANT_SSH_PASSWORD", ""))
    p.add_argument("--local", type=Path, default=DEFAULT_LOCAL_SRC)
    p.add_argument("--remote", default=DEFAULT_REMOTE_ROOT)
    p.add_argument("--only", default="bpc_v4", help="仅同步子目录，空=整个 src")
    args = p.parse_args()

    if not args.password:
        raise SystemExit("请设置环境变量 QUANT_SSH_PASSWORD 或 --password")

    local = args.local
    if args.only:
        local = local / args.only
    if not local.is_dir():
        raise SystemExit(f"本地目录不存在: {local}")

    remote = args.remote if not args.only else f"{args.remote.rstrip('/')}/{args.only}"

    print(f"同步: {local} -> {args.user}@{args.host}:{args.port}:{remote}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    sftp = client.open_sftp()
    try:
        n_files, n_dirs = _upload_tree(sftp, local, remote)
        print(f"完成: {n_files} 文件, {n_dirs} 目录")
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
