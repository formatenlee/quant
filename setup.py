"""本地开发安装：src/ 映射为 quant_cursor 包（与服务器 PYTHONPATH=quant 一致）"""

from pathlib import Path

from setuptools import setup

SRC = Path(__file__).parent / "src"

packages = ["quant_cursor"]
for init in SRC.rglob("__init__.py"):
    rel = init.parent.relative_to(SRC)
    if rel.parts:
        packages.append("quant_cursor." + ".".join(rel.parts))

setup(
    name="quant_cursor",
    version="0.1.0",
    package_dir={"quant_cursor": "src"},
    packages=packages,
    python_requires=">=3.10",
)
