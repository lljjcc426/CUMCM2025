from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def project_root(anchor: str) -> Path:
    return Path(anchor).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_matching(src_dir: Path, dst_dir: Path, filenames: list[str]) -> None:
    ensure_dir(dst_dir)
    for name in filenames:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected file: {src}")
        shutil.copy2(src, dst_dir / name)


def run_python(script_path: Path, args: list[str] | None = None, cwd: Path | None = None) -> None:
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


@contextmanager
def temporary_workspace(prefix: str):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield Path(tmp)
