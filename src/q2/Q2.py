from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.legacy_runner import ensure_dir, project_root, run_python


ROOT = project_root(__file__)
LEGACY_SCRIPT = ROOT / "archive" / "legacy_scripts" / "q2_legacy.py"
DEFAULT_DATA = ROOT / "data" / "processed" / "question1" / "清洗后_完整观测.csv"
DEFAULT_OUTDIR = ROOT / "results" / "q2"


def with_default_arg(args: list[str], flag: str, value: Path) -> list[str]:
    if any(arg == flag or arg.startswith(f"{flag}=") for arg in args):
        return args
    return [flag, str(value), *args]


def main() -> None:
    cli_args = sys.argv[1:]
    cli_args = with_default_arg(cli_args, "--outdir", ensure_dir(DEFAULT_OUTDIR))
    cli_args = with_default_arg(cli_args, "--data", DEFAULT_DATA)
    run_python(LEGACY_SCRIPT, cli_args, cwd=ROOT)


if __name__ == "__main__":
    main()
