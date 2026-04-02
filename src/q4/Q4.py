from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.legacy_runner import copy_matching, ensure_dir, project_root, run_python, temporary_workspace


ROOT = project_root(__file__)
LEGACY_SCRIPT = ROOT / "archive" / "legacy_scripts" / "q4_legacy.py"
DEFAULT_DATA = ROOT / "data" / "processed" / "question4" / "清洗后_完整观测.csv"
RESULTS_DIR = ensure_dir(ROOT / "results" / "q4")
OUTPUTS = [
    "adv_valid_阈值寻优.csv",
    "adv_test_综合评估.csv",
    "adv_BestChoice.txt",
]


def with_default_arg(args: list[str], flag: str, value: Path) -> list[str]:
    if any(arg == flag or arg.startswith(f"{flag}=") for arg in args):
        return args
    return [flag, str(value), *args]


def main() -> None:
    cli_args = with_default_arg(sys.argv[1:], "--csv", DEFAULT_DATA)
    with temporary_workspace("q4_legacy_") as tmp:
        script_path = tmp / "legacy_q4.py"
        shutil.copy2(LEGACY_SCRIPT, script_path)
        run_python(script_path, cli_args, cwd=tmp)
        copy_matching(tmp, RESULTS_DIR, OUTPUTS)


if __name__ == "__main__":
    main()
