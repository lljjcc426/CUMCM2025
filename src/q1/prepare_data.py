from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.legacy_runner import copy_matching, ensure_dir, project_root, run_python, temporary_workspace


ROOT = project_root(__file__)
LEGACY_SCRIPT = ROOT / "archive" / "legacy_scripts" / "q1_prepare_legacy.py"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ensure_dir(ROOT / "data" / "processed" / "question1")
RESULTS_DIR = ensure_dir(ROOT / "results" / "q1")
INPUTS = ["完整观测.csv", "男胎达标.csv"]
PROCESSED_OUTPUTS = [
    "清洗后_完整观测.csv",
    "清洗后_男胎达标.csv",
    "完整观测_补充_含本次是否达标列.csv",
    "男胎达标_补充_含删失信息.csv",
    "异常值清单.csv",
    "高BMI_重度肥胖清单.csv",
]


def main() -> None:
    with temporary_workspace("q1_prepare_legacy_") as tmp:
        base_dir = tmp
        script_dir = base_dir / "question1"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "legacy_prepare.py"
        shutil.copy2(LEGACY_SCRIPT, script_path)
        copy_matching(INTERIM_DIR, base_dir, INPUTS)
        run_python(script_path, cwd=script_dir)
        copy_matching(script_dir, PROCESSED_DIR, PROCESSED_OUTPUTS)
        copy_matching(script_dir, RESULTS_DIR, ["预处理重置_日志.txt"])


if __name__ == "__main__":
    main()
