from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.legacy_runner import copy_matching, ensure_dir, project_root, run_python, temporary_workspace


ROOT = project_root(__file__)
LEGACY_SCRIPT = ROOT / "archive" / "legacy_scripts" / "q1_legacy.py"
DATA_DIR = ROOT / "data" / "processed" / "question1"
RESULTS_DIR = ensure_dir(ROOT / "results" / "q1")
INPUTS = ["清洗后_完整观测.csv", "清洗后_男胎达标.csv"]
OUTPUTS = [
    "Q1_MixedLM_摘要.txt",
    "Q1_OLS_含重度肥胖标记_摘要.txt",
    "Q1_OLS_摘要.txt",
    "Q1_VIF_线性全部观测.csv",
    "Q1_参数与P值.csv",
    "Q1_模型对比.csv",
    "Q1_运行摘要.txt",
]


def main() -> None:
    with temporary_workspace("q1_legacy_") as tmp:
        work_dir = tmp / "question1"
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "legacy_q1.py"
        shutil.copy2(LEGACY_SCRIPT, script_path)
        copy_matching(DATA_DIR, work_dir, INPUTS)
        run_python(script_path, cwd=work_dir)
        copy_matching(work_dir / "问题一_重跑结果", RESULTS_DIR, OUTPUTS)


if __name__ == "__main__":
    main()
