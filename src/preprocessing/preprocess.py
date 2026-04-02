from __future__ import annotations

from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.legacy_runner import copy_matching, ensure_dir, project_root, run_python, temporary_workspace


ROOT = project_root(__file__)
LEGACY_SCRIPT = ROOT / "archive" / "legacy_scripts" / "preprocess_legacy.py"
RAW_FILE = ROOT / "data" / "raw" / "附件.xlsx"
INTERIM_DIR = ensure_dir(ROOT / "data" / "interim")
OUTPUTS = ["完整观测.csv", "男胎达标.csv", "女胎分类.csv"]


def main() -> None:
    with temporary_workspace("preprocess_legacy_") as tmp:
        script_path = tmp / "legacy_preprocess.py"
        shutil.copy2(LEGACY_SCRIPT, script_path)
        shutil.copy2(RAW_FILE, tmp / "附件.xlsx")
        run_python(script_path, cwd=tmp)
        copy_matching(tmp, INTERIM_DIR, OUTPUTS)


if __name__ == "__main__":
    main()
