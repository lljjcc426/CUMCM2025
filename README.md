# CUMCM 2025 National Finals Project

2025 全国大学生数学建模竞赛国赛 C 题项目仓库。

这个仓库已经按 GitHub 项目的常见方式整理为 `src / data / results / docs / figures / archive` 结构，便于协作、复现和后续维护。

## Repository Layout

```text
.
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ archive/
│  ├─ legacy_scripts/      # 原始脚本存档
│  ├─ packages/            # 压缩包、安装包、历史打包文件
│  ├─ ide/                 # IDE 配置存档
│  └─ temp/                # 临时文件存档
├─ data/
│  ├─ raw/                 # 原始数据
│  ├─ interim/             # 初步预处理输出
│  └─ processed/
│     ├─ question1/        # 问题一相关清洗数据
│     └─ question4/        # 问题四建模输入
├─ docs/
│  ├─ problem-statement/   # 赛题与格式要求
│  └─ drafts/              # 论文草稿
├─ figures/
│  └─ paper/               # 论文配图
├─ results/
│  ├─ q1/
│  ├─ q2/
│  ├─ q3/
│  └─ q4/
└─ src/
   ├─ common/              # 公共辅助代码
   ├─ preprocessing/       # 总预处理入口
   ├─ q1/
   ├─ q2/
   ├─ q3/
   └─ q4/
```

## What Lives Where

### `data/`

- `data/raw/附件.xlsx`：原始 Excel 数据。
- `data/interim/`：总预处理阶段生成的中间数据。
- `data/processed/question1/`：问题一和后续生存分析依赖的清洗结果。
- `data/processed/question4/`：问题四分类模型输入数据。

### `src/`

- `src/preprocessing/preprocess.py`：运行总预处理，生成 `data/interim/` 数据。
- `src/q1/prepare_data.py`：运行问题一补充预处理，生成 `data/processed/question1/`。
- `src/q1/Q1.py`：运行问题一模型并输出到 `results/q1/`。
- `src/q2/Q2.py`：运行问题二分析并输出到 `results/q2/`。
- `src/q3/Q3.py`：运行问题三分析并输出到 `results/q3/`。
- `src/q4/Q4.py`：运行问题四分类模型并输出到 `results/q4/`。

说明：

- `src/` 下的脚本是新的 GitHub 风格入口。
- 原始脚本已保存在 `archive/legacy_scripts/` 中，作为历史实现保留。
- 新入口会把当前仓库结构映射到旧脚本预期的输入输出方式，因此目录更整洁，同时尽量不破坏原有逻辑。

### `results/`

- `results/q1/`：OLS、MixedLM、VIF、参数表、运行摘要等。
- `results/q2/`：BMI 两组主结果汇总表和报告。
- `results/q3/`：K=2/K=3 分组结果与报告。
- `results/q4/`：阈值寻优、综合评估、推荐部署方案。

### `docs/`

- `docs/problem-statement/`：题目 PDF 和格式要求。
- `docs/drafts/`：当前论文草稿版本。

### `figures/`

- `figures/paper/`：流程图、CDF 图、成本曲线、混淆矩阵等论文图片。

## Quick Start

建议在仓库根目录执行：

```powershell
python src/preprocessing/preprocess.py
python src/q1/prepare_data.py
python src/q1/Q1.py
python src/q2/Q2.py
python src/q3/Q3.py --cuts 30,35
python src/q4/Q4.py --csv data/processed/question4/清洗后_完整观测.csv
```

## Existing Findings Snapshot

基于当前已有结果文件，可以快速把握项目结论：

- 问题一：Y 染色体浓度与孕周正相关，与孕妇 BMI 负相关。
- 问题二：`BMI < 35` 相比 `BMI >= 35`，高 BMI 组 p90 更晚、RMST 更大、期望成本更高。
- 问题三：K=3 分组下，高 BMI 组推荐检测时点更晚、成本更高。
- 问题四：现有最优部署方案文件显示逻辑回归方案在测试集上达到较好的召回和 AUC 表现。

## Suggested Workflow

1. 先读 `docs/problem-statement/` 中的赛题和格式要求。
2. 查看 `docs/drafts/` 中论文当前版本。
3. 从 `src/preprocessing/` 和 `src/q1/` 开始理解数据处理链路。
4. 再查看 `src/q2/`、`src/q3/`、`src/q4/` 对应方法与 `results/` 输出。
5. 最后对照 `figures/paper/` 完成论文修订或仓库说明补充。

## Dependencies

主要依赖如下：

- numpy
- pandas
- statsmodels
- scikit-learn
- lifelines
- xgboost
- openpyxl

可使用：

```powershell
pip install -r requirements.txt
```

