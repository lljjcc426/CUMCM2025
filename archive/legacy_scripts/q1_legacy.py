import os
import warnings
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
IN_FULL = os.path.join(BASE, "清洗后_完整观测.csv")
IN_MALE = os.path.join(BASE, "清洗后_男胎达标.csv")   # 可选
OUTDIR  = os.path.join(BASE, "问题一_重跑结果")
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(IN_FULL):
    raise FileNotFoundError(f"未找到输入文件：{IN_FULL}。请先运行预处理脚本生成该文件。")

df_full = pd.read_csv(IN_FULL, encoding="utf-8-sig")

need_cols = {"孕妇代码", "孕周数", "孕妇BMI", "Y染色体浓度"}
miss = need_cols - set(df_full.columns)
if miss:
    raise KeyError(f"清洗后_完整观测.csv 缺少列：{miss}")

has_obese = "重度肥胖标记" in df_full.columns

male_ids = None
if os.path.exists(IN_MALE):
    try:
        df_male = pd.read_csv(IN_MALE, encoding="utf-8-sig")
        if "孕妇代码" in df_male.columns:
            male_ids = set(df_male["孕妇代码"].dropna().astype(str).unique().tolist())
    except Exception:
        male_ids = None

if male_ids is None or len(male_ids) == 0:
    tmp = (df_full
           .dropna(subset=["孕妇代码", "Y染色体浓度"])
           .groupby("孕妇代码")["Y染色体浓度"].max()
           .reset_index())
    male_ids = set(tmp.loc[tmp["Y染色体浓度"] >= 0.01, "孕妇代码"].astype(str).tolist())

df_full["孕妇代码"] = df_full["孕妇代码"].astype(str)
df = df_full[df_full["孕妇代码"].isin(male_ids)].copy()

df = df.dropna(subset=["孕周数", "孕妇BMI", "Y染色体浓度"])
df = df[(df["孕周数"] >= 6) & (df["孕周数"] <= 40) & (df["孕妇BMI"] > 0)].copy()
# 两种样本策略
all_obs  = df.copy()
last_obs = (df.sort_values(["孕妇代码", "孕周数"])
              .groupby("孕妇代码", as_index=False).tail(1)
              .reset_index(drop=True))
# ===== OLS（全部观测·线性）=====
formula_lin = "Y染色体浓度 ~ 孕周数 + 孕妇BMI"
ols = smf.ols(formula_lin, all_obs).fit()
ols_rob = ols.get_robustcov_results(cov_type="HC3")
# ===== 混合效应模型 =====
mixed_summary_txt = None
mixed_fit = None
try:
    mixed = smf.mixedlm("Y染色体浓度 ~ 孕周数 + 孕妇BMI", all_obs, groups=all_obs["孕妇代码"])
    try:
        mixed_fit = mixed.fit(reml=True, method="lbfgs", maxiter=500)
    except Exception:
        mixed_fit = mixed.fit(reml=False, method="lbfgs", maxiter=500)
    mixed_summary_txt = str(mixed_fit.summary())
except Exception as e:
    mixed_summary_txt = f"[混合效应拟合失败] {repr(e)}"

if has_obese:
    try:
        sens_model = smf.ols("Y染色体浓度 ~ 孕周数 + 孕妇BMI + 重度肥胖标记", all_obs).fit()
        sens_rob = sens_model.get_robustcov_results(cov_type="HC3")
        with open(os.path.join(OUTDIR, "Q1_OLS_含重度肥胖标记_摘要.txt"), "w", encoding="utf-8") as f:
            f.write("【含重度肥胖标记的OLS（HC3稳健协方差）】\n" + str(sens_rob.summary()))
    except Exception as e:
        pass

with open(os.path.join(OUTDIR, "Q1_OLS_摘要.txt"), "w", encoding="utf-8") as f:
    f.write(str(ols.summary()))
if mixed_summary_txt:
    with open(os.path.join(OUTDIR, "Q1_MixedLM_摘要.txt"), "w", encoding="utf-8") as f:
        f.write(mixed_summary_txt)

rows_cmp = [
    {"模型": "OLS_全部观测_线性", "样本量": int(ols.nobs),
     "R2": float(ols.rsquared), "AIC": float(ols.aic), "BIC": float(ols.bic)}
]
if mixed_fit is not None:
    rows_cmp.append({
        "模型": "MixedLM_随机截距(孕妇)", "样本量": int(mixed_fit.nobs),
        "R2": np.nan,
        "AIC": float(getattr(mixed_fit, "aic", np.nan)),
        "BIC": float(getattr(mixed_fit, "bic", np.nan))
    })
pd.DataFrame(rows_cmp).to_csv(os.path.join(OUTDIR, "Q1_模型对比.csv"), index=False, encoding="utf-8-sig")

params_idx = list(ols.params.index)
p_tbl = pd.DataFrame({
    "参数": params_idx,
    "估计值": np.asarray(ols.params.values),
    "p值(常规)": np.asarray(ols.pvalues),
    "p值(HC3稳健)": np.asarray(ols_rob.pvalues)
})
p_tbl.to_csv(os.path.join(OUTDIR, "Q1_参数与P值.csv"), index=False, encoding="utf-8-sig")

def calc_vif(dfX, cols):
    X = add_constant(dfX[cols], has_constant="add")
    out = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        try:
            out.append({"变量": c, "VIF": variance_inflation_factor(X.values, i)})
        except Exception:
            out.append({"变量": c, "VIF": np.nan})
    return pd.DataFrame(out)

calc_vif(all_obs, ["孕周数", "孕妇BMI"]).to_csv(
    os.path.join(OUTDIR, "Q1_VIF_线性全部观测.csv"),
    index=False, encoding="utf-8-sig"
)

with open(os.path.join(OUTDIR, "Q1_运行摘要.txt"), "w", encoding="utf-8") as f:
    f.write("【问题一（重跑）关键信息】\n")
    f.write(f"- 清洗后_完整观测.csv 条数：{len(df_full)}\n")
    f.write(f"- 推断男胎孕妇数：{len(male_ids)}\n")
    f.write(f"- 建模用观测条数（全部观测）：{len(all_obs)}；不同孕妇：{all_obs['孕妇代码'].nunique()}\n")
    f.write(f"- 建模用观测条数（每人一次）：{len(last_obs)}；不同孕妇：{last_obs['孕妇代码'].nunique()}\n")
    f.write("\n[OLS_全部观测_线性]\n")
    f.write(str(ols.summary()))
    f.write("\n\n[混合效应模型]\n")
    f.write(mixed_summary_txt or "[无混合效应结果]")

print("=== Q1 重跑完成 ===")
print(f"全部观测：{len(all_obs)}；每人一次：{len(last_obs)}；输出目录：{OUTDIR}")
