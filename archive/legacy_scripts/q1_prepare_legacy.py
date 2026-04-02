import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

Y_THRESHOLD = 0.04              # 达标阈值
GEST_WEEK_MIN, GEST_WEEK_MAX = 10.0, 30.0
BMI_MIN_PLAUSIBLE, BMI_MAX_PLAUSIBLE = 15.0, 50.0

# 异常值处理策略
BMI_OUTLIER_RULE = "IQR_remove"     # "IQR_remove" 或 "none"
Y_WINSORIZE_Q = (0.01, 0.99)        # 对 Y 浓度做分位截尾；设为 None 则关闭
SEVERE_OBESE_BMI = 40.0

# 输入/输出路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 问题一及数据预处理/
DATA_DIR = os.path.dirname(BASE_DIR)                   # 上级目录：放原始CSV的目录
PATH_FULL_RAW = os.path.join(DATA_DIR, "完整观测.csv")
PATH_PASS_RAW = os.path.join(DATA_DIR, "男胎达标.csv")
OUT_FULL_CLEAN = os.path.join(BASE_DIR, "清洗后_完整观测.csv")
OUT_PASS_CLEAN = os.path.join(BASE_DIR, "清洗后_男胎达标.csv")
OUT_OOLIST     = os.path.join(BASE_DIR, "异常值清单.csv")
OUT_LOG        = os.path.join(BASE_DIR, "预处理重置_日志.txt")
OUT_SEV_OBESE  = os.path.join(BASE_DIR, "高BMI_重度肥胖清单.csv")
OUT_PASS_SUPP  = os.path.join(BASE_DIR, "男胎达标_补充_含删失信息.csv")
OUT_FULL_FLAG  = os.path.join(BASE_DIR, "完整观测_补充_含本次是否达标列.csv")

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path}")
    return pd.read_csv(path)

def parse_gest_week(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if m:
        w = float(m.group(1)); d = float(m.group(2))
        return float(w + d/7.0)
    try:
        return float(s)
    except:
        return np.nan

def iqr_bounds(series: pd.Series):
    q1 = series.quantile(0.25, interpolation="linear")
    q3 = series.quantile(0.75, interpolation="linear")
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper, q1, q3, iqr

def winsorize(series: pd.Series, q_low=0.01, q_high=0.99):
    lo = series.quantile(q_low, interpolation="linear")
    hi = series.quantile(q_high, interpolation="linear")
    capped = series.clip(lower=lo, upper=hi)
    return capped, lo, hi

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = [f"[{ts}] 启动重置版预处理（含重度肥胖标记）..."]

    df_full = load_csv(PATH_FULL_RAW)
    df_pass = load_csv(PATH_PASS_RAW)

    rename_map = {}
    if "孕周" in df_full.columns and "孕周数" not in df_full.columns:
        rename_map["孕周"] = "孕周数"
    if rename_map:
        df_full = df_full.rename(columns=rename_map)

    required_cols = ["孕妇代码", "孕周数", "孕妇BMI", "Y染色体浓度"]
    for c in required_cols:
        if c not in df_full.columns:
            raise KeyError(f"完整观测.csv 缺少所需列：{c}")

    df_full["孕周数"] = df_full["孕周数"].apply(parse_gest_week)
    df_full["孕妇BMI"] = pd.to_numeric(df_full["孕妇BMI"], errors="coerce")
    df_full["Y染色体浓度"] = pd.to_numeric(df_full["Y染色体浓度"], errors="coerce")

    mask_plausible = (
        df_full["孕周数"].between(GEST_WEEK_MIN, GEST_WEEK_MAX, inclusive="both") &
        df_full["孕妇BMI"].between(BMI_MIN_PLAUSIBLE, BMI_MAX_PLAUSIBLE, inclusive="both") &
        (df_full["Y染色体浓度"].notna())
    )
    removed_plausible = df_full.loc[~mask_plausible].copy()
    df_full = df_full.loc[mask_plausible].copy()

    oolist = []  # 异常/截尾记录清单
    if BMI_OUTLIER_RULE == "IQR_remove":
        lower, upper, q1, q3, iqr = iqr_bounds(df_full["孕妇BMI"].dropna())
        mask_bmi_ok = df_full["孕妇BMI"].between(lower, upper, inclusive="both")
        removed_bmi = df_full.loc[~mask_bmi_ok].copy()
        if not removed_bmi.empty:
            removed_bmi["异常类型"] = "BMI_IQR_剔除"
            removed_bmi["备注"] = f"IQR边界[{round(lower,2)}, {round(upper,2)}]"
            oolist.append(removed_bmi)
        df_full = df_full.loc[mask_bmi_ok].copy()
        logs.append(f"BMI IQR 剔除：{len(removed_bmi)} 条（边界≈[{lower:.2f}, {upper:.2f}]）")
    else:
        logs.append("BMI 异常值处理：未启用（BMI_OUTLIER_RULE=none）")

    if Y_WINSORIZE_Q is not None:
        ql, qh = Y_WINSORIZE_Q
        y_before = df_full["Y染色体浓度"].copy()
        y_after, y_lo, y_hi = winsorize(df_full["Y染色体浓度"], q_low=ql, q_high=qh)
        changed = df_full.loc[(y_before != y_after)].copy()
        if not changed.empty:
            changed["异常类型"] = "Y浓度_截尾"
            changed["备注"] = f"winsorize到[{round(y_lo,4)}, {round(y_hi,4)}]"
            oolist.append(changed)
        df_full["Y染色体浓度"] = y_after
        logs.append(f"Y 浓度 winsorize：{len(changed)} 条（区间≈[{y_lo:.4f}, {y_hi:.4f}]）")
    else:
        logs.append("Y 浓度截尾：未启用")

    if not removed_plausible.empty:
        removed_plausible["异常类型"] = "合理范围_剔除"
        removed_plausible["备注"] = f"孕周∉[{GEST_WEEK_MIN},{GEST_WEEK_MAX}] 或 BMI∉[{BMI_MIN_PLAUSIBLE},{BMI_MAX_PLAUSIBLE}] 或 Y缺失"
        oolist.append(removed_plausible)

    df_full["重度肥胖标记"] = (df_full["孕妇BMI"] >= SEVERE_OBESE_BMI).astype(int)
    severe_df = df_full[df_full["重度肥胖标记"] == 1].copy()
    severe_df.to_csv(OUT_SEV_OBESE, index=False, encoding="utf-8-sig")
    logs.append(f"高 BMI（≥{SEVERE_OBESE_BMI}）清单输出：{OUT_SEV_OBESE}（{len(severe_df)} 条）")

    if oolist:
        oolist_df = pd.concat(oolist, axis=0, ignore_index=True)
        oolist_df.to_csv(OUT_OOLIST, index=False, encoding="utf-8-sig")
        logs.append(f"异常值清单输出：{OUT_OOLIST}（共 {len(oolist_df)} 条）")
    else:
        pd.DataFrame(columns=list(df_full.columns) + ["异常类型", "备注"]).to_csv(OUT_OOLIST, index=False, encoding="utf-8-sig")
        logs.append(f"异常值清单为空：{OUT_OOLIST}")

    df_full.to_csv(OUT_FULL_CLEAN, index=False, encoding="utf-8-sig")
    logs.append(f"已生成：{OUT_FULL_CLEAN}（清洗后）")

    df_full_flag = df_full.copy()
    df_full_flag["是否达标_本次"] = (df_full_flag["Y染色体浓度"] >= Y_THRESHOLD).astype(int)
    df_full_flag.to_csv(OUT_FULL_FLAG, index=False, encoding="utf-8-sig")
    logs.append(f"已生成：{OUT_FULL_FLAG}（逐次本次是否达标）")

    kept_ids = df_full["孕妇代码"].dropna().unique().tolist()
    if "孕妇代码" not in df_pass.columns:
        raise KeyError("男胎达标.csv 缺少列：孕妇代码")
    if "是否达标" not in df_pass.columns:
        raise KeyError("男胎达标.csv 缺少列：是否达标")

    df_pass_clean = df_pass[df_pass["孕妇代码"].isin(kept_ids)].copy()
    # 规范达标孕周
    if "达标孕周" in df_pass_clean.columns:
        df_pass_clean["达标孕周"] = df_pass_clean["达标孕周"].apply(parse_gest_week)

    # 每位孕妇的“最后检测孕周”
    last_week = (
        df_full.groupby("孕妇代码", as_index=False)["孕周数"]
        .max()
        .rename(columns={"孕周数": "最后检测孕周"})
    )
    # 同步 BMI 与重度肥胖标记（取该孕妇所有记录的 BMI 中位，标记取“任一记录为1则为1”）
    bmi_agg = (
        df_full.groupby("孕妇代码")["孕妇BMI"].median()
        .reset_index()
        .rename(columns={"孕妇BMI": "孕妇BMI"})
    )
    obese_flag = (
        df_full.groupby("孕妇代码")["重度肥胖标记"].max()
        .reset_index()
        .rename(columns={"重度肥胖标记": "重度肥胖标记"})
    )

    df_pass_supp = (
        df_pass_clean
        .merge(last_week, on="孕妇代码", how="left")
        .merge(bmi_agg, on="孕妇代码", how="left")
        .merge(obese_flag, on="孕妇代码", how="left")
    )
    # 删失标记：已达标=0，未达标=1
    df_pass_supp["是否删失"] = np.where(df_pass_supp["是否达标"] == 1, 0, 1)
    # 观测时间：已达标用“达标孕周”，未达标用“最后检测孕周”
    if "达标孕周" not in df_pass_supp.columns:
        df_pass_supp["达标孕周"] = np.nan
    df_pass_supp["观测时间"] = df_pass_supp.apply(
        lambda r: r["达标孕周"] if pd.notnull(r["达标孕周"]) else r["最后检测孕周"],
        axis=1
    )
    # 写出清洗后的男胎达标表 + 补充删失
    df_pass_clean.to_csv(OUT_PASS_CLEAN, index=False, encoding="utf-8-sig")
    df_pass_supp.to_csv(OUT_PASS_SUPP, index=False, encoding="utf-8-sig")
    logs.append(f"已生成：{OUT_PASS_CLEAN}（清洗后 男胎达标）")
    logs.append(f"已生成：{OUT_PASS_SUPP}（含BMI/重度肥胖标记/删失/观测时间）")

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    print("\n".join(logs))

if __name__ == "__main__":
    main()
