import os, re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "附件.xlsx")  # 原始Excel

def parse_weeks(x):
    """把 '12+3'/'12周+3天'/'12w+3d'/12.3/12 转为 浮点周"""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try: return float(s)
        except: return np.nan
    nums = re.findall(r"\d+", s)
    if not nums: return np.nan
    w = float(nums[0]); d = float(nums[1]) if len(nums) > 1 else 0.0
    if d > 7: d = d % 7
    return w + d/7.0

def to_number(x, allow_ge=True):
    """把 '≥3' → 3, 其他不可转→NaN"""
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if allow_ge: s = s.replace("≥", "")
    try: return float(s)
    except: return np.nan

def pct_to_ratio(series: pd.Series) -> pd.Series:
    """把明显是百分数（1~100 为主）列缩放为 0~1 比例；否则原样返回"""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return x
    frac_pct = (x.dropna().between(1, 100)).mean()
    if frac_pct > 0.7:  # 大多数在1~100之间，按百分比处理
        return x / 100.0
    return x

def encode_ivf(x):
    """IVF妊娠 -> 1；其他或缺失 -> 0"""
    if pd.isna(x): return 0
    s = str(x).strip().upper()
    return 1 if ("IVF" in s or "是" == s or s == "1") else 0

def run_preprocess():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f"未找到Excel：{SRC}")
    df = pd.read_excel(SRC)
    df.columns = [str(c).strip() for c in df.columns]

    out = pd.DataFrame()
    out["孕妇代码"] = df["孕妇代码"]

    if df["检测孕周"].dtype == object:
        out["孕周数"] = df["检测孕周"].map(parse_weeks)
    else:
        out["孕周数"] = pd.to_numeric(df["检测孕周"], errors="coerce")

    out["孕妇BMI"] = pd.to_numeric(df["孕妇BMI"], errors="coerce")
    out["年龄"]   = pd.to_numeric(df["年龄"], errors="coerce")
    out["是否IVF妊娠"] = df["IVF妊娠"].map(encode_ivf) if "IVF妊娠" in df.columns else 0
    out["Y染色体浓度"] = pd.to_numeric(df["Y染色体浓度"], errors="coerce")
    out["X染色体浓度"] = pd.to_numeric(df["X染色体浓度"], errors="coerce")
    out["Z13"] = pd.to_numeric(df["13号染色体的Z值"], errors="coerce")
    out["Z18"] = pd.to_numeric(df["18号染色体的Z值"], errors="coerce")
    out["Z21"] = pd.to_numeric(df["21号染色体的Z值"], errors="coerce")
    out["ZX"]  = pd.to_numeric(df["X染色体的Z值"], errors="coerce")
    out["GC总"] = pct_to_ratio(df["GC含量"]) if "GC含量" in df.columns else np.nan
    out["GC13"] = pct_to_ratio(df["13号染色体的GC含量"]) if "13号染色体的GC含量" in df.columns else np.nan
    out["GC18"] = pct_to_ratio(df["18号染色体的GC含量"]) if "18号染色体的GC含量" in df.columns else np.nan
    out["GC21"] = pct_to_ratio(df["21号染色体的GC含量"]) if "21号染色体的GC含量" in df.columns else np.nan
    out["总读段"]     = pd.to_numeric(df["原始读段数"], errors="coerce")
    out["唯一比对段"] = pd.to_numeric(df["唯一比对的读段数"], errors="coerce")
    out["比对比例"]   = pct_to_ratio(df["在参考基因组上比对的比例"])
    out["重复比例"]   = pct_to_ratio(df["重复读段的比例"])
    out["过滤比例"]   = pct_to_ratio(df["被过滤掉读段数的比例"])
    out["孕次"] = df["怀孕次数"].map(lambda x: to_number(x, allow_ge=True)) if "怀孕次数" in df.columns else np.nan
    out["产次"] = df["生产次数"].map(lambda x: to_number(x, allow_ge=True)) if "生产次数" in df.columns else np.nan
    out["异常标签AB"] = df["染色体的非整倍体"].astype(str).str.strip()
    out["是否非整倍体"] = (~out["异常标签AB"].isin(["", "正常", "阴性", "无异常"])).astype(int)
    out["出生结果AE"] = df["胎儿是否健康"] if "胎儿是否健康" in df.columns else np.nan
    # 性别推断：通过 Y 指标
    y_z = pd.to_numeric(df["Y染色体的Z值"], errors="coerce")
    y_f = out["Y染色体浓度"]
    is_male = ( (y_f.notna() & (y_f > 0)) | (y_z.notna() & (np.abs(y_z) > 0)) )
    out["胎儿性别"] = np.where(is_male, "男", "女")
    # 基本清洗
    out.loc[out["孕妇BMI"] < 0, "孕妇BMI"] = np.nan
    out.loc[(out["孕周数"] < 4) | (out["孕周数"] > 45), "孕周数"] = np.nan

    full_path  = os.path.join(BASE, "完整观测.csv")
    male_path  = os.path.join(BASE, "男胎达标.csv")
    female_path= os.path.join(BASE, "女胎分类.csv")
    # 输出1：完整观测.csv
    full = out.copy()
    full.to_csv(full_path, index=False, encoding="utf-8-sig")
    # 输出2：男胎达标.csv（每人一行）
    male = full.query("胎儿性别=='男'").copy()
    res = []
    for pid, sub in male.groupby("孕妇代码"):
        sub = sub.sort_values("孕周数")
        row = {"孕妇代码": pid}
        reached = sub[sub["Y染色体浓度"] >= 0.04]
        if not reached.empty:
            row["达标孕周"] = float(reached.iloc[0]["孕周数"])
            row["是否达标"] = 1
        else:
            row["达标孕周"] = np.nan
            row["是否达标"] = 0
        base_row = sub.iloc[0]
        row["孕妇BMI"]     = float(base_row["孕妇BMI"]) if pd.notna(base_row["孕妇BMI"]) else np.nan
        row["年龄"]        = float(base_row["年龄"]) if pd.notna(base_row["年龄"]) else np.nan
        row["是否IVF妊娠"] = int(base_row["是否IVF妊娠"]) if pd.notna(base_row["是否IVF妊娠"]) else 0
        res.append(row)
    male_out = pd.DataFrame(res)
    male_out.to_csv(male_path, index=False, encoding="utf-8-sig")
    # 输出3：女胎分类.csv（逐次检测记录）
    female = full.query("胎儿性别=='女'").copy()
    keep = ["孕妇代码","孕妇BMI","年龄","是否IVF妊娠",
            "X染色体浓度","ZX","Z13","Z18","Z21",
            "GC总","GC13","GC18","GC21",
            "总读段","唯一比对段","比对比例","重复比例","过滤比例",
            "异常标签AB","是否非整倍体","出生结果AE"]
    keep = [c for c in keep if c in female.columns]
    female_out = female[keep].copy()
    female_out.to_csv(female_path, index=False, encoding="utf-8-sig")
    # 打印统计
    print("\n=== 预处理完成（无子目录版） ===")
    print(f"完整观测：{len(full)} 条；男胎：{(full['胎儿性别']=='男').sum()}；女胎：{(full['胎儿性别']=='女').sum()}")
    print(f"男胎达标表：{len(male_out)} 人；其中达标 {(male_out['是否达标']==1).sum()} 人")
    print(f"女胎分类表：{len(female_out)} 条")
    print("已输出至脚本同级目录：")
    print(" -", full_path)
    print(" -", male_path)
    print(" -", female_path)

if __name__ == "__main__":
    run_preprocess()