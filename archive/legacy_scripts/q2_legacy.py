import os, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

def resolve_data_path(cli_path):
    if cli_path and os.path.exists(cli_path):
        return cli_path
    cand = os.path.join(os.getcwd(), "清洗后_完整观测.csv")
    if os.path.exists(cand): return cand
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "清洗后_完整观测.csv")
    if os.path.exists(cand): return cand
    cand = os.path.join(os.path.dirname(here), "问题一及数据预处理", "清洗后_完整观测.csv")
    if os.path.exists(cand): return cand
    root = os.path.dirname(here)
    hits = glob.glob(os.path.join(root, "**", "*清洗后*观测*.csv"), recursive=True)
    hits = [h for h in hits if os.path.exists(h)]
    if hits:
        hits.sort(key=lambda p: len(p))
        return hits[0]
    raise FileNotFoundError("找不到数据文件：请用 --data 指定，或把 '清洗后_完整观测.csv' 放到脚本同目录。")

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="清洗后_完整观测.csv 的路径", default=None)
parser.add_argument("--outdir", help="输出目录", default=None)
args = parser.parse_args()

PATH_DATA = resolve_data_path(args.data)
OUTDIR    = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "问题二生成")
os.makedirs(OUTDIR, exist_ok=True)
print(f"[INFO] 使用数据：{PATH_DATA}")
print(f"[INFO] 输出目录：{OUTDIR}")

Y_THRESHOLD = 0.04     # Y% 达标阈值
BMI_CUT    = 35.0      # K=2 的固定切点
TAU_RMST   = 26.0      # RMST 积分上限
T_MIN, T_MAX, T_STEP = 10.0, 26.0, 0.1
W_TIME, C_RETEST = 1.0, 10.0   # 等待成本系数 / 复检成本

from statsmodels.duration.survfunc import SurvfuncRight
HAS_LIFELINES = True
try:
    from lifelines.statistics import logrank_test
    from lifelines import WeibullFitter
except Exception:
    HAS_LIFELINES = False

def smart_col(df: pd.DataFrame, keys):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None

def build_patient_events(df, y_thresh=Y_THRESHOLD):
    col_id   = smart_col(df, ["孕妇代码","id","pid","编号"])
    col_week = smart_col(df, ["孕周数","孕周","检测孕周","gest","week"])
    col_y    = smart_col(df, ["y染色体浓度","y浓度","y%","ff","胎儿浓度","胎儿份额"])
    col_bmi  = smart_col(df, ["孕妇bmi","bmi"])
    col_h    = smart_col(df, ["身高","height"])
    col_w    = smart_col(df, ["体重","weight"])
    if col_id is None or col_week is None or col_y is None:
        raise ValueError("找不到必要列：孕妇代码/孕周/Y浓度，请检查数据。")
    if col_bmi is None:
        if col_h is None or col_w is None:
            raise ValueError("找不到 BMI 列，且无身高/体重可计算。")
        h = pd.to_numeric(df[col_h], errors="coerce")
        if h.mean() > 100: h = h/100.0
        df["__BMI__"] = pd.to_numeric(df[col_w], errors="coerce") / (h**2)
        col_bmi = "__BMI__"
    use = df[[col_id, col_week, col_y, col_bmi]].copy()
    use.columns = ["pid","week","yval","bmi"]
    for c in ["pid","week","yval","bmi"]:
        if c!="pid": use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(subset=["pid","week","yval","bmi"])
    rows = []
    for pid, sub in use.groupby("pid"):
        sub = sub.sort_values("week")
        hit = sub[sub["yval"] >= y_thresh]
        if not hit.empty:
            t_event, event = float(hit.iloc[0]["week"]), 1
        else:
            t_event, event = float(sub["week"].max()), 0
        rows.append({"pid": str(pid), "event_time": t_event, "event": int(event),
                     "bmi": float(sub["bmi"].median())})
    return pd.DataFrame(rows).dropna(subset=["event_time","event","bmi"])

def q_from_km(sub, q=0.5):
    sf = SurvfuncRight(sub["event_time"], sub["event"])
    t, S = np.array(sf.surv_times), np.array(sf.surv_prob)
    if len(t)==0: return np.nan
    F = 1 - S
    idx = np.searchsorted(F, q, side="left")
    idx = np.clip(idx, 0, len(t)-1)
    return float(t[idx])

def rmst_tau(sub, tau=TAU_RMST):
    sf = SurvfuncRight(sub["event_time"], sub["event"])
    tt = np.array(sf.surv_times); SS = np.array(sf.surv_prob)
    t = [T_MIN]; S = [1.0]
    mask = tt <= tau
    t.extend(tt[mask].tolist()); S.extend(SS[mask].tolist())
    S_end = SS[mask][-1] if mask.sum()>0 else (SS[0] if len(SS)>0 else 1.0)
    if len(t)==1 or t[-1] < tau: t.append(tau); S.append(S_end)
    return float(np.trapz(np.array(S), np.array(t)))

def optimal_t_by_cost(sub, w_time=W_TIME, c_retest=C_RETEST,
                      t_min=T_MIN, t_max=T_MAX, step=T_STEP):
    sf = SurvfuncRight(sub["event_time"], sub["event"])
    t_grid = np.arange(t_min, t_max+1e-9, step)
    F = []
    tt = np.array(sf.sur_times) if hasattr(sf, "sur_times") else np.array(sf.surv_times)
    SS = np.array(sf.surv_prob)
    for t in t_grid:
        j = np.searchsorted(tt, t, side="right")-1
        S_t = 1.0 if j < 0 else SS[min(j, len(SS)-1)]
        F.append(1.0 - S_t)
    F = np.array(F)
    cost = w_time*(t_grid - t_grid.min()) + (1.0 - F)*c_retest
    k = int(np.argmin(cost))
    return float(t_grid[k]), float(cost[k])

def main():
    raw = pd.read_csv(PATH_DATA, encoding="utf-8-sig")
    # 仅保留男胎（如存在性别列）
    def _smart_col(df, ks):
        ks = [k.lower() for k in ks]
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in ks): return c
        return None
    col_sex = _smart_col(raw, ["胎儿性别","性别"])
    if col_sex is not None:
        raw = raw[raw[col_sex].astype(str).str.contains("男")].copy()
    # 事件表
    pat = build_patient_events(raw, y_thresh=Y_THRESHOLD)
    # K=2 固定切点分组
    mask = pat["bmi"] < BMI_CUT
    left, right = pat[mask].copy(), pat[~mask].copy()
    nL, nR = len(left), len(right)
    if nL==0 or nR==0:
        raise RuntimeError("某一侧为空，请检查 BMI 阈值与数据。")
    # 统计：log-rank
    stat, pval = np.nan, np.nan
    if HAS_LIFELINES:
        r = logrank_test(
            left["event_time"], right["event_time"],
            event_observed_A=left["event"], event_observed_B=right["event"]
        )
        stat = float(r.test_statistic) if r.test_statistic is not None else np.nan
        pval = float(r.p_value) if r.p_value is not None else np.nan
    # 组内统计与最优时点
    p50_L, p90_L = q_from_km(left, 0.5), q_from_km(left, 0.9)
    p50_R, p90_R = q_from_km(right,0.5), q_from_km(right,0.9)
    rmstL, rmstR  = rmst_tau(left), rmst_tau(right)
    tL, cL = optimal_t_by_cost(left); tR, cR = optimal_t_by_cost(right)
    # 汇总主结果表
    rows = [
        {"组别":"BMI<35",  "N":nL, "p50_KM":p50_L, "p90_KM":p90_L,
         "RMST_τ26":rmstL, "t*":tL, "期望成本":cL},
        {"组别":"BMI≥35", "N":nR, "p50_KM":p50_R, "p90_KM":p90_R,
         "RMST_τ26":rmstR, "t*":tR, "期望成本":cR},
    ]
    main_tbl = pd.DataFrame(rows)
    main_tbl_path = os.path.join(OUTDIR, "Q2_K2_35_主结果_汇总表.csv")
    main_tbl.to_csv(main_tbl_path, index=False, encoding="utf-8-sig")
    # 文本报告
    report = []
    report.append("【K=2（BMI<35 vs BMI≥35）主结果】")
    report.append(f"- N_left={nL}, N_right={nR}")
    if HAS_LIFELINES:
        report.append(f"- log-rank 统计量={stat:.3f}，p值={pval:.4g}")
    else:
        report.append("- lifelines 不可用：未计算 log-rank p 值")
    report.append(f"- 组内 p50/p90（KM）：<35:{p50_L:.2f}/{p90_L:.2f}；≥35:{p50_R:.2f}/{p90_R:.2f}")
    report.append(f"- 组内 RMST(τ=26)：<35:{rmstL:.2f}；≥35:{rmstR:.2f}")
    report.append(f"- 组内最优 t*（W={W_TIME}, C={C_RETEST}）：<35:{tL:.2f}；≥35:{tR:.2f}")
    report.append(f"- 组内期望成本：<35:{cL:.2f}；≥35:{cR:.2f}")
    with open(os.path.join(OUTDIR, "Q2_K2_35_主结果_报告.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=== K=2（35）主结果生成完成（无可视化版） ===")
    if HAS_LIFELINES: print(f"log-rank={stat:.3f}, p={pval:.4g}")
    print(f"主结果表：{main_tbl_path}")
    print(f"输出目录：{OUTDIR}")

if __name__ == "__main__":
    main()
