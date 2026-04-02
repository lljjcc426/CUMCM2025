import os, sys, glob, argparse, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

T0 = time.perf_counter()
def log(msg, *, flush=True):
    print(f"[{time.perf_counter()-T0:8.2f}s] {msg}")
    if flush: sys.stdout.flush()

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
    if hits:
        hits.sort(key=lambda p: len(p))
        return hits[0]
    raise FileNotFoundError("找不到数据文件：请用 --data 指定，或把 '清洗后_完整观测.csv' 放到脚本同目录。")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, help="清洗后_完整观测.csv 路径")
parser.add_argument("--outdir", default=None, help="输出目录（默认脚本旁 '问题三_PlanC_K3_常识'）")
parser.add_argument("--cuts", default="30,35", help="两阈值，逗号分隔，如 '30,35'")
parser.add_argument("--localsearch", action="store_true", help="对 (c1,c2) 做局部微调网格搜索并以BIC挑选最优")
parser.add_argument("--grid_step", type=float, default=0.2, help="局部网格步长（默认0.2）")
parser.add_argument("--grid_span", type=float, default=1.0, help="以 cuts±span 形成网格（默认±1.0）")
parser.add_argument("--tau", type=float, default=26.0, help="成本曲线/计算上限周数")
parser.add_argument("--w", type=float, default=1.0, help="等待成本权重 W")
parser.add_argument("--c", type=float, default=10.0, help="复检成本 C")
parser.add_argument("--seed", type=int, default=2025, help="随机种子")
args = parser.parse_args()

PATH_DATA = resolve_data_path(args.data)
OUTDIR = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "问题三_PlanC_K3_常识")
os.makedirs(OUTDIR, exist_ok=True)
log(f"使用数据：{PATH_DATA}")
log(f"输出目录：{OUTDIR}")
Y_THRESHOLD = 0.04
T_MIN, T_MAX, T_STEP = 10.0, float(args.tau), 0.1
W_TIME, C_RETEST = float(args.w), float(args.c)
rng = np.random.default_rng(int(args.seed))
# ============== lifelines ==============
try:
    from lifelines import WeibullAFTFitter, LogLogisticAFTFitter
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False
    log("错误：未检测到 lifelines（建议 pip/conda 安装 lifelines>=0.27）")
    sys.exit(2)

def smart_col(df: pd.DataFrame, keys):
    keys = [k.lower() for k in keys]
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None
# ============== 构建区间删失数据 ==============
def build_interval_dataset(df, y_thresh=Y_THRESHOLD, t_min=T_MIN):
    col_id   = smart_col(df, ["孕妇代码","id","pid","编号"])
    col_week = smart_col(df, ["孕周数","孕周","检测孕周","gest","week"])
    col_y    = smart_col(df, ["y染色体浓度","y浓度","y%","ff","胎儿浓度","胎儿份额"])
    col_bmi  = smart_col(df, ["孕妇bmi","bmi"])
    col_h    = smart_col(df, ["身高","height"])
    col_w    = smart_col(df, ["体重","weight"])
    col_age  = smart_col(df, ["年龄","age"])
    col_ivf  = smart_col(df, ["ivf","试管","体外受精"])
    if col_id is None or col_week is None or col_y is None:
        raise ValueError("缺必要列：孕妇代码/孕周/Y浓度")
    if col_bmi is None:
        if col_h is None or col_w is None:
            raise ValueError("缺 BMI 且无身高/体重可推算")
        h = pd.to_numeric(df[col_h], errors="coerce")
        if h.mean() > 100: h = h/100.0
        df["__BMI__"] = pd.to_numeric(df[col_w], errors="coerce") / (h**2)
        col_bmi = "__BMI__"
    use = df[[col_id, col_week, col_y, col_bmi]].copy()
    use.columns = ["pid","week","yval","bmi"]
    for c in ["week","yval","bmi"]:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(subset=["pid","week","yval","bmi"])
    rows, cov_rows = [], []
    for pid, sub in use.groupby("pid"):
        sub = sub.sort_values("week")
        hit = sub[sub["yval"] >= y_thresh]
        if not hit.empty:
            R = float(hit.iloc[0]["week"])
            before = sub[sub["week"] < R]
            before_not = before[before["yval"] < y_thresh]
            L = float(before_not["week"].max()) if not before_not.empty else t_min
            lo, hi = L, R
        else:
            lo, hi = float(sub["week"].max()), np.inf
        row_cov = {"pid": pid, "bmi": float(sub["bmi"].median())}
        if col_age:
            row_cov["age"] = float(pd.to_numeric(df.loc[sub.index, col_age], errors="coerce").median())
        else:
            row_cov["age"] = np.nan
        if col_ivf:
            iv = df.loc[sub.index, col_ivf].astype(str)
            row_cov["ivf"] = 1.0 if iv.str.contains("1|是|Y|y|True|IVF", regex=True).any() else 0.0
        else:
            row_cov["ivf"] = 0.0
        rows.append({"pid": pid, "lower": float(lo), "upper": float(hi)})
        cov_rows.append(row_cov)
    df_int = pd.DataFrame(rows)
    cov_df = pd.DataFrame(cov_rows).set_index("pid").reset_index()
    return df_int, cov_df
# ============== 生成 K=3 组别与哑变量 ==============
def add_group_k3(cov_df, c1, c2):
    # 0:<c1 ; 1: c1<=bmi<c2 ; 2: >=c2
    g = np.where(cov_df["bmi"] < c1, 0, np.where(cov_df["bmi"] < c2, 1, 2))
    covX = cov_df.copy()
    covX["G"] = g
    dummies = pd.get_dummies(covX["G"], prefix="G", drop_first=True)  # 基组=0，保留 G_1, G_2
    covX = pd.concat([covX.drop(columns=["G"]), dummies], axis=1)
    return covX, g
# ============== AFT 拟合（返回 BIC 等） ==============
def fit_aft_k3(df_int, cov_df, c1, c2, dist="weibull"):
    aft = WeibullAFTFitter(penalizer=0.01) if dist=="weibull" else LogLogisticAFTFitter(penalizer=0.01)
    covX, _ = add_group_k3(cov_df, c1, c2)
    X_cols = [c for c in covX.columns if c not in ["pid","bmi"]]
    dff = df_int.merge(covX[["pid"]+X_cols], on="pid", how="inner")
    dff = dff.rename(columns={"lower":"lower_bound","upper":"upper_bound"})
    formula = " + ".join(X_cols) if X_cols else None
    aft.fit_interval_censoring(
        dff,
        lower_bound_col="lower_bound",
        upper_bound_col="upper_bound",
        ancillary=True,
        show_progress=False,
        formula=formula,
        fit_options={"maxiter": 200, "ftol": 1e-6},
    )
    ll = float(aft.log_likelihood_)
    p  = int(np.prod(aft.params_.shape))
    N  = len(dff)
    bic = -2*ll + p*np.log(max(N,1))
    return aft, bic, ll, p, N
# ============== 预测 F(t)（按组别 0/1/2） ==============
def pred_F_group(aft, cov_df, c1, c2, group_id, t_grid):
    covX, g = add_group_k3(cov_df, c1, c2)
    mask = (g == group_id)
    sub = covX.loc[mask].copy()
    if sub.empty:
        return np.zeros_like(t_grid)
    # 确保 dummy 列存在
    for col in ["G_1","G_2"]:
        if col not in sub.columns: sub[col] = 0
    S_acc = np.zeros_like(t_grid, dtype=float); n = len(sub)
    for i in range(n):
        row = sub.iloc[[i]]
        try:
            S_vals = np.asarray(aft.predict_survival_function(row, times=t_grid)).ravel()
        except Exception:
            S_vals = np.ones_like(t_grid)
        S_acc += S_vals
    return 1.0 - S_acc/max(n,1)

def q_from_F(t_grid, F, q=0.5):
    k = int(np.searchsorted(F, q, side="left"))
    k = np.clip(k, 0, len(F)-1)
    return float(t_grid[k])

def optimal_t(F, t_grid, W=W_TIME, C=C_RETEST):
    cost = W*(t_grid - t_grid.min()) + (1.0 - F)*C
    k = int(np.argmin(cost))
    return float(t_grid[k]), float(cost[k])

def main():
    # 读取与筛选
    log("读取数据…")
    df = pd.read_csv(PATH_DATA, encoding="utf-8-sig")
    log(f"原始行数：{len(df)}")
    col_sex = smart_col(df, ["胎儿性别","性别"])
    if col_sex is not None:
        before = len(df)
        df = df[df[col_sex].astype(str).str.contains("男")].copy()
        log(f"筛选男胎：{before} -> {len(df)} 行")
    # 区间删失构建
    log("构建区间删失与协变量表…")
    t0 = time.perf_counter()
    df_int, cov_df = build_interval_dataset(df, y_thresh=Y_THRESHOLD, t_min=T_MIN)
    cov_df = cov_df.dropna(subset=["bmi"]).reset_index(drop=True)
    log(f"区间删失样本：{len(df_int)}；协变量样本：{len(cov_df)}；耗时 {time.perf_counter()-t0:.2f}s")
    # 初始 cuts
    try:
        c1, c2 = [float(x) for x in args.cuts.split(",")]
    except Exception:
        c1, c2 = 30.0, 35.0
    if c1 >= c2:
        raise ValueError("cuts需满足 c1<c2，例如 '30,35'")
    best = {}
    t_grid = np.arange(T_MIN, T_MAX + 1e-9, T_STEP)
    if args.localsearch:
        lo1, hi1 = c1 - args.grid_span, c1 + args.grid_span
        lo2, hi2 = c2 - args.grid_span, c2 + args.grid_span
        grid1 = np.arange(lo1, hi1 + 1e-9, args.grid_step)
        grid2 = np.arange(lo2, hi2 + 1e-9, args.grid_step)
        log(f"局部网格：c1∈[{lo1:.1f},{hi1:.1f}] step={args.grid_step}；c2∈[{lo2:.1f},{hi2:.1f}] step={args.grid_step}")
        total = len(grid1)*len(grid2)*2  # 两种分布
        done  = 0
        rows  = []
        for dname in ["weibull","loglogistic"]:
            for x in grid1:
                for y in grid2:
                    if x >= y:
                        continue
                    try:
                        aft, bic, ll, p, N = fit_aft_k3(df_int, cov_df, x, y, dist=dname)
                        rows.append({"dist": dname, "c1": x, "c2": y, "bic": bic, "ll": ll, "p": p, "N": N})
                    except Exception as e:
                        rows.append({"dist": dname, "c1": x, "c2": y, "bic": np.nan, "ll": np.nan, "p": np.nan, "N": 0})
                        log(f"  [WARN] {dname} c1={x:.2f}, c2={y:.2f} 拟合失败：{e}")
                    done += 1
                    if done % 10 == 0 or done == total:
                        log(f"  网格进度：{done}/{total}（{100*done/total:.1f}%）")
        grid_df = pd.DataFrame(rows)
        grid_path = os.path.join(OUTDIR, "PlanC_K3_局部网格_BIC.csv")
        grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
        log(f"网格结果写入：{grid_path}")
        valid = grid_df.dropna(subset=["bic"])
        if valid.empty:
            log("[WARN] 网格均失败，回退到默认 cuts")
        else:
            idx = valid["bic"].idxmin()
            c1, c2 = float(valid.loc[idx, "c1"]), float(valid.loc[idx, "c2"])
            dist_used = valid.loc[idx, "dist"]
            log(f"BIC 最优网格：dist={dist_used}, c1≈{c1:.2f}, c2≈{c2:.2f}，BIC={float(valid.loc[idx,'bic']):.3f}")
            best["dist"] = dist_used
    # 主拟合（两分布各一份）
    for dname in ["weibull","loglogistic"]:
        try:
            t1 = time.perf_counter()
            aft, bic, ll, p, N = fit_aft_k3(df_int, cov_df, c1, c2, dist=dname)
            best[dname] = {"aft": aft, "bic": bic}
            log(f"AFT({dname}) 拟合完成（c1={c1:.2f}, c2={c2:.2f}），BIC={bic:.3f}；耗时 {time.perf_counter()-t1:.2f}s")
        except Exception as e:
            log(f"[WARN] AFT({dname}) 拟合失败：{e}")

    if "aft" not in (best.get("weibull") or {}):
        if "aft" not in (best.get("loglogistic") or {}):
            raise RuntimeError("两种分布均未能拟合成功。")
        dist_used = "loglogistic"
    else:
        if "aft" in (best.get("loglogistic") or {}):
            dist_used = "weibull" if best["weibull"]["bic"] <= best["loglogistic"]["bic"] else "loglogistic"
        else:
            dist_used = "weibull"
    if "dist" in best:  # 若网格已确定分布，则覆写
        dist_used = best["dist"]
    aft = best[dist_used]["aft"]
    log(f"使用分布：{dist_used}；阈值：c1={c1:.2f}, c2={c2:.2f}")
    # 预测三组 CDF 与 t*
    t_grid = np.arange(T_MIN, T_MAX + 1e-9, T_STEP)
    F0 = pred_F_group(aft, cov_df, c1, c2, 0, t_grid)  # <c1
    F1 = pred_F_group(aft, cov_df, c1, c2, 1, t_grid)  # c1<= <c2
    F2 = pred_F_group(aft, cov_df, c1, c2, 2, t_grid)  # >=c2
    t0_star, c0 = optimal_t(F0, t_grid); t1_star, c1_star = optimal_t(F1, t_grid); t2_star, c2_star = optimal_t(F2, t_grid)

    g_all = np.where(cov_df["bmi"] < c1, 0, np.where(cov_df["bmi"] < c2, 1, 2))
    tbl = pd.DataFrame([
        {"组别": f"BMI<{c1:.2f}",           "N": int((g_all==0).sum()),
         "p50": q_from_F(t_grid,F0,0.5),    "p90": q_from_F(t_grid,F0,0.9),
         "t*": t0_star, "期望成本": c0},
        {"组别": f"{c1:.2f}≤BMI<{c2:.2f}",  "N": int((g_all==1).sum()),
         "p50": q_from_F(t_grid,F1,0.5),    "p90": q_from_F(t_grid,F1,0.9),
         "t*": t1_star, "期望成本": c1_star},
        {"组别": f"BMI≥{c2:.2f}",           "N": int((g_all==2).sum()),
         "p50": q_from_F(t_grid,F2,0.5),    "p90": q_from_F(t_grid,F2,0.9),
         "t*": t2_star, "期望成本": c2_star},
    ])
    out_tbl = os.path.join(OUTDIR, "PlanC_K3_主结果_汇总.csv")
    tbl.to_csv(out_tbl, index=False, encoding="utf-8-sig")
    log(f"主结果表：{out_tbl}")

    rep_lines = []
    rep_lines.append("【Q3 Plan C：K=3（医学常识 <30 / 30–35 / ≥35）· 区间删失AFT】")
    rep_lines.append(f"- 数据：{PATH_DATA}")
    rep_lines.append(f"- 使用分布：{dist_used}；阈值：c1={c1:.2f}，c2={c2:.2f}；W={W_TIME}，C={C_RETEST}；t_min={T_MIN}；τ={T_MAX}")
    rep_lines.append("- 三组关键指标：")
    for _, r in tbl.iterrows():
        rep_lines.append(f"  · {r['组别']}：N={int(r['N'])}，p50={float(r['p50']):.2f}，p90={float(r['p90']):.2f}，t*={float(r['t*']):.2f}，期望成本={float(r['期望成本']):.2f}")
    if args.localsearch:
        rep_lines.append(f"- 已进行局部网格搜索：步长={args.grid_step}，span=±{args.grid_span}，以BIC选择最优阈值组合。")
    rep_path = os.path.join(OUTDIR, "PlanC_K3_报告.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))
    log(f"报告已写入：{rep_path}")
    log("=== Plan C 完成（无图版）===")
    log(f"输出目录：{OUTDIR}")

if __name__ == "__main__":
    main()
