import os, sys, re, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, roc_auc_score,precision_recall_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
# ===== CLI =====
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="清洗后_完整观测.csv", help="输入数据文件")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--recall-min", type=float, default=0.0, help="测试阶段的最低召回约束（如 0.70）")
parser.add_argument("--zmax-bins", type=int, default=3, help="Zmax 分段数量（用于分组阈值）")
args = parser.parse_args()
np.random.seed(args.seed)
workdir = os.path.abspath(os.path.dirname(__file__) or ".")

def smart_col(df, keys):
    ks = [k.lower() for k in keys]
    for c in df.columns:
        if any(k in str(c).lower() for k in ks): return c
    return None

def parse_week(s):
    if pd.isna(s): return np.nan
    s2 = str(s).strip().lower()
    m = re.match(r"(\d+)\s*w\+?(\d+)?", s2)
    if m:
        w = int(m.group(1)); d = int(m.group(2)) if m.group(2) else 0
        return w + d/7.0
    if re.match(r"^\d+(\.\d+)?$", s2.replace('+','.')):
        try: return float(s2.replace('+','.'))
        except: return np.nan
    return np.nan

def ensure_ratio01(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.quantile(0.95) > 1.5: s = s/100.0
    return s.clip(0,1)

def winsorize(s, ql=0.01, qh=0.99):
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile(ql), x.quantile(qh)
    return x.clip(lower=lo, upper=hi)

if not os.path.exists(args.csv):
    print(f"【错误】找不到输入文件：{args.csv}", file=sys.stderr); sys.exit(2)
try:
    df = pd.read_csv(args.csv, encoding="utf-8-sig", low_memory=False)
except:
    df = pd.read_csv(args.csv, encoding="utf-8", low_memory=False)

TEXT_HINT = ("结果","判定","结论","风险","阳性","阴性","非整倍体","aneuploid","ab","汇总","总结")
EXCLUDE_HINT = ("z","z值","浓度","gc","读段","比例","rate","reads","比对","重复","过滤","count","数","测序")
POS_WORDS = r"(异常|阳性|高风险|非整倍体|trisomy|t(?:13|18|21)|13三体|18三体|21三体|xx[xy]|xyy|xxx|45x|单体|缺失|重复|微缺失|微重复)"
NEG_WORDS = r"(正常|阴性|低风险|未见异常|通过|合格|—|－|-|/|无|na|n/a|null|none|空|negative|normal|low)"

def looks_like_text_result(colname, series):
    name = str(colname).lower()
    if not any(h in name for h in [h.lower() for h in TEXT_HINT]): return False
    if any(h in name for h in [h.lower() for h in EXCLUDE_HINT]): return False
    s = series.astype(str).str.strip().str.lower()
    return (series.dtype == "object") or (s.nunique(dropna=False) <= max(30, int(0.1*len(series))))

tri_cols = {}
for key in ("13","18","21","t13","t18","t21"):
    for c in df.columns:
        if key in str(c).lower() and looks_like_text_result(c, df[c]):
            tri_cols.setdefault(key, c)
cand_13 = tri_cols.get("13", tri_cols.get("t13"))
cand_18 = tri_cols.get("18", tri_cols.get("t18"))
cand_21 = tri_cols.get("21", tri_cols.get("t21"))

def to_pos_neg(series):
    s = series.astype(str).str.strip().str.lower()
    neg = s.str.contains(NEG_WORDS, regex=True, na=False)
    pos = s.str.contains(POS_WORDS, regex=True, na=False)
    y = np.where(pos, 1, 0)
    y = np.where(neg, 0, y)
    return y

y = None; label_source = ""
if any([cand_13, cand_18, cand_21]):
    parts = []
    if cand_13 is not None: parts.append(to_pos_neg(df[cand_13]))
    if cand_18 is not None: parts.append(to_pos_neg(df[cand_18]))
    if cand_21 is not None: parts.append(to_pos_neg(df[cand_21]))
    y = np.maximum.reduce(parts); label_source = f"[来源] 文本结果列合成：{cand_13 or '-'} / {cand_18 or '-'} / {cand_21 or '-'}"

if y is None:
    col_ab_exact = next((c for c in df.columns if str(c).strip().upper() == "AB"), None)
    col_ab = col_ab_exact or next(
        (c for c in df.columns if any(k in str(c).lower() for k in ["非整倍体","ab","汇总","总结","判定","结论"])), None
    )
    if not col_ab:
        print("【错误】未找到 13/18/21 文本列或 AB/汇总列。", file=sys.stderr); sys.exit(3)
    s = df[col_ab].astype(str).str.strip().str.lower()
    neg = s.str.contains(NEG_WORDS, regex=True, na=False)
    pos = s.str.contains(POS_WORDS, regex=True, na=False)
    y = np.where(pos, 1, 0); y = np.where(neg, 0, y)
    y = np.where((s=="")|(s.isna()), 0, y)
    label_source = f"[来源] AB/汇总列：{col_ab}（关键词解析）"

pos_cnt, neg_cnt = int((y==1).sum()), int((y==0).sum())
print(label_source)
print(f"[标签分布] 正类(异常)={pos_cnt}；负类(正常)={neg_cnt}；总样本={len(y)}")
if pos_cnt==0 or neg_cnt==0:
    print("【中止】单类别标签，无法监督学习。", file=sys.stderr); sys.exit(4)
# ===== 特征工程 =====
def numcol(df, pats):
    c = smart_col(df, pats)
    return pd.to_numeric(df[c], errors="coerce") if c else None

col_week   = smart_col(df, ["检测孕周","孕周","孕周数"])
col_height = smart_col(df, ["身高","height"])
col_weight = smart_col(df, ["体重","weight"])
col_bmi    = smart_col(df, ["孕妇bmi","bmi"])
col_age    = smart_col(df, ["年龄","age"])
col_ivf    = smart_col(df, ["ivf","试管","体外受精","是否ivf"])

Z13 = numcol(df, ["13号染色体的z值","13 z","z13","q"])
Z18 = numcol(df, ["18号染色体的z值","18 z","z18","r"])
Z21 = numcol(df, ["21号染色体的z值","21 z","z21","s"])
Zx  = numcol(df, ["x染色体的z值","x z","zx","t"])
Xconc = numcol(df, ["x染色体浓度","x 浓度","xconc","w"])

GC_all = numcol(df, ["gc含量","gc overall","总体gc","p"])
GC13   = numcol(df, ["13号染色体的gc含量","13 gc","x"])
GC18   = numcol(df, ["18号染色体的gc含量","18 gc","y"])
GC21   = numcol(df, ["21号染色体的gc含量","21 gc","z"])

Reads0 = numcol(df, ["原始测序数据的总读段数","原始读段数","总读段","raw reads","l"])
UniqRd = numcol(df, ["唯一比对的读段数","unique mapped","o"])
MapRate= numcol(df, ["在参考基因组上比对的比例","比对比例","map rate","align rate","m"])
DupRate= numcol(df, ["重复读段的比例","dup rate","n"])
FiltRt = numcol(df, ["被过滤掉的读段数的比例","被过滤掉读段数的比例","filter rate","aa"])

week = df[col_week].apply(parse_week) if col_week else pd.Series([np.nan]*len(df))
if col_bmi and df[col_bmi].notna().any():
    bmi = pd.to_numeric(df[col_bmi], errors="coerce")
else:
    if not (col_height and col_weight): raise ValueError("无 BMI 列且无身高/体重可计算 BMI。")
    h = pd.to_numeric(df[col_height], errors="coerce"); h = h/100.0 if h.mean()>100 else h
    w = pd.to_numeric(df[col_weight], errors="coerce"); bmi = w/(h**2)
bmi = bmi.clip(15, 50)
age = pd.to_numeric(df[col_age], errors="coerce") if col_age else pd.Series([np.nan]*len(df))
ivf = df[col_ivf].astype(str).str.contains(r"1|是|y|true|体外|ivf", case=False, regex=True).astype(int) if col_ivf else pd.Series([0]*len(df))

for nm in ["GC_all","GC13","GC18","GC21","MapRate","DupRate","FiltRt"]:
    if eval(nm) is not None: locals()[nm] = ensure_ratio01(locals()[nm])

feat = {
    "孕周数": week, "孕妇BMI": bmi, "年龄": age, "IVF": ivf,
    "Z13": Z13, "Z18": Z18, "Z21": Z21, "Zx": Zx, "X浓度": Xconc,
    "GC_all": GC_all, "GC13": GC13, "GC18": GC18, "GC21": GC21,
    "Reads0": Reads0, "UniqReads": UniqRd, "MapRate": MapRate, "DupRate": DupRate, "FilterRate": FiltRt
}
X_df = pd.DataFrame({k:v for k,v in feat.items() if v is not None})
# Z衍生、比值、交互
for z in ["Z13","Z18","Z21","Zx"]:
    if z in X_df.columns: X_df[f"abs{z}"] = X_df[z].abs()
X_df["Zmax"] = pd.concat([X_df.get("absZ13",pd.Series([np.nan]*len(X_df))),
                          X_df.get("absZ18",pd.Series([np.nan]*len(X_df))),
                          X_df.get("absZ21",pd.Series([np.nan]*len(X_df)))], axis=1).max(axis=1)
X_df["Zsum"] = X_df.get("absZ13",0).fillna(0)+X_df.get("absZ18",0).fillna(0)+X_df.get("absZ21",0).fillna(0)
X_df["Zvar"] = pd.concat([X_df.get("Z13",pd.Series([np.nan]*len(X_df))),
                          X_df.get("Z18",pd.Series([np.nan]*len(X_df))),
                          X_df.get("Z21",pd.Series([np.nan]*len(X_df)))], axis=1).var(axis=1)
if "UniqReads" in X_df.columns and "Reads0" in X_df.columns:
    X_df["unique_ratio"] = (X_df["UniqReads"]/X_df["Reads0"]).replace([np.inf,-np.inf], np.nan)
for g in ["GC13","GC18","GC21"]:
    if g in X_df.columns and "GC_all" in X_df.columns:
        X_df[f"{g}_ratio"] = (X_df[g]/X_df["GC_all"]).replace([np.inf,-np.inf], np.nan)
if "孕妇BMI" in X_df.columns and "孕周数" in X_df.columns:
    X_df["BMI_week"] = X_df["孕妇BMI"] * X_df["孕周数"]
# winsorize + 中位数填补
for c in X_df.columns:
    if c == "IVF": continue
    X_df[c] = winsorize(X_df[c])
X_df = X_df.fillna(X_df.median(numeric_only=True))
y = y.astype(int)
# ===== 分层切分 =====
def stratified_split_with_check(X, y, seed0=42, max_try=200):
    for k in range(max_try):
        seed = seed0 + k
        X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        if len(np.unique(y_te))<2: continue
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=seed)
        if len(np.unique(y_tr))<2 or len(np.unique(y_va))<2: continue
        return X_tr, X_va, X_te, y_tr, y_va, y_te, seed
    return None

spl = stratified_split_with_check(X_df.values, y, args.seed, 200)
if spl is None:
    print("【中止】分层切分失败。", file=sys.stderr); sys.exit(5)
X_tr, X_va, X_te, y_tr, y_va, y_te, used_seed = spl
print(f"[分层切分] seed={used_seed}；train/valid/test={len(y_tr)}/{len(y_va)}/{len(y_te)}；正类计数={y_tr.sum()}/{y_va.sum()}/{y_te.sum()}")
# ===== 模型 & 概率校准 & 堆叠 =====
mdl_lr = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=used_seed))])
mdl_rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=used_seed, n_jobs=-1)

HAS_XGB=True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB=False
if HAS_XGB:
    pos = int((y_tr==1).sum()); neg = int((y_tr==0).sum())
    spw = max(1.0, neg/max(pos,1))
    mdl_xgb = XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                            random_state=used_seed, tree_method="hist", eval_metric="auc",
                            scale_pos_weight=spw)
# 训练 base
mdl_lr.fit(X_tr, y_tr)
mdl_rf.fit(X_tr, y_tr)
if HAS_XGB: mdl_xgb.fit(X_tr, y_tr)
# 概率校准（基于验证集）
cal_rf  = CalibratedClassifierCV(mdl_rf, method="isotonic", cv="prefit"); cal_rf.fit(X_va, y_va)
cal_xgb = None
if HAS_XGB:
    cal_xgb = CalibratedClassifierCV(mdl_xgb, method="isotonic", cv="prefit"); cal_xgb.fit(X_va, y_va)
# 验证集概率（供堆叠学习）
p_lr_va  = mdl_lr.predict_proba(X_va)[:,1]
p_rf_va  = cal_rf.predict_proba(X_va)[:,1]
p_xgb_va = (cal_xgb.predict_proba(X_va)[:,1] if HAS_XGB else np.zeros_like(p_lr_va))
# 堆叠二层：用验证集概率训练元分类器 LR
X_stack_tr = np.vstack([p_lr_va, p_rf_va, p_xgb_va]).T
meta = Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=used_seed))])
meta.fit(X_stack_tr, y_va)
# ===== 阈值策略 =====
def scan_best_thr(y_true, y_prob, recall_min=0.0, prefer="F2"):
    grid = np.linspace(0.05, 0.95, 181)
    best = {"thr":0.5, "F1":0.0, "F2":0.0, "Recall":0.0, "Precision":0.0}
    for t in grid:
        yhat = (y_prob>=t).astype(int)
        tp = np.sum((y_true==1)&(yhat==1))
        fp = np.sum((y_true==0)&(yhat==1))
        fn = np.sum((y_true==1)&(yhat==0))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        if rec < recall_min: continue
        F1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        b2 = 4.0; F2 = (1+b2)*prec*rec/(b2*prec+rec) if (prec+rec)>0 else 0.0
        key = "F2" if prefer=="F2" else "F1"
        if (F2 if key=="F2" else F1) > best[key]:
            best.update({"thr":t, "F1":F1, "F2":F2, "Recall":rec, "Precision":prec})
    return best
# 在验证集为每个模型找三种阈值：默认0.5 / F1* / F2*(含召回约束)
valid_rows=[]
cands = {}
def eval_valid(name, y_prob):
    vF1 = scan_best_thr(y_va, y_prob, recall_min=0.0,   prefer="F1")
    vF2 = scan_best_thr(y_va, y_prob, recall_min=args.recall_min, prefer="F2")
    valid_rows.append({"模型":name, "F1_thr":vF1["thr"], "F1(valid)":vF1["F1"],
                       "F2_thr":vF2["thr"], "F2(valid)":vF2["F2"], "Recall_min":args.recall_min})
    cands[name]={"F1_thr":vF1["thr"], "F2_thr":vF2["thr"]}

p_lr  = p_lr_va
p_rf  = p_rf_va
p_xgb = p_xgb_va
p_meta= meta.predict_proba(X_stack_tr)[:,1]

eval_valid("LR", p_lr)
eval_valid("RF_cal", p_rf)
if HAS_XGB: eval_valid("XGB_cal", p_xgb)
eval_valid("STACK", p_meta)

vd = pd.DataFrame(valid_rows).sort_values(["F2(valid)","F1(valid)"], ascending=False)
vd.to_csv(os.path.join(workdir,"adv_valid_阈值寻优.csv"), index=False, encoding="utf-8-sig")
print("[输出] adv_valid_阈值寻优.csv")
# ===== 测试集评估（含分组阈值）=====
p_lr_te  = mdl_lr.predict_proba(X_te)[:,1]
p_rf_te  = cal_rf.predict_proba(X_te)[:,1]
p_xgb_te = (cal_xgb.predict_proba(X_te)[:,1] if HAS_XGB else np.zeros_like(p_lr_te))
p_meta_te= meta.predict_proba(np.vstack([p_lr_te, p_rf_te, p_xgb_te]).T)[:,1]

def aucs(y_true, prob):
    auc_roc = roc_auc_score(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    auc_pr = average_precision_score(y_true, prob)  # 与PR面积一致方向
    return float(auc_roc), float(auc_pr)

def metrics_row(name, tag, thr, yte, prob, auc_roc, auc_pr):
    yhat = (prob>=thr).astype(int)
    cm = confusion_matrix(yte, yhat); tn, fp, fn, tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    F1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    b2=4.0; F2=(1+b2)*prec*rec/(b2*prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tn+tp)/np.sum(cm)
    return {"模型":name,"阈值策略":tag,"阈值":thr,"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
            "Accuracy":acc,"Precision":prec,"Recall":rec,"F1":F1,"F2":F2,"AUC_ROC":auc_roc,"AUC_PR":auc_pr}

def learn_group_thresholds(zmax_va, prob_va, y_va, bins, recall_min, prefer="F2"):
    edges = np.quantile(zmax_va, np.linspace(0,1,bins+1))
    edges[0], edges[-1] = -np.inf, np.inf
    thr_map = {}
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        m = (zmax_va>=lo)&(zmax_va<hi)
        if m.sum()<5:
            thr_map[i]=(lo,hi,0.5); continue
        best = scan_best_thr(y_va[m], prob_va[m], recall_min=recall_min, prefer=prefer)
        thr_map[i]=(lo,hi,best["thr"])
    return thr_map, edges

def apply_group_thresholds(zmax, prob, thr_map):
    yhat = np.zeros_like(prob, dtype=int)
    for i,(lo,hi,thr) in thr_map.items():
        m = (zmax>=lo)&(zmax<hi)
        yhat[m] = (prob[m] >= thr).astype(int)
    return yhat

rows=[]
probs = {"LR":p_lr_te, "RF_cal":p_rf_te, "STACK":p_meta_te}
if HAS_XGB: probs["XGB_cal"]=p_xgb_te
# 逐模型评估：默认0.5 / F1* / F2*(召回约束) / 分组阈值
for name, prob in probs.items():
    auc_roc, auc_pr = aucs(y_te, prob)
    v = cands[name]
    rows.append(metrics_row(name,"默认0.5",0.5, y_te, prob, auc_roc, auc_pr))
    rows.append(metrics_row(name,"F1最优", float(v["F1_thr"]), y_te, prob, auc_roc, auc_pr))
    rows.append(metrics_row(name,f"F2最优@Recall≥{args.recall_min:.2f}",
                            float(v["F2_thr"]), y_te, prob, auc_roc, auc_pr))
    # 分组阈值（按 Zmax 分段学习->应用）
    zmax_va = X_df.iloc[:len(y_tr)+len(y_va)]["Zmax"].values[len(y_tr):]
    zmax_te = X_df.iloc[len(y_tr)+len(y_va):]["Zmax"].values
    base_prob_va = (p_lr if name=="LR" else p_rf if name=="RF_cal" else
                    (p_xgb if (HAS_XGB and name=="XGB_cal") else p_meta))
    thr_map, edges = learn_group_thresholds(zmax_va, base_prob_va, y_va, bins=args.zmax_bins,
                                            recall_min=args.recall_min, prefer="F2")
    yhat = apply_group_thresholds(zmax_te, prob, thr_map)
    cm = confusion_matrix(y_te, yhat); tn, fp, fn, tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    F1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    b2=4.0; F2=(1+b2)*prec*rec/(b2*prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tn+tp)/np.sum(cm)
    rows.append({"模型":name,"阈值策略":f"分组阈值(Zmax,bins={args.zmax_bins})","阈值":np.nan,
                 "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
                 "Accuracy":acc,"Precision":prec,"Recall":rec,"F1":F1,"F2":F2,
                 "AUC_ROC":auc_roc,"AUC_PR":auc_pr})

adv_test = pd.DataFrame(rows).sort_values(["Recall","F2","AUC_ROC"], ascending=False)
adv_test.to_csv(os.path.join(workdir,"adv_test_综合评估.csv"), index=False, encoding="utf-8-sig")
print("[输出] adv_test_综合评估.csv")
# 推荐部署位点（带召回约束）
best = adv_test.iloc[0]
with open(os.path.join(workdir,"adv_BestChoice.txt"),"w",encoding="utf-8") as f:
    f.write("【问题四·进阶版 推荐部署方案】\n")
    f.write(f"模型：{best['模型']}\n阈值策略：{best['阈值策略']}，阈值：{best['阈值']}\n")
    f.write(f"测试集：Recall={best['Recall']:.3f}，Precision={best['Precision']:.3f}\n")
    f.write(f"F2={best['F2']:.3f}，F1={best['F1']:.3f}\n")
    f.write(f"AUC(ROC)={best['AUC_ROC']:.3f}，AUC(PR)={best['AUC_PR']:.3f}\n")
    f.write(f"混淆矩阵：TN={int(best['TN'])}, FP={int(best['FP'])}, FN={int(best['FN'])}, TP={int(best['TP'])}\n")
print("[完成] adv_BestChoice.txt 已生成（无任何图片文件）。")
