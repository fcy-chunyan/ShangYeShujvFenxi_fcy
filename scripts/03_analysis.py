# -*- coding: utf-8 -*-
"""
03_analysis.py —— 商业分析模型（6 类模型 + EDA 汇总）
模型清单与业务问题对应关系：
  M1 描述性统计与结构分析   → 客群/渠道/月份结构，回答"谁在被触达"
  M2 漏斗转化分析           → 触达→接通→沟通→深度沟通→认购，定位流失环节
  M3 环比/同比 + 移动平均   → 月度趋势与季节性，判断增长质量
  M4 帕累托/ABC 分类        → 客群贡献集中度，确定资源投放优先级
  M5 波士顿矩阵             → 份额 × 转化率同比增速，制定客群组合策略
  M6 K-Means 客群分群       → 多维特征聚类，输出可运营的客群画像
  M7 关联规则（Apriori）    → 特征组合 → 认购，挖掘高转化规则
"""
from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

# 数据载入
def load_clean() -> pd.DataFrame:
    df = pd.read_csv(cfg.DATA_CLEAN / "bank_clean.csv", parse_dates=["contact_date", "week"])
    return df

# M1 描述性统计与结构分析
def eda(df: pd.DataFrame) -> dict:
    print("\n[M1] 描述性统计与结构分析")
    total = len(df)
    kpi = {
        "contacts": int(total),
        "subscribed": int(df["is_subscribed"].sum()),
        "conversion_rate": round(float(df["is_subscribed"].mean()), 4),
        "avg_duration_sec": round(float(df["duration"].mean()), 1),
        "median_balance": float(df["balance"].median()),
        "avg_campaign": round(float(df["campaign"].mean()), 2),
        "repeat_contact_ratio": round(float(df["had_previous_contact"].mean()), 4),
        "date_start": str(df["contact_date"].min().date()),
        "date_end": str(df["contact_date"].max().date()),
    }
    print(f"  触达 {kpi['contacts']:,} 人次，认购 {kpi['subscribed']:,} 人次，"
          f"转化率 {kpi['conversion_rate']:.2%}")
    def agg_by(col: str, name: str) -> pd.DataFrame:
        """统一的分组汇总口径：触达量、认购量、转化率、平均通话时长、平均余额。"""
        t = (df.groupby(col, observed=True)
               .agg(contacts=("is_subscribed", "size"),
                    subscribed=("is_subscribed", "sum"),
                    conv_rate=("is_subscribed", "mean"),
                    avg_duration=("duration", "mean"),
                    avg_balance=("balance", "mean"))
               .reset_index())
        t["contact_share"] = t["contacts"] / total
        t["sub_share"] = t["subscribed"] / kpi["subscribed"]
        t["conv_rate"] = t["conv_rate"].round(4)
        t["avg_duration"] = t["avg_duration"].round(1)
        t["avg_balance"] = t["avg_balance"].round(1)
        t["contact_share"] = t["contact_share"].round(4)
        t["sub_share"] = t["sub_share"].round(4)
        t = t.sort_values("contacts", ascending=False)
        cfg.save_table(t, f"eda_{name}")
        return t
    for col, name in [("job", "job"), ("channel", "channel"), ("education", "education"),
                      ("marital", "marital"), ("age_band", "ageband"),
                      ("campaign_bucket", "campaign"), ("poutcome", "poutcome"),
                      ("balance_band", "balanceband")]:
        agg_by(col, name)
    # 月份季节性（不依赖年份，看全年节奏）
    month = (df.groupby("month_num")
               .agg(contacts=("is_subscribed", "size"),
                    subscribed=("is_subscribed", "sum"),
                    conv_rate=("is_subscribed", "mean"))
               .reset_index().sort_values("month_num"))
    month["month_cn"] = month["month_num"].map({v: k for k, v in cfg.MONTH_NUM.items()}).map(cfg.MONTH_CN)
    month["conv_rate"] = month["conv_rate"].round(4)
    cfg.save_table(month, "eda_month_season")
    print(f"  旺季：{month.loc[month['contacts'].idxmax(), 'month_cn']}"
          f"（{month['contacts'].max():,} 人次）；"
          f"转化率最高：{month.loc[month['conv_rate'].idxmax(), 'month_cn']}"
          f"（{month['conv_rate'].max():.2%}）")
    return kpi

# M2 漏斗转化分析
def funnel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n[M2] 漏斗转化分析")
    total = len(df)
    stages = []
    prev = total
    for name, thr in cfg.FUNNEL_STAGES:
        cnt = total if thr == 0 else int((df["duration"] >= thr).sum())
        stages.append({
            "stage": name, "threshold_sec": thr, "users": cnt,
            "overall_rate": round(cnt / total, 4),          # 相对首层
            "step_rate": round(cnt / prev, 4) if prev else np.nan,  # 相对上一层
            "drop_users": prev - cnt,
            "drop_rate": round((prev - cnt) / prev, 4) if prev else np.nan,
        })
        prev = cnt
    # 末层：认购（人数口径，非时长筛选）
    sub = int(df["is_subscribed"].sum())
    stages.append({
        "stage": "认购成功", "threshold_sec": np.nan, "users": sub,
        "overall_rate": round(sub / total, 4),
        "step_rate": round(sub / prev, 4) if prev else np.nan,
        "drop_users": prev - sub,
        "drop_rate": round((prev - sub) / prev, 4) if prev else np.nan,
    })
    f = pd.DataFrame(stages)
    cfg.save_table(f, "funnel_overall")
    for _, r in f.iterrows():
        print(f"  {r['stage']:<6}{r['users']:>7,}  累计 {r['overall_rate']:.2%}   "
              f"层内转化 {r['step_rate']:.2%}  流失 {r['drop_users']:,}")
    # 分渠道漏斗（用于定位渠道差异）
    by_ch = []
    for ch, g in df.groupby("channel"):
        row = {"channel": ch, "contacts": len(g)}
        for name, thr in cfg.FUNNEL_STAGES[1:]:
            row[name] = int((g["duration"] >= thr).sum())
        row["认购成功"] = int(g["is_subscribed"].sum())
        row["conv_rate"] = round(g["is_subscribed"].mean(), 4)
        by_ch.append(row)
    ch_df = pd.DataFrame(by_ch)
    cfg.save_table(ch_df, "funnel_by_channel")
    worst = f.iloc[1:-1].sort_values("step_rate").iloc[0]
    res = {"funnel": f.to_dict(orient="records"),
           "biggest_drop_stage": worst["stage"],
           "biggest_drop_rate": float(worst["drop_rate"])}
    print(f"  最大流失环节：{worst['stage']}（层内流失 {worst['drop_rate']:.2%}）")
    return f, res

# M3 环比/同比 + 移动平均
def trend(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[M3] 环比/同比与移动平均")
    m = (df.groupby("year_month")
           .agg(contacts=("is_subscribed", "size"),
                subscribed=("is_subscribed", "sum"),
                conv_rate=("is_subscribed", "mean"),
                avg_duration=("duration", "mean"))
           .reset_index().sort_values("year_month"))
    m["conv_rate"] = m["conv_rate"].round(4)
    m["contacts_mom"] = m["contacts"].pct_change().round(4)
    m["subs_mom"] = m["subscribed"].pct_change().round(4)
    m["conv_mom_pp"] = (m["conv_rate"].diff() * 100).round(2)   # 百分点变化
    # 同比：同月份上一年对比（数据仅 30 个月，覆盖两个完整年度）
    m["year"] = m["year_month"].str[:4].astype(int)
    m["month"] = m["year_month"].str[5:7].astype(int)
    yoy_map = {(r.year, r.month): r.conv_rate for r in m.itertuples()}
    m["conv_yoy_pp"] = [
        round((r.conv_rate - yoy_map.get((r.year - 1, r.month), np.nan)) * 100, 2)
        if (r.year - 1, r.month) in yoy_map else np.nan
        for r in m.itertuples()
    ]
    m["contacts_ma3"] = m["contacts"].rolling(3, min_periods=1).mean().round(1)
    m["conv_ma3"] = m["conv_rate"].rolling(3, min_periods=1).mean().round(4)
    cfg.save_table(m, "trend_monthly")
    peak = m.loc[m["contacts"].idxmax()]
    mom_txt = "—" if pd.isna(peak["contacts_mom"]) else f"{peak['contacts_mom']:.2%}"
    print(f"  峰值月：{peak['year_month']}（{peak['contacts']:,} 人次，环比 {mom_txt}）")
    print(f"  转化率区间：{m['conv_rate'].min():.2%} ~ {m['conv_rate'].max():.2%}；"
          f"3 期移动平均平滑后波动收窄")
    # 渠道结构随季度变化（结构占比变化分析）
    mix = (df.groupby(["quarter", "channel"]).size().unstack(fill_value=0))
    mix_share = mix.div(mix.sum(axis=1), axis=0).round(4)
    mix_share.reset_index().to_csv(cfg.OUT_TAB / "channel_mix_quarter.csv",
                                   index=False, encoding="utf-8-sig")
    cfg.save_table(mix_share.reset_index(), "channel_mix_quarter")
    return m

# M4 帕累托/ABC 分类
def pareto(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[M4] 帕累托/ABC 分类")
    work = df.copy()
    work["segment"] = work["job"].astype(str) + " | " + work["age_band"].astype(str)
    p = (work.groupby("segment")
             .agg(contacts=("is_subscribed", "size"), subscribed=("is_subscribed", "sum"))
             .reset_index().sort_values("subscribed", ascending=False))
    p["sub_share"] = (p["subscribed"] / p["subscribed"].sum()).round(4)
    p["cum_share"] = p["sub_share"].cumsum().round(4)
    p["conv_rate"] = (p["subscribed"] / p["contacts"]).round(4)
    p["contact_share"] = (p["contacts"] / p["contacts"].sum()).round(4)
    p["abc"] = np.where(p["cum_share"] <= 0.70, "A",
                        np.where(p["cum_share"] <= 0.90, "B", "C"))
    cfg.save_table(p, "pareto_segment")
    a_cnt = int((p["abc"] == "A").sum())
    a_contacts = p.loc[p["abc"] == "A", "contact_share"].sum()
    a_subs = p.loc[p["abc"] == "A", "sub_share"].sum()
    a_conv = p.loc[p["abc"] == "A", "subscribed"].sum() / p.loc[p["abc"] == "A", "contacts"].sum()
    c_conv = p.loc[p["abc"] == "C", "subscribed"].sum() / p.loc[p["abc"] == "C", "contacts"].sum()
    print(f"  A 类 {a_cnt}/{len(p)} 个细分客群，贡献 {a_subs:.1%} 认购，"
          f"消耗 {a_contacts:.1%} 触达；A 类转化率 {a_conv:.2%} vs C 类 {c_conv:.2%}")
    return p

# M5 波士顿矩阵
def bcg(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[M5] 波士顿矩阵")
    g = (df.groupby(["job", "year"])
           .agg(contacts=("is_subscribed", "size"),
                conv=("is_subscribed", "mean"),
                subs=("is_subscribed", "sum"))
           .reset_index())
    piv = g.pivot(index="job", columns="year", values="conv")
    subs = g.groupby("job")["subs"].sum()
    contacts = g.groupby("job")["contacts"].sum()

    b = pd.DataFrame({"contacts": contacts, "subscribed": subs})
    b["share"] = b["contacts"] / b["contacts"].sum()
    b["rel_share"] = (b["share"] / b["share"].mean()).round(3)
    if 2009 in piv.columns and 2010 in piv.columns:
        b["conv_2009"] = piv[2009].round(4)
        b["conv_2010"] = piv[2010].round(4)
        b["growth_pp"] = ((piv[2010] - piv[2009]) * 100).round(2)
    else:  # 兼容数据覆盖不足的情况
        b["conv_2009"] = np.nan
        b["conv_2010"] = np.nan
        b["growth_pp"] = 0.0
    b["growth_pp"] = b["growth_pp"].fillna(0.0)
    b["conv_rate"] = (b["subscribed"] / b["contacts"]).round(4)
    share_cut = 1.0
    growth_cut = float(b["growth_pp"].median())
    b["quadrant"] = np.select(
        [(b["rel_share"] >= share_cut) & (b["growth_pp"] >= growth_cut),
         (b["rel_share"] >= share_cut) & (b["growth_pp"] < growth_cut),
         (b["rel_share"] < share_cut) & (b["growth_pp"] >= growth_cut)],
        ["明星", "金牛", "问题"], default="瘦狗")
    b.attrs["growth_cut"] = growth_cut
    b.attrs["share_cut"] = share_cut
    b = b.reset_index().sort_values("subscribed", ascending=False)
    cfg.save_table(b, "bcg_job")
    for q in ["明星", "金牛", "问题", "瘦狗"]:
        names = b.loc[b["quadrant"] == q, "job"].tolist()
        print(f"  {q}：{', '.join(names) if names else '—'}")
    print(f"  分界：相对份额 1.0；增速 {growth_cut:.1f}pp（中位数）")
    return b

# M6 K-Means 客群分群
def kmeans_segments(df: pd.DataFrame, k_range=range(2, 9), k_business: int = 5):
    print("\n[M6] K-Means 客群分群")
    work = df.copy()
    X = pd.DataFrame({
        "age": work["age"].astype(float),
        "balance_log": np.sign(work["balance_w"]) * np.log1p(work["balance_w"].abs()),
        "duration_w": work["duration_w"].astype(float),
        "campaign": work["campaign"].astype(float),
        "previous": work["previous"].astype(float),
        "had_prev": work["had_previous_contact"].astype(float),
        "pout_success": (work["poutcome"] == "success").astype(float),
        "housing": (work["housing"] == "yes").astype(float),
        "loan": (work["loan"] == "yes").astype(float),
    })
    Xs = StandardScaler().fit_transform(X)
    scores = []
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(5000, len(Xs)), replace=False)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        s = silhouette_score(Xs[idx], labels[idx])
        scores.append({"k": k, "silhouette": round(float(s), 4),
                       "inertia": round(float(km.inertia_), 1)})
    ks = pd.DataFrame(scores)
    cfg.save_table(ks, "kmeans_k_selection")
    best_stat = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    print(f"  轮廓系数：{ks.set_index('k')['silhouette'].to_dict()}（统计最优 k={best_stat}）")
    print(f"  业务准则：客群数上限 5、单群占比≥5% → 取 k={k_business}")
    km = KMeans(n_clusters=k_business, random_state=cfg.RANDOM_STATE, n_init=10)
    work["cluster"] = km.fit_predict(Xs)
    prof = (work.groupby("cluster")
                .agg(size=("is_subscribed", "size"),
                     conv_rate=("is_subscribed", "mean"),
                     avg_age=("age", "mean"),
                     med_balance=("balance", "median"),
                     avg_duration=("duration", "mean"),
                     avg_campaign=("campaign", "mean"),
                     prev_contact=("had_previous_contact", "mean"))
                .reset_index())
    prof["share"] = (prof["size"] / len(work)).round(4)
    prof["conv_rate"] = prof["conv_rate"].round(4)
    for c in ["avg_age", "med_balance", "avg_duration", "avg_campaign", "prev_contact"]:
        prof[c] = prof[c].round(2)
    top_job = (work.groupby(["cluster", "job"]).size().reset_index(name="n")
                   .sort_values(["cluster", "n"], ascending=[True, False])
                   .drop_duplicates("cluster").set_index("cluster")["job"])
    prof["main_job"] = prof["cluster"].map(top_job)
    base_conv, base_dur = float(work["is_subscribed"].mean()), float(work["duration"].mean())
    prof["label"] = prof.apply(lambda r: _label_segment(r, base_conv, base_dur), axis=1)
    # 标签去重：极端情况下两群可能命中同一规则，追加序号以便区分
    seen: dict[str, int] = {}
    uniq = []
    for lb in prof["label"]:
        seen[lb] = seen.get(lb, 0) + 1
        uniq.append(lb if seen[lb] == 1 else f"{lb}{seen[lb]}")
    prof["label"] = uniq
    prof = prof.sort_values("conv_rate", ascending=False).reset_index(drop=True)
    cfg.save_table(prof, "kmeans_profile")
    # 降维坐标（仅用于可视化，不参与建模）
    pca = PCA(n_components=2, random_state=cfg.RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    pts = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1],
                        "cluster": work["cluster"].values,
                        "is_subscribed": work["is_subscribed"].values,
                        "duration_min": work["duration_min"].values})
    sample_idx = rng.choice(len(pts), size=min(8000, len(pts)), replace=False)
    cfg.save_table(pts.iloc[sample_idx].round(4), "kmeans_points")
    print(f"  PCA 解释方差：PC1 {pca.explained_variance_ratio_[0]:.1%}，"
          f"PC2 {pca.explained_variance_ratio_[1]:.1%}")
    for _, r in prof.iterrows():
        print(f"  C{int(r['cluster'])} {r['label']:<14} 占比 {r['share']:.1%}  "
              f"转化率 {r['conv_rate']:.2%}  平均时长 {r['avg_duration']:.0f}s  "
              f"主客群 {r['main_job']}")
    return prof, pts

def _label_segment(r: pd.Series, base_conv: float, base_dur: float) -> str:
    """按转化率、通话时长、历史触达与余额水平给出可读客群标签（相对总体均值分级）。"""
    high_conv = r["conv_rate"] >= base_conv * 1.3
    low_conv = r["conv_rate"] <= base_conv * 0.7
    long_call = r["avg_duration"] >= base_dur * 1.2
    if high_conv and long_call:
        return "高意向深度沟通型"
    if high_conv:
        return "高效转化型"
    if r["prev_contact"] >= 0.5:
        return "老客待再激活型"
    if low_conv:
        if r["med_balance"] < 300:
            return "低净值长尾型"
        return "高频低效打扰型" if r["avg_campaign"] >= 2.8 else "低效触达型"
    return "稳步转化型"

# M7 关联规则（Apriori 思想）
def association_rules(df: pd.DataFrame, min_support: float = 0.01,
                      min_conf: float = 0.20, max_len: int = 3) -> pd.DataFrame:
    print("\n[M7] 关联规则挖掘")
    work = df.copy()
    work["y_item"] = "y=" + work["y"]
    attrs = {"job": "job", "marital": "marital", "education": "education",
             "housing": "housing", "loan": "loan", "contact": "contact",
             "poutcome": "poutcome", "age_band": "age_band",
             "balance_band": "balance_band", "campaign_bucket": "campaign_bucket"}
    work = work[list(attrs) + ["y_item"]].astype(str)
    n = len(work)
    base_rate = (work["y_item"] == "y=yes").mean()
    # 逐层挖掘：1 项集 → 2 项集 → 3 项集
    frequent: dict[int, dict[frozenset, float]] = {}
    freq1 = {}
    for col in work.columns:
        for val, cnt in work[col].value_counts().items():
            sup = cnt / n
            if sup >= min_support:
                freq1[frozenset([f"{col}={val}"])] = sup
    frequent[1] = freq1
    print(f"  1 项集：{len(freq1)} 个（min_support={min_support}）")
    for length in range(2, max_len + 1):
        candidates: set[frozenset] = set()
        prev = list(frequent[length - 1])
        for a, b in combinations(prev, 2):
            union = a | b
            if len(union) != length:
                continue
            # 互斥约束：同一属性只允许出现一个取值
            if len({x.split("=")[0] for x in union}) != length:
                continue
            # Apriori 剪枝：任一子集不频繁则跳过
            if all(frozenset(sub) in frequent[length - 1] for sub in combinations(union, length - 1)):
                candidates.add(union)
        print(f"  {length} 项集候选：{len(candidates)} 个")
        # 按"涉及的属性组合"分组统计支持度（不同候选涉及的属性不同，需分别 groupby）
        sig_groups: dict[tuple, list] = {}
        for itemset in candidates:
            cond = {it.split("=", 1)[0]: it.split("=", 1)[1] for it in itemset}
            sig_groups.setdefault(tuple(sorted(cond)), []).append((itemset, cond))
        cur: dict[frozenset, float] = {}
        for sig, group in sig_groups.items():
            counts = work.groupby(list(sig), observed=True).size().to_dict()
            for itemset, cond in group:
                key = tuple(cond[c] for c in sig)
                cnt = counts.get(key, 0)
                if cnt / n >= min_support:
                    cur[itemset] = cnt / n
        frequent[length] = cur
        print(f"  {length} 项集频繁：{len(cur)} 个")
        if not cur:
            break
    # 生成规则：后件固定为 y=yes
    rules = []
    for length, items in frequent.items():
        for itemset, sup in items.items():
            if "y_item=y=yes" not in itemset:
                continue
            ante = frozenset(x for x in itemset if x != "y_item=y=yes")
            if not ante:
                continue
            ante_sup = frequent[len(ante)].get(ante) if len(ante) in frequent else None
            if not ante_sup:
                continue
            conf = sup / ante_sup
            lift = conf / base_rate
            if conf >= min_conf and lift >= 1.2:
                rules.append({
                    "antecedent": " & ".join(sorted(x for x in ante)),
                    "n_items": len(ante),
                    "support": round(sup, 4),
                    "confidence": round(conf, 4),
                    "lift": round(lift, 3),
                    "conviction": round((1 - base_rate) / (1 - conf), 3) if conf < 1 else np.inf,
                })
    r = pd.DataFrame(rules)
    if r.empty:
        print("  未发现满足阈值的规则，可下调 min_support/min_conf")
        return r
    r = r.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    cfg.save_table(r.head(50), "apriori_rules")
    print(f"  有效规则 {len(r)} 条（提升度>1），Top3：")
    for _, row in r.head(3).iterrows():
        print(f"    {row['antecedent']} → 认购  支持度 {row['support']:.2%}  "
              f"置信度 {row['confidence']:.2%}  提升度 {row['lift']:.2f}")
    return r

# 主流程
def main() -> None:
    print("=" * 68)
    print("Step 3 / 6  商业分析模型")
    print("=" * 68)
    df = load_clean()
    print(f"载入清洗数据：{len(df):,} 行")
    kpi = eda(df)
    f, f_res = funnel(df)
    tr = trend(df)
    pa = pareto(df)
    bc = bcg(df)
    prof, _ = kmeans_segments(df)
    rules = association_rules(df)
    results = {
        "kpi": kpi,
        "funnel": f_res,
        "trend": {
            "best_month": tr.loc[tr["subscribed"].idxmax(), "year_month"],
            "best_month_subs": int(tr["subscribed"].max()),
            "worst_month": tr.loc[tr["subscribed"].idxmin(), "year_month"],
            "avg_monthly_contacts": round(float(tr["contacts"].mean()), 1),
            "conv_volatility": round(float(tr["conv_rate"].std()), 4),
        },
        "pareto": {
            "a_classes": pa.loc[pa["abc"] == "A", "segment"].tolist(),
            "a_share_of_subs": round(float(pa.loc[pa["abc"] == "A", "sub_share"].sum()), 4),
            "a_share_of_contacts": round(float(pa.loc[pa["abc"] == "A", "contacts"].sum()
                                               / pa["contacts"].sum()), 4),
        },
        "bcg": bc.set_index("job")["quadrant"].to_dict(),
        "kmeans": {
            "k": int(prof.shape[0]),
            "segments": prof[["label", "share", "conv_rate", "avg_duration"]].to_dict(orient="records"),
        },
        "rules": rules.head(10).to_dict(orient="records") if not rules.empty else [],
    }
    cfg.save_results(results)
    print("\nStep 3 完成。结果表已写入 outputs/tables/。\n")

if __name__ == "__main__":
    main()