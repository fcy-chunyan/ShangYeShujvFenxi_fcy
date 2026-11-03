"""
05_visuals.py —— 全部静态图表
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
PC = cfg.COLORS
PAL = cfg.PALETTE

# 通用工具
def pct(x, _=None):
    return f"{x:.0%}"
def money(x, _=None):
    return f"{x:,.0f}"
def label_bars(ax, bars, fmt="{:.0f}", offset=0.02, color="#374151", fontsize=9):
    """在柱形末端标注数值（横向/纵向通用）。"""
    for b in bars:
        w, h = b.get_width(), b.get_height()
        if h == 0:
            continue
        if w < h:   # 竖向柱
            ax.text(b.get_x() + w / 2, h + offset * max(h, 1), fmt.format(h),
                    ha="center", va="bottom", fontsize=fontsize, color=color)
        else:       # 横向柱
            ax.text(w + offset * max(w, 1), b.get_y() + h / 2, fmt.format(w),
                    va="center", ha="left", fontsize=fontsize, color=color)

def binary_treemap(values, x, y, w, h, horizontal=True):
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [(x, y, w, h)]
    total = sum(values)
    acc, split = 0, 1
    for i, v in enumerate(values[:-1]):
        acc += v
        if acc >= total / 2:
            split = i + 1
            break
    left, right = values[:split], values[split:]
    ratio = sum(left) / total
    rects = []
    if horizontal:
        wl = w * ratio
        rects += binary_treemap(left, x, y, wl, h, not horizontal)
        rects += binary_treemap(right, x + wl, y, w - wl, h, not horizontal)
    else:
        hl = h * ratio
        rects += binary_treemap(left, x, y, w, hl, not horizontal)
        rects += binary_treemap(right, x, y + hl, w, h - hl, not horizontal)
    return rects

def load_all():
    df = pd.read_csv(cfg.DATA_CLEAN / "bank_clean.csv", parse_dates=["contact_date", "week"])
    tab = {p.stem: pd.read_csv(p) for p in cfg.OUT_TAB.glob("*.csv")}
    return df, tab

# A. 模板复现
def fig01_combo(df, tab):
    """组合图：柱形（触达量）+ 折线（转化率），双坐标轴。"""
    m = tab["trend_monthly"]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(m))
    bars = ax.bar(x, m["contacts"], color=PC["light"], edgecolor=PC["primary"],
                  linewidth=0.8, label="触达人次", width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(m["year_month"], rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("触达人次（人）")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax2 = ax.twinx()
    ax2.plot(x, m["conv_rate"], color=PC["accent"], marker="o", markersize=4.5,
             linewidth=2.2, label="转化率", zorder=5)
    ax2.set_ylabel("认购转化率（%）")
    ax2.yaxis.set_major_formatter(FuncFormatter(pct))
    ax2.set_ylim(0, max(m["conv_rate"]) * 1.25)
    ax2.grid(False)
    # 图例合并到一处，避免双轴图例分散
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=9)
    cfg.style_axes(ax, title="月度触达量与认购转化率（2008-05—2010-11）",
                   xlabel="月份", ylabel="触达人次（人）")
    ax.text(0.01, -0.28, "注：柱形为当月触达人次，折线为当月认购转化率；2010 年外呼量收缩但转化率显著抬升。",
            transform=ax.transAxes, fontsize=8.5, color=PC["gray"])
    cfg.save_fig(fig, "fig01_月度触达与转化_组合图")

def fig02_clustered(df, tab):
    """簇状柱形图：各职业按婚姻状况分组的转化率。"""
    g = (df.groupby(["job", "marital"])["is_subscribed"].mean().unstack()
           .loc[df.groupby("job")["is_subscribed"].size().sort_values(ascending=False).index])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(g))
    w = 0.26
    for i, col in enumerate(["married", "single", "divorced"]):
        name = {"married": "已婚", "single": "单身", "divorced": "离异"}[col]
        ax.bar(x + (i - 1) * w, g[col], width=w, label=name, color=PAL[i],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.legend(title="婚姻状况", ncol=3, loc="upper right")
    cfg.style_axes(ax, xlabel="职业", ylabel="认购转化率（%）",
                   title="各职业分婚姻状况的转化率（簇状柱形图）")
    cfg.save_fig(fig, "fig02_职业婚姻转化率_簇状柱形图")

def fig03_barh(df, tab):
    """条形图：转化率 Top10 职业，附总体均值参考线。"""
    t = (df.groupby("job")["is_subscribed"].agg(["mean", "size"])
           .rename(columns={"mean": "conv", "size": "n"}))
    t = t[t["n"] >= 200].sort_values("conv").tail(10)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    colors = [PC["primary"] if v >= df["is_subscribed"].mean() else PC["light"] for v in t["conv"]]
    bars = ax.barh(t.index, t["conv"], color=colors, edgecolor=PC["dark"], linewidth=0.6, height=0.62)
    ax.axvline(df["is_subscribed"].mean(), color=PC["accent"], linestyle="--", linewidth=1.6,
               label=f"总体均值 {df['is_subscribed'].mean():.1%}")
    label_bars(ax, bars, fmt="{:.1%}", offset=0.004, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlim(0, t["conv"].max() * 1.22)
    ax.legend(loc="lower right")
    cfg.style_axes(ax, xlabel="认购转化率（%）", ylabel="", title="转化率 Top10 职业（条形图）",
                   grid_axis="x")
    cfg.save_fig(fig, "fig03_职业转化率Top10_条形图")

def fig04_stacked(df, tab):
    """100% 堆积柱形图：各学历层次的认购结构。"""
    g = (df.groupby(["education", "y"]).size().unstack(fill_value=0))
    g = g.loc[["primary", "secondary", "tertiary", "unknown"]]
    share = g.div(g.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(share))
    b1 = ax.bar(x, share["no"], color=PC["gray"], label="未认购", width=0.55)
    b2 = ax.bar(x, share["yes"], bottom=share["no"], color=PC["green"], label="已认购", width=0.55)
    for i, (no, yes) in enumerate(zip(share["no"], share["yes"])):
        ax.text(i, no / 2, f"{no:.1%}", ha="center", va="center", color="white", fontsize=9.5)
        ax.text(i, no + yes / 2, f"{yes:.1%}", ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["小学/初中", "高中/中专", "高等教育", "未知"])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.legend(loc="lower right", ncol=2)
    cfg.style_axes(ax, xlabel="学历层次", ylabel="占比（%）",
                   title="各学历层次的认购结构（100% 堆积柱形图）")
    cfg.save_fig(fig, "fig04_学历认购结构_堆积柱形图")

def fig05_donut(df, tab):
    """圆环图：认购客户的渠道来源结构。"""
    s = df.loc[df["is_subscribed"] == 1, "channel"].value_counts()
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    wedges, texts, autotexts = ax.pie(
        s.values, labels=s.index, autopct=lambda p: f"{p:.1f}%",
        colors=[PC["primary"], PC["accent"], PC["gray"]], startangle=90,
        pctdistance=0.78, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11})
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.text(0, 0.08, f"{s.sum():,}", ha="center", va="center", fontsize=22,
            fontweight="bold", color=PC["dark"])
    ax.text(0, -0.14, "认购客户数", ha="center", va="center", fontsize=10, color=PC["gray"])
    ax.set_title("认购客户的渠道来源结构（圆环图）", pad=14)
    cfg.save_fig(fig, "fig05_认购渠道结构_圆环图")

def fig06_area(df, tab):
    """面积图：各渠道月度认购量的累计走势。"""
    m = (df[df["is_subscribed"] == 1].groupby(["year_month", "channel"]).size()
           .unstack(fill_value=0).cumsum())
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(m))
    ax.stackplot(x, [m[c] for c in m.columns], labels=m.columns,
                 colors=[PC["primary"], PC["accent"], PC["gray"]], alpha=0.85)
    step = 3
    ax.set_xticks(x[::step])
    ax.set_xticklabels(m.index[::step], rotation=45, ha="right", fontsize=8.5)
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(loc="upper left", title="渠道")
    cfg.style_axes(ax, xlabel="月份", ylabel="累计认购客户数（人）",
                   title="分渠道累计认购走势（面积图）")
    cfg.save_fig(fig, "fig06_累计认购走势_面积图")

def fig07_scatter(df, tab):
    """散点图：年龄与通话时长的关系（按是否认购着色）。"""
    s = df.sample(n=min(6000, len(df)), random_state=cfg.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    no = s[s["is_subscribed"] == 0]
    yes = s[s["is_subscribed"] == 1]
    ax.scatter(no["age"], no["duration_min"], s=10, c=PC["gray"], alpha=0.28, label="未认购", linewidths=0)
    ax.scatter(yes["age"], yes["duration_min"], s=14, c=PC["green"], alpha=0.65, label="已认购", linewidths=0)
    ax.set_ylim(0, 25)
    ax.legend(markerscale=2, loc="upper right")
    ax.axhline(df["duration_min"].median(), color=PC["accent"], linestyle="--", linewidth=1.2)
    ax.text(df["age"].max() - 1, df["duration_min"].median() + 0.5,
            f"时长中位数 {df['duration_min'].median():.1f} 分钟", ha="right",
            fontsize=9, color=PC["accent"])
    cfg.style_axes(ax, xlabel="年龄（岁）", ylabel="通话时长（分钟）",
                   title="年龄与通话时长分布（散点图，随机抽样 6,000 条）", grid_axis="both")
    cfg.save_fig(fig, "fig07_年龄与通话时长_散点图")

def fig08_radar(df, tab):
    """雷达图：主要职业客群的多维特征对比（归一化后）。"""
    jobs = ["management", "technician", "blue-collar", "admin.", "retired", "services"]
    metrics = {"转化率": "is_subscribed", "平均时长": "duration", "平均余额": "balance",
               "触达规模": "is_subscribed", "平均年龄": "age"}
    raw = pd.DataFrame({
        "转化率": df.groupby("job")["is_subscribed"].mean(),
        "平均时长": df.groupby("job")["duration"].mean(),
        "平均余额": df.groupby("job")["balance"].mean(),
        "触达规模": df.groupby("job").size(),
        "平均年龄": df.groupby("job")["age"].mean(),
    }).loc[jobs]
    norm = (raw - raw.min()) / (raw.max() - raw.min())
    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.4, 7.0), subplot_kw=dict(polar=True))
    for i, job in enumerate(jobs):
        vals = norm.loc[job].tolist() + [norm.loc[job].iloc[0]]
        ax.plot(angles, vals, linewidth=2, label=job, color=PAL[i])
        ax.fill(angles, vals, color=PAL[i], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=8, color=PC["gray"])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=9)
    ax.set_title("主要职业客群多维特征对比（雷达图，Min-Max 归一化）", pad=24)
    cfg.save_fig(fig, "fig08_客群多维对比_雷达图")

def fig09_treemap(df, tab):
    """矩形树图：职业对认购量的贡献结构。"""
    s = df[df["is_subscribed"] == 1].groupby("job").size().sort_values(ascending=False)
    rects = binary_treemap(s.values.tolist(), 0, 0, 100, 100, True)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, (r, (name, val)) in enumerate(zip(rects, s.items())):
        x, y, w, h = r
        ax.add_patch(Rectangle((x, y), w, h, facecolor=PAL[i % len(PAL)],
                               edgecolor="white", linewidth=2, alpha=0.9))
        # 面积过小的矩形不写字，避免文字溢出（具体数值见 outputs/tables/eda_job.csv）
        if w * h > 90 and w > 7 and h > 6:
            fs = min(13, max(8, (w * h) ** 0.22))
            ax.text(x + w / 2, y + h / 2, f"{name}\n{val:,}", ha="center", va="center",
                    color="white", fontsize=fs, fontweight="bold", linespacing=1.5,
                    clip_on=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("职业客群的认购贡献结构（矩形树图，面积=认购人数）", pad=12)
    cfg.save_fig(fig, "fig09_职业认购贡献_矩形树图")

def fig10_waterfall(df, tab):
    """瀑布图：从触达到认购的逐层流失拆解。"""
    f = tab["funnel_overall"].reset_index(drop=True)
    drop_names = ["未接通/秒挂", "沟通不足（<120s）", "深度沟通不足（<300s）", "深度沟通未转化"]
    steps, values, kinds = [], [], []
    for i in range(len(f)):
        steps.append(f.loc[i, "stage"]); values.append(int(f.loc[i, "users"])); kinds.append("total")
        if i < len(f) - 1:
            steps.append(drop_names[i])
            values.append(-int(f.loc[i, "users"] - f.loc[i + 1, "users"]))
            kinds.append("drop")
    fig, ax = plt.subplots(figsize=(12, 5.6))
    cum = 0
    for i, (s, v, k) in enumerate(zip(steps, values, kinds)):
        if k == "total":
            color = PC["green"] if s == "认购成功" else PC["primary"]
            ax.bar(i, v, bottom=0, color=color, width=0.6)
            ax.text(i, v + 800, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
            cum = v
        else:
            ax.bar(i, v, bottom=cum, color=PC["red"], width=0.6)
            ax.text(i, cum + 800, f"{v:,}", ha="center", fontsize=9, color=PC["red"])
            cum = cum + v
        if i < len(steps) - 1:
            ax.plot([i + 0.3, i + 0.7], [cum, cum], color=PC["gray"], linewidth=1, linestyle=":")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=28, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    cfg.style_axes(ax, ylabel="人数（人）", title="营销漏斗逐层流失拆解（瀑布图）")
    cfg.save_fig(fig, "fig10_转化流失拆解_瀑布图")

def fig11_heatmap(df, tab):
    """热力图：月份 × 职业的转化率（样本量 <100 的单元格留空）。"""
    import seaborn as sns
    g = df.groupby(["job", "month_num"])["is_subscribed"].agg(["mean", "size"])
    conv = g["mean"].unstack()
    size = g["size"].unstack()
    conv = conv.where(size >= 100)
    order = df.groupby("job")["is_subscribed"].size().sort_values(ascending=False).index
    conv = conv.loc[order]
    cn = list(cfg.MONTH_CN.values())          # 按 1—12 月顺序排列
    conv.columns = [cn[int(k) - 1] for k in conv.columns]
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    sns.heatmap(conv, annot=True, fmt=".0%", cmap="YlGnBu", linewidths=2, linecolor="white",
                cbar_kws={"label": "认购转化率"}, ax=ax, annot_kws={"fontsize": 8})
    ax.set_xlabel("月份")
    ax.set_ylabel("职业")
    ax.set_title("月份 × 职业转化率热力图（样本量 <100 的组合留空）", pad=12)
    cfg.save_fig(fig, "fig11_月份职业转化_热力图")

def fig12_box(df, tab):
    """箱线图：认购与未认购客户的通话时长分布。"""
    import seaborn as sns
    d = df.copy()
    d["认购结果"] = np.where(d["is_subscribed"] == 1, "已认购", "未认购")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    # 显式传入 hue 并关闭图例，与 seaborn 0.14 的新签名保持一致
    sns.boxplot(data=d, x="认购结果", y="duration_min", hue="认购结果", ax=ax, width=0.42,
                palette={"已认购": PC["green"], "未认购": PC["gray"]}, showfliers=False,
                legend=False, medianprops={"color": "white", "linewidth": 2})
    for i, grp in enumerate(["未认购", "已认购"]):
        med = d.loc[d["认购结果"] == grp, "duration_min"].median()
        ax.text(i, med + 0.4, f"中位数 {med:.1f} 分钟", ha="center", fontsize=9.5,
                color="#374151", fontweight="bold")
    ax.set_ylim(0, 14)
    cfg.style_axes(ax, xlabel="", ylabel="通话时长（分钟）",
                   title="认购与未认购客户的通话时长分布（箱线图，隐藏离群点）")
    cfg.save_fig(fig, "fig12_通话时长分布_箱线图")

def fig13_rose(df, tab):
    """玫瑰图：12 个月的触达量季节分布。"""
    m = df.groupby("month_num").size().reindex(range(1, 13), fill_value=0)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw=dict(polar=True))
    bars = ax.bar(angles, m.values, width=2 * np.pi / 12 * 0.92,
                  color=[PAL[i % len(PAL)] for i in range(12)], edgecolor="white", linewidth=1.2)
    ax.set_xticks(angles)
    ax.set_xticklabels([cfg.MONTH_CN[k] for k in cfg.MONTH_NUM], fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("月度触达量季节分布（玫瑰图）", pad=26)
    for ang, v in zip(angles, m.values):
        ax.text(ang, v + 700, f"{v:,}", ha="center", va="center", fontsize=8.5, color="#374151")
    cfg.save_fig(fig, "fig13_月度触达季节分布_玫瑰图")

def fig14_stackarea(df, tab):
    """堆叠面积图：季度渠道结构占比变化。"""
    mix = tab["channel_mix_quarter"]
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    x = np.arange(len(mix))
    ax.stackplot(x, [mix[c] for c in ["手机", "座机", "未知"]], labels=["手机", "座机", "未知"],
                 colors=[PC["primary"], PC["accent"], PC["gray"]], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(mix["quarter"], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.legend(loc="upper right", ncol=3, title="联系方式")
    cfg.style_axes(ax, xlabel="季度", ylabel="触达占比（%）",
                   title="分季度渠道结构变化（堆叠面积图）")
    cfg.save_fig(fig, "fig14_渠道结构演变_堆叠面积图")

def fig15_bullet(df, tab):
    """子弹图：核心 KPI 与行业基准/目标的对比。"""
    kpis = [
        ("认购转化率", df["is_subscribed"].mean(), 0.15, 0.10),
        ("有效沟通率(≥120s)", (df["duration"] >= 120).mean(), 0.80, 0.65),
        ("深度沟通率(≥300s)", (df["duration"] >= 300).mean(), 0.35, 0.27),
        ("老客再触达率", df["had_previous_contact"].mean(), 0.25, 0.18),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (name, val, target, base) in zip(axes, kpis):
        # 每个指标单独设定量程，避免"转化率 11.7% 对 100% 量程"时柱体过短
        xmax = min(1.0, max(target, val) * 1.28)
        ax.barh([0], [xmax], color="#EEF2F7", height=0.62)
        ax.barh([0], [base], color=PC["light"], height=0.40)
        ax.barh([0], [val], color=PC["primary"], height=0.22)
        ax.plot([target, target], [-0.34, 0.34], color=PC["accent"], linewidth=3)
        ax.text(target, 0.40, f"目标 {target:.0%}", ha="center", fontsize=9, color=PC["accent"])
        ax.text(xmax * 0.02, -0.50, f"实际 {val:.1%}", fontsize=11.5, fontweight="bold",
                color=PC["dark"])
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.68, 0.62)
        ax.set_yticks([])
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_xticklabels(["0", f"{xmax/2:.0%}", f"{xmax:.0%}"], fontsize=8)
        ax.set_title(name, fontsize=11)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(False)
    fig.suptitle("核心运营指标达成情况（子弹图：深色=实际，浅色=行业基准，竖线=目标）",
                 fontsize=13, fontweight="bold", y=1.02)
    cfg.save_fig(fig, "fig15_核心指标达成_子弹图")

# B. 业务模型图
def fig16_funnel_by_channel(df, tab):
    """分渠道漏斗图（业务模型图 1，对应 M2）。"""
    f = tab["funnel_by_channel"].set_index("channel")
    stages = ["contacts", "有效接通", "有效沟通", "深度沟通", "认购成功"]
    stage_cn = ["触达名单", "有效接通", "有效沟通", "深度沟通", "认购成功"]
    channels = ["手机", "座机", "未知"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    for ax, ch in zip(axes, channels):
        vals = [f.loc[ch, s] for s in stages]
        maxv = vals[0]
        for i, (v, name) in enumerate(zip(vals, stage_cn)):
            half = max(v / maxv / 2 * 0.92, 0.105)   # 设最小宽度，保证末层文字不溢出
            top, bot = 1 - i * 0.17, 1 - (i + 1) * 0.17 + 0.02
            poly = Polygon([(0.5 - half, top), (0.5 + half, top),
                            (0.5 + half * 0.96, bot), (0.5 - half * 0.96, bot)],
                           facecolor=PAL[i], edgecolor="white", linewidth=1.4, alpha=0.92)
            ax.add_patch(poly)
            ax.text(0.5, (top + bot) / 2, f"{name}\n{v:,}", ha="center", va="center",
                    color="white", fontsize=9.5 if half < 0.13 else 10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.06)
        ax.axis("off")
        ax.set_title(f"{ch}渠道（转化率 {f.loc[ch,'conv_rate']:.1%}）", fontsize=11.5)
    fig.suptitle("分渠道营销漏斗对比（漏斗图）", fontsize=13.5, fontweight="bold", y=1.0)
    cfg.save_fig(fig, "fig16_分渠道营销漏斗_漏斗图")

def fig17_trend(df, tab):
    """环比/同比与移动平均（业务模型图 2，对应 M3）。"""
    m = tab["trend_monthly"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(len(m))
    ax = axes[0]
    ax.bar(x, m["contacts"], color=PC["light"], edgecolor=PC["primary"], linewidth=0.7, label="触达人次")
    ax.plot(x, m["contacts_ma3"], color=PC["accent"], linewidth=2.2, label="3 期移动平均")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(loc="upper right", ncol=2)
    cfg.style_axes(ax, ylabel="触达人次（人）", title="触达量的环比/同比与移动平均（业务模型 M3）")
    ax2 = axes[1]
    colors = [PC["green"] if v >= 0 else PC["red"] for v in m["conv_mom_pp"].fillna(0)]
    ax2.bar(x, m["conv_mom_pp"].fillna(0), color=colors, alpha=0.55, label="转化率环比变化（pp）")
    ax2.plot(x, m["conv_rate"], color=PC["dark"], marker="o", markersize=3.6, linewidth=1.8,
             label="当月转化率")
    ax2.plot(x, m["conv_ma3"], color=PC["accent"], linewidth=2, linestyle="--", label="3 期移动平均")
    ax2.axhline(0, color=PC["gray"], linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(m["year_month"], rotation=45, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(FuncFormatter(pct))
    ax2.legend(loc="upper left", ncol=3, fontsize=8.5)
    cfg.style_axes(ax2, xlabel="月份", ylabel="转化率（%）")
    cfg.save_fig(fig, "fig17_环比同比与移动平均_趋势图")

def fig18_bcg(df, tab):
    """波士顿矩阵四象限气泡图（业务模型图 3，对应 M5）。"""
    b = tab["bcg_job"]
    growth_cut = b["growth_pp"].median()
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    qcolor = {"明星": PC["primary"], "金牛": PC["green"], "问题": PC["accent"], "瘦狗": PC["gray"]}
    for q, g in b.groupby("quadrant"):
        ax.scatter(g["rel_share"], g["growth_pp"], s=g["subscribed"] * 0.9 + 40,
                   color=qcolor[q], alpha=0.75, edgecolor="white", linewidth=1.5, label=q)
    ax.axvline(1.0, color=PC["dark"], linestyle="--", linewidth=1.2)
    ax.axhline(growth_cut, color=PC["dark"], linestyle="--", linewidth=1.2)
    offsets = {
        "unemployed": (-14, 2, "right"), "entrepreneur": (14, 2, "left"),
        "services": (0, 16, "center"), "blue-collar": (-12, 18, "center"),
        "management": (-34, 14, "center"), "technician": (0, 16, "center"),
        "admin.": (0, 16, "center"), "self-employed": (0, 16, "center"),
        "student": (-16, 14, "center"), "housemaid": (16, -18, "center"),
        "retired": (0, 16, "center"), "unknown": (14, 8, "center"),
    }
    for _, r in b.iterrows():
        dx, dy, ha = offsets.get(r["job"], (0, 14, "center"))
        ax.annotate(r["job"], (r["rel_share"], r["growth_pp"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=8.8, color="#374151")

    ax.text(0.02, 0.96, "问题（低份额·高增速）", transform=ax.transAxes, fontsize=9.5,
            color=PC["accent"], va="top")
    ax.text(0.98, 0.96, "明星（高份额·高增速）", transform=ax.transAxes, fontsize=9.5,
            color=PC["primary"], va="top", ha="right")
    ax.text(0.02, 0.04, "瘦狗（低份额·低增速）", transform=ax.transAxes, fontsize=9.5,
            color=PC["gray"])
    ax.text(0.98, 0.04, "金牛（高份额·低增速）", transform=ax.transAxes, fontsize=9.5,
            color=PC["green"], ha="right")
    ax.set_xlim(0, max(b["rel_share"]) * 1.12)
    ax.set_ylim(min(b["growth_pp"]) - 5, max(b["growth_pp"]) + 5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=qcolor[q],
                      markersize=9, label=q) for q in ["明星", "金牛", "问题", "瘦狗"]]
    ax.legend(handles=handles, title="象限", loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, title_fontsize=9)
    cfg.style_axes(ax, xlabel="相对触达份额（客群占比 / 平均占比）",
                   ylabel="转化率同比变化（百分点）",
                   title="职业客群波士顿矩阵（气泡大小=认购人数）", grid_axis="both")
    ax.text(0.01, -0.16, f"注：增速分界取客群中位数 {growth_cut:.1f}pp（全部客群同比为正，以 0 为界会退化为单象限）。",
            transform=ax.transAxes, fontsize=8.5, color=PC["gray"])
    cfg.save_fig(fig, "fig18_客群波士顿矩阵_四象限气泡图")

def fig19_pareto(df, tab):
    """帕累托/ABC 分类图（业务模型图 4，对应 M4）。"""
    p = tab["pareto_segment"].head(20).copy()
    short = {"management": "Mgmt", "blue-collar": "Blue", "technician": "Tech",
             "admin.": "Admin", "services": "Serv", "retired": "Retired",
             "self-employed": "SelfEmp", "entrepreneur": "Entrep", "unemployed": "Unemp",
             "housemaid": "House", "student": "Student", "unknown": "Unk"}
    p["short"] = p["segment"].apply(
        lambda s: f"{short.get(s.split(' | ')[0], s.split(' | ')[0])} | {s.split(' | ')[1]}")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(p))
    cmap = {"A": PC["primary"], "B": PC["secondary"], "C": PC["gray"]}
    bars = ax.bar(x, p["subscribed"], color=[cmap[c] for c in p["abc"]],
                  edgecolor="white", linewidth=0.8, width=0.72, label="认购人数")
    ax.set_xticks(x)
    ax.set_xticklabels(p["short"], rotation=55, ha="right", fontsize=8.4)
    ax.set_ylabel("认购人数（人）")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax2 = ax.twinx()
    ax2.plot(x, p["cum_share"], color=PC["accent"], marker="o", markersize=3.4, linewidth=2)
    ax2.axhline(0.7, color=PC["red"], linestyle="--", linewidth=1)
    ax2.axhline(0.9, color=PC["gray"], linestyle=":", linewidth=1)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("累计认购占比（%）")
    ax2.yaxis.set_major_formatter(FuncFormatter(pct))
    ax2.grid(False)
    ax2.text(0.3, 0.715, "70% 分界（A/B）", ha="left", fontsize=8.5, color=PC["red"])
    ax2.text(0.3, 0.915, "90% 分界（B/C）", ha="left", fontsize=8.5, color=PC["gray"])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap[k], label=f"{k} 类") for k in "ABC"],
              loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, title="ABC 分类")
    cfg.style_axes(ax, xlabel="细分客群（职业 | 年龄段，按认购量降序，展示前 20 个）",
                   ylabel="认购人数（人）",
                   title="细分客群认购贡献帕累托图与 ABC 分类（业务模型 M4）")
    cfg.save_fig(fig, "fig19_客群贡献帕累托_ABC分类")

def fig20_kmeans(df, tab):
    """K-Means 分群 PCA 散点（业务模型图 5，对应 M6）。"""
    pts = tab["kmeans_points"].copy()
    prof = tab["kmeans_profile"]
    # 截取 1%—99% 分位范围，避免极少数离群点把主体压成一团
    for c in ("pc1", "pc2"):
        lo, hi = pts[c].quantile([0.01, 0.99])
        pts = pts[(pts[c] >= lo) & (pts[c] <= hi)]
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    for _, row in prof.iterrows():
        c = int(row["cluster"])
        g = pts[pts["cluster"] == c]
        ax.scatter(g["pc1"], g["pc2"], s=9, alpha=0.35, color=PAL[c % len(PAL)],
                   linewidths=0,
                   label=f"C{c} {row['label']}（转化率 {row['conv_rate']:.1%}）")
        ax.scatter(g["pc1"].mean(), g["pc2"].mean(), s=210, marker="X",
                   color=PAL[c % len(PAL)], edgecolor="white", linewidth=1.6, zorder=5)
        ax.annotate(f"C{c}", (g["pc1"].mean(), g["pc2"].mean()),
                    xytext=(9, 7), textcoords="offset points", fontsize=9.5,
                    fontweight="bold", color="#1F2937",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=PAL[c % len(PAL)], alpha=0.9))
    ax.set_xlabel("主成分 1（PC1）")
    ax.set_ylabel("主成分 2（PC2）")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.6, title="客群（k=5）")
    cfg.style_axes(ax, title="K-Means 客群分群结果（PCA 降维散点，截取 1%—99% 分位）",
                   grid_axis="both")
    cfg.save_fig(fig, "fig20_KMeans客群分群_PCA散点图")

def fig21_segment_radar(df, tab):
    """分群画像雷达图（业务模型图 6，对应 M6）。"""
    prof = tab["kmeans_profile"].copy()
    metrics = ["conv_rate", "avg_duration", "med_balance", "avg_campaign", "share"]
    labels = ["转化率", "通话时长", "余额中位数", "联系次数", "客群规模"]
    norm = prof[metrics].copy()
    for c in metrics:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - lo) / (hi - lo) if hi > lo else 0.5
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(norm.iterrows()):
        vals = row[metrics].tolist() + [row[metrics[0]]]
        ax.plot(angles, vals, linewidth=2, color=PAL[i], label=f"C{int(prof.iloc[i]['cluster'])} {prof.iloc[i]['label']}")
        ax.fill(angles, vals, color=PAL[i], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=8, color=PC["gray"])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8.5)
    ax.set_title("五类客群画像对比（雷达图，归一化）", pad=26)
    cfg.save_fig(fig, "fig21_客群画像对比_雷达图")

def fig22_rules(df, tab):
    """关联规则气泡图（业务模型图 7，对应 M7）。"""
    r = tab["apriori_rules"].copy()
    r = r.sort_values("lift", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    sc = ax.scatter(r["support"], r["confidence"], s=r["lift"] * 32,
                    c=r["lift"], cmap="YlOrRd", alpha=0.85, edgecolor="#7F1D1D", linewidth=0.7)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("提升度（Lift）")
    # 提升度最高的 3 条规则放在右下角空白区，避免与散点、标题相互遮挡
    top = r.head(3).reset_index(drop=True)
    lines = ["Top3 规则（后件=认购）"]
    for i, row in top.iterrows():
        ante = row["antecedent"].replace(" & ", " + ")
        lines.append(f"{i+1}) {ante}\n    提升度 {row['lift']:.2f}｜置信度 {row['confidence']:.1%}")
    ax.text(0.985, 0.04, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.2, color="#374151", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#D1D5DB", alpha=0.95))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    cfg.style_axes(ax, xlabel="支持度（Support）", ylabel="置信度（Confidence）",
                   title="关联规则支持度—置信度—提升度（气泡图，后件=认购）", grid_axis="both")
    ax.text(0.01, -0.15, "注：仅展示提升度 Top20 规则；气泡越大、颜色越深表示提升度越高。",
            transform=ax.transAxes, fontsize=8.5, color=PC["gray"])
    cfg.save_fig(fig, "fig22_关联规则_提升度气泡图")

def fig23_sarima(df, tab):
    """SARIMA 预测与 95% 置信区间（业务模型图 8，对应 M4 预测）。"""
    f = tab["forecast_weekly"].copy()
    f["week"] = pd.to_datetime(f["week"])
    hist = f[f["type"] == "history"]
    fut = f[f["type"] == "forecast"]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(hist["week"], hist["forecast"], color=PC["primary"], linewidth=1.3, label="历史周认购量")
    ax.plot(fut["week"], fut["forecast"], color=PC["accent"], linewidth=2.4, marker="o",
            markersize=4, label="预测值")
    ax.fill_between(fut["week"], fut["lower_95"], fut["upper_95"], color=PC["accent"],
                    alpha=0.22, label="95% 置信区间")
    ax.axvline(fut["week"].min(), color=PC["gray"], linestyle="--", linewidth=1.2)
    ax.annotate("预测起点", xy=(fut["week"].min(), 0.80), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color=PC["gray"],
                rotation=90, va="center")
    m = tab["forecast_metrics"].set_index("model")
    sarima_name = [i for i in m.index if i.startswith("SARIMA")][0]
    ax.text(0.02, 0.95, f"{sarima_name} 回测：MAPE {m.loc[sarima_name,'MAPE_%']}%  "
                        f"RMSE {m.loc[sarima_name,'RMSE']}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PC["light"]))
    ax.set_ylabel("周认购量（人）")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    cfg.style_axes(ax, xlabel="周（以周一为起点）", ylabel="周认购量（人）",
                   title="周认购量 SARIMA 预测与 95% 置信区间（业务模型 M4-预测）")
    cfg.save_fig(fig, "fig23_SARIMA预测与置信区间")

def fig24_decompose(df, tab):
    """月度序列季节性分解（业务模型图 9）。"""
    d = tab["seasonal_decompose"].copy()
    d["month"] = pd.to_datetime(d["month"])
    fig, axes = plt.subplots(4, 1, figsize=(11.2, 8.4), sharex=True)
    items = [("observed", "观测值", PC["primary"]),
             ("trend", "趋势项", PC["dark"]),
             ("seasonal", "季节项", PC["accent"]),
             ("resid", "残差项", PC["gray"])]
    for ax, (col, name, color) in zip(axes, items):
        ax.plot(d["month"], d[col], color=color, linewidth=1.8)
        if col == "resid":
            ax.axhline(0, color=PC["gray"], linewidth=0.9)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(axis="y", color="#EEF2F7")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("月份")
    axes[0].set_title("月度认购量加性分解（观测 = 趋势 + 季节 + 残差）", pad=12)
    cfg.save_fig(fig, "fig24_月度序列季节性分解")

def main():
    print("=" * 68)
    print("Step 5 / 6  图表生成")
    print("=" * 68)
    df, tab = load_all()
    need = ["trend_monthly", "funnel_overall", "funnel_by_channel", "channel_mix_quarter",
            "bcg_job", "pareto_segment", "kmeans_profile", "kmeans_points",
            "apriori_rules", "forecast_weekly", "forecast_metrics", "seasonal_decompose"]
    missing = [n for n in need if n not in tab]
    if missing:
        raise SystemExit(f"缺少分析结果表：{missing}，请先运行 03_analysis.py 与 04_forecast.py")
    print("— 模板复现 —")
    fig01_combo(df, tab); fig02_clustered(df, tab); fig03_barh(df, tab)
    fig04_stacked(df, tab); fig05_donut(df, tab); fig06_area(df, tab)
    fig07_scatter(df, tab); fig08_radar(df, tab); fig09_treemap(df, tab)
    fig10_waterfall(df, tab); fig11_heatmap(df, tab); fig12_box(df, tab)
    fig13_rose(df, tab); fig14_stackarea(df, tab); fig15_bullet(df, tab)
    print("— 业务模型图 —")
    fig16_funnel_by_channel(df, tab); fig17_trend(df, tab); fig18_bcg(df, tab)
    fig19_pareto(df, tab); fig20_kmeans(df, tab); fig21_segment_radar(df, tab)
    fig22_rules(df, tab); fig23_sarima(df, tab); fig24_decompose(df, tab)
    print(f"\n共生成 {len(list(cfg.OUT_FIG.glob('*.png')))} 张图表，目录：{cfg.OUT_FIG}")
    print("Step 5 完成。\n")

if __name__ == "__main__":
    main()