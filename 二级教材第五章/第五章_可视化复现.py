from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\2《CDA 二级认证教材：量化策略分析》\第5章-数据可视化"
)
OUTPUT_DIR = BASE_DIR / "复现结果" / "量化策略可视化"

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})

def save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_stock(data_dir: Path, output_dir: Path) -> int:
    stock = pd.read_excel(data_dir / "2017股价.xlsx")
    required = ["日期", "收盘价", "开盘价", "最高价", "最低价"]
    missing = set(required).difference(stock.columns)
    if missing:
        raise ValueError(f"股价表缺少字段: {', '.join(sorted(missing))}")
    stock["日期"] = pd.to_datetime(stock["日期"])
    stock = stock.sort_values("日期").reset_index(drop=True)
    save_table(stock, output_dir, "股价_原始数据")
    fig, ax = plt.subplots(figsize=(10, 4))
    series = [("收盘价", "-*"), ("开盘价", "-o"), ("最高价", "-v"), ("最低价", "-^")]
    for column, marker in series:
        ax.plot(range(len(stock)), stock[column], marker, label=column)
    ax.set(title="股价走势图", xlabel="时间", ylabel="股价")
    ax.set_xticks(range(len(stock)), stock["日期"].dt.strftime("%Y-%m-%d"), rotation=30, ha="right")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "图表_股价线图")
    return 1

def plot_basic_examples(output_dir: Path) -> int:
    charts = 0
    rng = np.random.RandomState(1)
    votes = pd.DataFrame({"方案": ["plan A", "plan B", "plan C", "plan D", "plan E"], "投票人数": rng.randint(10, 100, 5)})
    save_table(votes, output_dir, "投票方案_数据")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(votes["投票人数"], labels=votes["方案"], autopct="%.2f%%", textprops={"fontsize": 12})
    ax.set_title("方案投票占比饼图")
    save_figure(fig, output_dir, "图表_投票饼图")
    charts += 1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(votes["方案"], votes["投票人数"], width=0.6, color="#277DA1")
    ax.set(title="各方案投票人数柱形图", xlabel="方案", ylabel="投票人数")
    save_figure(fig, output_dir, "图表_投票柱形图")
    charts += 1
    sorted_votes = votes.sort_values("投票人数")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_votes["方案"], sorted_votes["投票人数"], height=0.6, color="#43AA8B")
    ax.set(title="各方案投票人数条形图", xlabel="投票人数", ylabel="方案")
    save_figure(fig, output_dir, "图表_投票条形图")
    charts += 1
    rng = np.random.RandomState(1)
    scores = rng.normal(loc=80, scale=20, size=200).astype(int)
    save_table(pd.DataFrame({"分数": scores}), output_dir, "直方图_分数数据")
    bins = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=bins, color="#F8961E", edgecolor="white")
    ax.set_xticks(bins)
    ax.set(title="分数分布直方图", xlabel="分数区间", ylabel="频数")
    save_figure(fig, output_dir, "图表_分数直方图")
    charts += 1
    production = pd.DataFrame({"产量": [2200, 3300, 4400, 5500, 5700, 7300, 8000], "降水量": [22, 39, 55, 66, 105, 100, 122]})
    save_table(production, output_dir, "散点图_产量降水数据")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(production["降水量"], production["产量"], color="#E76F51", s=55)
    ax.set(title="降水量与产量散点图", xlabel="降水量", ylabel="产量")
    save_figure(fig, output_dir, "图表_产量降水散点图")
    charts += 1
    rng = np.random.RandomState(1)
    scores_by_subject = pd.DataFrame({"语文": rng.randint(60, 150, 10), "英语": rng.randint(40, 150, 10), "数学": rng.randint(20, 150, 10)})
    save_table(scores_by_subject, output_dir, "箱线图_学科成绩数据")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([scores_by_subject[column] for column in ("语文", "英语", "数学")], tick_labels=["语文", "英语", "数学"])
    ax.set(title="各学科成绩箱线图", ylabel="成绩")
    save_figure(fig, output_dir, "图表_学科成绩箱线图")
    charts += 1
    return charts

def plot_seaborn_examples(output_dir: Path) -> int:
    rng = np.random.RandomState(1)
    scores = rng.normal(loc=80, scale=20, size=200).astype(int)
    grid = sns.displot(scores, bins=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130], kde=True, height=5, aspect=1.5)
    grid.set_axis_labels("分数", "密度 / 频数")
    grid.fig.suptitle("带核密度估计的分数分布", y=1.02)
    save_figure(grid.fig, output_dir, "图表_直方图与核密度")
    rng = np.random.RandomState(0)
    values = rng.multivariate_normal([0, 1], [(1, 0.5), (0.5, 1)], 200)
    bivariate = pd.DataFrame(values, columns=["x", "y"])
    save_table(bivariate, output_dir, "二元分布_模拟数据")
    grid = sns.jointplot(data=bivariate, x="x", y="y", height=7)
    grid.fig.suptitle("二元联合分布", y=1.02)
    save_figure(grid.fig, output_dir, "图表_二元联合散点")
    grid = sns.relplot(data=bivariate, x="x", y="y", height=5, aspect=1.3)
    grid.fig.suptitle("x 与 y 关系图", y=1.02)
    save_figure(grid.fig, output_dir, "图表_二元关系图")
    grid = sns.jointplot(data=bivariate, x="x", y="y", kind="kde", height=7)
    grid.fig.suptitle("二元核密度联合分布", y=1.02)
    save_figure(grid.fig, output_dir, "图表_二元联合核密度")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.kdeplot(data=bivariate, x="x", y="y", fill=True, levels=12, cmap="mako", ax=ax)
    ax.set_title("二元分布核密度图")
    save_figure(fig, output_dir, "图表_二元核密度热区")
    rng = np.random.RandomState(0)
    heatmap_data = pd.DataFrame(rng.uniform(-1, 1, (7, 7)), index=list("abcdefg"), columns=list("ABCDEFG"))
    save_table(heatmap_data.rename_axis("行").reset_index(), output_dir, "热图_矩阵数据")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=False, cmap="vlag", center=0, ax=ax)
    ax.set_title("随机矩阵热图")
    save_figure(fig, output_dir, "图表_矩阵热图")
    return 6

def analyze_house_prices(data_dir: Path, output_dir: Path) -> int:
    data = pd.read_csv(data_dir / "sndHsPr.csv")
    required = {"AREA", "price"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"房价数据缺少字段: {', '.join(sorted(missing))}")
    price = pd.to_numeric(data["price"], errors="coerce").dropna()
    area = pd.to_numeric(data.loc[price.index, "AREA"], errors="coerce")
    valid = area.notna()
    paired = pd.DataFrame({"AREA": area[valid], "price": price[valid]})
    descriptive = price.agg(["count", "mean", "median", "std", "min", "max", "var", "skew", "kurtosis"]).rename_axis("统计量").rename("数值").reset_index()
    descriptive.loc[len(descriptive)] = ["全距", price.max() - price.min()]
    save_table(descriptive, output_dir, "房价_描述统计")
    quantiles = price.quantile([0.25, 0.5, 0.75]).rename_axis("分位数").rename("房价").reset_index()
    save_table(quantiles, output_dir, "房价_四分位数")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(price, bins=20, color="#277DA1", edgecolor="white")
    ax.set(title="房价分布直方图", xlabel="房价", ylabel="频数")
    save_figure(fig, output_dir, "图表_房价分布直方图")
    correlation_frames = []
    for method in ("pearson", "spearman", "kendall"):
        matrix = paired.corr(method=method)
        save_table(matrix.rename_axis("变量").reset_index(), output_dir, f"房价面积_相关系数_{method}")
        correlation_frames.append(matrix.stack().rename("相关系数").rename_axis(["变量1", "变量2"]).reset_index().assign(方法=method))
    correlations = pd.concat(correlation_frames, ignore_index=True)
    save_table(correlations, output_dir, "房价面积_相关系数汇总")
    return 1

def main() -> None:
    parser = argparse.ArgumentParser(description="复现量化策略教材第5章的描述统计与可视化示例")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="输入数据目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="复现结果目录")
    args = parser.parse_args()
    configure_charts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_count = plot_stock(args.data_dir, args.output_dir)
    chart_count += plot_basic_examples(args.output_dir)
    chart_count += plot_seaborn_examples(args.output_dir)
    chart_count += analyze_house_prices(args.data_dir, args.output_dir)
    print(f"复现完成：生成 {chart_count} 张图表及对应 CSV，输出目录：{args.output_dir.resolve()}")

if __name__ == "__main__":
    main()