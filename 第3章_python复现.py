"""使用 pandas 和 matplotlib 复现教材第3章的12张案例图表。"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

CHAPTER_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\1《CDA 一级认证教材：商业数据分析》\第3章 商业数据分析框架"
)
OUTPUT_DIR = Path(__file__).parent / "复现结果"
COLORS = ["#277DA1", "#F8961E", "#43AA8B", "#E76F51", "#577590", "#90BE6D"]

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def require_columns(frame: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} 缺少字段: {', '.join(sorted(missing))}")

def complete_workbook(chapter_dir: Path, folder: str) -> Path:
    case_dir = chapter_dir / folder
    matches = sorted(case_dir.glob("*完整*.xlsx")) if case_dir.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"{case_dir} 中应有且仅有一个完整案例工作簿，实际找到 {len(matches)} 个")
    return matches[0]

def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(f"工作簿 {path.name} 中不存在工作表 {sheet_name!r}") from exc
    return frame.dropna(axis=1, how="all").dropna(how="all").copy()

def save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def save_chart(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def add_growth_columns(frame: pd.DataFrame, previous: str, current: str) -> pd.DataFrame:
    result = frame.copy()
    result["同比差额"] = result[current] - result[previous]
    result["同比增长率_完整案例公式"] = np.where(
        result[current].ne(0), result["同比差额"] / result[current], np.nan
    )
    result["标准同比增长率"] = np.where(
        result[previous].ne(0), result["同比差额"] / result[previous], np.nan
    )
    return result

def analyze_revenue(chapter_dir: Path, output_dir: Path) -> int:
    source = complete_workbook(chapter_dir, "案例1：收入趋势分析")
    data = read_sheet(source, "原始数据")
    require_columns(data, ["订单年份", "订单月份", "销售省份", "销售额"], source.name)
    for column in ("订单年份", "订单月份", "销售额"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["订单年份", "订单月份", "销售额"])
    data["订单年份"] = data["订单年份"].astype(int)
    data["订单月份"] = data["订单月份"].astype(int)
    data["销售省份"] = data["销售省份"].astype(str).str.strip()

    annual = data.groupby("订单年份", as_index=False)["销售额"].sum().sort_values("订单年份")
    save_table(annual, output_dir, "收入趋势_年度销售额")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(annual["订单年份"], annual["销售额"], color=COLORS[0], marker="o", linewidth=2)
    ax.set(title="年度销售趋势分析", xlabel="年份", ylabel="销售额")
    ax.set_xticks(annual["订单年份"])
    ax.grid(axis="y", alpha=0.22)
    save_chart(fig, output_dir, "收入趋势_年度销售趋势")

    monthly_2020 = (
        data.loc[data["订单年份"].eq(2020)]
        .groupby("订单月份")["销售额"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
        .rename_axis("月份")
        .reset_index()
    )
    save_table(monthly_2020, output_dir, "收入趋势_2020月销售额")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(monthly_2020["月份"], monthly_2020["销售额"], color=COLORS[0], marker="o", linewidth=2)
    ax.set(title="2020年月销售趋势分析", xlabel="月份", ylabel="销售额")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.22)
    save_chart(fig, output_dir, "收入趋势_2020月销售趋势")

    monthly_pivot = (
        data.loc[data["订单年份"].isin([2019, 2020])]
        .pivot_table(index="订单月份", columns="订单年份", values="销售额", aggfunc="sum", fill_value=0)
        .reindex(index=range(1, 13), columns=[2019, 2020], fill_value=0)
    )
    monthly_yoy = add_growth_columns(
        monthly_pivot.rename_axis("月份").reset_index().rename(columns={2019: "2019年销售额", 2020: "2020年销售额"}),
        "2019年销售额",
        "2020年销售额",
    )
    save_table(monthly_yoy, output_dir, "收入趋势_2020月同比")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(monthly_yoy["月份"], monthly_yoy["同比增长率_完整案例公式"], color=COLORS[1], marker="o", linewidth=2)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="2020年销售同比增长率趋势分析", xlabel="月份", ylabel="同比增长率")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.22)
    save_chart(fig, output_dir, "收入趋势_2020月同比增长率")

    november = data.loc[data["订单月份"].eq(11) & data["订单年份"].isin([2019, 2020])]
    province_pivot = (
        november.pivot_table(index="销售省份", columns="订单年份", values="销售额", aggfunc="sum", fill_value=0)
        .reindex(columns=[2019, 2020], fill_value=0)
        .rename(columns={2019: "2019年销售额", 2020: "2020年销售额"})
        .reset_index()
    )
    province_yoy = add_growth_columns(province_pivot, "2019年销售额", "2020年销售额")
    province_yoy = province_yoy.sort_values("同比增长率_完整案例公式", na_position="first")
    save_table(province_yoy, output_dir, "收入趋势_2020年11月省份同比")
    fig, ax = plt.subplots(figsize=(10, 9))
    values = province_yoy["同比增长率_完整案例公式"]
    colors = [COLORS[2] if pd.notna(value) and value >= 0 else COLORS[3] for value in values]
    ax.barh(province_yoy["销售省份"], values, color=colors)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="2020年11月各省份同比增长情况分析", xlabel="同比增长率（完整案例公式）", ylabel="省份")
    ax.grid(axis="x", alpha=0.2)
    save_chart(fig, output_dir, "收入趋势_2020年11月省份同比")

    sichuan_pivot = (
        data.loc[data["销售省份"].eq("四川") & data["订单年份"].isin([2019, 2020])]
        .pivot_table(index="订单月份", columns="订单年份", values="销售额", aggfunc="sum", fill_value=0)
        .reindex(index=range(1, 13), columns=[2019, 2020], fill_value=0)
    )
    sichuan_yoy = add_growth_columns(
        sichuan_pivot.rename_axis("月份").reset_index().rename(columns={2019: "2019年销售额", 2020: "2020年销售额"}),
        "2019年销售额",
        "2020年销售额",
    )
    save_table(sichuan_yoy, output_dir, "收入趋势_四川省2020月同比")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(sichuan_yoy["月份"], sichuan_yoy["同比增长率_完整案例公式"], color=COLORS[4], marker="o", linewidth=2)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="四川省2020年各月同比增长情况", xlabel="月份", ylabel="同比增长率")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.22)
    save_chart(fig, output_dir, "收入趋势_四川省2020月同比")
    return 5

def normalize_columns(frame: pd.DataFrame, columns: Sequence[str], source: str) -> pd.DataFrame:
    if len(frame.columns) < len(columns):
        raise ValueError(f"{source} 列数不足，预期至少 {len(columns)} 列")
    result = frame.iloc[:, : len(columns)].copy()
    result.columns = list(columns)
    return result.dropna(how="all")

def parse_month(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

def analyze_finance(chapter_dir: Path, output_dir: Path) -> int:
    source = complete_workbook(chapter_dir, "案例2：财务费用分析")
    total = normalize_columns(
        read_sheet(source, "总费用数据-原始数据"), ["年份", "月份", "金额"], source.name
    )
    periods = normalize_columns(
        read_sheet(source, "财务三大期间费用-原始数据"), ["年份", "月份", "期间费用项", "金额"], source.name
    )
    management = normalize_columns(
        read_sheet(source, "管理费用-原始数据"), ["年份", "月份", "管理费用项", "金额"], source.name
    )
    for frame in (total, periods, management):
        frame["年份"] = pd.to_numeric(frame["年份"], errors="coerce")
        frame["月份"] = parse_month(frame["月份"])
        frame["金额"] = pd.to_numeric(frame["金额"], errors="coerce")
    periods["期间费用项"] = periods["期间费用项"].astype(str).str.strip()
    management["管理费用项"] = management["管理费用项"].astype(str).str.strip()
    total = total.dropna(subset=["年份", "月份", "金额"])
    periods = periods.dropna(subset=["年份", "月份", "金额"])
    management = management.dropna(subset=["年份", "月份", "金额"])

    total_pair = total.loc[total["年份"].eq(2020) & total["月份"].isin([6, 7])]
    total_chart_data = total_pair.pivot_table(index="月份", values="金额", aggfunc="sum").reindex([6, 7], fill_value=0)
    total_chart_data.index = ["6月", "7月"]
    total_chart_data = total_chart_data.rename_axis("月份").reset_index()
    save_table(total_chart_data, output_dir, "财务费用_总费用月度对比")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(total_chart_data["月份"], total_chart_data["金额"], color=COLORS[:2], width=0.58)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set(title="总财务费用对比", xlabel="月份", ylabel="金额")
    ax.grid(axis="y", alpha=0.2)
    save_chart(fig, output_dir, "财务费用_总费用对比")

    period_pair = periods.loc[periods["年份"].eq(2020) & periods["月份"].isin([6, 7])]
    period_pivot = period_pair.pivot_table(
        index="期间费用项", columns="月份", values="金额", aggfunc="sum", fill_value=0
    ).reindex(columns=[6, 7], fill_value=0)
    period_report = period_pivot.rename(columns={6: "6月金额", 7: "7月金额"}).reset_index()
    period_report["环比增减额"] = period_report["7月金额"] - period_report["6月金额"]
    save_table(period_report, output_dir, "财务费用_三大期间费用")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    index = np.arange(len(period_report))
    width = 0.34
    ax.bar(index - width / 2, period_report["6月金额"], width, label="6月", color=COLORS[0])
    ax.bar(index + width / 2, period_report["7月金额"], width, label="7月", color=COLORS[1])
    ax.set_xticks(index, period_report["期间费用项"], rotation=12, ha="right")
    ax.set(title="三大期间财务费用7月较6月增长情况", xlabel="期间费用项目", ylabel="金额")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_chart(fig, output_dir, "财务费用_三大期间费用对比")

    admin_pair = management.loc[management["年份"].eq(2020) & management["月份"].isin([6, 7])]
    admin_pivot = admin_pair.pivot_table(
        index="管理费用项", columns="月份", values="金额", aggfunc="sum", fill_value=0
    ).reindex(columns=[6, 7], fill_value=0)
    admin_report = admin_pivot.rename(columns={6: "6月金额", 7: "7月金额"}).reset_index()
    admin_report["环比增长率"] = np.where(
        admin_report["6月金额"].ne(0),
        (admin_report["7月金额"] - admin_report["6月金额"]) / admin_report["6月金额"],
        np.nan,
    )
    admin_report = admin_report.sort_values("环比增长率")
    save_table(admin_report, output_dir, "财务费用_管理费用环比")
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = [COLORS[2] if value >= 0 else COLORS[3] for value in admin_report["环比增长率"].fillna(0)]
    ax.barh(admin_report["管理费用项"], admin_report["环比增长率"], color=bar_colors)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="管理费用7月环比增长分析", xlabel="环比增长率", ylabel="管理费用项目")
    ax.grid(axis="x", alpha=0.2)
    save_chart(fig, output_dir, "财务费用_管理费用环比")
    return 3

def analyze_marketing(chapter_dir: Path, output_dir: Path) -> int:
    source = complete_workbook(chapter_dir, "案例3：电商平台的精准营销")
    data = read_sheet(source, "原始数据")
    require_columns(data, ["客户ID", "性别", "年龄", "品类", "支付方式"], source.name)
    data["年龄"] = pd.to_numeric(data["年龄"], errors="coerce")
    for column in ("客户ID", "性别", "品类", "支付方式"):
        data[column] = data[column].astype(str).str.strip()
    data = data.dropna(subset=["年龄"])
    data["年龄段"] = pd.cut(
        data["年龄"], bins=[-np.inf, 25, 35, 49, np.inf],
        labels=["18-25岁", "26-35岁", "36-49岁", "50岁以上"],
    )

    gender = data.groupby("性别", as_index=False, observed=True)["客户ID"].nunique().rename(columns={"客户ID": "客户数"})
    save_table(gender, output_dir, "精准营销_购买客户性别分布")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.pie(gender["客户数"], labels=gender["性别"], autopct="%1.1f%%", startangle=90, colors=COLORS[: len(gender)])
    ax.set_title("购买客户的性别分布")
    save_chart(fig, output_dir, "精准营销_购买客户性别分布")

    female = data.loc[data["性别"].eq("Female")]
    female_ages = (
        female.groupby("年龄段", observed=False)["客户ID"].nunique()
        .reindex(["18-25岁", "26-35岁", "36-49岁", "50岁以上"], fill_value=0)
        .rename("客户数")
        .rename_axis("年龄段")
        .reset_index()
    )
    save_table(female_ages, output_dir, "精准营销_女性客户年龄段分布")
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.pie(female_ages["客户数"], labels=female_ages["年龄段"], autopct="%1.1f%%", startangle=90, colors=COLORS[:4])
    ax.set_title("女性消费客户的年龄段分布")
    save_chart(fig, output_dir, "精准营销_女性客户年龄段分布")

    older_female = female.loc[female["年龄"].ge(50)]
    categories = (
        older_female.groupby("品类", as_index=False)["客户ID"].nunique()
        .rename(columns={"客户ID": "客户数"})
        .sort_values("客户数")
    )
    save_table(categories, output_dir, "精准营销_50岁以上女性品类分布")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(categories["品类"], categories["客户数"], color=COLORS[0])
    ax.set(title="50岁以上的女性客户的消费品类分布", xlabel="客户数", ylabel="品类")
    ax.grid(axis="x", alpha=0.2)
    save_chart(fig, output_dir, "精准营销_50岁以上女性品类分布")

    target_categories = older_female.loc[older_female["品类"].isin(["Clothing", "Cosmetics"])]
    payment = (
        target_categories.groupby("支付方式", as_index=False)["客户ID"].nunique()
        .rename(columns={"客户ID": "客户数"})
        .sort_values("客户数")
    )
    save_table(payment, output_dir, "精准营销_50岁以上女性服饰护肤支付方式")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(payment["支付方式"], payment["客户数"], color=COLORS[1])
    ax.set(title="50岁以上女性服饰和护肤品消费人群的支付方式分布", xlabel="客户数", ylabel="支付方式")
    ax.grid(axis="x", alpha=0.2)
    save_chart(fig, output_dir, "精准营销_50岁以上女性服饰护肤支付方式")
    return 4

def main() -> None:
    parser = argparse.ArgumentParser(description="复现教材第3章商业数据分析框架案例图表")
    parser.add_argument("--chapter-dir", type=Path, default=CHAPTER_DIR, help="第3章附件目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="图表和汇总表输出目录")
    parser.add_argument("--task", choices=("all", "revenue", "finance", "marketing"), default="all", help="运行全部或指定案例")
    args = parser.parse_args()
    configure_charts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runners = {
        "revenue": analyze_revenue,
        "finance": analyze_finance,
        "marketing": analyze_marketing,
    }
    selected = runners.items() if args.task == "all" else [(args.task, runners[args.task])]
    chart_count = 0
    for name, runner in selected:
        chart_count += runner(args.chapter_dir, args.output_dir)
        print(f"案例 {name} 复现完成")
    print(f"共生成 {chart_count} 张图表及对应汇总 CSV，输出至 {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()