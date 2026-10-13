"""按 Power BI 日期切片器复现维度模型中的两类销售趋势图。"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import FuncFormatter

FACT_SHEET = "产品销售数据表"
DATE_SHEET = "日期表"
MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

def load_daily_sales(workbook: Path) -> pd.DataFrame:
    """连接销售事实与日期维度，返回日期粒度的销售额总和、均值。"""
    fact, date_dimension = pd.read_excel(
        workbook,
        sheet_name=[FACT_SHEET, DATE_SHEET],
    ).values()
    fact = fact.rename(columns={" 销售单价": "销售单价"})
    required_fact = {"订单日期", "销售额"}
    missing = required_fact.difference(fact.columns)
    if missing:
        raise ValueError(f"销售事实表缺少字段: {sorted(missing)}")
    if "日期" not in date_dimension.columns or "年度" not in date_dimension.columns:
        raise ValueError("日期表必须包含“日期”和“年度”字段")
    fact["订单日期"] = pd.to_datetime(fact["订单日期"])
    date_dimension["日期"] = pd.to_datetime(date_dimension["日期"])
    fact["销售额"] = pd.to_numeric(fact["销售额"], errors="raise")
    date_columns = ["日期", "年度"]
    if "月" in date_dimension.columns:
        date_columns.append("月")
    model = fact.merge(
        date_dimension[date_columns],
        left_on="订单日期",
        right_on="日期",
        how="left",
        validate="many_to_one",
        indicator="_date_match",
    )
    unmatched = model["_date_match"].ne("both")
    if unmatched.any():
        missing_dates = model.loc[unmatched, "订单日期"].drop_duplicates().tolist()
        raise ValueError(f"有 {len(missing_dates)} 个订单日期未匹配到日期维度")
    daily = (
        model.groupby(["年度", "订单日期"], as_index=False)
        .agg(销售额总和=("销售额", "sum"), 销售额平均值=("销售额", "mean"))
        .sort_values("订单日期")
    )
    daily["月"] = daily["订单日期"].dt.month
    return daily

def format_ten_thousands(value: float, _position: int) -> str:
    return f"{value / 10_000:g} 万" if value else "0"

def save_yearly_sales_charts(daily: pd.DataFrame, output_dir: Path) -> int:
    """每個所选年度绘制一张按日销售额总和的面积图。"""
    chart_dir = output_dir / "按年销售额"
    chart_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for year, rows in daily.groupby("年度", sort=True):
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = rows["订单日期"]
        values = rows["销售额总和"]
        ax.fill_between(dates, values, color="#32c7bd", alpha=0.48)
        ax.plot(dates, values, color="#00b9ad", linewidth=1.5)
        ax.set_title(f"{year} 年每日销售额")
        ax.set_xlabel("订单日期")
        ax.set_ylabel("销售额")
        ax.yaxis.set_major_formatter(FuncFormatter(format_ten_thousands))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(chart_dir / f"{year}年_每日销售额.png", dpi=160)
        plt.close(fig)
        count += 1
    return count

def save_monthly_average_charts(daily: pd.DataFrame, output_dir: Path) -> int:
    """每个有销售记录的年月绘制日均销售额折线与当月平均参考线。"""
    chart_dir = output_dir / "按年月日均销售额"
    chart_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for (year, month), rows in daily.groupby(["年度", "月"], sort=True):
        rows = rows.sort_values("订单日期")
        monthly_average = rows["销售额平均值"].mean()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(
            rows["订单日期"],
            rows["销售额平均值"],
            color="#00b9ad",
            linewidth=1.6,
        )
        ax.axhline(monthly_average, color="#83ddd6", linewidth=1.8)
        ax.set_title(f"{year} 年 {month} 月每日平均销售额")
        ax.set_xlabel("订单日期")
        ax.set_ylabel("销售额平均值")
        ax.set_xlim(rows["订单日期"].min(), rows["订单日期"].max())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position: f"{mdates.num2date(value).month}月{mdates.num2date(value).day}日"
            )
        )
        ax.yaxis.set_major_formatter(FuncFormatter(format_ten_thousands))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        month_name = MONTH_NAMES[int(month)]
        fig.savefig(chart_dir / f"{year}年{month:02d}月_{month_name}_日均销售额.png", dpi=160)
        plt.close(fig)
        count += 1
    return count

def main() -> None:
    parser = argparse.ArgumentParser(description="复现维度模型的年度及年月销售图")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(
            r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\1《CDA 一级认证教材：商业数据分析》\第1章 数据分析思维\维度模型演示\商场销售数据表.xlsx"
        ),
        help="维度模型 Excel 工作簿路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\商业数据分析代码\第一章_python分析\复现结果\维度模型"),
        help="图表输出目录",
    )
    args = parser.parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_sales = load_daily_sales(args.workbook)
    yearly_count = save_yearly_sales_charts(daily_sales, args.output_dir)
    monthly_count = save_monthly_average_charts(daily_sales, args.output_dir)
    print(f"已生成 {yearly_count} 张年度销售额图、{monthly_count} 张年月日均销售额图。")
    print(f"图表目录: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()