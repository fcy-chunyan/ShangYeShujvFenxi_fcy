from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
CHAPTER_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\2《CDA 二级认证教材：量化策略分析》\第6章-市场调研与数据预处理"
)
DATA_DIR = CHAPTER_DIR / "2 数据预处理基础"
OUTPUT_DIR = Path(__file__).parent / "复现结果"

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def save_table(frame: pd.DataFrame, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def save_chart(fig: plt.Figure, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def run(data_dir: Path, output_dir: Path) -> int:
    source = data_dir / "teleco_camp_orig.csv"
    camp = pd.read_csv(source)
    required = {"AvgIncome", "AvgHomeValue"}
    missing = required.difference(camp.columns)
    if missing:
        raise ValueError(f"{source.name} 缺少字段: {', '.join(sorted(missing))}")
    income_before = pd.to_numeric(camp["AvgIncome"], errors="coerce")
    home_value = pd.to_numeric(camp["AvgHomeValue"], errors="coerce")
    before_stats = pd.DataFrame(
        {"AvgIncome": income_before.describe(), "AvgHomeValue": home_value.describe()}
    ).rename_axis("统计量").reset_index()
    save_table(before_stats, "预处理_替换前描述统计", output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(income_before.dropna(), bins=20, color="#277DA1", edgecolor="white")
    ax.set(title="AvgIncome 清理前分布", xlabel="AvgIncome", ylabel="频数")
    save_chart(fig, "预处理_AvgIncome清理前直方图", output_dir)
    zero_count = int(income_before.eq(0).sum())
    camp["AvgIncome"] = income_before.replace(0, np.nan)
    income_after = camp["AvgIncome"]
    after_stats = pd.DataFrame(
        {"AvgIncome": income_after.describe(), "AvgHomeValue": home_value.describe()}
    ).rename_axis("统计量").reset_index()
    save_table(after_stats, "预处理_替换后描述统计", output_dir)
    save_table(camp, "预处理_清理后数据", output_dir)
    missing_summary = pd.DataFrame(
        [
            {"字段": "AvgIncome", "替换为缺失的零值数量": zero_count, "处理后缺失值数量": int(income_after.isna().sum())},
            {"字段": "AvgHomeValue", "替换为缺失的零值数量": 0, "处理后缺失值数量": int(home_value.isna().sum())},
        ]
    )
    save_table(missing_summary, "预处理_缺失值处理汇总", output_dir)
    valid_income = income_after.dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        valid_income,
        bins=20,
        range=(valid_income.min(), valid_income.max()),
        color="#43AA8B",
        edgecolor="white",
    )
    ax.set(title="AvgIncome 将零值设为缺失后分布", xlabel="AvgIncome", ylabel="频数")
    save_chart(fig, "预处理_AvgIncome清理后直方图", output_dir)
    return 2

def main() -> None:
    parser = argparse.ArgumentParser(description="复现第6章数据预处理示例")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="预处理数据目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="结果输出目录")
    args = parser.parse_args()
    configure_charts()
    chart_count = run(args.data_dir, args.output_dir)
    print(f"数据预处理复现完成：{chart_count} 张图表，输出至 {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()