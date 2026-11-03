"""第一章：销售汇总与银行关系模型分析。
复现教材中的销售趋势、信用卡类型与持卡人性别关系，以及每年新增贷款性别分布。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_auto(path: Path, **kwargs: object) -> pd.DataFrame:
    """读取常见 UTF-8 与中文 Windows 编码的教材 CSV。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")

def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

def analyze_bank_model(chapter_dir: Path, output_dir: Path) -> None:
    """按明确的外键路径连接银行表，复现课件的两项关系模型分析。"""
    bank_dir = chapter_dir / "关系模型演示-不建议用poweBI操作关系型数据模型"
    clients = read_csv_auto(bank_dir / "clients.csv", parse_dates=["birth_date"])
    dispositions = read_csv_auto(bank_dir / "disp.csv")
    cards = read_csv_auto(bank_dir / "card.csv", parse_dates=["issued"])
    loans = read_csv_auto(bank_dir / "loans.csv", parse_dates=["date"])
    cards = cards.rename(columns={"type": "card_type"})
    dispositions = dispositions.rename(columns={"type": "disposition_type"})
    card_holders = (
        cards.merge(dispositions, on="disp_id", how="left", validate="many_to_one")
        .merge(clients[["client_id", "sex"]], on="client_id", how="left", validate="many_to_one")
    )
    card_holders = card_holders.dropna(subset=["sex"])
    card_mix = (
        card_holders.groupby(["card_type", "sex"], as_index=False)
        .agg(card_count=("card_id", "nunique"), holder_count=("client_id", "nunique"))
    )
    card_mix["share_within_sex"] = card_mix["card_count"] / card_mix.groupby("sex")["card_count"].transform("sum")
    configure_chinese_font()
    card_chart = card_mix.pivot(index="card_type", columns="sex", values="card_count").fillna(0)
    card_sex_order = [sex for sex in ["男", "女"] if sex in card_chart.columns]
    card_chart = card_chart[card_sex_order]
    card_type_order = [kind for kind in ["普通卡", "青年卡", "金卡"] if kind in card_chart.index]
    card_chart = card_chart.reindex(card_type_order)
    ax = card_chart.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5.5),
        color=["#1689F8", "#1729A5"][: len(card_sex_order)],
        width=0.72,
    )
    ax.set(title="信用卡类型计数（按性别）", xlabel="type", ylabel="sex 的计数")
    ax.legend(title="sex", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "信用卡类型与性别堆积柱状图.png", dpi=160)
    plt.close(ax.figure)
    owners = dispositions.loc[dispositions["disposition_type"].eq("所有者"), ["account_id", "client_id"]]
    loan_owners = (
        loans.merge(owners, on="account_id", how="inner", validate="many_to_many")
        .merge(clients[["client_id", "sex"]], on="client_id", how="left", validate="many_to_one")
    )
    loan_owners["year"] = loan_owners["date"].dt.year
    yearly_loans = (
        loan_owners.dropna(subset=["sex"])
        .groupby(["year", "sex"], as_index=False)
        .agg(loan_amount=("amount", "sum"))
    )
    loan_chart = yearly_loans.pivot(index="year", columns="sex", values="loan_amount").fillna(0)
    loan_sex_order = [sex for sex in ["男", "女"] if sex in loan_chart.columns]
    loan_chart = loan_chart[loan_sex_order].sort_index()
    years = loan_chart.index.to_numpy()
    amounts = [loan_chart[sex].to_numpy() for sex in loan_sex_order]
    colors = ["#1689F8", "#1729A5"][: len(loan_sex_order)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.stackplot(years, *amounts, labels=loan_sex_order, colors=colors, alpha=0.38)
    cumulative = loan_chart.cumsum(axis=1)
    for sex, color in zip(loan_sex_order, colors):
        ax.plot(years, cumulative[sex], color=color, linewidth=1.8)
    ax.set(title="贷款金额总和（按年和性别）", xlabel="年", ylabel="amount 的总和")
    ax.set_xticks(years, [f"{year}年" for year in years])
    ax.legend(title="sex", frameon=False)
    ax.grid(linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "年度新增贷款性别金额面积图.png", dpi=160)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="运行第一章销售与银行关系模型分析")
    default_bank_dir = Path(
        r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\1《CDA 一级认证教材：商业数据分析》\第1章 数据分析思维\关系模型演示-不建议用poweBI操作关系型数据模型"
    )
    default_output_dir = Path(r"D:\商业数据分析代码\第一章_python分析\复现结果")
    parser.add_argument(
        "--chapter-dir",
        type=Path,
        default=default_bank_dir,
        help="银行关系模型数据目录（默认使用本机教材附件目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="结果输出目录",
    )
    args = parser.parse_args()
    bank_dir = args.chapter_dir
    sales_dir = bank_dir.parent if bank_dir.name == "关系模型演示-不建议用poweBI操作关系型数据模型" else bank_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyze_bank_model(sales_dir, args.output_dir)
    print(f"两张关系模型图已生成: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()