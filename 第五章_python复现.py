"""复现教材第5章银行 Power BI 看板与 WPS 归因分析图表。"""
from __future__ import annotations
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from wordcloud import WordCloud

CHAPTER_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\1《CDA 一级认证教材：商业数据分析》\第5章 业务数据分析"
)
OUTPUT_DIR = Path(__file__).parent / "复现结果"
COLORS = ["#277DA1", "#F8961E", "#43AA8B", "#E76F51", "#577590", "#90BE6D"]
PBIX_TYPES = {
    "cardVisual": "指标卡",
    "decompositionTreeVisual": "分解树",
    "clusteredColumnChart": "簇状柱形图",
    "lineClusteredColumnComboChart": "折线和簇状柱形图",
    "WordCloud1447959067750": "词云",
}
PBIX_LABELS = [
    "购买金额与月目标达成率",
    "购买金额分解树",
    "产品类别与风险偏好购买金额",
    "产品类别客单与期限均值",
    "选择本行原因词云",
    "购买目的词云",
]
NS_CHART = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
NS_DRAWING = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def find_file(chapter_dir: Path, filename: str) -> Path:
    matches = list(chapter_dir.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"附件中应有且仅有一个 {filename!r}，实际找到 {len(matches)} 个")
    return matches[0]

def save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def save_chart(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def read_pbix_visual_inventory(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        layout = json.loads(archive.read("Report/Layout").decode("utf-16-le").lstrip("\ufeff"))
    records: list[dict[str, object]] = []
    for page in layout.get("sections", []):
        for index, container in enumerate(page.get("visualContainers", []), start=1):
            config = json.loads(container.get("config", "{}"))
            visual = config.get("singleVisual", {})
            projections = visual.get("projections", {})
            records.append(
                {
                    "页面": page.get("displayName", ""),
                    "序号": index,
                    "视觉类型": PBIX_TYPES.get(visual.get("visualType"), visual.get("visualType", "")),
                    "原始类型": visual.get("visualType", ""),
                    "字段绑定": json.dumps(
                        {role: [item.get("queryRef", "") for item in fields] for role, fields in projections.items()},
                        ensure_ascii=False,
                    ),
                    "位置": json.dumps(
                        {key: container.get(key) for key in ("x", "y", "width", "height")},
                        ensure_ascii=False,
                    ),
                }
            )
    return pd.DataFrame(records)

def read_bank_data(chapter_dir: Path) -> tuple[pd.DataFrame, float]:
    workbook = find_file(chapter_dir, "银行案例数据 - 学员用.xlsx")
    sales = pd.read_excel(workbook, sheet_name="销售数据")
    category_map = pd.read_excel(workbook, sheet_name="品类说明表").set_index("品类编号")["品类名称"]
    risk_map = pd.read_excel(workbook, sheet_name="风险偏好说明表").set_index("风险编号")["风险说明"]
    education_map = pd.read_excel(workbook, sheet_name="学历说明表").set_index("学历编号")["学历名称"]
    sales["品类名称"] = sales["品类编号"].map(category_map)
    sales["风险说明"] = sales["风险编号"].map(risk_map)
    sales["学历名称"] = sales["学历编号"].map(education_map)
    sales["购买金额"] = pd.to_numeric(sales["购买金额"], errors="coerce")
    sales["期限"] = pd.to_numeric(sales["期限"], errors="coerce")
    sales = sales.dropna(subset=["购买金额", "期限", "品类名称", "风险说明"])

    business = pd.read_excel(workbook, sheet_name="业务描述", header=None).fillna("")
    description = " ".join(str(value) for value in business.to_numpy().ravel() if str(value).strip())
    match = re.search(r"目标(?:为|是)\s*(\d+)\s*万", description)
    if not match:
        raise ValueError("业务描述中未找到以‘万’为单位的销售目标，无法复现目标达成率指标")
    target = float(match.group(1)) * 10_000
    return sales, target

def find_chinese_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("未找到微软雅黑或黑体字体，无法生成中文词云")

def analyze_powerbi(chapter_dir: Path, output_dir: Path) -> int:
    pbix = find_file(chapter_dir, "案例结果.pbix")
    inventory = read_pbix_visual_inventory(pbix)
    if len(inventory) != len(PBIX_LABELS):
        raise ValueError(f"PBIX 视觉对象数预期为 {len(PBIX_LABELS)}，实际为 {len(inventory)}")
    inventory["复现图表"] = PBIX_LABELS
    save_table(inventory, output_dir, "PowerBI_视觉对象清单")

    data, target = read_bank_data(chapter_dir)
    total_sales = data["购买金额"].sum()
    target_rate = total_sales / target if target else np.nan
    kpis = pd.DataFrame(
        [
            {"指标": "累计购买金额", "数值": total_sales, "说明": "学员数据表购买金额求和"},
            {"指标": "月目标", "数值": target, "说明": "从业务描述中提取"},
            {"指标": "月目标达成率", "数值": target_rate, "说明": "累计购买金额 / 月目标"},
        ]
    )
    save_table(kpis, output_dir, "PowerBI_指标卡")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    positions = [0.25, 0.75]
    values = [f"{total_sales / 10000:,.1f} 万元", f"{target_rate:.1%}"]
    labels = ["累计购买金额", "月目标达成率"]
    for x, label, value, color in zip(positions, labels, values, COLORS[:2]):
        ax.text(x, 0.7, label, ha="center", va="center", fontsize=14, color="#555555", transform=ax.transAxes)
        ax.text(x, 0.34, value, ha="center", va="center", fontsize=25, fontweight="bold", color=color, transform=ax.transAxes)
    save_chart(fig, output_dir, "PowerBI_指标卡")

    dimensions = ["家庭年收入", "年龄", "学历名称", "性别"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    decomp_records: list[pd.DataFrame] = []
    for ax, dimension in zip(axes.flat, dimensions):
        summary = data.groupby(dimension, dropna=False, as_index=False)["购买金额"].sum().sort_values("购买金额")
        summary["拆解维度"] = dimension
        decomp_records.append(summary.rename(columns={dimension: "维度值"}))
        ax.barh(summary[dimension].astype(str), summary["购买金额"], color=COLORS[dimensions.index(dimension)])
        ax.set_title(f"按{dimension}拆解购买金额")
        ax.set_xlabel("购买金额")
        ax.grid(axis="x", alpha=0.18)
    save_table(pd.concat(decomp_records, ignore_index=True), output_dir, "PowerBI_分解树数据")
    fig.suptitle("购买金额分解树字段拆解", y=1.01, fontsize=15)
    save_chart(fig, output_dir, "PowerBI_购买金额分解树")

    risk_summary = (
        data.groupby(["品类名称", "风险说明"], as_index=False)["购买金额"]
        .sum()
        .sort_values("品类名称")
    )
    save_table(risk_summary, output_dir, "PowerBI_品类风险购买金额")
    risk_pivot = risk_summary.pivot(index="品类名称", columns="风险说明", values="购买金额").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    risk_pivot.plot(kind="bar", ax=ax, color=COLORS[: len(risk_pivot.columns)], width=0.82)
    ax.set(title="按产品类别和风险偏好的购买金额", xlabel="产品类别", ylabel="购买金额")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="风险偏好", frameon=False)
    ax.grid(axis="y", alpha=0.18)
    save_chart(fig, output_dir, "PowerBI_品类风险购买金额")

    category_summary = (
        data.groupby("品类名称", as_index=False)
        .agg(客单均值=("购买金额", "mean"), 期限均值=("期限", "mean"), 记录数=("购买金额", "size"))
        .sort_values("客单均值", ascending=False)
    )
    save_table(category_summary, output_dir, "PowerBI_品类客单与期限")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(category_summary))
    ax.bar(x, category_summary["客单均值"], color=COLORS[0], label="客单均值")
    ax.set_ylabel("平均购买金额")
    ax.set_xticks(x, category_summary["品类名称"], rotation=30, ha="right")
    secondary = ax.twinx()
    secondary.plot(x, category_summary["期限均值"], color=COLORS[3], marker="o", linewidth=2, label="期限均值")
    secondary.set_ylabel("平均期限")
    ax.set_title("产品类别客单与期限均值")
    ax.grid(axis="y", alpha=0.18)
    lines, line_labels = secondary.get_legend_handles_labels()
    bars, bar_labels = ax.get_legend_handles_labels()
    ax.legend(bars + lines, bar_labels + line_labels, frameon=False, loc="upper right")
    save_chart(fig, output_dir, "PowerBI_品类客单与期限")

    font_path = find_chinese_font()
    wordcloud_specs = [
        ("选择本行的原因", "PowerBI_选择本行原因词云", "选择本行原因"),
        ("购买目的", "PowerBI_购买目的词云", "购买目的"),
    ]
    for column, filename, table_name in wordcloud_specs:
        frequencies = data[column].dropna().astype(str).str.strip().value_counts()
        frequencies = frequencies[frequencies.index != ""]
        summary = frequencies.rename_axis("文本").rename("计数").reset_index()
        save_table(summary, output_dir, f"PowerBI_{table_name}_频数")
        cloud = WordCloud(
            font_path=font_path,
            width=1200,
            height=620,
            background_color="white",
            colormap="viridis",
            collocations=False,
            prefer_horizontal=0.9,
            max_words=100,
            random_state=42,
        ).generate_from_frequencies(frequencies.to_dict())
        fig, ax = plt.subplots(figsize=(10, 5.4))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(table_name, fontsize=14, pad=12)
        save_chart(fig, output_dir, filename)
    return len(PBIX_LABELS)

def cached_values(parent: ET.Element | None) -> list[str]:
    if parent is None:
        return []
    return [node.text for node in parent.findall(".//c:v", NS_CHART) if node.text is not None]

def read_docx_chart_cache(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with ZipFile(path) as archive:
        chart_names = sorted(
            name for name in archive.namelist() if re.fullmatch(r"word/charts/chart\d+\.xml", name)
        )
        for chart_name in chart_names:
            root = ET.fromstring(archive.read(chart_name))
            for series_index, series in enumerate(root.findall(".//c:ser", NS_CHART), start=1):
                title_element = series.find("c:tx", NS_CHART)
                title_values = cached_values(title_element)
                if not title_values and title_element is not None:
                    title_values = [node.text for node in title_element.findall(".//a:t", NS_DRAWING) if node.text]
                categories = cached_values(series.find("c:cat", NS_CHART))
                values = cached_values(series.find("c:val", NS_CHART))
                records.append(
                    {
                        "图表": chart_name,
                        "系列序号": series_index,
                        "系列名称": title_values[-1] if title_values else f"系列{series_index}",
                        "类别": categories,
                        "数值": values,
                    }
                )
    return records

def read_xlsx_chart_cache(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with ZipFile(path) as archive:
        chart_names = sorted(
            name for name in archive.namelist() if re.fullmatch(r"xl/charts/chart\d+\.xml", name)
        )
        for chart_name in chart_names:
            root = ET.fromstring(archive.read(chart_name))
            for series_index, series in enumerate(root.findall(".//c:ser", NS_CHART), start=1):
                title_element = series.find("c:tx", NS_CHART)
                title_values = cached_values(title_element)
                if not title_values and title_element is not None:
                    title_values = [node.text for node in title_element.findall(".//a:t", NS_DRAWING) if node.text]
                categories = cached_values(series.find("c:cat", NS_CHART))
                values = cached_values(series.find("c:val", NS_CHART))
                records.append(
                    {
                        "图表": chart_name,
                        "系列序号": series_index,
                        "系列名称": title_values[-1] if title_values else f"系列{series_index}",
                        "类别": categories,
                        "数值": values,
                    }
                )
    return records

def verify_chart_caches(
    docx_path: Path,
    xlsx_path: Path,
    monthly: pd.DataFrame,
    channel_pivot: pd.DataFrame,
) -> pd.DataFrame:
    caches = [("Word", read_docx_chart_cache(docx_path)), ("Excel", read_xlsx_chart_cache(xlsx_path))]
    month_labels = monthly["申请年月"].tolist()
    expected_monthly = dict(zip(month_labels, monthly["贷款申请量"].astype(int)))
    for source_name, cache in caches:
        total_series = next(
            (item for item in cache if "贷款申请量" in str(item["系列名称"])), None
        )
        if total_series is None:
            raise ValueError(f"{source_name} 附件图表中未找到‘贷款申请量’系列")
        actual_monthly = dict(zip(total_series["类别"], [int(float(value)) for value in total_series["数值"]]))
        if expected_monthly != actual_monthly:
            raise ValueError(f"源数据重算的月度申请量与 {source_name} 图表缓存不一致")
        for channel in channel_pivot.columns:
            series = next((item for item in cache if str(item["系列名称"]).strip() == channel), None)
            if series is None:
                raise ValueError(f"{source_name} 附件图表中未找到渠道系列 {channel}")
            actual = dict(zip(series["类别"], [int(float(value)) for value in series["数值"]]))
            expected = dict(zip(month_labels, channel_pivot[channel].astype(int)))
            if actual != expected:
                raise ValueError(f"源数据重算的 {channel} 申请量与 {source_name} 图表缓存不一致")
    flat = [
        {
            "附件类型": source_name,
            "图表": item["图表"],
            "系列名称": item["系列名称"],
            "类别": category,
            "数值": value,
        }
        for source_name, cache in caches
        for item in cache
        for category, value in zip(item["类别"], item["数值"])
    ]
    return pd.DataFrame(flat)

def analyze_attribution(chapter_dir: Path, output_dir: Path) -> int:
    source = find_file(chapter_dir, "归因分析数据.xlsx")
    data = pd.read_excel(source, sheet_name="贷款申请表")
    required = ["申请书编号（主键）", "申请日期", "申请渠道"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"{source.name} 缺少字段: {', '.join(sorted(missing))}")
    data["申请日期"] = pd.to_datetime(data["申请日期"], errors="coerce")
    data = data.dropna(subset=["申请书编号（主键）", "申请日期", "申请渠道"])
    data["月份"] = data["申请日期"].dt.to_period("M")
    data["申请年月"] = data["月份"].map(lambda value: f"{value.year}年{value.month}月")
    months = sorted(data["月份"].unique())
    labels = [f"{value.year}年{value.month}月" for value in months]
    monthly = (
        data.groupby(["月份", "申请年月"], as_index=False)["申请书编号（主键）"]
        .nunique()
        .rename(columns={"申请书编号（主键）": "贷款申请量"})
        .sort_values("月份")
    )
    monthly["环比变化率"] = monthly["贷款申请量"].pct_change()
    save_table(monthly.drop(columns="月份"), output_dir, "归因_月度贷款申请量")

    channel_pivot = (
        data.pivot_table(
            index="月份", columns="申请渠道", values="申请书编号（主键）", aggfunc="nunique", fill_value=0
        )
        .reindex(months, fill_value=0)
    )
    channel_pivot.index = labels
    channel_pivot.index.name = "申请年月"
    channel_table = channel_pivot.reset_index()
    save_table(channel_table, output_dir, "归因_申请渠道月度分布")

    report = find_file(chapter_dir, "2021年5月贷款申请情况月度分析报告.docx")
    result_workbook = find_file(chapter_dir, "归因分析 - 结果.xlsx")
    cache_table = verify_chart_caches(report, result_workbook, monthly, channel_pivot)
    save_table(cache_table, output_dir, "归因_Word图表缓存核对")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(monthly))
    ax.bar(x, monthly["贷款申请量"], color=COLORS[0], alpha=0.82, width=0.62, label="贷款申请量")
    ax.set_xticks(x, monthly["申请年月"], rotation=25, ha="right")
    ax.set_ylabel("申请量")
    rate_axis = ax.twinx()
    rate_axis.plot(x, monthly["环比变化率"], marker="o", color=COLORS[3], linewidth=2, label="环比变化率")
    rate_axis.yaxis.set_major_formatter(PercentFormatter(1))
    rate_axis.set_ylabel("环比变化率")
    ax.set_title("贷款申请量及环比变化趋势")
    ax.grid(axis="y", alpha=0.2)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = rate_axis.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, frameon=False, loc="upper left")
    save_chart(fig, output_dir, "归因_月度申请量趋势")

    fig, ax = plt.subplots(figsize=(11, 5.4))
    channel_pivot.plot(kind="bar", ax=ax, color=COLORS[: len(channel_pivot.columns)], width=0.78)
    ax.set(title="贷款申请量渠道构成趋势", xlabel="申请年月", ylabel="申请量")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="申请渠道", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_chart(fig, output_dir, "归因_渠道月度分布")

    april = data.loc[data["月份"].astype(str).eq("2021-04")]
    may = data.loc[data["月份"].astype(str).eq("2021-05")]
    april_total = april["申请书编号（主键）"].nunique()
    may_total = may["申请书编号（主键）"].nunique()
    april_channel = april.groupby("申请渠道")["申请书编号（主键）"].nunique()
    may_channel = may.groupby("申请渠道")["申请书编号（主键）"].nunique()
    channel_order = list(channel_pivot.columns)
    waterfall = pd.DataFrame({"申请渠道": channel_order})
    waterfall["4月申请量"] = waterfall["申请渠道"].map(april_channel).fillna(0).astype(int)
    waterfall["5月申请量"] = waterfall["申请渠道"].map(may_channel).fillna(0).astype(int)
    waterfall["变化量"] = waterfall["5月申请量"] - waterfall["4月申请量"]
    net_change = may_total - april_total
    waterfall["净变化贡献率"] = waterfall["变化量"] / net_change if net_change else np.nan
    waterfall["上期总量"] = april_total
    waterfall["本期总量"] = may_total
    save_table(waterfall, output_dir, "归因_渠道变化贡献")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(0, april_total, color="#577590", width=0.66)
    running = april_total
    labels_waterfall = ["4月总量"]
    for index, row in enumerate(waterfall.itertuples(index=False), start=1):
        change = int(row.变化量)
        bottom = running if change >= 0 else running + change
        ax.bar(index, abs(change), bottom=bottom, color=COLORS[2] if change >= 0 else COLORS[3], width=0.66)
        running += change
        labels_waterfall.append(str(row.申请渠道))
    ax.bar(len(waterfall) + 1, may_total, color=COLORS[0], width=0.66)
    labels_waterfall.append("5月总量")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(range(len(labels_waterfall)), labels_waterfall, rotation=20, ha="right")
    ax.set(title="4月至5月贷款申请量渠道归因", ylabel="贷款申请量变化")
    ax.grid(axis="y", alpha=0.2)
    for index, row in enumerate(waterfall.itertuples(index=False), start=1):
        ax.text(index, max(row.变化量 + april_total, 0), f"{row.变化量:+d}", ha="center", va="bottom", fontsize=9)
    save_chart(fig, output_dir, "归因_渠道变化瀑布图")
    return 3

def main() -> None:
    parser = argparse.ArgumentParser(description="复现教材第5章银行 Power BI 看板与 WPS 归因分析图表")
    parser.add_argument("--chapter-dir", type=Path, default=CHAPTER_DIR, help="第5章附件目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="PNG 与 CSV 输出目录")
    parser.add_argument("--task", choices=("all", "powerbi", "attribution"), default="all", help="全部或指定案例")
    args = parser.parse_args()
    configure_charts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_count = 0
    if args.task in ("all", "powerbi"):
        chart_count += analyze_powerbi(args.chapter_dir, args.output_dir)
        print("Power BI 银行看板复现完成：6 个视觉对象")
    if args.task in ("all", "attribution"):
        chart_count += analyze_attribution(args.chapter_dir, args.output_dir)
        print("WPS 归因分析复现完成：3 张图表，Word 与 Excel 内嵌系列缓存校验通过")
    print(f"共生成 {chart_count} 张图表及对应 CSV，输出至 {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()