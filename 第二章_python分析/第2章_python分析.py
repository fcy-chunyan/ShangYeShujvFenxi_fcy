"""将教材第2章的电子表格和 Power BI 分析改写为 pandas/matplotlib。"""
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
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\1《CDA 一级认证教材：商业数据分析》\第2章 数据分析方法"
)
OUTPUT_DIR = Path(__file__).parent / "复现结果"

def read_csv_auto(path: Path) -> pd.DataFrame:
    """兼容教材数据常见的 UTF-8 与中文 Windows 编码。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")

def read_table(path: Path, sheet_name: str | int = 0, header: int | None = 0) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_csv_auto(path)
    return pd.read_excel(path, sheet_name=sheet_name, header=header)

def require_columns(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{source.name} 缺少字段: {', '.join(sorted(missing))}")

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def save_chart(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def _workbook(chapter_dir: Path, relative: str) -> Path:
    path = chapter_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"未找到教材文件: {path}")
    return path

def analyze_boston(chapter_dir: Path, output_dir: Path) -> pd.DataFrame:
    """以月销售额份额和市场增速划分品牌波士顿矩阵。"""
    source = _workbook(chapter_dir, "1、分析的基础范式/data-无答案/1-1波士顿矩阵.xlsx")
    sales = read_table(source, sheet_name="销售表")
    require_columns(sales, ["销售日期", "品牌", "单价", "销量"], source)
    sales["销售日期"] = pd.to_datetime(sales["销售日期"])
    sales["销售额"] = pd.to_numeric(sales["单价"]) * pd.to_numeric(sales["销量"])
    sales["月份"] = sales["销售日期"].dt.to_period("M").dt.to_timestamp()
    monthly = sales.groupby(["月份", "品牌"], as_index=False)["销售额"].sum()
    monthly["品牌市场份额"] = monthly["销售额"] / monthly.groupby("月份")["销售额"].transform("sum")
    market = monthly.groupby("月份")["销售额"].sum().sort_index()
    market_growth = market.pct_change()
    monthly["市场环比增长率"] = monthly["月份"].map(market_growth)
    summary = monthly.groupby("品牌", as_index=False).agg(
        销售额=("销售额", "sum"),
        平均市场份额=("品牌市场份额", "mean"),
        平均市场增长率=("市场环比增长率", "mean"),
    )
    share_mid = summary["平均市场份额"].mean()
    growth_mid = market_growth.mean()
    summary["品牌分类"] = np.select(
        [
            summary["平均市场份额"].ge(share_mid) & summary["平均市场增长率"].ge(growth_mid),
            summary["平均市场份额"].ge(share_mid) & summary["平均市场增长率"].lt(growth_mid),
            summary["平均市场份额"].lt(share_mid) & summary["平均市场增长率"].ge(growth_mid),
        ],
        ["明星", "现金牛", "问题"],
        default="瘦狗",
    )
    save_table(monthly, output_dir, "波士顿矩阵_月品牌明细")
    save_table(summary, output_dir, "波士顿矩阵_品牌分类")
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = 300 + 1400 * summary["平均市场份额"]
    ax.scatter(summary["平均市场份额"], summary["平均市场增长率"], s=sizes, alpha=0.65, color="#1689a7")
    for row in summary.itertuples():
        ax.annotate(str(row.品牌), (row.平均市场份额, row.平均市场增长率), xytext=(5, 4), textcoords="offset points")
    ax.axvline(share_mid, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(growth_mid, color="#777777", linestyle="--", linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="品牌波士顿矩阵", xlabel="平均月市场份额", ylabel="平均市场环比增长率")
    ax.grid(alpha=0.2)
    save_chart(fig, output_dir, "波士顿矩阵")
    return summary

def analyze_rfm(chapter_dir: Path, output_dir: Path, analysis_date: str | None = None) -> pd.DataFrame:
    """按用户汇总最近购买间隔、去重订单数、消费金额，并按四分位数分档。"""
    source = _workbook(chapter_dir, "1、分析的基础范式/data-无答案/1-2RFM模型.xlsx")
    transactions = read_table(source, sheet_name="原始数据")
    require_columns(transactions, ["流水号", "用户编号", "销售日期", "销售金额"], source)
    transactions["销售日期"] = pd.to_datetime(transactions["销售日期"])
    transactions["销售金额"] = pd.to_numeric(transactions["销售金额"])
    as_of = pd.Timestamp(analysis_date) if analysis_date else transactions["销售日期"].max().normalize() + pd.Timedelta(days=1)
    customers = transactions.groupby("用户编号").agg(
        最近消费日期=("销售日期", "max"),
        消费次数=("流水号", "nunique"),
        消费金额=("销售金额", "sum"),
    )
    customers["R_最近消费间隔天数"] = (as_of - customers["最近消费日期"].dt.normalize()).dt.days
    q1, q3 = customers["R_最近消费间隔天数"].quantile([0.25, 0.75])
    customers["R得分"] = np.select([customers["R_最近消费间隔天数"].le(q1), customers["R_最近消费间隔天数"].ge(q3)], [3, 1], default=2)
    for metric, score in [("消费次数", "F得分"), ("消费金额", "M得分")]:
        q1, q3 = customers[metric].quantile([0.25, 0.75])
        customers[score] = np.select([customers[metric].le(q1), customers[metric].ge(q3)], [1, 3], default=2)
    customers["RFM总分"] = customers[["R得分", "F得分", "M得分"]].sum(axis=1)
    customers["用户分层"] = pd.cut(customers["RFM总分"], [0, 3, 5, 7, 9], labels=["低价值", "待发展", "重要发展", "重要价值"], include_lowest=True)
    customers = customers.reset_index()
    customers["分析时点"] = as_of
    save_table(customers, output_dir, "RFM用户分层")
    counts = customers["用户分层"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#1689a7")
    ax.set(title="RFM 用户分层", xlabel="用户分层", ylabel="用户数")
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, output_dir, "RFM用户分层")
    return customers

def analyze_loyalty(chapter_dir: Path, output_dir: Path, analysis_date: str = "2021-01-01") -> pd.DataFrame:
    """依教材定义标记最近 13 周潜在用户与易流失用户。"""
    source = _workbook(chapter_dir, "1、分析的基础范式/data-无答案/1-3用户忠诚度模型.xlsx")
    transactions = read_table(source, sheet_name="销售流水表")
    require_columns(transactions, ["会员编号", "订单编号", "消费日期"], source)
    transactions["消费日期"] = pd.to_datetime(transactions["消费日期"])
    as_of = pd.Timestamp(analysis_date)
    cutoff = as_of - pd.Timedelta(weeks=13)
    transactions["窗口"] = np.where(transactions["消费日期"].ge(cutoff), "最近13周", "此前")
    counts = transactions.groupby(["会员编号", "窗口"])["订单编号"].nunique().unstack(fill_value=0)
    for period in ("此前", "最近13周"):
        if period not in counts:
            counts[period] = 0
    potential = counts["此前"].eq(0) & counts["最近13周"].gt(0)
    at_risk = counts["此前"].gt(0) & counts["最近13周"].eq(0)
    segments = counts.loc[potential | at_risk, ["此前", "最近13周"]].copy()
    segments["用户类别"] = np.where(potential.loc[segments.index], "潜在用户", "易流失用户")
    segments["分析时点"] = as_of
    segments["最近13周起始日"] = cutoff
    segments = segments.reset_index()
    save_table(segments, output_dir, "忠诚度_潜在与易流失用户")
    return segments

def analyze_cohort(chapter_dir: Path, output_dir: Path) -> pd.DataFrame:
    """按注册日建立同期群，计算用户在注册后第 N 日的留存率。"""
    source = _workbook(chapter_dir, "1、分析的基础范式/data-无答案/1-4同期群分析.xlsx")
    logins = read_table(source, sheet_name="用户登录明细")
    require_columns(logins, ["用户编号", "注册日期", "登录时点"], source)
    logins["注册日期"] = pd.to_datetime(logins["注册日期"]).dt.normalize()
    logins["登录日期"] = pd.to_datetime(logins["登录时点"]).dt.normalize()
    logins["距注册日"] = (logins["登录日期"] - logins["注册日期"]).dt.days
    active = logins.loc[logins["距注册日"].ge(0), ["用户编号", "注册日期", "距注册日"]].drop_duplicates()
    cohort_sizes = active.groupby("注册日期")["用户编号"].nunique()
    retained = active.groupby(["注册日期", "距注册日"])["用户编号"].nunique().unstack(fill_value=0)
    rates = retained.div(cohort_sizes, axis=0)
    rates.columns = [f"第{day}日" for day in rates.columns]
    rates.index.name = "注册日期"
    result = rates.reset_index()
    save_table(result, output_dir, "同期群_日留存率")
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(rates.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set(title="注册日同期群留存率", xlabel="距注册天数", ylabel="注册日期")
    ax.set_xticks(range(len(rates.columns)), rates.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rates.index)), [date.strftime("%m-%d") for date in rates.index])
    fig.colorbar(image, ax=ax, format=PercentFormatter(1), label="留存率")
    save_chart(fig, output_dir, "同期群日留存热力图")
    registrations = logins.groupby("注册日期")["用户编号"].nunique().rename("注册用户数").reset_index()
    registrations["环比增长率"] = registrations["注册用户数"].pct_change()
    save_table(registrations, output_dir, "同期群_注册趋势")
    return result

def analyze_funnel(chapter_dir: Path, output_dir: Path) -> pd.DataFrame:
    source = _workbook(chapter_dir, "1、分析的基础范式/data-无答案/2-1漏斗图.xlsx")
    raw = read_table(source, sheet_name=0, header=2)
    raw = raw.dropna(subset=["阶段", "用户数"])
    raw["用户数"] = pd.to_numeric(raw["用户数"])
    raw["阶段转化率"] = raw["用户数"] / raw["用户数"].shift(1)
    raw["总体转化率"] = raw["用户数"] / raw["用户数"].iloc[0]
    save_table(raw, output_dir, "销售漏斗")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(raw["阶段"], raw["用户数"], color="#1689a7")
    ax.invert_yaxis()
    for index, row in raw.iterrows():
        ax.text(row["用户数"], index, f"  {row['用户数']:g}  ({row['总体转化率']:.1%})", va="center")
    ax.set(title="销售漏斗", xlabel="用户数", ylabel="阶段")
    save_chart(fig, output_dir, "销售漏斗")
    return raw

def analyze_correlation(chapter_dir: Path, output_dir: Path) -> pd.DataFrame:
    source = _workbook(chapter_dir, "1、分析的基础范式/data-无答案/3-1相关分析.xlsx")
    data = read_table(source, sheet_name="原始数据")
    require_columns(data, ["x", "y"], source)
    data = data[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
    correlation = data["x"].corr(data["y"])
    save_table(data, output_dir, "相关分析_原始数据")
    pd.DataFrame([{"样本数": len(data), "Pearson相关系数": correlation}]).to_csv(
        output_dir / "相关分析_统计结果.csv", index=False, encoding="utf-8-sig"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["x"], data["y"], alpha=0.75, color="#1689a7")
    slope, intercept = np.polyfit(data["x"], data["y"], 1)
    x_line = np.linspace(data["x"].min(), data["x"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#db5a42", linewidth=1.5)
    ax.set(title=f"相关分析（Pearson r={correlation:.3f}）", xlabel="x", ylabel="y")
    ax.grid(alpha=0.2)
    save_chart(fig, output_dir, "相关分析")
    return data

def analyze_relationship_model(chapter_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    source = _workbook(chapter_dir, "5、PowerBI基本操作/创建数据模型示例.xlsx")
    fact = read_table(source, sheet_name="销售业绩表")
    customers = read_table(source, sheet_name="客户信息表")
    cities = read_table(source, sheet_name="城市信息表")
    products = read_table(source, sheet_name="产品信息表")
    model = fact.merge(customers, on="客户ID", how="left", validate="many_to_one")
    model = model.merge(cities, on="城市编号", how="left", validate="many_to_one")
    model = model.merge(products, on="产品编号", how="left", validate="many_to_one")
    if model[["客户名称", "城市名", "产品"]].isna().any().any():
        raise ValueError("关系模型中存在未匹配的客户、城市或产品外键")
    reports = {
        "模型_区域产品销售": model.groupby(["区域名", "产品系列"], as_index=False)["金额（$M）"].sum(),
        "模型_企业规模销售": model.groupby("企业规模", as_index=False)["金额（$M）"].agg(销售额="sum", 交易数="size"),
        "模型_城市销售": model.groupby(["城市名", "城市经理"], as_index=False)["金额（$M）"].sum(),
    }
    for name, report in reports.items():
        save_table(report, output_dir, name)
    reports["明细"] = model
    save_table(model, output_dir, "模型_关联明细")
    return reports

def analyze_retail_orders(chapter_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    source = _workbook(chapter_dir, "5、PowerBI基本操作/零售订单表.xlsx")
    orders = read_table(source)
    require_columns(orders, ["订单号", "订单时间", "实际支付", "城市", "数量", "商品类别"], source)
    orders["订单时间"] = pd.to_datetime(orders["订单时间"])
    orders["实际支付"] = pd.to_numeric(orders["实际支付"], errors="coerce").fillna(0)
    orders["数量"] = pd.to_numeric(orders["数量"], errors="coerce").fillna(0)
    orders["月份"] = orders["订单时间"].dt.to_period("M").astype(str)
    reports = {
        "零售_月度趋势": orders.groupby("月份", as_index=False).agg(销售额=("实际支付", "sum"), 订单数=("订单号", "nunique"), 销量=("数量", "sum")),
        "零售_城市销售": orders.groupby("城市", as_index=False).agg(销售额=("实际支付", "sum"), 订单数=("订单号", "nunique")).sort_values("销售额", ascending=False),
        "零售_品类销售": orders.groupby("商品类别", as_index=False).agg(销售额=("实际支付", "sum"), 订单数=("订单号", "nunique")).sort_values("销售额", ascending=False),
    }
    for name, report in reports.items():
        save_table(report, output_dir, name)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reports["零售_月度趋势"]["月份"], reports["零售_月度趋势"]["销售额"], marker="o", color="#1689a7")
    ax.set(title="零售订单月度销售额", xlabel="月份", ylabel="实际支付")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    save_chart(fig, output_dir, "零售月度销售趋势")
    return reports

def analyze_etl(chapter_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    """以 pandas 展示数据清洗、拆分为多行、横向关联和纵向追加。"""
    etl_dir = chapter_dir / "5、PowerBI基本操作/ETL练习文件"
    raw_file = _workbook(chapter_dir, "5、PowerBI基本操作/ETL练习文件/ETL演示数据/数据清洗示例数据.xlsx")
    raw = read_table(raw_file, sheet_name="sheet1", header=1).dropna(how="all").copy()
    cleaned = raw.drop_duplicates().dropna(axis=1, how="all")
    for column in ("实付金额", "下单数量", "year", "month", "day"):
        if column in cleaned:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    reports: dict[str, pd.DataFrame] = {"清洗数据": cleaned}
    split_file = _workbook(chapter_dir, "5、PowerBI基本操作/ETL练习文件/ETL演示数据/拆分列示例数据.xlsx")
    split_source = read_table(split_file)
    split_col = "订单号"
    reports["拆分列为多行"] = split_source.assign(**{split_col: split_source[split_col].astype(str).str.split(",")}).explode(split_col, ignore_index=True)
    append_files = sorted((etl_dir / "合并演示数据").glob("纵向合并演示-*.xlsx"))
    append_frames = [read_table(path) for path in append_files]
    reports["纵向追加"] = pd.concat(append_frames, ignore_index=True, sort=False) if append_frames else pd.DataFrame()
    merge_file = _workbook(chapter_dir, "5、PowerBI基本操作/ETL练习文件/合并演示数据/BI横向合并练习文件 .xlsx")
    students = read_table(merge_file, sheet_name="学生表")
    enrollments = read_table(merge_file, sheet_name="选课表")
    courses = read_table(merge_file, sheet_name="课程表")
    reports["横向关联"] = enrollments.merge(students, on="学号", how="left", validate="many_to_one").merge(courses, on="课程号", how="left", validate="many_to_one")
    for name, report in reports.items():
        save_table(report, output_dir, f"ETL_{name}")
    return reports

def analyze_clothing_retail(
    chapter_dir: Path,
    output_dir: Path,
    store_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """按教材 Power BI 服装零售案例复现关系模型、指标与报表视图。"""
    source = _workbook(chapter_dir, "5、PowerBI基本操作/案例2/服装零售-原始数据.xlsx")
    read_sheet = lambda sheet: read_table(source, sheet_name=sheet)
    orders = read_sheet("零售单")
    details = read_sheet("零售单明细")
    attributes = read_sheet("属性表")
    colors = read_sheet("颜色表")[["颜色id", "颜色名称"]]
    sizes = read_sheet("尺码表")[["尺码id", "尺码名称"]]
    staff = read_sheet("员工表")[["员工id", "员工工号", "员工姓名"]]
    sku = read_sheet("SKU表")
    styles = read_sheet("款号表")
    stores = read_sheet("店铺表")[["店铺id", "店铺名称", "商圈", "店铺面积"]]

    brand = attributes.rename(columns={"属性id": "品牌id", "属性名称": "品牌"})
    category = attributes.rename(columns={"属性id": "大类id", "属性名称": "大类"})
    sku = sku.merge(styles, on="款号id", how="left", validate="many_to_one")
    sku = sku.merge(brand, on="品牌id", how="left", validate="many_to_one")
    sku = sku.merge(category, on="大类id", how="left", validate="many_to_one")
    sku = sku.merge(colors, on="颜色id", how="left", validate="many_to_one")
    sku = sku.merge(sizes, on="尺码id", how="left", validate="many_to_one")
    model = details.merge(sku, left_on="商品id", right_on="SKUid", how="left", validate="many_to_one")
    model = model.merge(orders, on="订单id", how="left", validate="many_to_one", suffixes=("_明细", "_订单"))
    model = model.merge(staff, on="员工id", how="left", validate="many_to_one")
    model = model.merge(stores, on="店铺id", how="left", validate="many_to_one")
    model["订单日期"] = pd.to_datetime(model["订单日期"])
    model["成交金额"] = pd.to_numeric(model["成交金额_明细"], errors="coerce").fillna(0)
    model["零售金额"] = pd.to_numeric(model["零售金额_明细"], errors="coerce").fillna(0)
    model["数量明细"] = pd.to_numeric(model["数量_明细"], errors="coerce").fillna(0)
    if model["订单日期"].isna().any():
        raise ValueError("服装零售明细存在未匹配订单日期的订单")
    model["月份"] = model["订单日期"].dt.to_period("M").dt.to_timestamp()
    model["日期"] = model["订单日期"].dt.normalize()
    pieces_by_order = model.groupby("订单id")["数量明细"].sum()
    model["联单件数"] = model["订单id"].map(pieces_by_order)
    model["联单类型"] = np.where(model["联单件数"].le(1), "1单1件", model["联单件数"].round().astype(int).astype(str) + "联单")
    model["坪效"] = model["成交金额"] / pd.to_numeric(model["店铺面积"], errors="coerce")

    filtered = model
    if store_name:
        filtered = filtered.loc[filtered["店铺名称"].eq(store_name)]
        if filtered.empty:
            available = ", ".join(sorted(model["店铺名称"].dropna().unique()))
            raise ValueError(f"找不到店铺 {store_name!r}；可选店铺：{available}")
    minimum = pd.Timestamp(start_date).normalize() if start_date else filtered["日期"].min()
    maximum = pd.Timestamp(end_date).normalize() if end_date else filtered["日期"].max()
    if minimum > maximum:
        raise ValueError("开始日期不能晚于结束日期")
    current = filtered.loc[filtered["日期"].between(minimum, maximum)].copy()
    duration = (maximum - minimum).days + 1
    previous_start = minimum - pd.Timedelta(days=duration)
    previous = filtered.loc[filtered["日期"].between(previous_start, minimum - pd.Timedelta(days=1))]

    def summarize(frame: pd.DataFrame) -> dict[str, float]:
        sales = frame["成交金额"].sum()
        retail = frame["零售金额"].sum()
        pieces = frame["数量明细"].sum()
        order_count = frame["订单id"].nunique()
        member_count = frame.loc[frame["会员id"].notna(), "会员id"].nunique()
        member_sales = frame.loc[frame["会员id"].notna(), "成交金额"].sum()
        area = frame.drop_duplicates("店铺id")["店铺面积"].pipe(pd.to_numeric, errors="coerce").sum()
        return {
            "成交金额": sales,
            "零售金额": retail,
            "折扣率": sales / retail if retail else np.nan,
            "会员人数": member_count,
            "会员人均消费": member_sales / member_count if member_count else np.nan,
            "订单数": order_count,
            "客单价": sales / order_count if order_count else np.nan,
            "件数": pieces,
            "件单价": sales / pieces if pieces else np.nan,
            "连带率": pieces / order_count if order_count else np.nan,
            "坪效": sales / area if area else np.nan,
        }

    current_metrics = summarize(current)
    previous_metrics = summarize(previous)
    kpis = pd.DataFrame(
        [
            {
                "指标": metric,
                "当前值": value,
                "上期值": previous_metrics[metric],
                "环比": value / previous_metrics[metric] - 1
                if previous_metrics[metric] not in (0, np.nan) and pd.notna(previous_metrics[metric])
                else np.nan,
            }
            for metric, value in current_metrics.items()
        ]
    )
    daily = current.groupby("日期", as_index=False).agg(
        成交金额=("成交金额", "sum"),
        零售金额=("零售金额", "sum"),
        订单数=("订单id", "nunique"),
        件数=("数量明细", "sum"),
        会员人数=("会员id", "nunique"),
    )
    daily["折扣率"] = daily["成交金额"] / daily["零售金额"].replace(0, np.nan)
    daily["成交金额环比"] = daily["成交金额"].pct_change()
    by_category = current.groupby("大类", as_index=False).agg(成交金额=("成交金额", "sum"), 件数=("数量明细", "sum"), 订单数=("订单id", "nunique")).sort_values("成交金额", ascending=False)
    by_brand = current.groupby("品牌", as_index=False).agg(成交金额=("成交金额", "sum"), 折扣率=("成交金额", "sum")).sort_values("成交金额", ascending=False)
    brand_retail = current.groupby("品牌")["零售金额"].sum()
    by_brand["折扣率"] = by_brand["品牌"].map(by_brand.set_index("品牌")["成交金额"]) / by_brand["品牌"].map(brand_retail).replace(0, np.nan)
    by_staff = current.groupby(["员工工号", "员工姓名"], as_index=False).agg(成交金额=("成交金额", "sum"), 订单数=("订单id", "nunique"), 件数=("数量明细", "sum")).sort_values("成交金额", ascending=False)
    by_store = current.groupby(["店铺名称", "商圈"], as_index=False).agg(成交金额=("成交金额", "sum"), 订单数=("订单id", "nunique"), 件数=("数量明细", "sum"))
    by_store["坪效"] = by_store["店铺名称"].map(current.drop_duplicates("店铺名称").set_index("店铺名称")["店铺面积"].pipe(pd.to_numeric, errors="coerce"))
    by_store["坪效"] = by_store["成交金额"] / by_store["坪效"]
    order_mix = current.drop_duplicates("订单id").groupby("联单类型", as_index=False).agg(订单数=("订单id", "nunique"))
    order_mix["订单占比"] = order_mix["订单数"] / order_mix["订单数"].sum()
    reports = {
        "服装零售_KPI及环比": kpis,
        "服装零售_每日趋势": daily,
        "服装零售_品类销售": by_category,
        "服装零售_品牌销售": by_brand,
        "服装零售_员工销售": by_staff,
        "服装零售_店铺销售": by_store,
        "服装零售_联单结构": order_mix,
        "服装零售_筛选明细": current,
    }
    for name, report in reports.items():
        save_table(report, output_dir, name)
    save_table(current, output_dir, "服装零售_筛选明细")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily["日期"], daily["成交金额"], color="#303f9f", marker="o", linewidth=1.8)
    ax.set(title="服装零售成交金额趋势", xlabel="日期", ylabel="成交金额")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m月%d日"))
    ax.grid(axis="y", alpha=0.2)
    save_chart(fig, output_dir, "服装零售_成交金额趋势")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, frame, label in zip(axes, [by_staff.head(8), by_category.head(8), order_mix], ["员工成交金额", "大类成交金额", "订单数按联单类型"]):
        if label == "订单数按联单类型":
            axis.pie(frame["订单数"], labels=frame["联单类型"], autopct="%1.0f%%", startangle=90)
        else:
            category_column = "员工姓名" if label == "员工成交金额" else "大类"
            axis.barh(frame.iloc[::-1][category_column], frame.iloc[::-1]["成交金额"], color="#303f9f")
        axis.set_title(label)
    save_chart(fig, output_dir, "服装零售_销售构成")
    return reports

def method_trend(data: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return data.groupby(x, as_index=False)[y].sum().sort_values(x)

def method_compare(data: pd.DataFrame, group: str, value: str) -> pd.DataFrame:
    return data.groupby(group, as_index=False)[value].agg(样本数="size", 合计="sum", 均值="mean")

def method_composition(data: pd.DataFrame, group: str, value: str) -> pd.DataFrame:
    result = data.groupby(group, as_index=False)[value].sum()
    result["占比"] = result[value] / result[value].sum()
    return result.sort_values(value, ascending=False)

def method_distribution(data: pd.DataFrame, value: str) -> pd.DataFrame:
    values = pd.to_numeric(data[value], errors="coerce").dropna()
    return pd.DataFrame([values.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])])

def method_relationship(data: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return data[[x, y]].apply(pd.to_numeric, errors="coerce").corr(method="pearson").reset_index(names="指标")

def method_flow(data: pd.DataFrame, source: str, target: str, value: str) -> pd.DataFrame:
    return data.groupby([source, target], as_index=False)[value].sum().sort_values(value, ascending=False)

def run_fundamentals(chapter_dir: Path, output_dir: Path) -> None:
    results = [
        analyze_boston(chapter_dir, output_dir),
        analyze_rfm(chapter_dir, output_dir),
        analyze_loyalty(chapter_dir, output_dir),
        analyze_cohort(chapter_dir, output_dir),
        analyze_funnel(chapter_dir, output_dir),
        analyze_correlation(chapter_dir, output_dir),
    ]
    print(f"基础分析范式完成：{len(results)} 项，输出至 {output_dir.resolve()}")

def run_powerbi(
    chapter_dir: Path,
    output_dir: Path,
    store_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    model = analyze_relationship_model(chapter_dir, output_dir)
    retail = analyze_retail_orders(chapter_dir, output_dir)
    etl = analyze_etl(chapter_dir, output_dir)
    clothing = analyze_clothing_retail(chapter_dir, output_dir, store_name, start_date, end_date)
    print(
        f"Power BI 案例完成：关系模型 {len(model['明细']):,} 行、一般零售汇总 {sum(len(frame) for frame in retail.values()):,} 行、"
        f"服装零售明细 {len(clothing['服装零售_筛选明细']):,} 行、ETL {len(etl)} 项。"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="教材第2章商业分析案例的 Python 复现")
    parser.add_argument("--chapter-dir", type=Path, default=CHAPTER_DIR, help="教材第2章附件目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="分析结果目录")
    parser.add_argument("--task", choices=("all", "fundamentals", "powerbi"), default="all", help="运行全部、基础分析范式或 Power BI 案例")
    parser.add_argument("--store-name", help="服装零售店铺筛选，例如 CRYSTAL百联")
    parser.add_argument("--start-date", help="服装零售筛选起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", help="服装零售筛选结束日期，格式 YYYY-MM-DD")
    args = parser.parse_args()
    configure_charts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task in ("all", "fundamentals"):
        run_fundamentals(args.chapter_dir, args.output_dir)
    if args.task in ("all", "powerbi"):
        run_powerbi(args.chapter_dir, args.output_dir, args.store_name, args.start_date, args.end_date)

if __name__ == "__main__":
    main()