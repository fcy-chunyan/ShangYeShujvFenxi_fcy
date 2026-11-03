# -*- coding: utf-8 -*-
"""
02_clean.py —— 数据清洗与治理（原始层 → 清洗层）
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "poutcome"]
UNKNOWN_COLS = ["job", "education", "contact", "poutcome"]

def rebuild_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    重建联系日期。
    原始数据只有 day（1–31）与 month（英文缩写），没有年份。依据 UCI 官方说明，
    数据采集区间为 2008 年 5 月至 2010 年 11 月，且记录按时间先后排列（月份为连续区块）。
    因此按“月份序号回退即跨年”的规则递增年份，得到 year_month 与 contact_date。
    局限：仅用于月度/周度趋势聚合，不用于“星期几”类推断（原始数据未提供星期）。
    """
    month_num = df["month"].map(cfg.MONTH_NUM).astype(int)
    years, year = [], cfg.START_YEAR
    prev = month_num.iloc[0]
    for m in month_num:
        if m < prev:          # 月份回退（如 12 月 → 1 月）判定为跨年
            year += 1
        years.append(year)
        prev = m
    df["year"] = years
    df["month_num"] = month_num
    # 个别月份不存在 31 日，非法日期先置空再回填为当月最后一天
    date_str = (df["year"].astype(str) + "-" + df["month_num"].astype(str).str.zfill(2)
                + "-" + df["day"].astype(str).str.zfill(2))
    dt = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    month_end = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month_num"].astype(str).str.zfill(2) + "-01"
    ) + pd.offsets.MonthEnd(0)
    df["contact_date"] = dt.fillna(month_end)
    df["year_month"] = df["contact_date"].dt.to_period("M").astype(str)
    df["quarter"] = df["contact_date"].dt.to_period("Q").astype(str)
    # 周起始日统一为周一（与 pd.date_range(freq="W-MON") 对齐，避免重采样错位）
    df["week"] = df["contact_date"] - pd.to_timedelta(df["contact_date"].dt.weekday, unit="D")
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """派生分析用指标。"""
    # 目标变量
    df["is_subscribed"] = (df["y"] == "yes").astype(int)
    # 通话时长（分钟）与缩尾值：缩尾抑制极端长通话对均值/聚类的拉扯
    df["duration_min"] = (df["duration"] / 60).round(2)
    p99 = df["duration"].quantile(0.99)
    df["duration_w"] = df["duration"].clip(upper=p99)
    # 余额缩尾，保留符号，避免极端负值/巨额存款主导聚类
    p1, p99b = df["balance"].quantile(0.01), df["balance"].quantile(0.99)
    df["balance_w"] = df["balance"].clip(lower=p1, upper=p99b)
    # 分箱（业务可读）
    df["age_band"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 65, 100],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    df["balance_band"] = pd.cut(df["balance"], bins=[-np.inf, 0, 1000, 5000, np.inf],
                                labels=["负余额", "0-1千", "1千-5千", "5千以上"])
    df["campaign_bucket"] = pd.cut(df["campaign"], bins=[0, 1, 3, 6, np.inf],
                                   labels=["1次", "2-3次", "4-6次", "7次以上"])
    # 历史联系标记：pdays = -1 表示此前从未联系
    df["had_previous_contact"] = (df["pdays"] != -1).astype(int)
    df["pdays_clean"] = df["pdays"].where(df["pdays"] != -1, np.nan)
    # 业务性缺失标记
    for c in UNKNOWN_COLS:
        df[f"{c}_known"] = (df[c] != "unknown").astype(int)
    # 联系渠道合并（unknown 单列，避免误并入 telephone）
    df["channel"] = df["contact"].map({"cellular": "手机", "telephone": "座机"}).fillna("未知")
    return df

def main() -> None:
    print("=" * 68)
    print("Step 2 / 6  数据清洗与治理")
    print("=" * 68)
    df = pd.read_csv(cfg.RAW_MAIN, sep=";", dtype=str, keep_default_na=False)
    raw_rows = len(df)
    report: dict = {"raw_rows": raw_rows}
    # ---------- 1. 文本规范化 ----------
    for c in CAT_COLS + ["y", "month"]:
        df[c] = df[c].str.strip().str.lower()
    for c in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    # ---------- 2. 重复值 ----------
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = dup
    print(f"完全重复记录：{dup} 条，已删除")
    # ---------- 3. 无效通话 ----------
    zero_dur = int((df["duration"] <= 0).sum())
    df = df[df["duration"] > 0].reset_index(drop=True)
    report["zero_duration_removed"] = zero_dur
    print(f"通话时长为 0（未接通）：{zero_dur} 条，已剔除")
    # ---------- 4. 缺失与异常 ----------
    unknown_cnt = {c: int((df[c] == "unknown").sum()) for c in UNKNOWN_COLS}
    report["unknown_counts"] = unknown_cnt
    report["unknown_ratio"] = {c: round(v / len(df), 4) for c, v in unknown_cnt.items()}
    print(f"业务性缺失（unknown）：{unknown_cnt}（保留并打标，不删除）")
    q1, q3 = df["duration"].quantile([0.25, 0.75])
    report["duration_quartiles"] = {"p25": float(q1), "p50": float(df["duration"].median()),
                                   "p75": float(q3), "p99": float(df["duration"].quantile(0.99))}
    report["balance_range"] = {"min": int(df["balance"].min()), "max": int(df["balance"].max()),
                               "p1": float(df["balance"].quantile(0.01)),
                               "p99": float(df["balance"].quantile(0.99))}
    print(f"通话时长分位：P25={q1:.0f}s  P50={df['duration'].median():.0f}s  "
          f"P75={q3:.0f}s  P99={df['duration'].quantile(0.99):.0f}s（缩尾处理）")
    # ---------- 5. 日期重建与特征派生 ----------
    df = rebuild_dates(df)
    df = add_features(df)
    report["date_range"] = [str(df["contact_date"].min().date()), str(df["contact_date"].max().date())]
    report["n_months"] = int(df["year_month"].nunique())
    report["n_weeks"] = int(df["week"].nunique())
    print(f"日期重建区间：{report['date_range'][0]} ~ {report['date_range'][1]}"
          f"（{report['n_months']} 个月 / {report['n_weeks']} 周）")
    report["clean_rows"] = int(len(df))
    report["subscribed"] = int(df["is_subscribed"].sum())
    report["conversion_rate"] = round(float(df["is_subscribed"].mean()), 4)
    print(f"清洗后记录数：{len(df):,}（保留率 {len(df)/raw_rows:.2%}），"
          f"认购 {report['subscribed']:,} 条，转化率 {report['conversion_rate']:.2%}")
    # ---------- 6. 多格式导出 ----------
    csv_path = cfg.DATA_CLEAN / "bank_clean.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[clean] CSV  → {csv_path.name}（{csv_path.stat().st_size/1024/1024:.1f} MB）")
    json_path = cfg.DATA_CLEAN / "bank_clean.json"
    df.to_json(json_path, orient="records", force_ascii=False, indent=None,
               date_format="iso")
    print(f"[clean] JSON → {json_path.name}（{json_path.stat().st_size/1024/1024:.1f} MB）")
    db_path = cfg.DATA_CLEAN / "bank_clean.sqlite"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        # 日期/分箱列转字符串，保证 SQLite 兼容
        out = df.copy()
        for c in ["contact_date", "week"]:
            out[c] = out[c].astype(str)
        out.to_sql("bank_clean", conn, index=False, if_exists="replace")
        conn.execute("CREATE INDEX idx_ym ON bank_clean(year_month)")
        conn.execute("CREATE INDEX idx_job ON bank_clean(job)")
    print(f"[clean] SQLite → {db_path.name}")
    report["outputs"] = [csv_path.name, json_path.name, db_path.name]
    with open(cfg.OUT_TAB / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[log] 已保存 cleaning_report.json")
    print("Step 2 完成。\n")

if __name__ == "__main__":
    main()