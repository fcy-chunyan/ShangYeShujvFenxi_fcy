"""复现第10章 Power BI 中的 AirPassengers 两张预测折线图。"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CHAPTER_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\1《CDA 一级认证教材：商业数据分析》\第10章 使用效应分解方法做预报"
)
OUTPUT_DIR = Path(__file__).parent / "复现结果"
FORECAST_MONTHS = 24
SEASONAL_PERIODS = 12
CONFIDENCE_Z = 1.959963984540054
COLORS = {"history": "#277DA1", "forecast": "#F8961E", "interval": "#F9C784"}

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"AirPassengers.csv 缺少字段: {', '.join(sorted(missing))}")

def read_data(chapter_dir: Path) -> pd.DataFrame:
    source = chapter_dir / "AirPassengers.csv"
    if not source.is_file():
        raise FileNotFoundError(f"未找到数据文件: {source}")
    data = pd.read_csv(source, encoding="utf-8-sig")
    require_columns(data, ["DATE", "AIR", "ln_AIR"])
    data = data[["DATE", "AIR", "ln_AIR"]].copy()
    data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    for column in ("AIR", "ln_AIR"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().sort_values("DATE").set_index("DATE")
    data.index = data.index.to_period("M").to_timestamp()
    if data.index.has_duplicates:
        raise ValueError("数据中存在重复月份")
    expected = pd.date_range(data.index.min(), data.index.max(), freq="MS")
    if not data.index.equals(expected):
        raise ValueError("数据月份不连续，请检查 AirPassengers.csv")
    if (data[["AIR", "ln_AIR"]] <= 0).any().any():
        raise ValueError("AIR 和 ln_AIR 必须为正数")
    return data

def forecast_series(
    series: pd.Series,
    seasonal: str,
    forecast_months: int,
) -> pd.DataFrame:
    model = ExponentialSmoothing(
        series.astype(float),
        trend="add",
        seasonal=seasonal,
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    ).fit(optimized=True)
    forecast = model.forecast(forecast_months)
    residuals = (series - model.fittedvalues).dropna()
    residual_std = float(np.sqrt(np.mean(np.square(residuals))))
    steps = np.arange(1, forecast_months + 1)
    margin = CONFIDENCE_Z * residual_std * np.sqrt(steps)
    future_dates = pd.date_range(
        series.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )
    return pd.DataFrame(
        {
            "日期": future_dates,
            "预测值": np.asarray(forecast, dtype=float),
            "下限_95%近似": np.asarray(forecast, dtype=float) - margin,
            "上限_95%近似": np.asarray(forecast, dtype=float) + margin,
        }
    )

def save_forecast(
    history: pd.Series,
    forecast: pd.DataFrame,
    output_dir: Path,
    name: str,
    value_name: str,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(
        [
            pd.DataFrame({"日期": history.index, "实际值": history.to_numpy()}),
            forecast.rename(columns={"预测值": "预测值", "下限_95%近似": "下限_95%近似", "上限_95%近似": "上限_95%近似"}),
        ],
        ignore_index=True,
    )
    result.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(
        history.index,
        history,
        color=COLORS["history"],
        linewidth=1.8,
        label="实际值",
    )
    forecast_dates = pd.DatetimeIndex(forecast["日期"])
    forecast_values = forecast["预测值"].to_numpy(dtype=float)
    lower = forecast["下限_95%近似"].to_numpy(dtype=float)
    upper = forecast["上限_95%近似"].to_numpy(dtype=float)
    bridge_dates = history.index[-1:].append(forecast_dates)
    bridge_values = np.concatenate(([float(history.iloc[-1])], forecast_values))
    ax.fill_between(
        forecast_dates,
        lower,
        upper,
        color=COLORS["interval"],
        alpha=0.45,
        label="95% 近似置信区间",
    )
    ax.plot(
        bridge_dates,
        bridge_values,
        color=COLORS["forecast"],
        linewidth=2,
        label="预测值",
    )
    ax.axvline(history.index[-1], color="#777777", linewidth=0.8, linestyle="--")
    ax.set(title=title, xlabel="月份", ylabel=value_name)
    ax.xaxis.set_major_locator(YearLocator(2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def run(chapter_dir: Path, output_dir: Path) -> int:
    configure_charts()
    data = read_data(chapter_dir)
    tasks = [
        (
            "AIR",
            "mul",
            "AirPassengers_客运量预测",
            "客运量",
            "AirPassengers 月度客运量预测",
        ),
        (
            "ln_AIR",
            "add",
            "AirPassengers_对数客运量预测",
            "对数客运量",
            "AirPassengers 月度对数客运量预测",
        ),
    ]
    for column, seasonal, name, value_name, title in tasks:
        forecast = forecast_series(data[column], seasonal, FORECAST_MONTHS)
        save_forecast(data[column], forecast, output_dir, name, value_name, title)
        print(f"已输出 {name}，预测 {len(forecast)} 个月")
    return len(tasks)

def main() -> None:
    parser = argparse.ArgumentParser(description="复现第10章 AirPassengers Power BI 预测图")
    parser.add_argument("--chapter-dir", type=Path, default=CHAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.chapter_dir, args.output_dir)

if __name__ == "__main__":
    main()