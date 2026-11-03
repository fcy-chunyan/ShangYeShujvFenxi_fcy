"""
04_forecast.py —— SARIMA 时间序列预测与回测
业务问题：未来 8 周的认购量会是多少？用于排班、名单采购与业绩目标设定。
方法：
1. 以"周"为粒度（124 周），避免月度样本仅 30 期导致参数估计不稳；
2. 在 (p,d,q) 网格上用 AIC 选阶，并与 Holt-Winters 指数平滑做对照；
3. 留出最后 12 周做回测，输出 MAPE / RMSE，说明模型误差水平；
4. 全样本重估后预测未来 8 周，给出点预测与 95% 置信区间；
5. 另做月度 STL 分解，量化趋势与季节性强度。
"""
from __future__ import annotations
import itertools
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
warnings.filterwarnings("ignore")   # 抑制收敛警告，结果中会显式报告是否收敛
TEST_WEEKS = 12      # 回测窗口
FORECAST_WEEKS = 8   # 预测步长

def build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """按周聚合：触达量、认购量、转化率、平均通话时长。"""
    w = (df.groupby("week")
           .agg(contacts=("is_subscribed", "size"),
                subscribed=("is_subscribed", "sum"),
                conv_rate=("is_subscribed", "mean"),
                avg_duration=("duration", "mean"))
           .reset_index().sort_values("week"))
    # 补齐缺失周（若某周无外呼记录），保证时间序列等间隔
    full = pd.date_range(w["week"].min(), w["week"].max(), freq="W-MON")
    w = w.set_index("week").reindex(full).rename_axis("week").reset_index()
    w[["contacts", "subscribed"]] = w[["contacts", "subscribed"]].fillna(0)
    w["conv_rate"] = w["conv_rate"].fillna(0)
    w["avg_duration"] = w["avg_duration"].interpolate()
    return w

def grid_search(train: pd.Series, p_range=range(0, 3), q_range=range(0, 3)) -> tuple[tuple, pd.DataFrame]:
    rows = []
    best = (None, np.inf)
    for p, q in itertools.product(p_range, q_range):
        try:
            m = SARIMAX(train, order=(p, 1, q), trend="n",
                        enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False)
            rows.append({"p": p, "d": 1, "q": q, "aic": round(float(r.aic), 2),
                         "bic": round(float(r.bic), 2), "converged": bool(r.mle_retvals.get("converged", True))})
            if r.aic < best[1]:
                best = ((p, 1, q), r.aic)
        except Exception as e:              # 个别参数组合可能不收敛，记录后跳过
            rows.append({"p": p, "d": 1, "q": q, "aic": np.nan, "bic": np.nan,
                         "converged": False, "error": str(e)[:60]})
    return best[0], pd.DataFrame(rows)

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """回测误差：MAPE（%）、RMSE、MAE。MAPE 对 0 值做保护。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"MAPE_%": round(mape, 2), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}

def main() -> None:
    print("=" * 68)
    print("Step 4 / 6  SARIMA 预测与回测")
    print("=" * 68)
    df = pd.read_csv(cfg.DATA_CLEAN / "bank_clean.csv", parse_dates=["contact_date", "week"])
    weekly = build_weekly(df)
    print(f"周度序列：{len(weekly)} 周（{weekly['week'].min().date()} ~ {weekly['week'].max().date()}）")
    y = weekly.set_index("week")["subscribed"].astype(float)
    y.index = pd.DatetimeIndex(y.index, freq="W-MON")
    train, test = y.iloc[:-TEST_WEEKS], y.iloc[-TEST_WEEKS:]
    print(f"训练集 {len(train)} 周 / 回测集 {len(test)} 周")
    # ---------- 1. 选阶 ----------
    best_order, grid = grid_search(train)
    cfg.save_table(grid, "sarima_grid")
    print(f"AIC 最优阶数：SARIMA{best_order}")
    # ---------- 2. 回测 ----------
    model = SARIMAX(train, order=best_order, trend="n",
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = model.get_forecast(steps=TEST_WEEKS)
    pred_mean = pred.predicted_mean.values
    pred_ci = pred.conf_int(alpha=0.05).values
    m_sarima = metrics(test.values, pred_mean)
    # 对照模型：Holt 阻尼趋势（周序列仅 2.5 个年度周期，52 周季节项无法稳健估计，
    # 故只保留趋势项并加阻尼，作为一个"常规业务外推"的参照）
    try:
        hw = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                  seasonal=None, initialization_method="estimated").fit()
        hw_pred = hw.forecast(TEST_WEEKS).values
        m_hw = metrics(test.values, hw_pred)
    except Exception as e:
        print(f"Holt 趋势模型拟合失败（{e}），仅报告 SARIMA")
        hw_pred, m_hw = np.full(TEST_WEEKS, np.nan), {"MAPE_%": np.nan, "RMSE": np.nan, "MAE": np.nan}
    # 朴素基线：用训练集最后 4 周均值，衡量模型是否真的优于"拍平均数"
    naive = np.full(TEST_WEEKS, train.iloc[-4:].mean())
    m_naive = metrics(test.values, naive)
    comp = pd.DataFrame([
        {"model": f"SARIMA{best_order}", **m_sarima},
        {"model": "Holt 阻尼趋势", **m_hw},
        {"model": "朴素基线(近4周均值)", **m_naive},
    ])
    cfg.save_table(comp, "forecast_metrics")
    print(f"回测误差：SARIMA MAPE={m_sarima['MAPE_%']}% RMSE={m_sarima['RMSE']:.1f}；"
          f"HW MAPE={m_hw['MAPE_%']}%；朴素基线 MAPE={m_naive['MAPE_%']}%")
    # ---------- 3. 全样本重估 + 未来 8 周预测 ----------
    final = SARIMAX(y, order=best_order, trend="n",
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = final.get_forecast(steps=FORECAST_WEEKS)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=0.05)
    # 认购量不可能为负，置信区间下界截断到 0（论文中会说明该处理）
    future = pd.DataFrame({
        "week": fc_mean.index,
        "forecast": np.round(np.clip(fc_mean.values, 0, None), 1),
        "lower_95": np.round(np.clip(fc_ci.iloc[:, 0].values, 0, None), 1),
        "upper_95": np.round(fc_ci.iloc[:, 1].values, 1),
        "type": "forecast",
    })
    hist = pd.DataFrame({"week": y.index, "forecast": y.values,
                         "lower_95": np.nan, "upper_95": np.nan, "type": "history"})
    out = pd.concat([hist, future], ignore_index=True)
    cfg.save_table(out, "forecast_weekly")
    print(f"未来 {FORECAST_WEEKS} 周预测：周均 {fc_mean.mean():.1f} 人次，"
          f"区间 {max(fc_ci.iloc[:, 0].mean(), 0):.1f} ~ {fc_ci.iloc[:, 1].mean():.1f}（下界截断至 0）")
    # ---------- 4. 月度分解（趋势/季节/残差） ----------
    monthly = (df.groupby("year_month")
                 .agg(subscribed=("is_subscribed", "sum"),
                      contacts=("is_subscribed", "size"))
                 .reset_index().sort_values("year_month"))
    monthly.index = pd.PeriodIndex(monthly["year_month"], freq="M").to_timestamp()
    dec = seasonal_decompose(monthly["subscribed"], model="additive", period=12)
    dec_df = pd.DataFrame({
        "month": monthly["year_month"].values,
        "observed": dec.observed.values.round(2),
        "trend": dec.trend.values.round(2),
        "seasonal": dec.seasonal.values.round(2),
        "resid": dec.resid.values.round(2),
    })
    cfg.save_table(dec_df, "seasonal_decompose")
    seasonal_strength = max(0.0, 1 - np.nanvar(dec.resid) / np.nanvar(dec.observed - dec.trend))
    print(f"季节性强度（1 - Var(resid)/Var(detrended)）：{seasonal_strength:.2f}")
    res = {
        "order": list(best_order),
        "train_weeks": int(len(train)),
        "test_weeks": int(TEST_WEEKS),
        "backtest": m_sarima,
        "hw_backtest": m_hw,
        "naive_backtest": m_naive,
        "forecast_mean": round(float(fc_mean.mean()), 1),
        "forecast_ci": [round(float(max(fc_ci.iloc[:, 0].mean(), 0)), 1),
                        round(float(fc_ci.iloc[:, 1].mean()), 1)],
        "seasonal_strength": round(float(seasonal_strength), 3),
    }
    with open(cfg.OUT_TAB / "forecast_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[log] 已保存 forecast_results.json")
    print("Step 4 完成。\n")

if __name__ == "__main__":
    main()