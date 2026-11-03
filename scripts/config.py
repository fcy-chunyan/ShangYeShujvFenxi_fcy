# -*- coding: utf-8 -*-
"""
config.py —— 全局配置（路径、中文字体、配色、通用工具函数）

所有脚本共用本模块，保证：
1. 路径唯一来源，避免各脚本硬编码；
2. 中文字体、配色、分辨率（150dpi）统一；
3. 图表命名规范统一（figXX_中文名.png）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 路径配置
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_DASH = ROOT / "outputs" / "dashboard"
DOCS = ROOT / "docs"

for _p in (DATA_CLEAN, OUT_FIG, OUT_TAB, OUT_DASH, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# 原始数据文件名（UCI Bank Marketing，分号分隔）
RAW_MAIN = DATA_RAW / "bank-full.csv"
RAW_AUX = DATA_RAW / "bank-additional-full.csv"

# ---------------------------------------------------------------- 字体与绘图风格
# Windows 常见中文字体；如缺失会依次回退，避免出现“方块字”
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Source Han Sans SC", "Noto Sans CJK SC", "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110          # 屏幕显示
plt.rcParams["savefig.dpi"] = 150         # 作业要求：≥150dpi
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["axes.edgecolor"] = "#B8C2CC"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#E6EAF0"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["figure.facecolor"] = "white"

# ---------------------------------------------------------------- 配色方案
# 统一的商务蓝主色 + 强调橙，避免每张图各自配色导致风格割裂
COLORS = {
    "primary": "#2E5AAC",     # 主蓝
    "secondary": "#4C9BE8",   # 浅蓝
    "accent": "#F2994A",      # 强调橙
    "dark": "#1F3B63",        # 深蓝
    "gray": "#8A94A6",
    "light": "#D6E4F5",
    "green": "#2FA37C",
    "red": "#D9534F",
    "purple": "#7B61A8",
}
PALETTE = ["#2E5AAC", "#4C9BE8", "#F2994A", "#2FA37C", "#7B61A8",
           "#D9534F", "#8A94A6", "#5BC0DE", "#F7C948", "#3C6E71"]

# 业务含义固定的颜色（全篇一致）：认购/未认购
COLOR_YES = COLORS["green"]
COLOR_NO = COLORS["gray"]

# ---------------------------------------------------------------- 业务常量
MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
             "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_CN = {"jan": "1月", "feb": "2月", "mar": "3月", "apr": "4月", "may": "5月",
            "jun": "6月", "jul": "7月", "aug": "8月", "sep": "9月", "oct": "10月",
            "nov": "11月", "dec": "12月"}

# 数据起始年月：UCI 官方说明该数据集采集区间为 2008-05 至 2010-11
START_YEAR, START_MONTH = 2008, 5
RANDOM_STATE = 42          # 全局随机种子，保证抽样/聚类/建模可复现

# 漏斗分层的通话时长阈值（依据见 03_analysis.py 中的分位数论证）
FUNNEL_STAGES = [
    ("触达名单", 0),
    ("有效接通", 30),       # ≥30s：外呼业务通行的"接通"判定，排除秒挂/拒接
    ("有效沟通", 120),      # ≥120s，约等于 duration 的 P25
    ("深度沟通", 300),      # ≥300s，约等于 duration 的 P75
]


def save_fig(fig, name: str, tight: bool = True) -> Path:
    """按项目命名规范保存图表：outputs/figures/figXX_名称.png（150dpi）。"""
    path = OUT_FIG / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight" if tight else None,
                facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"[fig] 已保存 {path.name}")
    return path


def save_table(df, name: str) -> Path:
    """保存分析结果表到 outputs/tables/*.csv（utf-8-sig，Excel 可直接打开）。"""
    path = OUT_TAB / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[tab] 已保存 {path.name}")
    return path


def save_results(d: dict, name: str = "key_results.json") -> Path:
    """保存关键指标 JSON，供论文写作时引用（避免手工抄数出错）。"""
    path = OUT_TAB / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"[json] 已保存 {path.name}")
    return path


def style_axes(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    """统一坐标轴样式：只保留一个方向的网格，去掉上/右边框。"""
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis=grid_axis, color="#E6EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax
