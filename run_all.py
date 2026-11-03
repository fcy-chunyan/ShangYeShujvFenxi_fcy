# -*- coding: utf-8 -*-
"""
run_all.py —— 一键复现入口

按顺序执行 01—06 全部脚本，任一环节失败立即中止并打印错误。
用法：
    python run_all.py            # 全流程
    python run_all.py 1 2 3      # 只跑指定步骤（编号见 STEPS）

依赖：见 requirements.txt（pip install -r requirements.txt）
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

STEPS = [
    (1, "01_load.py", "数据加载与校验"),
    (2, "02_clean.py", "数据清洗与治理"),
    (3, "03_analysis.py", "商业分析模型（漏斗/趋势/帕累托/BCG/K-Means/关联规则）"),
    (4, "04_forecast.py", "SARIMA 预测与回测"),
    (5, "05_visuals.py", "生成 24 张静态图表（fig01–fig24：15 种模板复现 + 9 张业务模型图）"),
    (6, "06_dashboard.py", "生成交互式大屏 HTML，并自动截取大屏整体效果图（fig25）"),
]


def run(steps: list[int] | None = None) -> None:
    todo = [s for s in STEPS if steps is None or s[0] in steps]
    total = time.time()
    for no, script, desc in todo:
        path = SCRIPTS / script
        if not path.exists():
            raise FileNotFoundError(f"缺少脚本：{path}")
        print(f"\n{'=' * 68}\n[{no}/{len(STEPS)}] {desc}  ({script})\n{'=' * 68}")
        t0 = time.time()
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n步骤 {no} 失败（退出码 {result.returncode}），已中止。")
            sys.exit(result.returncode)
        print(f"—— 用时 {time.time() - t0:.1f}s")
    print(f"\n全部完成，总用时 {time.time() - total:.1f}s")
    print(f"图表：{ROOT / 'outputs' / 'figures'}")
    print(f"大屏：{ROOT / 'outputs' / 'dashboard' / 'dashboard.html'}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    run(args or None)
