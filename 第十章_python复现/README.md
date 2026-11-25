# 第10章 Python 图表复现

读取教材附件中的 `AirPassengers.csv`，复现 `效应分解法.pbix` 中的两张折线预测图：原始客运量 `AIR` 和对数客运量 `ln_AIR`。两张图均预测未来24个月，按月度周期12进行效应分解；原始序列使用乘法季节项，对数序列使用加法季节项。

## 运行

从工作区根目录运行：

```powershell
.\.venv\Scripts\python.exe .\第十章_python复现\第10章_python复现.py
```

默认从教材第十章附件目录读取 `AirPassengers.csv`，结果保存在 `第十章_python复现/复现结果/`。附件目录或结果目录不同时，可用 `--chapter-dir` 和 `--output-dir` 指定。依赖为 `pandas`、`matplotlib` 和 `statsmodels`。

每张图输出 PNG 及配套 CSV，CSV 含历史值、24个月预测值和区间。Power BI 的预测算法未公开为可直接调用的 Python 模型，因此此处使用 Holt-Winters 指数平滑实现趋势与季节效应分解；95%区间按拟合残差和预测步长计算，为近似区间，预测曲线不会与 Power BI 内置算法逐点完全相同。