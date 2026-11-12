# 第3章 Python 图表复现
本目录使用教材附件中的“完整案例”工作簿作为图表数据来源，基于 pandas 重新计算汇总结果，并以 matplotlib 输出 PNG。完整复现收入趋势分析 5 张、财务费用分析 3 张、电商精准营销 4 张，共 12 张图；每张图同时输出对应的 UTF-8-SIG CSV 汇总表。

## 运行
从工作区根目录运行全部案例：

```powershell
.\.venv\Scripts\python.exe .\第三章_python复现\第3章_python复现.py
```

只运行指定案例：

```powershell
.\.venv\Scripts\python.exe .\第三章_python复现\第3章_python复现.py --task revenue
.\.venv\Scripts\python.exe .\第三章_python复现\第3章_python复现.py --task finance
.\.venv\Scripts\python.exe .\第三章_python复现\第3章_python复现.py --task marketing
```

附件目录不在默认位置时，可通过 `--chapter-dir` 指定；可通过 `--output-dir` 更改结果目录。依赖为 `pandas`、`matplotlib` 和 `openpyxl`。

## 口径说明
- 收入分析包含年度销售额、2020年月销售额、2020年月同比、2020年11月省份同比、四川省月同比。为复现完整案例图表，图中的同比率采用其准备区公式 `(本期 - 上期) / 本期`；CSV 同时提供常规定义 `(本期 - 上期) / 上期` 的 `标准同比增长率`，便于核对。上期为零时标准同比率为空值。
- 财务分析读取完整案例工作簿内的分类费用明细。该工作簿中的财务分类数值与单独提供的原始数据工作簿有差异，因此以完整案例内用于制图的数据为准。
- 精准营销按客户 ID 去重统计。附件数据中每个客户 ID 唯一，因此与按记录计数的完整案例结果一致；年龄段按 18-25、26-35、36-49、50岁以上划分。