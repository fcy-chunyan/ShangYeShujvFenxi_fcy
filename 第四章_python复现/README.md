# 第4章 Python 数据处理复现

本目录复现教材第4章“数据整合和数据清洗”notebook 中的 pandas 示例。脚本读取教材附件 `sample.csv` 和 `data/big.csv`，复现数据读取、行列操作、条件筛选、横向连接、纵向合并、排序、分组汇总、透视表、条件赋值，以及重复值、缺失值、异常值和分箱处理。表格结果写为 UTF-8-SIG CSV，异常值盖帽前后的分布写为 PNG。

## 运行

从工作区根目录运行：

```powershell
.\.venv\Scripts\python.exe .\第四章_python复现\第四章_python复现.py
```

附件目录不在脚本默认路径时，可通过 `--chapter-dir` 指定；结果目录可通过 `--output-dir` 设置：

```powershell
.\.venv\Scripts\python.exe .\第四章_python复现\第四章_python复现.py --chapter-dir "D:\教材附件\第4章- 数据处理(pandas)" --output-dir ".\第四章_python复现\复现结果"
```

依赖为 `pandas`、`numpy` 和 `matplotlib`。复现结果默认写入本目录的 `复现结果`。

## 口径说明

- `big.csv` 按 notebook 中的 `chunksize=900` 分块读取，并保存每块行列数及合并后的数据；当前附件共 4,500 行，即 5 个完整分块。
- 读取示例保留原 notebook 的 `na_values='99999'`。附件中的 `9999` 不会因此被识别为缺失值。
- 随机数示例固定种子为 0，以保证生成的演示数据和图表可重复；盖帽法使用 1% 和 99% 分位数。
- notebook 中的展示和 Series 查询结果也导出为 CSV，行索引会作为“行号”列保留；空值在 CSV 中保存为空字段。