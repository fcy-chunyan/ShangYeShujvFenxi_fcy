# 第5章 Python 图表复现

本脚本读取教材第5章银行案例附件，重建 Power BI 看板中的 6 个视觉对象，并重绘 WPS 贷款归因月报中的 3 张图。输出为 PNG 图表和 UTF-8-SIG CSV 汇总表。

## 运行

```powershell
.\.venv\Scripts\python.exe .\第五章_python复现.py
```

指定案例运行：

```powershell
.\.venv\Scripts\python.exe .\第五章_python复现.py --task powerbi
.\.venv\Scripts\python.exe .\第五章_python复现.py --task attribution
```

附件目录不在默认位置时使用 `--chapter-dir`，输出目录可使用 `--output-dir` 设置。依赖为 `pandas`、`matplotlib`、`openpyxl` 和 `wordcloud`。

## 复现边界

- 脚本解析 `.pbix` 的 `Report/Layout`，导出视觉类型、字段绑定和尺寸清单；购买金额、产品类别、风险偏好、期限、原因、目的等指标由学员数据表重新聚合。指标卡的月目标从业务描述中提取。
- Power BI 分解树是交互式视觉对象，附件没有保存用户展开路径。Python 图以树中绑定的家庭年收入、年龄、学历、性别四个维度分别拆解购买金额，不伪造未保存的交互状态。
- 自定义词云按 `.pbix` 字段绑定显示原因和购买目的，词频由源数据计数。渲染使用系统微软雅黑/黑体；布局与 Power BI 自定义视觉对象不保证像素级相同。
- WPS 月报图依据申请明细重算，且脚本会将重算月度/渠道数量与 Word 中嵌入的图表缓存交叉校验。归因瀑布图从 2021 年 4 月到 5 月的总申请量及各渠道变化生成。
- PBIX 中的 DAX 度量表达式未作为可移植定义导出；目标达成率按业务描述中的月目标计算，展示为累计购买金额除以月目标。