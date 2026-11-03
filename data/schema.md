# 数据字典（UCI Bank Marketing）

## 1. 原始字段（bank-full.csv，17 字段）

| 字段 | 类型 | 含义 | 取值范围 / 说明 |
|---|---|---|---|
| age | int | 客户年龄 | 18–95 |
| job | category | 职业 | admin. / blue-collar / entrepreneur / housemaid / management / retired / self-employed / services / student / technician / unemployed / unknown |
| marital | category | 婚姻状况 | married / single / divorced |
| education | category | 学历 | primary / secondary / tertiary / unknown |
| default | category | 是否有违约记录 | yes / no |
| balance | int | 年均账户余额（欧元） | −8,019 ~ 102,127，可为负 |
| housing | category | 是否有房贷 | yes / no |
| loan | category | 是否有个人消费贷 | yes / no |
| contact | category | 联系方式 | cellular / telephone / unknown |
| day | int | 最后一次联系的日 | 1–31 |
| month | category | 最后一次联系的月 | jan–dec（英文缩写） |
| duration | int | 最后一次通话时长（秒） | 0–4,918；**该字段在实际外呼前不可知**，仅用于事后分析，不能作为预测特征上线 |
| campaign | int | 本次活动中对该客户的联系次数 | 1–63 |
| pdays | int | 距上次活动联系的天数 | −1 表示此前从未联系 |
| previous | int | 本次活动之前的联系次数 | 0–275 |
| poutcome | category | 上次活动的结果 | success / failure / other / unknown（unknown 表示无历史活动） |
| y | category | 是否认购定期存款（目标变量） | yes / no，正样本占比 11.70% |

## 2. 派生字段（bank_clean.csv 新增）

| 字段 | 类型 | 计算口径 | 用途 |
|---|---|---|---|
| is_subscribed | 0/1 | `y == "yes"` | 建模与所有转化率指标 |
| contact_date | date | 由 day + month + 重建年份组合 | 月度/周度聚合 |
| year / month_num / year_month / quarter / week | 时间 | 依据 UCI 标注区间（2008-05—2010-11）按月份连续性重建 | 趋势、同期群、季节性 |
| duration_min | float | duration / 60 | 可读性 |
| duration_w | float | duration 按 P99（1,269 秒）缩尾 | 聚类、回归，抑制极端值 |
| balance_w | float | balance 按 P1 / P99 缩尾 | 聚类，抑制巨额存款主导距离 |
| age_band | category | 18-25 / 26-35 / 36-45 / 46-55 / 56-65 / 65+ | 分群与帕累托细分 |
| balance_band | category | 负余额 / 0-1千 / 1千-5千 / 5千以上 | 结构分析 |
| campaign_bucket | category | 1次 / 2-3次 / 4-6次 / 7次以上 | 触达频次分析 |
| had_previous_contact | 0/1 | `pdays != -1` | 新老客区分 |
| pdays_clean | float | pdays 为 −1 时置空 | 描述统计（避免 −1 拉低均值） |
| {job,education,contact,poutcome}_known | 0/1 | 是否为非 unknown | 缺失指示变量，供模型区分"未知" |
| channel | category | cellular→手机 / telephone→座机 / unknown→未知 | 图表可读性 |

## 3. 主外键与关联逻辑

- **主表**：`bank-full.csv`，粒度 = "一次外呼接触"，无客户主键（UCI 数据已匿名化，不提供客户 ID），因此**不做 RFM 客户级建模**，改用"接触级 + 客群级"分析口径，避免伪造客户 ID。
- **辅助表**：`bank-additional-full.csv` 粒度同为一次接触，字段多出 `day_of_week` 与 5 个宏观经济指标（emp.var.rate、cons.price.idx、cons.conf.idx、euribor3m、nr.employed）。关联键为自然时间（年-月），用于解释"2010 年外呼量收缩、转化率上升"的宏观背景，不参与主表建模（两表样本量与抽样口径不同，直接拼接会重复计数）。
- **关联一致性校验**：主表与辅助表在 `month` 分布、`job` 分布上一致（同为该银行同一时期活动记录），年份区间一致，可安全用于背景对照。
