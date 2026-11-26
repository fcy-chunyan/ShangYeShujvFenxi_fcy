from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHAPTER_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\2《CDA 二级认证教材：量化策略分析》\第4章- 数据处理(pandas)"
)
OUTPUT_DIR = Path(__file__).parent / "复现结果"

def configure_charts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def save_chart(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def with_row_index(frame: pd.DataFrame, name: str = "行号") -> pd.DataFrame:
    result = frame.copy()
    result.index.name = name
    return result.reset_index()

def reproduce_reading(chapter_dir: Path, output_dir: Path) -> dict[str, int]:
    sample_path = chapter_dir / "sample.csv"
    big_path = chapter_dir / "data" / "big.csv"
    if not sample_path.is_file() or not big_path.is_file():
        raise FileNotFoundError(f"未找到 sample.csv 或 data/big.csv，请检查附件目录：{chapter_dir}")
    sample = pd.read_csv(sample_path)
    save_table(sample, output_dir, "读取_sample全量")
    save_table(pd.read_csv(sample_path, usecols=["id", "name"], nrows=2), output_dir, "读取_指定列前两行")
    chunk_frames = list(pd.read_csv(big_path, chunksize=900))
    chunk_info = pd.DataFrame(
        [{"分块序号": index, "行数": len(frame), "列数": len(frame.columns)} for index, frame in enumerate(chunk_frames, 1)]
    )
    save_table(chunk_info, output_dir, "读取_big分块信息")
    big_data = pd.concat(chunk_frames, ignore_index=True)
    save_table(big_data, output_dir, "读取_big分块合并")

    missing_value = pd.read_csv(sample_path, na_values="99999")
    save_table(missing_value, output_dir, "读取_指定缺失标记")
    utf8_data = pd.read_csv(sample_path, encoding="utf-8")
    save_table(utf8_data, output_dir, "读取_UTF8编码")
    utf8_data.to_csv(output_dir / "write.csv", encoding="utf-8", index=False)
    return {"sample_rows": len(sample), "big_rows": len(big_data), "big_chunks": len(chunk_frames)}

def reproduce_row_column_operations(output_dir: Path) -> None:
    np.random.seed(0)
    sample = pd.DataFrame(np.random.randn(4, 5), columns=list("abcde"))
    save_table(with_row_index(sample), output_dir, "行列_随机样例")
    save_table(with_row_index(sample["a"].to_frame()), output_dir, "行列_单列Series")
    save_table(with_row_index(sample.iloc[:, 0].to_frame()), output_dir, "行列_iloc取列")
    save_table(with_row_index(sample.loc[:, "a"].to_frame()), output_dir, "行列_loc取列")
    save_table(with_row_index(sample[["a"]]), output_dir, "行列_双括号取列")
    save_table(with_row_index(sample.loc[:1, :"b"]), output_dir, "行列_loc切片")
    save_table(with_row_index(sample.iloc[0:2, 0:2]), output_dir, "行列_iloc切片")
    sample["new_col"] = sample["a"] - sample["b"]
    save_table(with_row_index(sample), output_dir, "行列_新增列")
    assigned = sample.assign(new_col2=sample["a"] - sample["b"], new_col3=sample["a"] + sample["b"])
    save_table(with_row_index(assigned), output_dir, "行列_assign新增列")
    save_table(with_row_index(sample.drop(columns="a")), output_dir, "行列_删除单列")
    save_table(with_row_index(sample.drop(columns=["a", "b"])), output_dir, "行列_删除多列")

def reproduce_filtering(output_dir: Path) -> None:
    sample = pd.DataFrame(
        {"group": [1, 1, 1, 2, 1, 2], "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"], "score": [98, 78, 87, 77, 65, 67]}
    )
    masks = {
        "大于70": sample["score"] > 70,
        "大于70且group1": (sample["score"] > 70) & (sample["group"] == 1),
        "非group1": ~(sample["group"] == 1),
        "group1或group2": (sample["group"] == 1) | (sample["group"] == 2),
        "query大于90": sample.eval("score > 90"),
        "query两个组": sample.eval("(group == 2) | (group == 1)"),
        "分数70到80": sample["score"].between(70, 80, inclusive="both"),
        "姓名Bob或Lindy": sample["name"].isin(["Bob", "Lindy"]),
        "姓名包含M": sample["name"].str.contains("[M]+"),
    }
    save_table(with_row_index(sample), output_dir, "筛选_原始样例")
    for label, mask in masks.items():
        save_table(with_row_index(sample.loc[mask]), output_dir, f"筛选_{label}")

def reproduce_joins(output_dir: Path) -> None:
    left = pd.DataFrame({"col1": ["a", "b", "c"], "id": [1, 2, 3]})
    right = pd.DataFrame({"col2": ["d", "e"], "id": [4, 3]})
    for how in ("inner", "left", "right", "outer"):
        result = left.merge(right, how=how, on="id")
        save_table(result, output_dir, f"横向连接_merge_{how}")
    left_indexed = pd.DataFrame({"id1": [1, 2, 3], "col1": ["a", "b", "c"]}, index=[1, 2, 3])
    right_indexed = pd.DataFrame({"id2": [1, 2, 3], "col2": ["aa", "bb", "cc"]}, index=[1, 3, 2])
    save_table(with_row_index(pd.concat([left_indexed, right_indexed], axis=1)), output_dir, "横向连接_concat按索引")
    save_table(with_row_index(left_indexed.join(right_indexed)), output_dir, "横向连接_join按索引")
    first = pd.DataFrame({"id": [1, 1, 1, 2, 3, 4, 6], "col": ["a", "a", "b", "c", "v", "e", "q"]})
    second = pd.DataFrame({"id": [1, 2, 3, 3, 5], "col": ["x", "y", "z", "v", "w"]})
    appended = pd.concat([first, second], ignore_index=True)
    save_table(appended, output_dir, "纵向合并_concat")
    save_table(appended.drop_duplicates(), output_dir, "纵向合并_concat去重")
    renamed = first.rename(columns={"col": "new_col"})
    save_table(renamed, output_dir, "纵向合并_重命名列")
    save_table(pd.concat([first, renamed], ignore_index=True).drop_duplicates(), output_dir, "纵向合并_不同列名追加")

def reproduce_sorting_and_grouping(output_dir: Path) -> None:
    sorting = pd.DataFrame(
        {"group": [1, 1, 1, 2, 1, 2], "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"], "score": [98, 78, 87, 77, 77, np.nan]}
    )
    save_table(sorting.sort_values("score", ascending=False, na_position="last"), output_dir, "排序_按分数降序")
    save_table(sorting.sort_values(["group", "score"]), output_dir, "排序_按组和分数")
    students = pd.DataFrame(
        {
            "name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"],
            "class": [1, 1, 1, 2, 1, 2],
            "grade": [1, 1, 1, 2, 2, 2],
            "chinese": [88, 78, 86, 56, 77, 54],
            "math": [98, 78, 87, 77, 77, np.nan],
        }
    )
    save_table(students.groupby("grade")[["math"]].max().reset_index(), output_dir, "分组汇总_年级数学最高分")
    save_table(students.groupby(["grade", "class"])[["math"]].mean().reset_index(), output_dir, "分组汇总_年级班级均分")
    save_table(students.groupby("grade")[["math", "chinese"]].mean().reset_index(), output_dir, "分组汇总_各科均分")
    save_table(students.groupby("class")["math"].agg(["mean", "min", "max"]).reset_index(), output_dir, "分组汇总_聚合统计")

    multi_agg = students.groupby(["grade", "class"])[["math", "chinese"]].agg(["min", "max"])
    multi_agg.columns = [f"{subject}_{stat}" for subject, stat in multi_agg.columns]
    multi_agg = multi_agg.reset_index()
    save_table(multi_agg, output_dir, "分组汇总_多级列聚合")
    save_table(multi_agg[["grade", "class", "math_min"]], output_dir, "分组汇总_选择聚合列")
    save_table(multi_agg.loc[(multi_agg["grade"] == 2) & (multi_agg["class"] == 2), ["math_min"]], output_dir, "分组汇总_选择单个聚合值")

def reproduce_pivot_and_assignment(output_dir: Path) -> None:
    transactions = pd.DataFrame(
        {
            "Monetary": [3608, 420, 1894, 3503, 4567],
            "cust_id": [10001, 10001, 10002, 10002, 10003],
            "type": ["Normal", "Special_offer", "Normal", "Special_offer", "Special_offer"],
        }
    )
    pivot_mean = pd.pivot_table(transactions, index="cust_id", columns="type", values="Monetary")
    save_table(pivot_mean.reset_index(), output_dir, "透视表_默认均值")
    pivot_sum = pd.pivot_table(
        transactions, index="cust_id", columns="type", values="Monetary", fill_value=0, aggfunc="sum"
    )
    save_table(pivot_sum.reset_index(), output_dir, "透视表_求和填零")

    sample = pd.DataFrame(
        {"name": ["Bob", "Lindy", "Mark", "Miki", "Sully", "Rose"], "score": [99, 78, 999, 77, 77, np.nan], "group": [1, 1, 1, 2, 1, 2]}
    )
    save_table(sample["score"].replace(999, np.nan).to_frame(), output_dir, "赋值_replace单列")
    replaced = sample.replace({"score": {999: np.nan}, "name": {"Bob": np.nan}})
    save_table(replaced, output_dir, "赋值_replace多列")
    sample["class_n"] = sample["group"].map({1: "class1", 2: "class2"})
    save_table(sample, output_dir, "赋值_apply映射结果")

def reproduce_cleaning(output_dir: Path) -> None:
    duplicates = pd.DataFrame(
        {
            "group": [1, 1, 1, 2, 1, 2],
            "id": [1, 1, 1, 3, 4, 5],
            "name": ["Bob", "Bob", "Mark", "Miki", "Sully", "Rose"],
            "score": [99, 99, 87, 77, 77, np.nan],
        }
    )
    save_table(duplicates.loc[duplicates.duplicated()], output_dir, "重复值_重复记录")
    save_table(duplicates.drop_duplicates(), output_dir, "重复值_整行去重")
    save_table(duplicates.drop_duplicates("id"), output_dir, "重复值_按ID去重")

    missing = pd.DataFrame(
        {
            "group": [1, 1, np.nan, 2, 1, np.nan],
            "id": [1, 1, 1, 3, 4, np.nan],
            "name": ["Bob", "Bob", "Mark", "Miki", "Sully", np.nan],
            "score": [99, np.nan, 87, 77, 77, np.nan],
        }
    )
    missing_rate = missing.isna().mean().rename("缺失率").rename_axis("字段").reset_index()
    save_table(missing_rate, output_dir, "缺失值_各列缺失率")
    save_table(missing.assign(score=missing["score"].fillna(missing["score"].mean())), output_dir, "缺失值_均值填充")
    save_table(missing.assign(score=missing["score"].fillna(missing["score"].median())), output_dir, "缺失值_中位数填充")
    save_table(missing["score"].isna().astype(int).to_frame(), output_dir, "缺失值_缺失指示")

    np.random.seed(0)
    normal = pd.DataFrame({"normal": np.random.randn(1000)})
    lower, upper = normal["normal"].quantile([0.01, 0.99])
    capped = normal.clip(lower=lower, upper=upper, axis="columns")
    save_table(
        pd.DataFrame(
            [{"处理": "原始", "最小值": normal["normal"].min(), "最大值": normal["normal"].max(), "下限": lower, "上限": upper},
             {"处理": "1%和99%分位盖帽", "最小值": capped["normal"].min(), "最大值": capped["normal"].max(), "下限": lower, "上限": upper}]
        ),
        output_dir,
        "异常值_盖帽统计",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    normal["normal"].hist(bins=50, ax=axes[0], color="#277DA1")
    axes[0].set_title("盖帽前")
    capped["normal"].hist(bins=50, ax=axes[1], color="#F8961E")
    axes[1].set_title("1%和99%分位盖帽后")
    for ax in axes:
        ax.set_xlabel("数值")
        ax.set_ylabel("频数")
        ax.grid(axis="y", alpha=0.2)
    save_chart(fig, output_dir, "异常值_盖帽前后直方图")

    np.random.seed(0)
    values = pd.DataFrame({"normal": np.random.randn(10)})
    save_table(values, output_dir, "分箱_原始数据")
    save_table(values.assign(等宽五箱=pd.cut(values["normal"], bins=5).astype(str)), output_dir, "分箱_等宽五箱")
    save_table(values.assign(好坏标签=pd.cut(values["normal"], bins=2, labels=["bad", "good"]).astype(str)), output_dir, "分箱_好坏标签")
    quantile_edges = values["normal"].quantile([0, 0.5, 1]).to_numpy()
    quantile_bins = pd.cut(values["normal"], bins=quantile_edges, include_lowest=True)
    quantile_labels = pd.cut(
        values["normal"], bins=quantile_edges, labels=["bad", "good"], include_lowest=True
    )
    save_table(
        values.assign(分位数区间=quantile_bins.astype(str), 分位数标签=quantile_labels.astype(str)),
        output_dir,
        "分箱_分位数区间与标签",
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="复现教材第4章 pandas 数据整合和数据清洗示例")
    parser.add_argument("--chapter-dir", type=Path, default=CHAPTER_DIR, help="教材附件目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="结果输出目录")
    args = parser.parse_args()
    configure_charts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_summary = reproduce_reading(args.chapter_dir, args.output_dir)
    reproduce_row_column_operations(args.output_dir)
    reproduce_filtering(args.output_dir)
    reproduce_joins(args.output_dir)
    reproduce_sorting_and_grouping(args.output_dir)
    reproduce_pivot_and_assignment(args.output_dir)
    reproduce_cleaning(args.output_dir)
    print(
        "复现完成："
        f"sample.csv {input_summary['sample_rows']} 行；"
        f"big.csv {input_summary['big_rows']} 行，按 900 行分为 {input_summary['big_chunks']} 块；"
        f"结果目录：{args.output_dir}"
    )

if __name__ == "__main__":
    main()