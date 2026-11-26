from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

CHAPTER_DIR = Path(
    r"D:\商业数据分析\CDA教材资料2025\20260830 CDA教材资料2025\2《CDA 二级认证教材：量化策略分析》\第6章-市场调研与数据预处理"
)
DATA_DIR = CHAPTER_DIR / "1数据采集与处理 1 数据采集方法"
OUTPUT_DIR = Path(__file__).parent / "复现结果"

def sample_size(k: int | float, population_size: int) -> tuple[int, bool]:
    if isinstance(k, bool) or not isinstance(k, (int, float)) or k <= 0:
        raise ValueError("k 必须是正整数或 (0, 1) 之间的小数比例")
    if k >= 1:
        if not isinstance(k, int):
            raise ValueError("按数量抽样时 k 必须为整数")
        if k > population_size:
            raise ValueError(f"抽样数量 {k} 超过总体数量 {population_size}")
        return k, True
    return math.ceil(population_size * k), False

def get_sample(
    data: pd.DataFrame,
    sampling: str = "simple_random",
    k: int | float = 1,
    stratified_col: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """执行简单随机、分层或等距系统抽样。"""
    if data.empty:
        raise ValueError("输入数据不能为空")
    if sampling not in {"simple_random", "stratified", "systematic"}:
        raise ValueError("sampling 仅支持 simple_random、stratified、systematic")
    if sampling == "stratified":
        if not stratified_col:
            raise ValueError("分层抽样必须指定 stratified_col")
        missing = set(stratified_col).difference(data.columns)
        if missing:
            raise ValueError(f"分层字段不存在: {', '.join(sorted(missing))}")
        _, by_count = sample_size(k, len(data))
        rng = np.random.default_rng(random_state)
        samples = []
        for _, group in data.groupby(stratified_col, dropna=False, sort=True):
            n = int(k) if by_count else math.ceil(len(group) * float(k))
            if n > len(group):
                raise ValueError(f"分层样本量 {n} 超过该层总体数量 {len(group)}")
            positions = rng.choice(len(group), size=n, replace=False)
            samples.append(group.iloc[positions])
        return pd.concat(samples).sort_index().copy()
    n, _ = sample_size(k, len(data))
    if sampling == "simple_random":
        return data.sample(n=n, replace=False, random_state=random_state).copy()
    interval = len(data) / n
    positions = np.floor(np.arange(n) * interval).astype(int)
    return data.iloc[positions].copy()

def save_table(frame: pd.DataFrame, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def run(data_dir: Path, output_dir: Path, random_state: int) -> int:
    source = data_dir / "clients.csv"
    clients = pd.read_csv(source, encoding="gbk")
    if "Dist" not in clients.columns:
        raise ValueError(f"{source.name} 缺少分层字段 Dist")
    cases: list[tuple[str, str, int | float, list[str] | None]] = [
        ("简单随机_固定22", "simple_random", 22, None),
        ("简单随机_比例10pct", "simple_random", 0.1, None),
        ("分层抽样_每层2条", "stratified", 2, ["Dist"]),
        ("分层抽样_每层10pct", "stratified", 0.1, ["Dist"]),
        ("系统抽样_固定4", "systematic", 4, None),
        ("系统抽样_比例10pct", "systematic", 0.1, None),
    ]
    summary = []
    for offset, (name, method, k, strata) in enumerate(cases):
        seed = random_state + offset
        result = get_sample(clients, method, k, strata, seed)
        save_table(result, f"抽样_{name}", output_dir)
        row = {
            "抽样方法": method,
            "抽样方案": name,
            "总体数量": len(clients),
            "样本数量": len(result),
            "实际抽样比例": len(result) / len(clients),
            "随机种子": seed,
        }
        if method == "stratified":
            counts = result.groupby("Dist", dropna=False).size()
            row["覆盖层数"] = len(counts)
            row["每层样本量最小值"] = int(counts.min())
            row["每层样本量最大值"] = int(counts.max())
        summary.append(row)
    save_table(pd.DataFrame(summary), "抽样_方案汇总", output_dir)
    return len(cases)

def main() -> None:
    parser = argparse.ArgumentParser(description="复现第6章抽样方法示例")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="抽样数据目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    count = run(args.data_dir, args.output_dir, args.random_state)
    print(f"抽样复现完成：{count} 组样本，输出至 {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()