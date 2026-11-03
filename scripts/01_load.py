"""
01_load.py —— 数据加载与完整性校验（只读，不修改原始数据）
"""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

EXPECTED_COLUMNS = {
    "age": "int", "job": "cat", "marital": "cat", "education": "cat",
    "default": "cat", "balance": "int", "housing": "cat", "loan": "cat",
    "contact": "cat", "day": "int", "month": "cat", "duration": "int",
    "campaign": "int", "pdays": "int", "previous": "int",
    "poutcome": "cat", "y": "cat",
}
NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    """计算文件 MD5，作为数据版本指纹。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def validate(df: pd.DataFrame) -> list[str]:
    """字段完整性与取值域校验，返回问题列表（为空表示通过）。"""
    issues: list[str] = []
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols:
        issues.append(f"缺失字段：{sorted(missing_cols)}")
    if extra_cols:
        issues.append(f"多余字段：{sorted(extra_cols)}")
    # 数值字段必须可转为数值
    for c in NUMERIC_COLS:
        if c in df.columns:
            try:
                pd.to_numeric(df[c], errors="raise")
            except Exception:
                issues.append(f"字段 {c} 存在非数值内容")
    # 取值域检查：目标变量只能是 yes/no
    if "y" in df.columns:
        bad = set(df["y"].unique()) - {"yes", "no"}
        if bad:
            issues.append(f"目标变量 y 存在非法取值：{bad}")
    # 月份必须是英文三字母缩写
    if "month" in df.columns:
        bad = set(df["month"].unique()) - set(cfg.MONTH_NUM)
        if bad:
            issues.append(f"month 存在非法取值：{bad}")
    if len(df) < 500:
        issues.append(f"记录数 {len(df)} 少于作业要求的 500 条")
    return issues

def main() -> None:
    print("=" * 68)
    print("Step 1 / 6  数据加载与校验")
    print("=" * 68)
    if not cfg.RAW_MAIN.exists():
        raise FileNotFoundError(f"未找到主数据文件：{cfg.RAW_MAIN}")
    # 分号分隔、带表头；全部按字符串读入后再做类型转换，避免静默截断
    df = pd.read_csv(cfg.RAW_MAIN, sep=";", dtype=str, keep_default_na=False)
    print(f"读取主表：{cfg.RAW_MAIN.name}  形状={df.shape[0]:,} 行 × {df.shape[1]} 列")
    issues = validate(df)
    md5_main = md5_of(cfg.RAW_MAIN)
    # 基础画像
    info = {
        "file": cfg.RAW_MAIN.name,
        "md5": md5_main,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "columns": {},
        "issues": issues,
    }
    for c in df.columns:
        s = df[c]
        col_info = {
            "dtype": EXPECTED_COLUMNS.get(c, "unknown"),
            "n_unique": int(s.nunique()),
            "n_missing": int((s == "").sum()),
        }
        if c in NUMERIC_COLS:
            num = pd.to_numeric(s, errors="coerce")
            col_info.update({
                "min": float(num.min()), "max": float(num.max()),
                "mean": round(float(num.mean()), 3),
            })
        else:
            col_info["top_values"] = s.value_counts().head(12).to_dict()
        info["columns"][c] = col_info
    # 辅助数据源（可选）：存在则登记，用于宏观环境对照
    aux = {}
    if cfg.RAW_AUX.exists():
        aux_df = pd.read_csv(cfg.RAW_AUX, sep=";", dtype=str, keep_default_na=False)
        aux = {"file": cfg.RAW_AUX.name, "rows": int(aux_df.shape[0]),
               "cols": int(aux_df.shape[1]), "md5": md5_of(cfg.RAW_AUX)}
        print(f"读取辅助表：{cfg.RAW_AUX.name}  形状={aux_df.shape[0]:,} 行 × {aux_df.shape[1]} 列")
    # 控制台报告
    print(f"MD5：{md5_main}")
    print(f"字段数：{df.shape[1]}；空值单元格：{int((df == '').sum().sum())}；"
          f"完全重复行（原始层）：{int(df.duplicated().sum())}")
    print(f"目标变量分布：{df['y'].value_counts(normalize=True).round(4).to_dict()}")
    if issues:
        print("\n[校验未通过]")
        for it in issues:
            print("  -", it)
        raise SystemExit("数据校验失败，请检查原始文件。")
    print("校验结果：通过（字段完整、取值域合法、记录数达标）")
    # 落盘报告
    with open(cfg.DATA_RAW / "schema.json", "w", encoding="utf-8") as f:
        json.dump({"main": info, "aux": aux}, f, ensure_ascii=False, indent=2)
    lines = [
        "UCI Bank Marketing 数据加载报告",
        f"生成时间：{info['loaded_at']}",
        "",
        f"主表：{info['file']}（{info['rows']:,} 行 × {info['cols']} 列）",
        f"MD5：{md5_main}",
        f"空值：{int((df == '').sum().sum())}",
        f"原始层完全重复行：{int(df.duplicated().sum())}",
        "",
        "字段清单：",
    ]
    for c, meta in info["columns"].items():
        lines.append(f"  {c:<12}{meta['dtype']:<6}唯一值={meta['n_unique']:<5}{meta.get('top_values', '')}")
    if aux:
        lines += ["", f"辅助表：{aux['file']}（{aux['rows']:,} 行 × {aux['cols']} 列）",
                  f"辅助表 MD5：{aux['md5']}"]
    (cfg.DATA_RAW / "load_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[log] 已保存 load_report.txt / schema.json")
    print("Step 1 完成。\n")

if __name__ == "__main__":
    main()