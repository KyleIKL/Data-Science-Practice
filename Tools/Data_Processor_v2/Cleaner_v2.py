import pandas as pd
import numpy as np
import os
import json


# =========================
# 1. 读取文件
# =========================
def read_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".xlsx":
        return pd.read_excel(file_path)
    else:
        raise ValueError("只支持 .csv 或 .xlsx 文件")


# =========================
# 2. 判断是否像时间列
# =========================
def looks_like_datetime(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.astype(str).head(50)
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.6


# =========================
# 3. 平均文本长度
# =========================
def avg_text_length(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0
    return non_null.astype(str).map(len).mean()


# =========================
# 4. 探测数据结构
# =========================
def detect_structure(df):
    structure_report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "id_like_columns": [],
        "long_text_columns": [],
        "columns": {}
    }

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()

        missing_ratio = float(s.isna().mean())
        nunique = int(non_null.nunique()) if len(non_null) > 0 else 0
        unique_ratio = float(nunique / len(non_null)) if len(non_null) > 0 else 0
        dtype_str = str(s.dtype)

        if pd.api.types.is_numeric_dtype(s):
            inferred_type = "numeric"
            structure_report["numeric_columns"].append(col)
        elif looks_like_datetime(s):
            inferred_type = "datetime"
            structure_report["datetime_columns"].append(col)
        else:
            inferred_type = "categorical"
            structure_report["categorical_columns"].append(col)

        # ID-like：几乎每行唯一
        if len(non_null) > 0 and unique_ratio >= 0.95 and nunique > 10:
            structure_report["id_like_columns"].append(col)

        # 长文本列
        if not pd.api.types.is_numeric_dtype(s):
            if avg_text_length(s) >= 40:
                structure_report["long_text_columns"].append(col)

        structure_report["columns"][col] = {
            "dtype": dtype_str,
            "inferred_type": inferred_type,
            "missing_ratio": round(missing_ratio, 4),
            "nunique_non_null": nunique,
            "unique_ratio_non_null": round(unique_ratio, 4)
        }

    return structure_report


# =========================
# 5. 缺失值处理
# =========================
def handle_missing_values(df, structure_report):
    result = df.copy()

    numeric_cols = structure_report["numeric_columns"]
    categorical_cols = structure_report["categorical_columns"]
    datetime_cols = structure_report["datetime_columns"]
    id_like_cols = structure_report["id_like_columns"]
    long_text_cols = structure_report["long_text_columns"]

    report = {
        "dropped_rows_count": 0,
        "dropped_columns": {},
        "fill_values": {},
        "column_strategy": {},
        "final_shape": None
    }

    # -------------------------
    # 1. 先删高缺失行
    # -------------------------
    row_missing_ratio = result.isna().mean(axis=1)
    keep_mask = row_missing_ratio < 0.90
    report["dropped_rows_count"] = int((~keep_mask).sum())
    result = result.loc[keep_mask].reset_index(drop=True)

    # -------------------------
    # 2. 按列处理
    # -------------------------
    for col in list(result.columns):
        s = result[col]
        missing_ratio = s.isna().mean()
        lower_col = col.lower()

        # ===== ID-like列 =====
        if col in id_like_cols:
            if missing_ratio > 0:
                result.drop(columns=[col], inplace=True)
                report["dropped_columns"][col] = "id_like_with_missing"
                report["column_strategy"][col] = "drop_column"
            else:
                report["column_strategy"][col] = "keep_identifier"
            continue

        # ===== 长文本列 =====
        if col in long_text_cols:
            if missing_ratio > 0:
                result[col] = result[col].fillna("__MISSING_TEXT__")
                report["fill_values"][col] = "__MISSING_TEXT__"
                report["column_strategy"][col] = "fill_text_with___MISSING_TEXT__"
            else:
                report["column_strategy"][col] = "keep_text"
            continue

        # ===== 数值列 =====
        if col in numeric_cols:
            if missing_ratio > 0.60:
                result.drop(columns=[col], inplace=True)
                report["dropped_columns"][col] = "numeric_missing_too_high"
                report["column_strategy"][col] = "drop_column"
            else:
                if any(x in lower_col for x in ["qty", "quantity", "stock", "inventory", "count", "units"]):
                    fill_value = 0
                elif any(x in lower_col for x in ["discount", "coupon", "refund"]):
                    fill_value = 0
                elif any(x in lower_col for x in ["price", "amount", "sales", "revenue", "cost", "payment"]):
                    fill_value = float(s.median()) if not s.dropna().empty else 0.0
                elif any(x in lower_col for x in ["rating", "score", "review"]):
                    fill_value = float(s.median()) if not s.dropna().empty else 0.0
                else:
                    fill_value = float(s.median()) if not s.dropna().empty else 0.0

                result[col] = s.fillna(fill_value)
                report["fill_values"][col] = fill_value
                report["column_strategy"][col] = f"fill_numeric_with_{fill_value}"
            continue

        # ===== 时间列 =====
        if col in datetime_cols:
            if missing_ratio > 0.70:
                result.drop(columns=[col], inplace=True)
                report["dropped_columns"][col] = "datetime_missing_too_high"
                report["column_strategy"][col] = "drop_column"
            else:
                dt = pd.to_datetime(s, errors="coerce")
                if dt.dropna().empty:
                    result.drop(columns=[col], inplace=True)
                    report["dropped_columns"][col] = "datetime_parse_failed"
                    report["column_strategy"][col] = "drop_column"
                else:
                    fill_value = dt.mode().iloc[0] if not dt.mode().empty else dt.dropna().iloc[0]
                    result[col] = dt.fillna(fill_value)
                    report["fill_values"][col] = str(fill_value)
                    report["column_strategy"][col] = "fill_datetime_with_mode"
            continue

        # ===== 类别列 =====
        if col in categorical_cols:
            if missing_ratio > 0.70:
                result.drop(columns=[col], inplace=True)
                report["dropped_columns"][col] = "categorical_missing_too_high"
                report["column_strategy"][col] = "drop_column"
            else:
                fill_value = "__MISSING__"
                result[col] = s.fillna(fill_value)
                report["fill_values"][col] = fill_value
                report["column_strategy"][col] = "fill_categorical_with___MISSING__"
            continue

        # ===== 兜底 =====
        if missing_ratio > 0:
            result[col] = s.fillna("__MISSING__")
            report["fill_values"][col] = "__MISSING__"
            report["column_strategy"][col] = "fallback_fill"

    report["final_shape"] = result.shape
    return result, report


# =========================
# 6. 保存输出到文件夹
# =========================
def save_outputs(df, report, structure_report, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    data_path = os.path.join(output_folder, "cleaned_data.csv")
    report_path = os.path.join(output_folder, "missing_report.json")
    structure_path = os.path.join(output_folder, "structure_report.json")

    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    with open(structure_path, "w", encoding="utf-8") as f:
        json.dump(structure_report, f, ensure_ascii=False, indent=2, default=str)

    return data_path, report_path, structure_path


# =========================
# 7. 主程序
# =========================
def main():
    print("====== 电商数据缺失值处理工具（第一版）======")

    input_path = input("请输入数据文件地址（csv 或 xlsx）: ").strip().strip('"')
    output_folder = input("请输入保存文件夹地址: ").strip().strip('"')

    try:
        print("\n[1/4] 正在读取数据...")
        df = read_data(input_path)
        print(f"读取成功，数据形状: {df.shape}")

        print("\n[2/4] 正在探测数据结构...")
        structure_report = detect_structure(df)
        print("结构探测完成")

        print("\n[3/4] 正在处理缺失值...")
        cleaned_df, missing_report = handle_missing_values(df, structure_report)
        print(f"缺失值处理完成，处理后形状: {cleaned_df.shape}")

        print("\n[4/4] 正在保存结果...")
        data_path, report_path, structure_path = save_outputs(
            cleaned_df,
            missing_report,
            structure_report,
            output_folder
        )

        print("\n====== 处理完成 ======")
        print(f"清洗后数据已保存到: {data_path}")
        print(f"缺失值处理报告已保存到: {report_path}")
        print(f"结构探测报告已保存到: {structure_path}")

    except Exception as e:
        print("\n程序运行失败：")
        print(str(e))


if __name__ == "__main__":
    main()