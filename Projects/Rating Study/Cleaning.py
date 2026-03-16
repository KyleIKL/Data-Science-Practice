import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================================
# 1. Read CSV file (automatic encoding detection)
# =========================================

file_path = input("Enter CSV file path: ").strip()

encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"File loaded using encoding: {enc}")
        break
    except:
        continue

print("Original dataset shape:", df.shape)


# =========================================
# 2. Extract valid columns
# =========================================
# Remove:
# - columns with all missing values
# - identifier columns
# - URL columns
# - columns with only one unique value

df = df.dropna(axis=1, how="all").copy()

drop_cols = [
    "product_id",
    "sku",
    "product_url",
    "canonical_url",
    "image_url",
    "market_record_source",
    "size_record_source",
    "seen_markets",
    "size_labels",
    "best_for_ids",
    "badge_texts",
    "base_model_number",
    "model_number"
]

existing_drop_cols = [col for col in drop_cols if col in df.columns]

df = df.drop(columns=existing_drop_cols, errors="ignore")

print("\nDropped identifier / URL columns:")
print(existing_drop_cols)


# Remove columns with only one unique value
single_value_cols = []

for col in df.columns:
    if df[col].nunique(dropna=True) <= 1:
        single_value_cols.append(col)

df = df.drop(columns=single_value_cols, errors="ignore")

print("\nDropped single-value columns:")
print(single_value_cols)

print("\nRemaining columns:")
print(df.columns.tolist())


# =========================================
# 3. Clean dataset
# =========================================
# Remove rows containing missing values

shape_before = df.shape

df = df.dropna(axis=0, how="any").copy()

shape_after = df.shape

print("\nShape before cleaning:", shape_before)
print("Shape after cleaning:", shape_after)


# =========================================
# 4. Encode categorical variables
# =========================================

mapping_records = []

cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col].astype(str))

    for original, encoded in zip(le.classes_, le.transform(le.classes_)):

        mapping_records.append({
            "column_name": col,
            "original_value": original,
            "encoded_value": int(encoded)
        })


# Convert mapping to dataframe
encoding_mapping = pd.DataFrame(mapping_records)


# =========================================
# 5. Export results
# =========================================

df.to_csv(r"E:\Project95\Projects\Rating Study\encoded_dataset.csv", index=False, encoding="utf-8-sig")

encoding_mapping.to_excel(r"E:\Project95\Projects\Rating Study\encoding_mapping.xlsx", index=False, engine="openpyxl")

print("\nExport completed.")

print("\nEncoded dataset saved as:")
print("encoded_dataset.csv")

print("\nEncoding mapping saved as:")
print("encoding_mapping.xlsx")


# =========================================
# 6. Display mapping preview
# =========================================

print("\nEncoding mapping preview:")

print(encoding_mapping.head(20))