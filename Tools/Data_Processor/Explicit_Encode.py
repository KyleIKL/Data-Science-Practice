import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def preprocess_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Step 1: Convert date columns to datetime (handling separately)
    for column in df.columns:
        if df[column].dtype == 'object':  # Check for potential date columns
            try:
                df[column] = pd.to_datetime(df[column], errors='raise')  # Attempt to convert to datetime
            except Exception:
                pass  # If conversion fails, continue treating it as an object
    
    # Step 2: Separate numeric, categorical, and datetime columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns

    # Step 3: Handle missing values for numeric columns
    imputer_num = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    # Handle missing values for categorical columns (imputation with most frequent)
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # Step 4: Remove outliers for numeric columns (e.g., beyond 3 standard deviations)
    for column in numeric_cols:
        mean = df[column].mean()
        std_dev = df[column].std()
        df = df[(df[column] >= (mean - 3*std_dev)) & (df[column] <= (mean + 3*std_dev))]
    
    # Step 5: Encode categorical variables using LabelEncoder and store mappings
    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        # Perform encoding
        df[column] = le.fit_transform(df[column])
        
        # Store the original and encoded value mapping in a dictionary
        label_encoders[column] = {
            'original': le.classes_.tolist(),
            'encoded': le.transform(le.classes_).tolist()
        }
    
    # Step 6: Save the processed data to a CSV file
    processed_data_path = "processed_data.csv"
    df.to_csv(processed_data_path, index=False)
    
    # Step 7: Save the encoding mappings to an Excel file
    encoding_mappings_path = "encoding_mappings.xlsx"
    
    # Prepare the mapping data to be saved
    encoding_mappings = []

    for column, mapping in label_encoders.items():
        column_mappings = list(zip(mapping['original'], mapping['encoded']))
        for original, encoded in column_mappings:
            encoding_mappings.append([column, original, encoded])

    # Create a DataFrame for encoding mappings
    encoding_df = pd.DataFrame(encoding_mappings, columns=['Column', 'Original Value', 'Encoded Value'])
    
    # Save the mappings as Excel with two sheets
    with pd.ExcelWriter(encoding_mappings_path) as writer:
        encoding_df.to_excel(writer, sheet_name='Encoding Mappings', index=False)
        # Optionally, save the encoded columns too if needed
        encoded_columns_df = df[categorical_cols]
        encoded_columns_df.to_excel(writer, sheet_name='Encoded Columns', index=False)
    
    return processed_data_path, encoding_mappings_path

# Example usage:
file_path = input("Please enter the file path: ")
processed_data, encoding_mappings = preprocess_data(file_path)
print(f"Processed data saved at: {processed_data}")
print(f"Encoding mappings saved at: {encoding_mappings}")