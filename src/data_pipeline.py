import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    ORIGINAL_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    RANDOM_STATE
)

def load_data(path):
    print(f"Loading raw data from {path}...")
    # Read only required columns + ARR_DELAY to prevent memory blowup and speed up reading
    cols_to_read = ORIGINAL_COLS + ["ARR_DELAY", "CANCELLED", "DIVERTED"]
    df = pd.read_csv(path, usecols=cols_to_read)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    
    # 1. Target creation: DELAYED = 1 if ARR_DELAY >= 15 else 0
    # Drop rows where ARR_DELAY is missing
    initial_len = len(df)
    
    # Analyze cancellation and diversion records
    cancelled_count = df["CANCELLED"].sum() if "CANCELLED" in df.columns else 0
    diverted_count = df["DIVERTED"].sum() if "DIVERTED" in df.columns else 0
    print(f"Dataset has {cancelled_count} cancelled and {diverted_count} diverted records.")
    
    df = df.dropna(subset=["ARR_DELAY"])
    cleaned_len = len(df)
    print(f"Dropped {initial_len - cleaned_len} rows with missing ARR_DELAY (including cancellations/diversions).")
    print(f"Remaining clean rows: {cleaned_len}")
    
    # Define DELAYED target
    df[TARGET_COL] = (df["ARR_DELAY"] >= 15.0).astype(int)
    
    # Drop ARR_DELAY and other leakage/temporary columns
    df = df.drop(columns=["ARR_DELAY", "CANCELLED", "DIVERTED"], errors="ignore")
    
    # 2. Feature Engineering
    print("Extracting features from FL_DATE...")
    # Convert FL_DATE to datetime
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%Y-%m-%d")
    df["YEAR"] = df["FL_DATE"].dt.year
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY"] = df["FL_DATE"].dt.day
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek
    df = df.drop(columns=["FL_DATE"])
    
    print("Extracting features from CRS_DEP_TIME and CRS_ARR_TIME...")
    # Pre-departure times are represented as integers (HHMM format)
    # Fillna just in case, cast to int
    df["CRS_DEP_TIME"] = df["CRS_DEP_TIME"].fillna(0).astype(int)
    df["CRS_ARR_TIME"] = df["CRS_ARR_TIME"].fillna(0).astype(int)
    
    df["CRS_DEP_HOUR"] = df["CRS_DEP_TIME"] // 100
    df["CRS_DEP_MINUTE"] = df["CRS_DEP_TIME"] % 100
    
    df["CRS_ARR_HOUR"] = df["CRS_ARR_TIME"] // 100
    df["CRS_ARR_MINUTE"] = df["CRS_ARR_TIME"] % 100
    
    df = df.drop(columns=["CRS_DEP_TIME", "CRS_ARR_TIME"])
    
    # 3. Handle categorical dtypes for XGBoost enable_categorical=True
    print("Converting categorical columns to category dtype...")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).astype("category")
        
    # Fill numerical missing values
    for col in NUMERICAL_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            print(f"Filling missing values in numerical column {col} with median: {median_val}")
            df[col] = df[col].fillna(median_val)
            
    # Keep only target and final feature columns
    final_cols = CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL]
    df = df[final_cols]
    
    return df

def split_data(df):
    print("Splitting data into train/test sets...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    # Reassemble train and test sets to save them
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, test_df

def save_data(train_df, test_df):
    print(f"Saving train dataset to {TRAIN_DATA_PATH}...")
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    
    print(f"Saving test dataset to {TEST_DATA_PATH}...")
    test_df.to_parquet(TEST_DATA_PATH, index=False)
    print("Data saving complete.")

def main():
    df = load_data(RAW_DATA_PATH)
    processed_df = preprocess_data(df)
    
    # Verify class distribution
    counts = processed_df[TARGET_COL].value_counts()
    print("Class Distribution:")
    for cls, val in counts.items():
        pct = (val / len(processed_df)) * 100
        print(f"  Class {cls}: {val} rows ({pct:.2f}%)")
        
    train_df, test_df = split_data(processed_df)
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    save_data(train_df, test_df)
    print("Data processing pipeline executed successfully!")

if __name__ == "__main__":
    main()
