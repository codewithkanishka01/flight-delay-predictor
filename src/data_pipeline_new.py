import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config_new import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TEMPORAL_TRAIN_PATH,
    TEMPORAL_TEST_PATH,
    ORIGINAL_COLS,
    TARGET_COL,
    RANDOM_STATE
)

def load_data(path):
    print(f"Loading raw data from {path}...")
    # Read only original candidate columns + target/cleaning columns to save memory
    cols_to_read = ORIGINAL_COLS + ["ARR_DELAY", "CANCELLED", "DIVERTED"]
    df = pd.read_csv(path, usecols=cols_to_read)
    return df

def clean_data(df):
    print("Cleaning dataset...")
    # Drop cancelled and diverted flights, and rows missing ARR_DELAY
    df = df[df["CANCELLED"] == 0]
    df = df[df["DIVERTED"] == 0]
    df = df.dropna(subset=["ARR_DELAY"])
    
    # Create target column
    df[TARGET_COL] = (df["ARR_DELAY"] >= 15.0).astype(int)
    
    # Drop ARR_DELAY and temporary columns
    df = df.drop(columns=["ARR_DELAY", "CANCELLED", "DIVERTED"], errors="ignore")
    return df

def split_random(df):
    print("Performing 80/20 random stratified split...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df

def split_temporal(df):
    print("Performing temporal split...")
    # Convert FL_DATE to datetime to sort and split
    df_temp = df.copy()
    df_temp["FL_DATE_DT"] = pd.to_datetime(df_temp["FL_DATE"], format="%Y-%m-%d")
    df_temp = df_temp.sort_values(by="FL_DATE_DT").drop(columns=["FL_DATE_DT"])
    
    # Find chronological split index representing ~80% training data
    split_idx = int(len(df_temp) * 0.80)
    train_df = df_temp.iloc[:split_idx]
    test_df = df_temp.iloc[split_idx:]
    return train_df, test_df

def main():
    df_raw = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df_raw)
    
    # Random Split
    train_df, test_df = split_random(df_clean)
    train_df.to_parquet(TRAIN_DATA_PATH, index=True)
    test_df.to_parquet(TEST_DATA_PATH, index=True)
    print(f"Random split saved: Train shape {train_df.shape}, Test shape {test_df.shape}")
    
    # Temporal Split
    t_train_df, t_test_df = split_temporal(df_clean)
    t_train_df.to_parquet(TEMPORAL_TRAIN_PATH, index=True)
    t_test_df.to_parquet(TEMPORAL_TEST_PATH, index=True)
    print(f"Temporal split saved: Train shape {t_train_df.shape}, Test shape {t_test_df.shape}")

if __name__ == "__main__":
    main()
