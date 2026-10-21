import os
import pandas as pd
import numpy as np
import sys

# Add src to system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config_new import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COL,
    LEAKAGE_COLS,
    ORIGINAL_COLS
)
from preprocessing import engineer_features_df, get_preprocessing_pipeline

def run_sanity_check():
    print("=== STARTING PHASE 2 SANITY CHECK ===")
    
    # 1. Check if files exist
    if not (os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH)):
        print("Error: Train/test parquet datasets not found. Running data_pipeline_new.py first...")
        import data_pipeline_new
        data_pipeline_new.main()
        
    # 2. Load train and test data
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    # 3. Print shapes
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape:  {test_df.shape}")
    
    # 4. Assert train/test indices have zero overlap
    train_idx = set(train_df.index)
    test_idx = set(test_df.index)
    overlap = train_idx.intersection(test_idx)
    print(f"Index overlap count between train and test: {len(overlap)}")
    assert len(overlap) == 0, "Error: Index overlap found between train and test sets!"
    
    # 5. Check class distributions
    print("\nClass Distributions:")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        counts = df[TARGET_COL].value_counts()
        print(f"  {name} Set:")
        for cls, val in counts.items():
            pct = (val / len(df)) * 100
            print(f"    Class {cls}: {val} rows ({pct:.2f}%)")
            
    # 6. Verify no post-departure columns exist in features
    print("\nVerifying no post-departure columns in feature space...")
    all_features = [col for col in train_df.columns if col != TARGET_COL]
    leakage_found = [col for col in all_features if col in LEAKAGE_COLS]
    print(f"  Leakage columns found: {leakage_found}")
    assert len(leakage_found) == 0, f"Error: Leakage columns found in feature set: {leakage_found}"
    
    # 7. Apply feature engineering
    print("\nRunning feature engineering on split datasets...")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    X_train_eng = engineer_features_df(X_train)
    X_test_eng = engineer_features_df(X_test)
    print(f"  Engineered Train shape: {X_train_eng.shape}")
    print(f"  Engineered Test shape:  {X_test_eng.shape}")
    
    # 8. Preprocessing Fit Test
    print("\nTesting preprocessing pipeline fitting...")
    preprocessor = get_preprocessing_pipeline()
    
    # Fit strictly on train
    preprocessor.fit(X_train_eng)
    print("  Preprocessor fitted successfully on training data only.")
    
    # Transform both
    X_train_trans = preprocessor.transform(X_train_eng)
    X_test_trans = preprocessor.transform(X_test_eng)
    
    print(f"  Transformed Sparse Train shape: {X_train_trans.shape}")
    print(f"  Transformed Sparse Test shape:  {X_test_trans.shape}")
    
    # Verify that preprocessor state doesn't depend on test data (median check)
    median_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    print(f"  Fitted numerical medians: {median_imputer.statistics_}")
    
    print("\n=== SANITY CHECK PASSED SUCCESSFULLY ===")
    return True

if __name__ == "__main__":
    run_sanity_check()
