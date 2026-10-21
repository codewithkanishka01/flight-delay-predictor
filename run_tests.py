import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import unittest
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from config import MODEL_DIR, OUTPUT_DIR, TARGET_COL

def test_prediction_demo():
    print("Testing pre-departure flight delay predictor function...")
    
    # Load model and threshold
    model_path = os.path.join(MODEL_DIR, "xgb_model_tuned.json")
    threshold_path = os.path.join(MODEL_DIR, "threshold_config_10.json")
    
    if not os.path.exists(model_path) or not os.path.exists(threshold_path):
        print("Warning: Trained model or threshold config not found. Skipping prediction test.")
        return False
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    with open(threshold_path, 'r') as f:
        t_config = json.load(f)
        locked_threshold = t_config["best_threshold_accuracy"]
        
    # Mock pre-departure flight input dict (8 columns representing the 10 locked columns - times/dates are raw)
    mock_input = {
        "FL_DATE": "2023-11-20",
        "AIRLINE": "Delta Air Lines Inc.",
        "AIRLINE_CODE": "DL",
        "ORIGIN": "MSP",
        "ORIGIN_CITY": "Minneapolis, MN",
        "DEST": "SEA",
        "DEST_CITY": "Seattle, WA",
        "CRS_DEP_TIME": 2120,
        "CRS_ARR_TIME": 2315,
        "DISTANCE": 1399.0
    }
    
    df_input = pd.DataFrame([mock_input])
    df_proc = df_input.copy()
    
    # Feature extraction
    dt = pd.to_datetime(df_proc["FL_DATE"], format="%Y-%m-%d")
    df_proc["YEAR"] = dt.dt.year
    df_proc["MONTH"] = dt.dt.month
    df_proc["DAY"] = dt.dt.day
    df_proc["DAY_OF_WEEK"] = dt.dt.dayofweek
    df_proc = df_proc.drop(columns=["FL_DATE"])
    
    df_proc["CRS_DEP_TIME"] = df_proc["CRS_DEP_TIME"].fillna(0).astype(int)
    df_proc["CRS_DEP_HOUR"] = df_proc["CRS_DEP_TIME"] // 100
    df_proc["CRS_DEP_MINUTE"] = df_proc["CRS_DEP_TIME"] % 100
    df_proc = df_proc.drop(columns=["CRS_DEP_TIME"])
    
    df_proc["CRS_ARR_TIME"] = df_proc["CRS_ARR_TIME"].fillna(0).astype(int)
    df_proc["CRS_ARR_HOUR"] = df_proc["CRS_ARR_TIME"] // 100
    df_proc["CRS_ARR_MINUTE"] = df_proc["CRS_ARR_TIME"] % 100
    df_proc = df_proc.drop(columns=["CRS_ARR_TIME"])
    
    categorical_cols = ["AIRLINE", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY"]
    for col in df_proc.columns:
        if col in categorical_cols:
            df_proc[col] = df_proc[col].astype(str).astype("category")
            
    expected_cols = [
        "AIRLINE", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY",
        "DISTANCE", "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
        "CRS_DEP_HOUR", "CRS_DEP_MINUTE", "CRS_ARR_HOUR", "CRS_ARR_MINUTE"
    ]
    df_proc = df_proc[expected_cols]
    
    # Predict
    prob = model.predict_proba(df_proc)[0, 1]
    pred = 1 if prob >= locked_threshold else 0
    
    print(f"  Mock input predicted successfully!")
    print(f"  Prediction class: {pred} (Probability: {prob*100:.2f}%, Threshold: {locked_threshold*100:.2f}%)")
    return True

def main():
    print("=== STARTING ML PIPELINE INTEGRITY TESTS ===")
    
    # 1. Run unittest test suite
    print("\n1. Running Preprocessing and Leakage Unit Tests...")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='*_test.py')
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    if not test_result.wasSuccessful():
        print("\nERROR: Preprocessing unit tests failed!")
        sys.exit(1)
        
    # 2. Run mock prediction
    print("\n2. Running Mock Prediction Demo...")
    pred_ok = test_prediction_demo()
    
    if not pred_ok:
        print("\nERROR: Mock prediction failed!")
        sys.exit(1)
        
    print("\n=== ALL INTEGRITY TESTS PASSED SUCCESSFULLY ===")
    print("The Flight Delay Prediction ML project is working, leakage-free, and fully runnable.")

if __name__ == "__main__":
    main()
