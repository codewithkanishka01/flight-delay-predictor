import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def main():
    data_dir = "/Users/kanishkapandey/Desktop/ML/data"
    model_dir = "/Users/kanishkapandey/Desktop/ML/models"
    output_dir = "/Users/kanishkapandey/Desktop/ML/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load untouched test datasets
    print("Loading untouched test set...")
    X_test_10 = pd.read_parquet(os.path.join(data_dir, "X_test_10.parquet"))
    y_test_10 = pd.read_parquet(os.path.join(data_dir, "y_test_10.parquet")).iloc[:, 0]
    
    # 2. Load locked model
    model_path = os.path.join(model_dir, "xgb_model_tuned.json")
    if not os.path.exists(model_path):
        print(f"Error: Tuned XGBoost model not found at {model_path}. Run tuning first.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Loaded tuned XGBoost model successfully.")
    
    # 3. Load locked threshold (0.50)
    locked_threshold = 0.50
    print(f"Locked Accuracy-Optimized Threshold: {locked_threshold:.2f}")
    
    # 4. Predict probabilities and classes
    print("Running predictions on the untouched test set...")
    probabilities = model.predict_proba(X_test_10)[:, 1]
    predictions = (probabilities >= locked_threshold).astype(int)
    
    # 5. Compute metrics
    acc = accuracy_score(y_test_10, predictions)
    prec = precision_score(y_test_10, predictions, zero_division=0)
    rec = recall_score(y_test_10, predictions)
    f1 = f1_score(y_test_10, predictions)
    auc = roc_auc_score(y_test_10, probabilities)
    cm = confusion_matrix(y_test_10, predictions)
    
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    
    print("\n================ OFFICIAL FINAL TEST RESULTS ================")
    print(f"Final Test Accuracy:  {acc*100:.4f}%")
    print(f"Final Test Precision: {prec*100:.4f}%")
    print(f"Final Test Recall:    {rec*100:.4f}%")
    print(f"Final Test F1 Score:  {f1*100:.4f}%")
    print(f"Final Test ROC-AUC:   {auc*100:.4f}%")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nBreakdown:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    print("=============================================================")
    
    # Check 90% target
    print("\n=== GOAL TARGET ACHIEVEMENT REPORT ===")
    if acc >= 0.90:
        print("GENUINE 90%+ ACCURACY ACHIEVED: Yes")
    else:
        print("GENUINE 90%+ ACCURACY ACHIEVED: No")
        print("\nLimiting Factors & Scientific Explanation:")
        print("1. Target Leakage Prevention: Since we strictly avoided post-departure variables (such as departure delays, elapsed air time, or taxis), we only used scheduled pre-departure information. Pre-departure flight data has limited signal because most flight delays are caused by operational deviations that occur AFTER departure (like late-arriving aircraft, terminal congestion, or weather changes).")
        print("2. Class Imbalance: The target distribution is highly skewed (81.68% class 0, 18.32% class 1). Predicting class 0 for all instances yields an accuracy of 81.68%. To exceed 90% accuracy, the model must be highly precise in predicting class 1, which is challenging without real-time post-boarding details.")
        
    # Save final results to JSON
    results_summary = {
        "final_test_accuracy": float(acc),
        "final_test_precision": float(prec),
        "final_test_recall": float(rec),
        "final_test_f1": float(f1),
        "final_test_roc_auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
        "genuine_90_achieved": bool(acc >= 0.90)
    }
    
    results_path = os.path.join(output_dir, "final_test_10_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\nOfficial final test results saved to {results_path}")

if __name__ == "__main__":
    main()
