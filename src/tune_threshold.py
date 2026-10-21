import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import (
    TRAIN_DATA_PATH,
    TARGET_COL,
    RANDOM_STATE,
    TUNING_ROWS,
    MODEL_DIR
)

def main():
    # Load optimized parameters
    params_path = os.path.join(MODEL_DIR, "best_params.json")
    if not os.path.exists(params_path):
        print(f"Error: optimized parameters not found at {params_path}. Run hyperparameter tuning first.")
        return
        
    with open(params_path, 'r') as f:
        best_params = json.load(f)
        
    print("Loaded best parameters:")
    print(json.dumps(best_params, indent=2))
    
    print(f"Loading training data for threshold tuning from {TRAIN_DATA_PATH}...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    
    # Stratified sampling of 300,000 rows
    print(f"Sampling {TUNING_ROWS} rows for validation and threshold tuning...")
    tune_df, _ = train_test_split(
        train_df,
        train_size=TUNING_ROWS,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COL]
    )
    
    X_tune = tune_df.drop(columns=[TARGET_COL])
    y_tune = tune_df[TARGET_COL]
    
    # 80% tuning train, 20% validation
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tune, y_tune, test_size=0.20, random_state=RANDOM_STATE, stratify=y_tune
    )
    
    # Train model on the 80% tuning training split using best parameters
    print("Training tuning model...")
    model = xgb.XGBClassifier(
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params
    )
    model.fit(X_tr, y_tr)
    
    # Predict probabilities on the validation split
    print("Predicting probabilities on validation set...")
    val_probs = model.predict_proba(X_va)[:, 1]
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []
    
    print("\nEvaluating thresholds:")
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(y_va, preds)
        prec = precision_score(y_va, preds, zero_division=0)
        rec = recall_score(y_va, preds)
        f1 = f1_score(y_va, preds)
        
        results.append({
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        print(f"  Threshold {t:.2f} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
    results_df = pd.DataFrame(results)
    
    # Lock the threshold that yields the maximum Accuracy
    best_idx = results_df["Accuracy"].idxmax()
    locked_threshold = results_df.loc[best_idx, "Threshold"]
    max_accuracy = results_df.loc[best_idx, "Accuracy"]
    
    print(f"\nLocked Decision Threshold: {locked_threshold:.2f} (Accuracy: {max_accuracy:.4f})")
    
    # Save the locked threshold to JSON
    threshold_config = {
        "locked_threshold": float(locked_threshold),
        "validation_accuracy": float(max_accuracy)
    }
    
    threshold_path = os.path.join(MODEL_DIR, "threshold_config.json")
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=4)
        
    print(f"Threshold configuration saved to {threshold_path}")

if __name__ == "__main__":
    main()
