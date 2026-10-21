import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
    
    # 1. Load prepared training data to re-create the validation set
    print("Loading prepared training data...")
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train_10.parquet"))
    y_train = pd.read_parquet(os.path.join(data_dir, "y_train_10.parquet")).iloc[:, 0]
    
    # Re-create the 300,000 stratified training subset
    X_opt_sub, _, y_opt_sub, _ = train_test_split(
        X_train, y_train, train_size=300000, random_state=42, stratify=y_train
    )
    
    # Extract the 20% validation split (60,000 rows)
    _, X_va_opt, _, y_va_opt = train_test_split(
        X_opt_sub, y_opt_sub, test_size=0.20, random_state=42, stratify=y_opt_sub
    )
    print(f"Validation set size: {X_va_opt.shape[0]} rows")
    
    # 2. Load the optimized model
    model_path = os.path.join(model_dir, "xgb_model_tuned.json")
    if not os.path.exists(model_path):
        print(f"Error: Tuned XGBoost model not found at {model_path}. Run hyperparameter optimization first.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Loaded tuned XGBoost model successfully.")
    
    # 3. Predict probabilities
    print("Predicting probabilities on validation set...")
    val_probs = model.predict_proba(X_va_opt)[:, 1]
    
    # Sweep thresholds
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    auc_score = roc_auc_score(y_va_opt, val_probs)
    
    results = []
    print("\nSweeping decision thresholds:")
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(y_va_opt, preds)
        prec = precision_score(y_va_opt, preds, zero_division=0)
        rec = recall_score(y_va_opt, preds)
        f1 = f1_score(y_va_opt, preds)
        
        results.append({
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        print(f"  Threshold {t:.2f} -> Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
        
    results_df = pd.DataFrame(results)
    
    # Find Best Accuracy Threshold
    best_acc_idx = results_df["Accuracy"].idxmax()
    best_acc_row = results_df.loc[best_acc_idx]
    best_threshold_accuracy = best_acc_row["Threshold"]
    
    # Find Best F1 Threshold
    best_f1_idx = results_df["F1 Score"].idxmax()
    best_f1_row = results_df.loc[best_f1_idx]
    best_threshold_f1 = best_f1_row["Threshold"]
    
    # Confusion matrix for best accuracy threshold
    best_acc_preds = (val_probs >= best_threshold_accuracy).astype(int)
    cm_acc = confusion_matrix(y_va_opt, best_acc_preds)
    
    print("\n================ THRESHOLD TUNING REPORT ================")
    print(f"ROC-AUC (Threshold-Independent): {auc_score*100:.2f}%")
    
    print("\n--- 1. Accuracy-Maximized Decision Boundary ---")
    print(f"Best Accuracy Threshold:  {best_threshold_accuracy:.2f}")
    print(f"Validation Accuracy:      {best_acc_row['Accuracy']*100:.2f}%")
    print(f"Precision at Threshold:   {best_acc_row['Precision']*100:.2f}%")
    print(f"Recall at Threshold:      {best_acc_row['Recall']*100:.2f}%")
    print(f"F1 Score at Threshold:    {best_acc_row['F1 Score']*100:.2f}%")
    print(f"Confusion Matrix:\n{cm_acc}")
    
    print("\n--- 2. F1-Score-Maximized Decision Boundary ---")
    print(f"Best F1 Threshold:        {best_threshold_f1:.2f}")
    print(f"Accuracy at F1-Max:       {best_f1_row['Accuracy']*100:.2f}%")
    print(f"Precision at F1-Max:      {best_f1_row['Precision']*100:.2f}%")
    print(f"Recall at F1-Max:         {best_f1_row['Recall']*100:.2f}%")
    print(f"F1 Score at F1-Max:       {best_f1_row['F1 Score']*100:.2f}%")
    print("=========================================================")
    
    # Save the locked accuracy-optimized threshold to JSON
    threshold_config = {
        "best_threshold_accuracy": float(best_threshold_accuracy),
        "best_threshold_f1": float(best_threshold_f1),
        "validation_accuracy": float(best_acc_row['Accuracy']),
        "roc_auc": float(auc_score)
    }
    
    config_path = os.path.join(model_dir, "threshold_config_10.json")
    with open(config_path, 'w') as f:
        json.dump(threshold_config, f, indent=4)
        
    print(f"\nSaved threshold configuration to {config_path}")

if __name__ == "__main__":
    main()
