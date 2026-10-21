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
from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COL,
    RANDOM_STATE,
    MODEL_DIR,
    OUTPUT_DIR
)

def main():
    # 1. Load optimized params and threshold
    params_path = os.path.join(MODEL_DIR, "best_params.json")
    threshold_path = os.path.join(MODEL_DIR, "threshold_config.json")
    
    if not os.path.exists(params_path) or not os.path.exists(threshold_path):
        print("Error: Hyperparameters or threshold config not found. Run tuning scripts first.")
        return
        
    with open(params_path, 'r') as f:
        best_params = json.load(f)
        
    with open(threshold_path, 'r') as f:
        threshold_config = json.load(f)
        
    locked_threshold = threshold_config["locked_threshold"]
    
    print("Optimal Hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"Locked Decision Threshold: {locked_threshold:.2f}")
    
    # 2. Load full train and test sets
    print(f"Loading full train set from {TRAIN_DATA_PATH}...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    print(f"Loading untouched test set from {TEST_DATA_PATH}...")
    test_df = pd.read_parquet(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    print(f"Shapes - Train features: {X_train.shape}, Test features: {X_test.shape}")
    
    # 3. Train final model on full training set
    print("Training final XGBoost model on full training dataset (this may take a few minutes)...")
    final_model = xgb.XGBClassifier(
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params
    )
    final_model.fit(X_train, y_train)
    
    # Save the trained model
    model_save_path = os.path.join(MODEL_DIR, "final_xgb_model.json")
    final_model.save_model(model_save_path)
    print(f"Saved trained final model to {model_save_path}")
    
    # 4. Evaluate once on the untouched test set
    print("Running predictions on the untouched test set...")
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= locked_threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    
    print("\n================ FINAL TEST EVALUATION RESULTS ================")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"ROC-AUC:   {auc*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print("===============================================================")
    
    # 5. Honest reporting of 90% accuracy target
    if acc >= 0.90:
        print("\nSUCCESS: The 90%+ classification accuracy target has been genuinely achieved!")
    else:
        print(f"\nHONEST REPORT: Final test accuracy is {acc*100:.2f}%, which is below the 90% target.")
        print("Limiting Factors Analysis:")
        print("1. Target Leakage Prevention: Since we strictly avoided post-departure variables (such as departure delays, elapsed air time, or taxis), we only used scheduled pre-departure information. Pre-departure flight data has limited signal because most flight delays are caused by operational deviations that occur AFTER departure (like late-arriving aircraft, terminal congestion, or weather changes).")
        print("2. Class Imbalance: The target distribution is highly skewed (81.68% class 0, 18.32% class 1). Predicting class 0 for all instances yields an accuracy of 81.68%. To exceed 90% accuracy, the model must be highly precise in predicting class 1, which is challenging without real-time post-boarding details.")
        
    # Save final metrics
    metrics_summary = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "ROC-AUC": float(auc),
        "Confusion_Matrix": cm.tolist()
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, "final_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nSaved final metrics summary to {metrics_path}")

if __name__ == "__main__":
    main()
