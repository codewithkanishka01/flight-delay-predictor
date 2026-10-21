import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from config import (
    TRAIN_DATA_PATH,
    CATEGORICAL_COLS,
    TARGET_COL,
    RANDOM_STATE,
    OUTPUT_DIR
)

def target_encode(X_train, y_train, X_val, cols):
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    global_mean = y_train.mean()
    
    for col in cols:
        train_series = X_train[col].astype(str)
        val_series = X_val[col].astype(str)
        
        means = y_train.groupby(train_series).mean()
        counts = train_series.value_counts()
        smooth_weight = 10
        
        # Compute smoothed target encoding mapping
        smoothed_vals = (means * counts + global_mean * smooth_weight) / (counts + smooth_weight)
        mapping = smoothed_vals.to_dict()
        
        X_train_enc[col] = train_series.map(mapping).fillna(global_mean).astype(float)
        X_val_enc[col] = val_series.map(mapping).fillna(global_mean).astype(float)
        
    return X_train_enc, X_val_enc


def evaluate_predictions(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, auc, cm

def main():
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    
    # Split into model training and validation (80/20)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Split completed. Train shape: {X_tr.shape}, Validation shape: {X_va.shape}")
    
    # 1. Target encode categoricals for Logistic Regression
    print("Applying target encoding for Logistic Regression...")
    X_tr_lr, X_va_lr = target_encode(X_tr, y_tr, X_va, CATEGORICAL_COLS)
    
    # Train Logistic Regression
    print("Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_tr_lr, y_tr)
    lr_preds = lr.predict(X_va_lr)
    lr_probs = lr.predict_proba(X_va_lr)[:, 1]
    
    lr_metrics = evaluate_predictions(y_va, lr_preds, lr_probs)
    
    # 2. Train XGBoost Baseline (Native Categorical support)
    print("Training XGBoost baseline...")
    # Using small estimators for baseline speed
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_clf.fit(X_tr, y_tr)
    xgb_preds = xgb_clf.predict(X_va)
    xgb_probs = xgb_clf.predict_proba(X_va)[:, 1]
    
    xgb_metrics = evaluate_predictions(y_va, xgb_preds, xgb_probs)
    
    # Report Results
    models_metrics = [
        ("Logistic Regression", *lr_metrics),
        ("XGBoost Baseline", *xgb_metrics)
    ]
    
    print("\n================ BASELINE MODELS COMPARISON ================")
    summary_data = []
    for name, acc, prec, rec, f1, auc, cm in models_metrics:
        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        summary_data.append([name, acc, prec, rec, f1, auc])
        
    summary_df = pd.DataFrame(
        summary_data, 
        columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison.csv"), index=False)
    print("\nBaseline results saved to outputs/baseline_comparison.csv")

if __name__ == "__main__":
    main()
