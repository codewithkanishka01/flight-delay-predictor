import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def target_encode_smooth(train_X, col, train_y, test_X, smoothing=10):
    global_mean = train_y.mean()
    col_str = train_X[col].astype(str)
    test_col_str = test_X[col].astype(str)
    
    means = train_y.groupby(col_str).mean()
    counts = col_str.value_counts()
    
    smoothed_vals = (means * counts + global_mean * smoothing) / (counts + smoothing)
    mapping = smoothed_vals.to_dict()
    
    train_encoded = col_str.map(mapping).fillna(global_mean)
    test_encoded = test_col_str.map(mapping).fillna(global_mean)
    return train_encoded, test_encoded

def main():
    data_dir = "/Users/kanishkapandey/Desktop/ML/data"
    output_dir = "/Users/kanishkapandey/Desktop/ML/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prepared datasets
    print("Loading prepared datasets...")
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train_10.parquet"))
    X_test = pd.read_parquet(os.path.join(data_dir, "X_test_10.parquet"))
    y_train = pd.read_parquet(os.path.join(data_dir, "y_train_10.parquet")).iloc[:, 0]
    y_test = pd.read_parquet(os.path.join(data_dir, "y_test_10.parquet")).iloc[:, 0]
    
    # Draw reproducible stratified training subset of 300,000 rows
    print("Drawing 300,000 row stratified training subset for baseline models...")
    X_tr_sub, _, y_tr_sub, _ = train_test_split(
        X_train, y_train, train_size=300000, random_state=42, stratify=y_train
    )
    print(f"Baseline Training sample size: {X_tr_sub.shape[0]}")
    print(f"Untouched Test sample size: {X_test.shape[0]}")
    
    categorical_cols = ['AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY']
    
    # ------------------ 1. Logistic Regression Baseline ------------------
    print("\nTraining Logistic Regression baseline...")
    X_tr_lr = X_tr_sub.copy()
    X_te_lr = X_test.copy()
    
    # Apply smoothed target encoding for LR
    for col in categorical_cols:
        train_enc, test_enc = target_encode_smooth(X_tr_sub, col, y_tr_sub, X_test)
        X_tr_lr[col] = train_enc
        X_te_lr[col] = test_enc
        
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    lr.fit(X_tr_lr, y_tr_sub)
    
    lr_probs = lr.predict_proba(X_te_lr)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    
    # ------------------ 2. Random Forest Baseline ------------------
    print("Training Random Forest baseline (max_depth=10, 50 estimators)...")
    X_tr_rf = X_tr_sub.copy()
    X_te_rf = X_test.copy()
    
    # Apply label encoding for RF
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on training partition, handle unseen categories in test by adding an 'unknown' class
        train_cats = X_tr_sub[col].astype(str).tolist()
        test_cats = X_test[col].astype(str).tolist()
        
        le.fit(train_cats + ['<unknown>'])
        
        X_tr_rf[col] = le.transform(train_cats)
        X_te_rf[col] = [val if val in le.classes_ else '<unknown>' for val in test_cats]
        X_te_rf[col] = le.transform(X_te_rf[col])
        
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr_rf, y_tr_sub)
    
    rf_probs = rf.predict_proba(X_te_rf)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    
    # ------------------ 3. XGBoost Baseline ------------------
    print("Training XGBoost baseline (enable_categorical=True)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_tr_sub, y_tr_sub)
    
    xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    
    # ------------------ Evaluation & Reporting ------------------
    models = {
        "Logistic Regression": (lr_preds, lr_probs),
        "Random Forest": (rf_preds, rf_probs),
        "XGBoost Baseline": (xgb_preds, xgb_probs)
    }
    
    comparison = []
    
    for name, (preds, probs) in models.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        comparison.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc
        })
        
        print(f"\n================ {name.upper()} RESULTS ================")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%")
        print(f"ROC-AUC:   {auc*100:.2f}%")
        print(f"Confusion Matrix:\n{cm}")
        print("=====================================================")
        
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(output_dir, "baseline_10_comparison.csv"), index=False)
    
    print("\n=== BASELINE MODELS COMPARISON TABLE ===")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    main()
