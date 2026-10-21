import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    os.makedirs(model_dir, exist_ok=True)
    
    # Load prepared datasets
    print("Loading prepared training data...")
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train_10.parquet"))
    y_train = pd.read_parquet(os.path.join(data_dir, "y_train_10.parquet")).iloc[:, 0]
    
    # Create reproducible stratified optimization subset (300,000 rows)
    print("Extracting 300,000 row stratified optimization subset...")
    X_opt_sub, _, y_opt_sub, _ = train_test_split(
        X_train, y_train, train_size=300000, random_state=42, stratify=y_train
    )
    
    # Split into 80% optimization training (240k) and 20% validation (60k)
    print("Splitting optimization subset 80/20 into train/validation...")
    X_tr_opt, X_va_opt, y_tr_opt, y_va_opt = train_test_split(
        X_opt_sub, y_opt_sub, test_size=0.20, random_state=42, stratify=y_opt_sub
    )
    print(f"  Optimization Train shape: {X_tr_opt.shape}")
    print(f"  Optimization Validation shape: {X_va_opt.shape}")
    
    # XGBoost Grid Search Space
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1, 0.15],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.3],
        'scale_pos_weight': [1.0, 1.5, 2.0, 3.0, 4.46]
    }
    
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        enable_categorical=True,
        n_jobs=-1
    )
    
    print("\nInitializing RandomizedSearchCV (10 iterations, 2-fold CV)...")
    rs = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=10,
        cv=2,
        scoring='accuracy',
        random_state=42,
        n_jobs=2,
        verbose=2
    )
    
    print("Starting randomized search (this may take a few minutes)...")
    rs.fit(X_tr_opt, y_tr_opt)
    
    print("\n=== SEARCH COMPLETE ===")
    print("Best Parameters:")
    for key, val in rs.best_params_.items():
        print(f"  {key}: {val}")
    print(f"Best CV Accuracy Score: {rs.best_score_*100:.4f}%")
    
    # Evaluate best estimator on the validation split (60k rows)
    best_clf = rs.best_estimator_
    
    # Save the best estimator model file
    model_save_path = os.path.join(model_dir, "xgb_model_tuned.json")
    best_clf.save_model(model_save_path)
    
    # Save the parameters as JSON for reference
    params_path = os.path.join(model_dir, "best_params_10.json")
    with open(params_path, 'w') as f:
        json.dump(rs.best_params_, f, indent=4)
        
    print(f"\nSaved tuned model to {model_save_path}")
    
    # Validation evaluations
    val_probs = best_clf.predict_proba(X_va_opt)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_va_opt, val_preds)
    prec = precision_score(y_va_opt, val_preds, zero_division=0)
    rec = recall_score(y_va_opt, val_preds)
    f1 = f1_score(y_va_opt, val_preds)
    auc = roc_auc_score(y_va_opt, val_probs)
    cm = confusion_matrix(y_va_opt, val_preds)
    
    print("\n================ TUNED MODEL VALIDATION RESULTS ================")
    print(f"Validation Accuracy:  {acc*100:.2f}%")
    print(f"Validation Precision: {prec*100:.2f}%")
    print(f"Validation Recall:    {rec*100:.2f}%")
    print(f"Validation F1 Score:  {f1*100:.2f}%")
    print(f"Validation ROC-AUC:   {auc*100:.2f}%")
    print(f"Validation Confusion Matrix:\n{cm}")
    print("================================================================")

if __name__ == "__main__":
    main()
