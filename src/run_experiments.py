import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def target_encode_smooth(train_X, col, train_y, smoothing=10):
    global_mean = train_y.mean()
    col_str = train_X[col].astype(str)
    means = train_y.groupby(col_str).mean()
    counts = col_str.value_counts()
    
    smoothed_vals = (means * counts + global_mean * smoothing) / (counts + smoothing)
    mapping = smoothed_vals.to_dict()
    
    return col_str.map(mapping).fillna(global_mean)

def engineer_features(df_in, active_cols):
    df_out = pd.DataFrame()
    for col in active_cols:
        if col == 'FL_DATE':
            dt = pd.to_datetime(df_in['FL_DATE'], format='%Y-%m-%d')
            df_out['YEAR'] = dt.dt.year.astype(int)
            df_out['MONTH'] = dt.dt.month.astype(int)
            df_out['DAY'] = dt.dt.day.astype(int)
            df_out['DAY_OF_WEEK'] = dt.dt.dayofweek.astype(int)
        elif col == 'CRS_DEP_TIME':
            times = df_in['CRS_DEP_TIME'].fillna(0).astype(int)
            df_out['CRS_DEP_HOUR'] = (times // 100).astype(int)
            df_out['CRS_DEP_MINUTE'] = (times % 100).astype(int)
        elif col == 'CRS_ARR_TIME':
            times = df_in['CRS_ARR_TIME'].fillna(0).astype(int)
            df_out['CRS_ARR_HOUR'] = (times // 100).astype(int)
            df_out['CRS_ARR_MINUTE'] = (times % 100).astype(int)
        elif col in ['AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 'AIRLINE_DOT', 'DOT_CODE']:
            df_out[col] = df_in[col].astype(str).astype('category')
        else:
            df_out[col] = df_in[col].fillna(df_in[col].median())
    return df_out

def run_experiment_pipeline(X_train_raw, y_train, X_test_raw, y_test, selected_cols, exp_name):
    print(f"\n--- Running Pipeline for {exp_name} ({len(selected_cols)} columns) ---")
    
    # 1. Feature Engineering
    X_tr_full = engineer_features(X_train_raw, selected_cols)
    X_te_full = engineer_features(X_test_raw, selected_cols)
    
    # 2. Extract reproducible 300,000 stratified training subset
    X_opt_sub, _, y_opt_sub, _ = train_test_split(
        X_tr_full, y_train, train_size=300000, random_state=42, stratify=y_train
    )
    
    # Split 80/20 train/validation for optimization
    X_tr_opt, X_va_opt, y_tr_opt, y_va_opt = train_test_split(
        X_opt_sub, y_opt_sub, test_size=0.20, random_state=42, stratify=y_opt_sub
    )
    
    # 3. XGBoost Hyperparameter Search (same grids as previous tuning)
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
    
    rs = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=10,
        cv=2,
        scoring='accuracy',
        random_state=42,
        n_jobs=2,
        verbose=0
    )
    rs.fit(X_tr_opt, y_tr_opt)
    
    best_params = rs.best_params_
    print(f"  Best Parameters: {best_params}")
    
    # 4. Fit model on full 300k training subset using optimal params
    tuned_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        enable_categorical=True,
        n_jobs=-1,
        **best_params
    )
    tuned_model.fit(X_opt_sub, y_opt_sub)
    
    # 5. Tune Threshold on Validation Split (60k)
    val_probs = tuned_model.predict_proba(X_va_opt)[:, 1]
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_acc = 0
    best_t_acc = 0.50
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(y_va_opt, preds)
        if acc > best_acc:
            best_acc = acc
            best_t_acc = t
            
    print(f"  Locked Accuracy Threshold: {best_t_acc:.2f} (Val Acc: {best_acc*100:.2f}%)")
    
    # 6. Evaluate once on untouched test set
    test_probs = tuned_model.predict_proba(X_te_full)[:, 1]
    test_preds = (test_probs >= best_t_acc).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    
    print(f"  Test Accuracy:  {acc*100:.2f}%")
    print(f"  Test Precision: {prec*100:.2f}%")
    print(f"  Test Recall:    {rec*100:.2f}%")
    print(f"  Test F1:        {f1*100:.2f}%")
    print(f"  Test ROC-AUC:   {auc*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Save model
    model_save_path = f"/Users/kanishkapandey/Desktop/ML/models/xgb_model_{exp_name}.json"
    tuned_model.save_model(model_save_path)
    
    return {
        "Model": exp_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc,
        "Selected Columns": selected_cols,
        "Threshold": best_t_acc
    }

def main():
    data_path = "/Users/kanishkapandey/Desktop/flights_sample_3m.csv"
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[df['CANCELLED'] == 0]
    df = df[df['DIVERTED'] == 0]
    df = df.dropna(subset=['ARR_DELAY'])
    df['DELAYED'] = (df['ARR_DELAY'] >= 15).astype(int)
    
    candidates = [
        'FL_DATE', 'AIRLINE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE', 'FL_NUMBER',
        'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 'CRS_DEP_TIME', 'CRS_ARR_TIME',
        'CRS_ELAPSED_TIME', 'DISTANCE'
    ]
    
    X = df[candidates]
    y = df['DELAYED']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # ANOVA F-Test Ranking
    print("Running target encoding for ANOVA F-Test...")
    X_train_encoded = pd.DataFrame()
    categorical_cols = ['FL_DATE', 'AIRLINE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY']
    
    for col in candidates:
        if col in categorical_cols:
            X_train_encoded[col] = target_encode_smooth(X_train, col, y_train)
        else:
            X_train_encoded[col] = X_train[col].fillna(X_train[col].median())
            
    f_vals, _ = f_classif(X_train_encoded, y_train)
    anova_df = pd.DataFrame({
        'Feature': candidates,
        'F_Score': f_vals
    }).sort_values(by='F_Score', ascending=False)
    
    ranked_features = anova_df['Feature'].tolist()
    print("\n=== ANOVA RANKING ===")
    print(anova_df)
    
    # Selected original column lists
    col_10 = ['FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 'DISTANCE']
    col_12 = ranked_features[:12]
    col_14 = ranked_features[:14] # Experiment B - maximum possible 14 columns
    
    # ------------------ Run Experiments ------------------
    # We load the tuned results of 10 columns from before, or run it here for fair comparison:
    results_10 = run_experiment_pipeline(X_train, y_train, X_test, y_test, col_10, "10_columns")
    results_12 = run_experiment_pipeline(X_train, y_train, X_test, y_test, col_12, "12_columns")
    results_14 = run_experiment_pipeline(X_train, y_train, X_test, y_test, col_14, "14_columns")
    
    # Compile comparison
    comparison = [results_10, results_12, results_14]
    comp_df = pd.DataFrame(comparison)
    
    output_path = "/Users/kanishkapandey/Desktop/ML/outputs/experiments_comparison.csv"
    comp_df.to_csv(output_path, index=False)
    
    print("\n=== EXPERIMENTS COMPARISON TABLE ===")
    print(comp_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Threshold']].to_string(index=False))
    
    print("\nSelected Columns Details:")
    print(f"10 selected columns: {col_10}")
    print(f"12 selected columns: {col_12}")
    print(f"14 selected columns (Experiment B maximum): {col_14}")

if __name__ == "__main__":
    main()
