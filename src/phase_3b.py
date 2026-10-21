import os
import json
import pandas as pd
import numpy as np
import time
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import joblib
import sys

# Add src to system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config_new import (
    TRAIN_DATA_PATH,
    TARGET_COL,
    OUTPUT_DIR,
    MODEL_DIR,
    RANDOM_STATE
)
from preprocessing import engineer_features_df, get_preprocessing_pipeline

def get_indices_from_names(feature_names_list, selected_names):
    feature_names_map = {name: idx for idx, name in enumerate(feature_names_list)}
    indices = []
    for name in selected_names:
        if name in feature_names_map:
            indices.append(feature_names_map[name])
        else:
            # Categorical value might be rare and missing from screening subset
            pass
    return indices

def main():
    print("=== STARTING PHASE 3B: LEAKAGE-FREE MODEL SCREENING ===")
    
    # Create screening models directory
    screening_model_dir = os.path.join(MODEL_DIR, "screening")
    os.makedirs(screening_model_dir, exist_ok=True)
    
    # 1. Load train dataset
    print("Loading train partition...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    
    # 2. Extract reproducible 300,000 stratified screening subset
    print("Extracting 300,000 row stratified subset for screening...")
    df_sub, _ = train_test_split(
        train_df, train_size=300000, random_state=RANDOM_STATE, stratify=train_df[TARGET_COL]
    )
    
    X_sub = df_sub.drop(columns=[TARGET_COL])
    y_sub = df_sub[TARGET_COL]
    
    # 3. Split 80/20 train/validation strictly inside the training partition
    print("Splitting subset 80/20 into train/validation splits...")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_sub, y_sub, test_size=0.20, random_state=RANDOM_STATE, stratify=y_sub
    )
    print(f"  Screening Train shape: {X_tr.shape}")
    print(f"  Screening Validation shape: {X_va.shape}")
    
    # 4. Feature Engineering
    print("Running feature engineering...")
    X_tr_eng = engineer_features_df(X_tr)
    X_va_eng = engineer_features_df(X_va)
    
    # 5. Fit Preprocessor strictly on training split
    print("Fitting ColumnTransformer preprocessor on screening train features...")
    preprocessor = get_preprocessing_pipeline()
    preprocessor.fit(X_tr_eng)
    
    # Transform both
    X_tr_trans = preprocessor.transform(X_tr_eng)
    X_va_trans = preprocessor.transform(X_va_eng)
    
    feature_names = preprocessor.get_feature_names_out()
    print(f"  Total sparse feature columns: {X_tr_trans.shape[1]}")
    
    # 6. Load feature selection names and map dynamically to current indices
    anova_path = os.path.join(OUTPUT_DIR, "anova_feature_ranking.csv")
    sfs_path = os.path.join(OUTPUT_DIR, "sfs_selected_features.json")
    
    if not (os.path.exists(anova_path) and os.path.exists(sfs_path)):
        print("Error: Feature selection files not found. Run Phase 3A first.")
        return
        
    # Get top 20 ANOVA feature names
    anova_df = pd.read_csv(anova_path)
    top_20_names = anova_df.iloc[:20]["Feature_Name"].tolist()
    
    # Get SFS feature names
    with open(sfs_path, "r") as f:
        sfs_names = json.load(f)
        
    # Map names to current one-hot index space
    top_20_indices = get_indices_from_names(feature_names, top_20_names)
    sfs_indices = get_indices_from_names(feature_names, sfs_names)
    
    print(f"  Mapped ANOVA Top 20 features: {len(top_20_indices)} features mapped.")
    print(f"  Mapped SFS Top 10 features: {len(sfs_indices)} features mapped.")
    
    # Feature representations dictionary
    feature_sets = {
        "Full encoded features (Sparse)": (X_tr_trans, X_va_trans),
        "ANOVA Top 20 features (Dense)": (X_tr_trans[:, top_20_indices].toarray(), X_va_trans[:, top_20_indices].toarray()),
        "SFS Top 10 features (Dense)": (X_tr_trans[:, sfs_indices].toarray(), X_va_trans[:, sfs_indices].toarray())
    }
    
    # List to store results
    results = []
    
    def evaluate_model(model, X_train_eval, X_val_eval, y_train_eval, y_val_eval, model_name, feat_set_name, bal_strategy):
        print(f"  Training {model_name} on {feat_set_name}...")
        t0 = time.time()
        model.fit(X_train_eval, y_train_eval)
        train_time = time.time() - t0
        
        # Predictions and Probabilities
        probs = model.predict_proba(X_val_eval)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        # Calculate Metrics
        acc = accuracy_score(y_val_eval, preds)
        prec = precision_score(y_val_eval, preds, zero_division=0)
        rec = recall_score(y_val_eval, preds)
        f1 = f1_score(y_val_eval, preds)
        auc = roc_auc_score(y_val_eval, probs)
        pr_auc = average_precision_score(y_val_eval, probs)
        cm = confusion_matrix(y_val_eval, preds)
        
        results.append({
            "Model": model_name,
            "Feature Set": feat_set_name,
            "Feature Selection": "ANOVA + SFS" if "SFS" in feat_set_name else ("ANOVA" if "ANOVA" in feat_set_name else "None"),
            "Training Size": X_train_eval.shape[0],
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "ROC-AUC": float(auc),
            "PR-AUC": float(pr_auc),
            "Training Time": float(train_time),
            "Class Weight / Balance Strategy": bal_strategy
        })
        
        # Save model checkpoint
        sanitized_name = f"{model_name.lower().replace(' ', '_')}_{feat_set_name.split()[0].lower()}_{bal_strategy.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join(screening_model_dir, sanitized_name))
        
        print(f"    Finished. Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ROC-AUC: {auc*100:.2f}%, Time: {train_time:.2f}s")
    
    # --- Experiment Runs ---
    
    # 1. Logistic Regression
    # 1A. Unweighted
    lr_unweighted = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    evaluate_model(
        lr_unweighted, 
        feature_sets["Full encoded features (Sparse)"][0], 
        feature_sets["Full encoded features (Sparse)"][1], 
        y_tr, y_va, 
        "Logistic Regression", 
        "Full encoded features (Sparse)", 
        "None"
    )
    
    # 1B. Class-Balanced
    lr_balanced = LogisticRegression(max_iter=300, class_weight="balanced", random_state=RANDOM_STATE)
    evaluate_model(
        lr_balanced, 
        feature_sets["Full encoded features (Sparse)"][0], 
        feature_sets["Full encoded features (Sparse)"][1], 
        y_tr, y_va, 
        "Logistic Regression", 
        "Full encoded features (Sparse)", 
        "class_weight='balanced'"
    )
    
    # 2. XGBoost
    # 2A. Unweighted
    xgb_unweighted = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        xgb_unweighted,
        feature_sets["Full encoded features (Sparse)"][0], 
        feature_sets["Full encoded features (Sparse)"][1],
        y_tr, y_va,
        "XGBoost",
        "Full encoded features (Sparse)",
        "None"
    )
    
    # 2B. Balanced (scale_pos_weight = 81.68 / 18.32 = 4.46)
    xgb_balanced = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=4.46, random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        xgb_balanced,
        feature_sets["Full encoded features (Sparse)"][0], 
        feature_sets["Full encoded features (Sparse)"][1],
        y_tr, y_va,
        "XGBoost",
        "Full encoded features (Sparse)",
        "scale_pos_weight=4.46"
    )
    
    # 3. Random Forest (ANOVA Top 20)
    # 3A. Unweighted
    rf_unweighted = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        rf_unweighted,
        feature_sets["ANOVA Top 20 features (Dense)"][0],
        feature_sets["ANOVA Top 20 features (Dense)"][1],
        y_tr, y_va,
        "Random Forest",
        "ANOVA Top 20 features (Dense)",
        "None"
    )
    
    # 3B. Balanced
    rf_balanced = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        rf_balanced,
        feature_sets["ANOVA Top 20 features (Dense)"][0],
        feature_sets["ANOVA Top 20 features (Dense)"][1],
        y_tr, y_va,
        "Random Forest",
        "ANOVA Top 20 features (Dense)",
        "class_weight='balanced'"
    )
    
    # 4. Extra Trees (ANOVA Top 20)
    # 4A. Unweighted
    et_unweighted = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        et_unweighted,
        feature_sets["ANOVA Top 20 features (Dense)"][0],
        feature_sets["ANOVA Top 20 features (Dense)"][1],
        y_tr, y_va,
        "Extra Trees",
        "ANOVA Top 20 features (Dense)",
        "None"
    )
    
    # 4B. Balanced
    et_balanced = ExtraTreesClassifier(n_estimators=50, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        et_balanced,
        feature_sets["ANOVA Top 20 features (Dense)"][0],
        feature_sets["ANOVA Top 20 features (Dense)"][1],
        y_tr, y_va,
        "Extra Trees",
        "ANOVA Top 20 features (Dense)",
        "class_weight='balanced'"
    )
    
    # 5. HistGradientBoosting (ANOVA Top 20, natively supports dense categorical/numerical representation)
    # 5A. Unweighted
    hgb_unweighted = HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE)
    evaluate_model(
        hgb_unweighted,
        feature_sets["ANOVA Top 20 features (Dense)"][0],
        feature_sets["ANOVA Top 20 features (Dense)"][1],
        y_tr, y_va,
        "HistGradientBoosting",
        "ANOVA Top 20 features (Dense)",
        "None"
    )
    
    # 5B. Balanced
    hgb_balanced = HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=RANDOM_STATE)
    evaluate_model(
        hgb_balanced,
        feature_sets["ANOVA Top 20 features (Dense)"][0],
        feature_sets["ANOVA Top 20 features (Dense)"][1],
        y_tr, y_va,
        "HistGradientBoosting",
        "ANOVA Top 20 features (Dense)",
        "class_weight='balanced'"
    )
    
    # 6. SFS-Selected Features compact check (Logistic Regression & XGBoost)
    # 6A. LR SFS
    lr_sfs = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    evaluate_model(
        lr_sfs,
        feature_sets["SFS Top 10 features (Dense)"][0],
        feature_sets["SFS Top 10 features (Dense)"][1],
        y_tr, y_va,
        "Logistic Regression",
        "SFS Top 10 features (Dense)",
        "None"
    )
    
    # 6B. XGBoost SFS
    xgb_sfs = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_model(
        xgb_sfs,
        feature_sets["SFS Top 10 features (Dense)"][0],
        feature_sets["SFS Top 10 features (Dense)"][1],
        y_tr, y_va,
        "XGBoost",
        "SFS Top 10 features (Dense)",
        "None"
    )
    
    # Save results to CSV & JSON
    comp_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "leakage_free_model_screening.csv")
    json_path = os.path.join(OUTPUT_DIR, "leakage_free_model_screening.json")
    
    comp_df.to_csv(csv_path, index=False)
    comp_df.to_json(json_path, orient="records", indent=4)
    print(f"\nModel screening completed. Results saved to {csv_path} and {json_path}")
    
    # Find Best Models by validation metrics
    best_acc_row = comp_df.loc[comp_df["Accuracy"].idxmax()]
    best_f1_row = comp_df.loc[comp_df["F1"].idxmax()]
    best_rec_row = comp_df.loc[comp_df["Recall"].idxmax()]
    best_auc_row = comp_df.loc[comp_df["ROC-AUC"].idxmax()]
    
    # Let's define the best overall/balanced candidate as the one with highest ROC-AUC 
    # among models that predict Class 1 (Recall > 5%)
    candidates_with_recall = comp_df[comp_df["Recall"] > 0.05]
    if len(candidates_with_recall) > 0:
        best_balanced_row = candidates_with_recall.loc[candidates_with_recall["ROC-AUC"].idxmax()]
    else:
        best_balanced_row = best_auc_row
        
    print("\n=================== SCREENING RESULTS SUMMARY ===================")
    print(f"1. Best Validation Accuracy:  {best_acc_row['Accuracy']*100:.2f}% (Model: {best_acc_row['Model']}, Feature Set: {best_acc_row['Feature Set']}, Balance: {best_acc_row['Class Weight / Balance Strategy']})")
    print(f"2. Best Validation F1 Score:  {best_f1_row['F1']*100:.2f}% (Model: {best_f1_row['Model']}, Feature Set: {best_f1_row['Feature Set']}, Balance: {best_f1_row['Class Weight / Balance Strategy']})")
    print(f"3. Best Validation Recall:    {best_rec_row['Recall']*100:.2f}% (Model: {best_rec_row['Model']}, Feature Set: {best_rec_row['Feature Set']}, Balance: {best_rec_row['Class Weight / Balance Strategy']})")
    print(f"4. Best Validation ROC-AUC:   {best_auc_row['ROC-AUC']*100:.2f}% (Model: {best_auc_row['Model']}, Feature Set: {best_auc_row['Feature Set']}, Balance: {best_auc_row['Class Weight / Balance Strategy']})")
    print(f"5. Best Overall Balanced Model: {best_balanced_row['Model']} on {best_balanced_row['Feature Set']} (Balance: {best_balanced_row['Class Weight / Balance Strategy']}, ROC-AUC: {best_balanced_row['ROC-AUC']*100:.2f}%, Recall: {best_balanced_row['Recall']*100:.2f}%)")
    
    print("\nVerification & Integrity Statement:")
    print("   - Untouched test set was NOT used for model training or validation.")
    print("   - Hyperparameter tuning (RandomizedSearchCV) was NOT performed.")
    print("   - Probability decision threshold tuning was NOT performed.")
    print("   - No full sparse feature matrix was converted to dense (only small ANOVA/SFS subsets).")
    print("================================================================")

if __name__ == "__main__":
    main()
