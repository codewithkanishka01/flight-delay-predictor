import os
import json
import pandas as pd
import numpy as np
import time
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import sys

# Add src to system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config_new import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COL,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    MODEL_DIR,
    RANDOM_STATE
)
from preprocessing import engineer_features_df, get_preprocessing_pipeline

def main():
    print("=== STARTING PHASE 3A: BASELINE & FEATURE SELECTION ===")
    
    # 1. Load train and test data
    print("Loading datasets...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    # Separate features and labels
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # 2. Evaluate Majority Class Baseline
    print("Evaluating Majority Class Baseline Classifier...")
    # Class 0 is the majority class (81.68%)
    majority_preds = np.zeros(len(y_test))
    
    acc = accuracy_score(y_test, majority_preds)
    prec = precision_score(y_test, majority_preds, zero_division=0)
    rec = recall_score(y_test, majority_preds)
    f1 = f1_score(y_test, majority_preds)
    cm = confusion_matrix(y_test, majority_preds)
    
    baseline_metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "Confusion_Matrix": cm.tolist()
    }
    
    baseline_path = os.path.join(OUTPUT_DIR, "leakage_free_baseline_metrics.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"Majority baseline evaluated. Metrics saved to {baseline_path}")
    
    # 3. Apply Preprocessing (Strictly on training partition first)
    print("Running feature engineering...")
    X_train_eng = engineer_features_df(X_train)
    X_test_eng = engineer_features_df(X_test)
    
    print("Fitting preprocessing ColumnTransformer on train features...")
    preprocessor = get_preprocessing_pipeline()
    preprocessor.fit(X_train_eng)
    
    print("Transforming train and test sets to sparse matrices...")
    X_train_trans = preprocessor.transform(X_train_eng)
    X_test_trans = preprocessor.transform(X_test_eng)
    
    feature_names = preprocessor.get_feature_names_out()
    num_features = len(feature_names)
    print(f"Total available engineered/encoded features: {num_features}")
    
    # 4. ANOVA feature selection using f_classif (Train partition only)
    print("Running ANOVA F-Test ranking on train features...")
    t0 = time.time()
    f_vals, p_vals = f_classif(X_train_trans, y_train)
    print(f"ANOVA calculation completed in {time.time() - t0:.2f} seconds.")
    
    # Create ANOVA ranking dataframe
    anova_df = pd.DataFrame({
        "Feature_Index": np.arange(num_features),
        "Feature_Name": feature_names,
        "F_Score": f_vals,
        "p_value": p_vals
    }).sort_values(by="F_Score", ascending=False)
    
    anova_path = os.path.join(OUTPUT_DIR, "anova_feature_ranking.csv")
    anova_df.to_csv(anova_path, index=False)
    print(f"ANOVA feature ranking saved to {anova_path}")
    
    # 5. Forward Sequential Feature Selection (FSFS)
    # SFS is highly expensive on 8,516 features. To prevent computational blowup,
    # we first filter down to the top 50 ANOVA features, and select the best 10 from them.
    top_50_indices = anova_df.iloc[:50]["Feature_Index"].values
    top_50_names = anova_df.iloc[:50]["Feature_Name"].values
    
    X_train_trans_filtered = X_train_trans[:, top_50_indices]
    
    print("\nExtracting 100,000 stratified training rows for SFS...")
    X_sfs_sub, _, y_sfs_sub, _ = train_test_split(
        X_train_trans_filtered, y_train, train_size=100000, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print("Initializing Forward Sequential Feature Selection (SFS) with Logistic Regression...")
    lr_estimator = LogisticRegression(max_iter=150, solver="lbfgs", random_state=RANDOM_STATE)
    
    # Select 10 features out of the top 50 ANOVA features
    sfs = SequentialFeatureSelector(
        estimator=lr_estimator,
        n_features_to_select=10,
        direction="forward",
        cv=2,
        n_jobs=-1
    )
    
    t0 = time.time()
    sfs.fit(X_sfs_sub, y_sfs_sub)
    print(f"SFS completed in {time.time() - t0:.2f} seconds.")
    
    # Retrieve names of SFS-selected features
    sfs_support = sfs.get_support()
    selected_features_names = list(top_50_names[sfs_support])
    
    sfs_path = os.path.join(OUTPUT_DIR, "sfs_selected_features.json")
    with open(sfs_path, "w") as f:
        json.dump(selected_features_names, f, indent=4)
    print(f"SFS-selected feature names saved to {sfs_path}")
    
    # Save the selected feature indices in the pipeline models folder
    feature_selection_path = os.path.join(MODEL_DIR, "feature_selection.joblib")
    import joblib
    joblib.dump({
        "top_50_indices": top_50_indices,
        "sfs_selected_indices": sfs_support,
        "selected_features_names": selected_features_names
    }, feature_selection_path)
    print(f"Feature selection state serialized to {feature_selection_path}")
    
    # 6. Printing summary report
    print("\n=================== PHASE 3A REPORT ===================")
    print("1. Majority Class Baseline Classifier Results:")
    print(f"   Accuracy:  {acc*100:.4f}%")
    print(f"   Precision: {prec*100:.4f}%")
    print(f"   Recall:    {rec*100:.4f}%")
    print(f"   F1 Score:  {f1*100:.4f}%")
    print(f"   Confusion Matrix:\n{cm}")
    
    print(f"\n2. Total Available Encoded/Scaled Features: {num_features}")
    
    print("\n3. Top 10 ANOVA Features:")
    print(anova_df.head(10)[["Feature_Name", "F_Score"]])
    
    print("\n4. Forward SFS Selected Features (Top 10):")
    for idx, name in enumerate(selected_features_names, 1):
        print(f"   {idx}. {name}")
        
    print("\n5. Test Data Leakage Prevention Statement:")
    print("   [VERIFIED] All feature selection metrics, ANOVA ranking, and SFS steps were fit")
    print("   strictly on the training partition. Untouched test data was never used.")
    print("=======================================================")

if __name__ == "__main__":
    main()
