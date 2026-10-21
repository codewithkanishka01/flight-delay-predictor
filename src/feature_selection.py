import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, roc_auc_score

def target_encode_smooth(train_X, col, train_y, smoothing=10):
    global_mean = train_y.mean()
    col_str = train_X[col].astype(str)
    means = train_y.groupby(col_str).mean()
    counts = col_str.value_counts()
    
    # Compute smoothed target encoding mapping
    smoothed_vals = (means * counts + global_mean * smoothing) / (counts + smoothing)
    mapping = smoothed_vals.to_dict()
    
    return col_str.map(mapping).fillna(global_mean)


def main():
    data_path = "/Users/kanishkapandey/Desktop/flights_sample_3m.csv"
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Target Ingestion & Cleaning
    print("Cleaning dataset (dropping cancelled and diverted flights)...")
    df = df[df['CANCELLED'] == 0]
    df = df[df['DIVERTED'] == 0]
    df = df.dropna(subset=['ARR_DELAY'])
    
    df['DELAYED'] = (df['ARR_DELAY'] >= 15).astype(int)
    print(f"Cleaned dataset shape: {df.shape}")
    
    # Define candidate original columns (12 pre-departure columns)
    original_candidates = [
        'FL_DATE', 'AIRLINE', 'AIRLINE_CODE', 'FL_NUMBER', 
        'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 
        'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE'
    ]
    
    X = df[original_candidates]
    y = df['DELAYED']
    
    # 2. Train-Test Split (80/20 Stratified) BEFORE feature selection
    print("Performing stratified split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. ANOVA F-Test Ranking on training data
    print("Running target encoding for ANOVA...")
    X_train_encoded = pd.DataFrame()
    categorical_cols = ['FL_DATE', 'AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY']
    
    for col in original_candidates:
        if col in categorical_cols:
            X_train_encoded[col] = target_encode_smooth(X_train, col, y_train)
        else:
            X_train_encoded[col] = X_train[col].fillna(X_train[col].median())
            
    print("Computing ANOVA F-scores...")
    f_values, p_values = f_classif(X_train_encoded, y_train)
    
    anova_df = pd.DataFrame({
        'Feature': original_candidates,
        'F_Score': f_values,
        'p_value': p_values
    }).sort_values(by='F_Score', ascending=False)
    
    print("\n=== ANOVA RANKING TABLE ===")
    print(anova_df.to_string(index=False))
    
    ranked_features = anova_df['Feature'].tolist()
    
    # 4. Forward Sequential Feature Selection (FSFS)
    # To run FSFS quickly, we take a 100k stratified subset of the training data
    print("\nRunning FSFS on a 100k stratified training subset...")
    subset_X, _, subset_y, _ = train_test_split(
        X_train, y_train, train_size=100000, random_state=42, stratify=y_train
    )
    
    # Split the subset 80/20 into train/validation for evaluation
    X_tr_fsfs, X_va_fsfs, y_tr_fsfs, y_va_fsfs = train_test_split(
        subset_X, subset_y, test_size=0.20, random_state=42, stratify=subset_y
    )
    
    # Helper to engineer features from original columns dynamically
    def engineer_subset(df_to_eng, active_cols):
        df_out = pd.DataFrame()
        for col in active_cols:
            if col == 'FL_DATE':
                dt = pd.to_datetime(df_to_eng['FL_DATE'], format='%Y-%m-%d')
                df_out['YEAR'] = dt.dt.year
                df_out['MONTH'] = dt.dt.month
                df_out['DAY'] = dt.dt.day
                df_out['DAY_OF_WEEK'] = dt.dt.dayofweek
            elif col == 'CRS_DEP_TIME':
                times = df_to_eng['CRS_DEP_TIME'].fillna(0).astype(int)
                df_out['CRS_DEP_HOUR'] = times // 100
                df_out['CRS_DEP_MINUTE'] = times % 100
            elif col == 'CRS_ARR_TIME':
                times = df_to_eng['CRS_ARR_TIME'].fillna(0).astype(int)
                df_out['CRS_ARR_HOUR'] = times // 100
                df_out['CRS_ARR_MINUTE'] = times % 100
            elif col in ['AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY']:
                df_out[col] = df_to_eng[col].astype(str).astype('category')
            else:
                df_out[col] = df_to_eng[col].fillna(df_to_eng[col].median())
        return df_out

    print("\nEvaluating sequential subsets:")
    results = []
    
    for i in range(2, 11):
        active_candidates = ranked_features[:i]
        
        # Engineer features
        X_tr_eng = engineer_subset(X_tr_fsfs, active_candidates)
        X_va_eng = engineer_subset(X_va_fsfs, active_candidates)
        
        # Train fast model
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            enable_categorical=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_eng, y_tr_fsfs)
        
        # Predict & Evaluate
        probs = model.predict_proba(X_va_eng)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_va_fsfs, preds)
        auc = roc_auc_score(y_va_fsfs, probs)
        
        results.append({
            'size': i,
            'features': ", ".join(active_candidates),
            'accuracy': acc,
            'auc': auc
        })
        print(f"  Subset size {i} -> Accuracy: {acc*100:.2f}%, ROC-AUC: {auc*100:.2f}%")
        
    print("\n=== FSFS RESULTS TABLE ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    final_10_features = ranked_features[:10]
    print("\n=== FINAL 10 SELECTED ORIGINAL COLUMNS ===")
    print(final_10_features)

if __name__ == "__main__":
    main()
