import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def engineer_features(df_in, final_10_cols):
    df_out = pd.DataFrame()
    for col in final_10_cols:
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
        elif col in ['AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY']:
            df_out[col] = df_in[col].astype(str).astype('category')
        else:
            df_out[col] = df_in[col].fillna(df_in[col].median())
            
    # Re-order columns for consistency
    expected_order = [
        'AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY',
        'DISTANCE', 'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK',
        'CRS_DEP_HOUR', 'CRS_DEP_MINUTE', 'CRS_ARR_HOUR', 'CRS_ARR_MINUTE'
    ]
    return df_out[expected_order]

def main():
    data_path = "/Users/kanishkapandey/Desktop/flights_sample_3m.csv"
    data_dir = "/Users/kanishkapandey/Desktop/ML/data"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Target Ingestion & Cleaning
    print("Cleaning dataset...")
    df = df[df['CANCELLED'] == 0]
    df = df[df['DIVERTED'] == 0]
    df = df.dropna(subset=['ARR_DELAY'])
    
    df['DELAYED'] = (df['ARR_DELAY'] >= 15).astype(int)
    
    final_10_cols = [
        'FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'AIRLINE', 'AIRLINE_CODE',
        'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 'DISTANCE'
    ]
    
    X = df[final_10_cols]
    y = df['DELAYED']
    
    # 80/20 Stratified Split
    print("Performing stratified split...")
    X_train_cand, X_test_cand, y_train_10, y_test_10 = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Feature Engineering
    print("Engineering features from locked 10 original columns...")
    X_train_10 = engineer_features(X_train_cand, final_10_cols)
    X_test_10 = engineer_features(X_test_cand, final_10_cols)
    
    # Save processed files
    print("Saving processed datasets to Parquet files...")
    X_train_10.to_parquet(os.path.join(data_dir, "X_train_10.parquet"))
    X_test_10.to_parquet(os.path.join(data_dir, "X_test_10.parquet"))
    y_train_10.to_frame().to_parquet(os.path.join(data_dir, "y_train_10.parquet"))
    y_test_10.to_frame().to_parquet(os.path.join(data_dir, "y_test_10.parquet"))
    
    # Verification and Printing
    print("\n=== DATASET VERIFICATION RESULTS ===")
    print(f"X_train_10 shape: {X_train_10.shape}")
    print(f"X_test_10 shape:  {X_test_10.shape}")
    print(f"y_train_10 shape: {y_train_10.shape}")
    print(f"y_test_10 shape:  {y_test_10.shape}")
    
    col_names = X_train_10.columns.tolist()
    print(f"\nAll Features ({len(col_names)}): {col_names}")
    
    categorical_cols = X_train_10.select_dtypes(include=['category']).columns.tolist()
    numerical_cols = X_train_10.select_dtypes(exclude=['category']).columns.tolist()
    
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    
    train_dist = y_train_10.value_counts(normalize=True).to_dict()
    test_dist = y_test_10.value_counts(normalize=True).to_dict()
    
    print("\nTarget Distribution:")
    print(f"  Train: 0 (On-Time) = {train_dist.get(0, 0)*100:.2f}%, 1 (Delayed) = {train_dist.get(1, 0)*100:.2f}%")
    print(f"  Test:  0 (On-Time) = {test_dist.get(0, 0)*100:.2f}%, 1 (Delayed) = {test_dist.get(1, 0)*100:.2f}%")

if __name__ == "__main__":
    main()
