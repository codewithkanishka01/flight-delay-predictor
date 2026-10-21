import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from config import (
    TRAIN_DATA_PATH,
    TARGET_COL,
    RANDOM_STATE,
    TUNING_ROWS,
    MODEL_DIR
)

def main():
    print(f"Loading training data for tuning from {TRAIN_DATA_PATH}...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    
    # Stratified sampling of 300,000 rows
    print(f"Sampling {TUNING_ROWS} rows for hyperparameter tuning...")
    tune_df, _ = train_test_split(
        train_df,
        train_size=TUNING_ROWS,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COL]
    )
    
    X_tune = tune_df.drop(columns=[TARGET_COL])
    y_tune = tune_df[TARGET_COL]
    
    # 80% tuning train, 20% validation
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tune, y_tune, test_size=0.20, random_state=RANDOM_STATE, stratify=y_tune
    )
    print(f"Tuning set shapes: Train {X_tr.shape}, Validation {X_va.shape}")
    
    # Parameter grid search space
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1.0, 2.0, 4.46]
    }
    
    # Initialize XGBoost model
    xgb_clf = xgb.XGBClassifier(
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    print("Initializing RandomizedSearchCV...")
    # Randomized Search with 3-fold cross validation
    rs = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting randomized search on subset...")
    rs.fit(X_tr, y_tr)
    
    print("\nSearch complete!")
    print("Best Parameters:")
    for param, val in rs.best_params_.items():
        print(f"  {param}: {val}")
    print(f"Best CV Accuracy Score: {rs.best_score_:.4f}")
    
    # Save best parameters to JSON
    params_path = os.path.join(MODEL_DIR, "best_params.json")
    with open(params_path, 'w') as f:
        json.dump(rs.best_params_, f, indent=4)
        
    print(f"Saved optimal parameters to {params_path}")

if __name__ == "__main__":
    main()
