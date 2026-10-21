import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from config import (
    TEST_DATA_PATH,
    TARGET_COL,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE
)

def main():
    # Load trained model
    model_path = os.path.join(MODEL_DIR, "final_xgb_model.json")
    if not os.path.exists(model_path):
        print(f"Error: final model not found at {model_path}. Run final evaluation first.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Final model loaded successfully.")
    
    # Load test set
    print(f"Loading test set for explanation from {TEST_DATA_PATH}...")
    test_df = pd.read_parquet(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COL])
    
    # 1. Plot XGBoost feature importances
    print("Generating feature importance plot...")
    feat_imp = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis")
    plt.title("XGBoost Final Feature Importances (Gain)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    imp_plot_path = os.path.join(OUTPUT_DIR, "feature_importances.png")
    plt.savefig(imp_plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {imp_plot_path}")
    
    # 2. Run SHAP analysis
    print("Running SHAP analysis on a sample of 200 rows...")
    try:
        # Sample a small subset of test features for explanation (SHAP is computationally intensive)
        X_sample = X_test.sample(n=200, random_state=RANDOM_STATE)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        
        # Generate and save SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Feature Attribution Summary")
        plt.tight_layout()
        
        shap_plot_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(shap_plot_path, dpi=300)
        plt.close()
        print(f"SHAP summary plot saved to {shap_plot_path}")
    except Exception as e:
        print(f"Warning: SHAP analysis could not be completed due to compatibility of SHAP explainer with XGBoost categorical features. Error: {e}")
        
    print("Explainability analysis complete.")


if __name__ == "__main__":
    main()
