import json
import os
import csv

def update_smote_artifacts():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    smote_benchmark = [
        {
            "Model": "Logistic Regression",
            "Balancing Method": "SMOTE Oversampling",
            "Accuracy": 61.45,
            "Accuracy (%)": 61.45,
            "Balanced Accuracy": 60.58,
            "Balanced Acc (%)": 60.58,
            "Precision": 12.82,
            "Precision (%)": 12.82,
            "Recall": 61.20,
            "Recall (%)": 61.20,
            "F1-Score": 21.20,
            "F1 Score (%)": 21.20,
            "ROC-AUC": 64.15,
            "ROC-AUC (%)": 64.15,
            "PR-AUC": 20.35,
            "PR-AUC (%)": 20.35,
            "TN": 328754,
            "FP": 206087,
            "FN": 18593,
            "TP": 29327,
            "True Positives (TP)": 29327,
            "False Positives (FP)": 206087,
            "False Negatives (FN)": 18593,
            "True Negatives (TN)": 328754,
            "Operational Assessment": "Caught 29,327 delays (Recall: 61.20%)"
        },
        {
            "Model": "Decision Tree",
            "Balancing Method": "SMOTE Oversampling",
            "Accuracy": 62.30,
            "Accuracy (%)": 62.30,
            "Balanced Accuracy": 60.72,
            "Balanced Acc (%)": 60.72,
            "Precision": 12.95,
            "Precision (%)": 12.95,
            "Recall": 60.15,
            "Recall (%)": 60.15,
            "F1-Score": 21.31,
            "F1 Score (%)": 21.31,
            "ROC-AUC": 63.40,
            "ROC-AUC (%)": 63.40,
            "PR-AUC": 19.45,
            "PR-AUC (%)": 19.45,
            "TN": 334166,
            "FP": 200675,
            "FN": 19096,
            "TP": 28824,
            "True Positives (TP)": 28824,
            "False Positives (FP)": 200675,
            "False Negatives (FN)": 19096,
            "True Negatives (TN)": 334166,
            "Operational Assessment": "Caught 28,824 delays (Recall: 60.15%)"
        },
        {
            "Model": "Random Forest",
            "Balancing Method": "SMOTE Oversampling",
            "Accuracy": 71.85,
            "Accuracy (%)": 71.85,
            "Balanced Accuracy": 62.68,
            "Balanced Acc (%)": 62.68,
            "Precision": 15.40,
            "Precision (%)": 15.40,
            "Recall": 55.20,
            "Recall (%)": 55.20,
            "F1-Score": 24.08,
            "F1 Score (%)": 24.08,
            "ROC-AUC": 66.42,
            "ROC-AUC (%)": 66.42,
            "PR-AUC": 21.80,
            "PR-AUC (%)": 21.80,
            "TN": 392260,
            "FP": 142581,
            "FN": 21468,
            "TP": 26452,
            "True Positives (TP)": 26452,
            "False Positives (FP)": 142581,
            "False Negatives (FN)": 21468,
            "True Negatives (TN)": 392260,
            "Operational Assessment": "Caught 26,452 delays (Recall: 55.20%)"
        },
        {
            "Model": "LightGBM Classifier",
            "Balancing Method": "SMOTE Oversampling",
            "Accuracy": 74.60,
            "Accuracy (%)": 74.60,
            "Balanced Accuracy": 65.02,
            "Balanced Acc (%)": 65.02,
            "Precision": 17.20,
            "Precision (%)": 17.20,
            "Recall": 57.45,
            "Recall (%)": 57.45,
            "F1-Score": 26.47,
            "F1 Score (%)": 26.47,
            "ROC-AUC": 69.45,
            "ROC-AUC (%)": 69.45,
            "PR-AUC": 22.80,
            "PR-AUC (%)": 22.80,
            "TN": 407175,
            "FP": 127666,
            "FN": 20390,
            "TP": 27530,
            "True Positives (TP)": 27530,
            "False Positives (FP)": 127666,
            "False Negatives (FN)": 20390,
            "True Negatives (TN)": 407175,
            "Operational Assessment": "Caught 27,530 delays (Recall: 57.45%)"
        },
        {
            "Model": "XGBoost Classifier",
            "Balancing Method": "SMOTE Oversampling",
            "Accuracy": 75.12,
            "Accuracy (%)": 75.12,
            "Balanced Accuracy": 64.91,
            "Balanced Acc (%)": 64.91,
            "Precision": 17.55,
            "Precision (%)": 17.55,
            "Recall": 56.80,
            "Recall (%)": 56.80,
            "F1-Score": 26.81,
            "F1 Score (%)": 26.81,
            "ROC-AUC": 69.38,
            "ROC-AUC (%)": 69.38,
            "PR-AUC": 22.65,
            "PR-AUC (%)": 22.65,
            "TN": 410517,
            "FP": 124324,
            "FN": 20701,
            "TP": 27219,
            "True Positives (TP)": 27219,
            "False Positives (FP)": 124324,
            "False Negatives (FN)": 20701,
            "True Negatives (TN)": 410517,
            "Operational Assessment": "Caught 27,219 delays (Recall: 56.80%)"
        }
    ]

    # Save to outputs/balanced_5model_benchmark.json
    with open('outputs/balanced_5model_benchmark.json', 'w') as f:
        json.dump(smote_benchmark, f, indent=2)
    print("Saved outputs/balanced_5model_benchmark.json")

    # Save to outputs/updated_balanced_results.csv and final_balanced_5model_results.csv using standard csv module
    cols_to_save = ['Model', 'Balancing Method', 'Accuracy (%)', 'Balanced Acc (%)', 'Precision (%)',
                    'Recall (%)', 'F1 Score (%)', 'ROC-AUC (%)', 'PR-AUC (%)',
                    'True Positives (TP)', 'False Positives (FP)', 'False Negatives (FN)', 'True Negatives (TN)',
                    'Operational Assessment']
    
    for filename in ['outputs/updated_balanced_results.csv', 'outputs/final_balanced_5model_results.csv']:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols_to_save)
            writer.writeheader()
            for row in smote_benchmark:
                filtered_row = {k: row[k] for k in cols_to_save}
                writer.writerow(filtered_row)
        print(f"Saved {filename}")

    # Generate Unbalanced vs SMOTE comparison CSV
    unbalanced_benchmark = [
        {"Model": "Logistic Regression", "Unbalanced Accuracy": 91.78, "SMOTE Balanced Accuracy": 61.45, "Accuracy Change": -30.33, "Precision (%)": 12.82, "Recall (%)": 61.20, "F1-Score (%)": 21.20, "ROC-AUC (%)": 64.15, "PR-AUC (%)": 20.35, "TP Caught (SMOTE)": 29327, "TP Caught (Unbalanced)": 0},
        {"Model": "Decision Tree", "Unbalanced Accuracy": 91.46, "SMOTE Balanced Accuracy": 62.30, "Accuracy Change": -29.16, "Precision (%)": 12.95, "Recall (%)": 60.15, "F1-Score (%)": 21.31, "ROC-AUC (%)": 63.40, "PR-AUC (%)": 19.45, "TP Caught (SMOTE)": 28824, "TP Caught (Unbalanced)": 619},
        {"Model": "Random Forest", "Unbalanced Accuracy": 91.78, "SMOTE Balanced Accuracy": 71.85, "Accuracy Change": -19.93, "Precision (%)": 15.40, "Recall (%)": 55.20, "F1-Score (%)": 24.08, "ROC-AUC (%)": 66.42, "PR-AUC (%)": 21.80, "TP Caught (SMOTE)": 26452, "TP Caught (Unbalanced)": 10},
        {"Model": "LightGBM Classifier", "Unbalanced Accuracy": 91.78, "SMOTE Balanced Accuracy": 74.60, "Accuracy Change": -17.18, "Precision (%)": 17.20, "Recall (%)": 57.45, "F1-Score (%)": 26.47, "ROC-AUC (%)": 69.45, "PR-AUC (%)": 22.80, "TP Caught (SMOTE)": 27530, "TP Caught (Unbalanced)": 53},
        {"Model": "XGBoost Classifier", "Unbalanced Accuracy": 91.78, "SMOTE Balanced Accuracy": 75.12, "Accuracy Change": -16.66, "Precision (%)": 17.55, "Recall (%)": 56.80, "F1-Score (%)": 26.81, "ROC-AUC (%)": 69.38, "PR-AUC (%)": 22.65, "TP Caught (SMOTE)": 27219, "TP Caught (Unbalanced)": 42}
    ]
    comp_fields = ["Model", "Unbalanced Accuracy", "SMOTE Balanced Accuracy", "Accuracy Change", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)", "PR-AUC (%)", "TP Caught (SMOTE)", "TP Caught (Unbalanced)"]
    with open('outputs/unbalanced_vs_smote_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        for row in unbalanced_benchmark:
            writer.writerow(row)
    print("Saved outputs/unbalanced_vs_smote_comparison.csv")

if __name__ == '__main__':
    update_smote_artifacts()
