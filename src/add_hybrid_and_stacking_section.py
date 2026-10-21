import json
import os
import ast

def add_hybrid_and_stacking():
    primary_notebook = 'abcflightDelay_colab_updated.ipynb'
    with open(primary_notebook, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded {primary_notebook} with {len(nb['cells'])} cells.")

    # 1. Update Cell 54 (Section 21 Markdown)
    sec21_md = r"""---
# SECTION 21: PROPOSED HYBRID & ENSEMBLE PIPELINE (SMOTE-BALANCED MULTI-ARCHITECTURE FUSION)

### 1. Hybrid & Ensemble Strategy Overview
To transcend the individual algorithmic limitations of single classifiers, we construct two multi-architecture ensembles using the **5 existing SMOTE-balanced models as base learners**:
1. **Logistic Regression** (Linear decision surface, calibrated baseline)
2. **Decision Tree** (Non-linear orthogonal axis splits)
3. **Random Forest** (Bagging ensemble, variance reduction)
4. **LightGBM Classifier** (Leaf-wise gradient boosting, high categorical capability)
5. **XGBoost Classifier** (Depth-wise gradient boosting, second-order loss regularization)

### 2. Multi-Architecture Ensembles Implemented:
* **Hybrid / Soft-Voting Model:**
  Computes the probability-level consensus across all 5 SMOTE base learners:
  $$\hat{P}_{\text{SoftVoting}} = \frac{1}{5} \sum_{m=1}^{5} P_m(X_{\text{test}})$$
  This averages the predicted class posterior distributions, smoothing decision boundaries and reducing model variance across diverse inductive biases.
* **Stacking Model (Stacked Generalization, Wolpert 1992):**
  * **Level-0 Base Estimators:** The 5 SMOTE-balanced models generate out-of-fold probability predictions on the training data.
  * **Level-1 Meta-Learner:** A regularized Logistic Regression classifier trained on the Level-0 probability meta-features, learning the optimal non-linear blending weights.
  * Evaluated on the untouched test partition using the base models' test predictions as meta-features.

### 3. Strict Methodological Protocols:
1. **Identical Train / Test Split:** Models utilize the exact same SMOTE-balanced training split (`X_tr_smote_d`, `X_tr_smote_s`, `y_tr_bal`) and untouched test set (`X_te_w_d`, `X_te_w_s`, `y_test`).
2. **Untouched Test Partition:** Evaluated on the untouched empirical test partition ($N = 582,761$).
3. **Reproducibility:** Seeded with `random_state=42`."""

    nb['cells'][54]['source'] = [line + "\n" for line in sec21_md.strip().split("\n")]
    print("Updated Cell 54 (Section 21 Markdown).")

    # 2. Update Cell 55 (Section 21 Code)
    sec21_code = r"""# =============================================================================
# SECTION 21: PROPOSED HYBRID & ENSEMBLE PIPELINE (SMOTE-BALANCED MULTI-ARCHITECTURE FUSION)
# =============================================================================
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

print("=" * 75)
print("SECTION 21: HYBRID & ENSEMBLE MODELS (SMOTE-BALANCED 5-MODEL BASE LEARNERS)")
print("=" * 75)

# Verify base learner test probabilities from Section 19 (SMOTE models)
p_lr  = balanced_probs['Logistic Regression']
p_dt  = balanced_probs['Decision Tree']
p_rf  = balanced_probs['Random Forest']
p_lgb = balanced_probs['LightGBM']
p_xgb = balanced_probs['XGBoost']

# -----------------------------------------------------------------------------
# 1. HYBRID / SOFT VOTING MODEL (Consensus of all 5 SMOTE base learners)
# -----------------------------------------------------------------------------
print("Evaluating Hybrid / Soft-Voting Ensemble (5 Base Learners)...")
t0_voting = time.time()
p_soft_voting = (p_lr + p_dt + p_rf + p_lgb + p_xgb) / 5.0
preds_soft_voting = (p_soft_voting >= 0.50).astype(int)
t_voting = time.time() - t0_voting

acc_v  = accuracy_score(y_test, preds_soft_voting) * 100
bal_v  = balanced_accuracy_score(y_test, preds_soft_voting) * 100
prec_v = precision_score(y_test, preds_soft_voting, zero_division=0) * 100
rec_v  = recall_score(y_test, preds_soft_voting, zero_division=0) * 100
f1_v   = f1_score(y_test, preds_soft_voting, zero_division=0) * 100
roc_v  = roc_auc_score(y_test, p_soft_voting) * 100
pr_v   = average_precision_score(y_test, p_soft_voting) * 100
cm_v   = confusion_matrix(y_test, preds_soft_voting)
tn_v, fp_v, fn_v, tp_v = cm_v.ravel()

print(f"  -> Hybrid / Soft Voting Done in {t_voting:.2f}s | Acc: {acc_v:.2f}% | Recall: {rec_v:.2f}% | ROC-AUC: {roc_v:.2f}% (TP: {tp_v:,})")

# -----------------------------------------------------------------------------
# 2. STACKING MODEL (Level-1 Meta-Learner on 5 Base Learners)
# -----------------------------------------------------------------------------
print("\nTraining Stacking Model (Level-1 Meta-Learner via Out-Of-Fold Cross-Validation)...")
t0_stack = time.time()
n_splits = 3  # Stratified cross-validation on SMOTE-balanced training partition
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

meta_X_tr = np.zeros((len(y_tr_bal), 5))

# Generate out-of-fold meta-features across the 5 diverse base learners
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_smote_d, y_tr_bal)):
    print(f"  ├── Generating OOF meta-features fold {fold+1}/{n_splits}...")
    # Model 1: Logistic Regression
    clf_lr_f = LogisticRegression(max_iter=200, random_state=42)
    clf_lr_f.fit(X_tr_smote_d[tr_idx], y_tr_bal[tr_idx])
    meta_X_tr[val_idx, 0] = clf_lr_f.predict_proba(X_tr_smote_d[val_idx])[:, 1]
    
    # Model 2: Decision Tree
    clf_dt_f = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42)
    clf_dt_f.fit(X_tr_smote_d[tr_idx], y_tr_bal[tr_idx])
    meta_X_tr[val_idx, 1] = clf_dt_f.predict_proba(X_tr_smote_d[val_idx])[:, 1]
    
    # Model 3: Random Forest
    clf_rf_f = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    clf_rf_f.fit(X_tr_smote_d[tr_idx], y_tr_bal[tr_idx])
    meta_X_tr[val_idx, 2] = clf_rf_f.predict_proba(X_tr_smote_d[val_idx])[:, 1]
    
    # Model 4: LightGBM
    clf_lgb_f = lgb.LGBMClassifier(n_estimators=80, num_leaves=31, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1)
    clf_lgb_f.fit(X_tr_smote_s[tr_idx], y_tr_bal[tr_idx])
    meta_X_tr[val_idx, 3] = clf_lgb_f.predict_proba(X_tr_smote_s[val_idx])[:, 1]
    
    # Model 5: XGBoost
    clf_xgb_f = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, eval_metric='logloss')
    clf_xgb_f.fit(X_tr_smote_s[tr_idx], y_tr_bal[tr_idx])
    meta_X_tr[val_idx, 4] = clf_xgb_f.predict_proba(X_tr_smote_s[val_idx])[:, 1]

# Assemble Level-1 Meta-Test Matrix from the already-trained full base models
meta_X_te = np.column_stack([p_lr, p_dt, p_rf, p_lgb, p_xgb])

# Train Level-1 Meta-Learner (Logistic Regression)
print("  └── Fitting Level-1 Logistic Regression Meta-Learner...")
meta_learner = LogisticRegression(max_iter=500, random_state=42)
meta_learner.fit(meta_X_tr, y_tr_bal)

# Evaluate on untouched test partition
p_stacking = meta_learner.predict_proba(meta_X_te)[:, 1]
preds_stacking = (p_stacking >= 0.50).astype(int)
t_stack = time.time() - t0_stack

acc_s  = accuracy_score(y_test, preds_stacking) * 100
bal_s  = balanced_accuracy_score(y_test, preds_stacking) * 100
prec_s = precision_score(y_test, preds_stacking, zero_division=0) * 100
rec_s  = recall_score(y_test, preds_stacking, zero_division=0) * 100
f1_s   = f1_score(y_test, preds_stacking, zero_division=0) * 100
roc_s  = roc_auc_score(y_test, p_stacking) * 100
pr_s   = average_precision_score(y_test, p_stacking) * 100
cm_s   = confusion_matrix(y_test, preds_stacking)
tn_s, fp_s, fn_s, tp_s = cm_s.ravel()

print(f"  -> Stacking Model Done in {t_stack:.2f}s | Acc: {acc_s:.2f}% | Recall: {rec_s:.2f}% | ROC-AUC: {roc_s:.2f}% (TP: {tp_s:,})")

# Store predictions
balanced_probs['Hybrid / Soft Voting'] = p_soft_voting
balanced_probs['Stacking Model'] = p_stacking

# -----------------------------------------------------------------------------
# 3. BUILD HYBRID & STACKING RESULTS DATAFRAME
# -----------------------------------------------------------------------------
hybrid_results_list = [
    {
        'Model': 'Hybrid / Soft Voting (5 Base Models)',
        'Balancing Method': 'SMOTE Oversampling',
        'Accuracy (%)': round(acc_v, 2),
        'Balanced Acc (%)': round(bal_v, 2),
        'Precision (%)': round(prec_v, 2),
        'Recall (%)': round(rec_v, 2),
        'F1 Score (%)': round(f1_v, 2),
        'ROC-AUC (%)': round(roc_v, 2),
        'PR-AUC (%)': round(pr_v, 2),
        'True Positives (TP)': int(tp_v),
        'False Positives (FP)': int(fp_v),
        'False Negatives (FN)': int(fn_v),
        'True Negatives (TN)': int(tn_v),
        'Operational Assessment': f'Caught {tp_v:,} delays (Recall: {rec_v:.2f}%)'
    },
    {
        'Model': 'Stacking Model (5 Base Models + LR Meta)',
        'Balancing Method': 'SMOTE Oversampling',
        'Accuracy (%)': round(acc_s, 2),
        'Balanced Acc (%)': round(bal_s, 2),
        'Precision (%)': round(prec_s, 2),
        'Recall (%)': round(rec_s, 2),
        'F1 Score (%)': round(f1_s, 2),
        'ROC-AUC (%)': round(roc_s, 2),
        'PR-AUC (%)': round(pr_s, 2),
        'True Positives (TP)': int(tp_s),
        'False Positives (FP)': int(fp_s),
        'False Negatives (FN)': int(fn_s),
        'True Negatives (TN)': int(tn_s),
        'Operational Assessment': f'Caught {tp_s:,} delays (Recall: {rec_s:.2f}%)'
    }
]

hybrid_results_df = pd.DataFrame(hybrid_results_list)
os.makedirs('outputs', exist_ok=True)
hybrid_results_df.to_csv('outputs/smote_hybrid_stacking_results.csv', index=False)
hybrid_results_df.to_csv('outputs/updated_hybrid_results.csv', index=False)

print("\n=== HYBRID & STACKING MODELS EVALUATION ON UNTOUCHED TEST SET ===")
display(hybrid_results_df)

# -----------------------------------------------------------------------------
# 4. FINAL COMPREHENSIVE 4-TIER COMPARISON TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON TABLE: OLD vs SMOTE vs HYBRID vs STACKING")
print("=" * 75)

comparison_all_rows = []

# 1. Old / Original Results (Unbalanced Baseline)
for _, r in unbalanced_results_df.iterrows():
    if 'Majority' in r['Model']:
        continue
    comparison_all_rows.append({
        'Category / Label': 'Old/Original Results',
        'Model Architecture': r['Model'],
        'Accuracy (%)': r['Accuracy (%)'],
        'Precision (%)': r['Precision (%)'],
        'Recall (%)': r['Recall (%)'],
        'F1-Score (%)': r['F1 Score (%)'],
        'ROC-AUC (%)': r['ROC-AUC (%)']
    })

# 2. SMOTE / Fixed Results (Balanced Baseline)
for _, r in balanced_results_df.iterrows():
    comparison_all_rows.append({
        'Category / Label': 'SMOTE/Fixed Results',
        'Model Architecture': r['Model'],
        'Accuracy (%)': r['Accuracy (%)'],
        'Precision (%)': r['Precision (%)'],
        'Recall (%)': r['Recall (%)'],
        'F1-Score (%)': r['F1 Score (%)'],
        'ROC-AUC (%)': r['ROC-AUC (%)']
    })

# 3. Hybrid Results
comparison_all_rows.append({
    'Category / Label': 'Hybrid Results',
    'Model Architecture': 'Hybrid / Soft Voting (5 Base Models)',
    'Accuracy (%)': round(acc_v, 2),
    'Precision (%)': round(prec_v, 2),
    'Recall (%)': round(rec_v, 2),
    'F1-Score (%)': round(f1_v, 2),
    'ROC-AUC (%)': round(roc_v, 2)
})

# 4. Stacking Results
comparison_all_rows.append({
    'Category / Label': 'Stacking Results',
    'Model Architecture': 'Stacking Model (5 Base Models + LR Meta)',
    'Accuracy (%)': round(acc_s, 2),
    'Precision (%)': round(prec_s, 2),
    'Recall (%)': round(rec_s, 2),
    'F1-Score (%)': round(f1_s, 2),
    'ROC-AUC (%)': round(roc_s, 2)
})

final_4tier_comparison_df = pd.DataFrame(comparison_all_rows)
final_4tier_comparison_df.to_csv('outputs/final_4tier_model_comparison.csv', index=False)
display(final_4tier_comparison_df)
print("\n[Artifact Saved] -> outputs/final_4tier_model_comparison.csv")"""

    nb['cells'][55]['source'] = [line + "\n" for line in sec21_code.strip().split("\n")]
    print("Updated Cell 55 (Section 21 Code).")

    # 3. Update Cell 56 (Section 22 Markdown)
    sec22_md = r"""---
# SECTION 22: HYBRID & ENSEMBLE ARCHITECTURAL ANALYSIS

### 1. Rationale for Multi-Architecture Fusion
In ensemble theory, combining classifiers with diverse inductive biases reduces both algorithmic variance and decision-boundary bias:
* **Logistic Regression:** Regularized linear hyper-plane, provides well-calibrated baseline probabilities.
* **Decision Tree:** Non-parametric axis-aligned partitioner capturing hard piecewise conditions.
* **Random Forest:** Bagging ensemble that dramatically reduces variance across feature subsamples.
* **LightGBM:** Fast leaf-wise (best-first) gradient booster using Exclusive Feature Bundling (EFB).
* **XGBoost:** Regularized depth-wise gradient booster with second-order Taylor approximations.

### 2. Architectural Comparison: Soft Voting vs. Stacking
* **Soft-Voting Ensemble:** Computes equal-weighted class posterior probabilities across all 5 models. Because each model excels in different feature subspaces, soft voting eliminates outlier errors from individual models.
* **Two-Stage Stacking Classifier:** Rather than assuming equal weights, Stacking trains a meta-learner (Logistic Regression) on out-of-fold probability vectors. The meta-learner learns to assign higher weights to high-discriminative gradient boosted learners while retaining stabilizing regularization from bagging and linear models."""

    nb['cells'][56]['source'] = [line + "\n" for line in sec22_md.strip().split("\n")]
    print("Updated Cell 56 (Section 22 Markdown).")

    # 4. Update Cell 69 (Section 29 Markdown)
    sec29_md = r"""---
# SECTION 29: FINAL MODEL COMPARISON

### Comprehensive Multi-Tier Model Audit Table
We aggregate all evaluated architectures into a unified comparison table clearly distinguishing:
1. **Old/Original Results (Section 18):** Demonstrating the natural accuracy baseline under extreme class imbalance.
2. **SMOTE/Fixed Results (Section 19):** Demonstrating the operational delay recall gains from training data SMOTE oversampling.
3. **Hybrid Results (Section 21):** Combining all 5 base learners via soft voting probability consensus.
4. **Stacking Results (Section 21):** Two-stage stacking meta-learning on Level-0 out-of-fold predictions."""

    nb['cells'][69]['source'] = [line + "\n" for line in sec29_md.strip().split("\n")]
    print("Updated Cell 69 (Section 29 Markdown).")

    # 5. Update Cell 70 (Section 29 Code)
    sec70_code = r"""# =============================================================================
# SECTION 29: FINAL MODEL COMPARISON TABLE & VISUALIZATION
# =============================================================================
print("=== FINAL 4-TIER MODEL COMPARISON BENCHMARK TABLE ===")
display(final_4tier_comparison_df)

# Visual Comparison: ROC-AUC Across Tiers
plt.figure(figsize=(14, 5))
plt.bar(final_4tier_comparison_df['Model Architecture'], final_4tier_comparison_df['ROC-AUC (%)'], color='#1565c0')
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.ylabel('ROC-AUC (%)', fontsize=11, fontweight='bold')
plt.title('Discrimination Ability (ROC-AUC %) Across All 4 Tiers', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
os.makedirs('outputs/figures', exist_ok=True)
plt.savefig('outputs/figures/final_model_comparison_bar.png', dpi=300)
plt.show()
print("[Figure Saved] -> outputs/figures/final_model_comparison_bar.png")"""

    nb['cells'][70]['source'] = [line + "\n" for line in sec70_code.strip().split("\n")]
    print("Updated Cell 70 (Section 29 Code).")

    # Validate syntax of updated code cells
    for c_idx in [55, 70]:
        code_str = "".join(nb['cells'][c_idx]['source'])
        try:
            ast.parse(code_str)
            print(f"[VALID SYNTAX] Cell {c_idx}")
        except SyntaxError as e:
            print(f"[SYNTAX ERROR] Cell {c_idx}: {e}")
            raise

    # Save to primary notebook
    with open(primary_notebook, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved {primary_notebook}")

    # Mirror to all other notebooks
    mirrors = [
        'flightDelay.ipynb',
        'flightDelay_colab_updated.ipynb',
        '123flights.ipynb'
    ]
    for mirror in mirrors:
        with open(mirror, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Mirrored to {mirror}")

if __name__ == '__main__':
    add_hybrid_and_stacking()
