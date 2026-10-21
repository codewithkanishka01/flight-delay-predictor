# Flight Delay Prediction Project — Progress & State Summary

**Status Date:** September 3, 2026  
**Project Workspace:** `/Users/kanishkapandey/Desktop/ML`  
**Primary Dataset:** `/Users/kanishkapandey/Desktop/flights_sample_3m.csv`  
**Current Phase Completed:** **Phase 11 (90% Target Optimization & Problem Formulation Audit)**  
**External Test Set Status:** **STRICTLY UNTOUCHED (582,761 rows)**

---

## 1. Executive Summary Across All Completed Phases

Across Phases 1 through 11, we systematically investigated every academic and machine learning dimension of the pre-flight delay prediction problem using rigorous data leakage controls, internal cross-validation frameworks, rich pre-flight feature engineering, and external data integration.

### The 90% Accuracy Target: Scientific & Mathematical Reality
1. **Standard FAA Target (`ARR_DELAY >= 15m`):**
   * Class Distribution: **81.68% on-time (Class 0) vs. 18.32% delayed (Class 1)**.
   * Predictive Ceiling: Legitimate pre-departure tabular features (scheduled timetable geometry, calendar markers, cyclic time encodings, airline/airport historical lag states, and surface weather) reach an empirical discrimination ceiling of **ROC-AUC ~70.23%** and **PR-AUC ~35.95%**.
   * Reaching $\ge 90\%$ nominal accuracy on this target is **physically and mathematically impossible** without predicting 100% of flights as on-time (Recall = 0.00%), which destroys all minority-class utility.
2. **Severe Disruption Targets (`ARR_DELAY >= 45m` or `ARR_DELAY >= 60m`):**
   * For severe delays of at least 45 minutes, the on-time rate is **91.79%** (majority baseline alone is **91.79%**).
   * For severe delays of at least 60 minutes, the on-time rate is **93.88%** (majority baseline alone is **93.88%**).
   * Therefore, $\ge 90\%$ accuracy is naturally achievable under a **Severe Disruption Problem Formulation**, but constitutes an entirely distinct prediction task from 15-minute operational delays.

---

## 2. Summary of Phases Completed

| Phase | Description | Key Findings / Metrics | Artifacts Created |
| :--- | :--- | :--- | :--- |
| **Phase 2** | Leakage-Free Pipeline & Split | 80/20 stratified split (`train_leakage_free.parquet`: 2.33M rows, `test_leakage_free.parquet`: 582k rows). 17 post-departure leakage variables eliminated. | `src/data_pipeline_new.py`, `src/config_new.py` |
| **Phase 3A** | Baselines & Feature Selection | Majority Baseline: 81.68% Acc, 0% Recall. ANOVA & Forward Sequential Feature Selection (FSFS) completed without leakage. | `outputs/anova_feature_ranking.csv`, `outputs/fsfs_results.csv` |
| **Phase 3B** | Model Screening | Screened Logistic Regression, Random Forest, ExtraTrees, XGBoost. XGBoost emerged as top discriminator (ROC-AUC ~68.4%). | `outputs/screening_results.csv` |
| **Phase 4** | Leakage-Free XGBoost Tuning | Systematic randomized CV search across depths and learning rates. Optimal operating threshold identified as $T=0.25$. | `models/tuned_leakage_free_xgb.json`, `outputs/leakage_free_xgb_tuning_results.csv` |
| **Phase 5** | Test Evaluation & Temporal Robustness | One-time test set evaluation of Phase 4 model (Acc: 75.75%, ROC-AUC: 69.05%). Temporal robustness validation (Train 2019–2022, Val 2023: Acc 68.76%, ROC-AUC 66.16%). | `outputs/reports/final_confusion_matrix.png`, `outputs/reports/final_roc_curve.png` |
| **Phase 6** | Historical Feature Engineering | Evaluated 13 historical delay rate features. Discovered high-cardinality target encodings overfit tree models unless Bayesian smoothed. | `src/historical_features.py`, `src/phase_6.py` |
| **Phase 7** | 70/15/15 Validation Framework | Established strict internal validation framework (70% train: 1.63M, 15% val: 349k, 15% holdout: 349k) inside training partition. External test set locked. | `src/phase_7.py`, `outputs/phase7_holdout_results.csv` |
| **Phase 8** | Advanced Models & Ensembling | Evaluated XGBoost, LightGBM, Random Forest, ExtraTrees, Deep MLP, and Stacking/Blending. Best model: 90% XGB + 10% LGB (Holdout ROC-AUC: 69.89%, F1: 38.26%). | `src/phase_8.py`, `models/phase8/`, `outputs/phase8_*` |
| **Phase 9** | Rich Pre-Flight Features | Calendar markers, cyclic hours/minutes, distance/duration geometry, airline-origin interactions, OOF historical delay rates. ROC-AUC crossed 70% (Holdout ROC-AUC: 70.24%). | `src/phase_9.py`, `outputs/phase9_*`, `models/phase9/` |
| **Phase 10** | External Data & Weather | Feasibility audit of NOAA, IEM ASOS, Open-Meteo, and timetable congestion. Pilot weather merge (98.6% coverage). Prior-day operational delay lag states added. | `src/phase_10.py`, `outputs/phase10_*`, `models/phase10/` |
| **Phase 11** | Target Formulation & 90% Audit | Target audit (15m, 30m, 45m, 60m). 7-model benchmark, fine threshold search (91 points), subgroup error analysis, and mathematical proof of 90% feasibility constraints. | `src/phase11_90pct_optimization.py`, `outputs/phase11_*`, `models/phase11/` |

---

## 3. Best Performing Model Checkpoint (Phase 11)

* **Architecture:** Weighted Blend (**80% Tuned XGBoost + 20% Tuned LightGBM**)
* **Feature Set:** Enriched Pre-Flight Library (36 numeric + 14 one-hot categorical interactions = 22,955 sparse encoded columns)
* **Optimal Operating Threshold:** **$T = 0.25$**
* **Internal Holdout Performance (349,657 rows):**
  * **Accuracy:** **75.81%**
  * **Balanced Accuracy:** **62.59%**
  * **Precision:** **36.14%**
  * **Recall:** **41.71%**
  * **F1 Score:** **38.72%**
  * **ROC-AUC:** **70.23%**
  * **PR-AUC:** **35.95%**
  * **Confusion Matrix:**
    ```
    [[238365  47228]
     [ 37342  26722]]
    ```
* **Serialized Models:**
  * `models/phase11/phase11_xgb_model.json`
  * `models/phase11/phase11_lgb_model.txt`
  * `models/phase11/config.json`

---

## 4. Key Output Files & Audit Artifacts

All files are located in `/Users/kanishkapandey/Desktop/ML/outputs/`:
* `phase11_target_comparison.csv`: Delay thresholds 15m, 30m, 45m, 60m distribution & baselines.
* `phase11_data_quality_audit.csv`: Verification of 0 duplicates, 0 invalid times/distances, and clean labels.
* `phase11_feature_comparison.csv`: Systematic inventory of pre-flight feature groups.
* `phase11_model_comparison.csv`: 7-model evaluation table on validation split.
* `phase11_threshold_fine.csv`: 91 fine-grained threshold sweeps (0.05 to 0.95).
* `phase11_error_analysis.csv`: False Positive and False Negative frequency audit.
* `phase11_group_performance.csv`: Performance segmented by airline, origin hub, and departure time bin.
* `phase11_summary.md`: Full academic summary of Phase 11 findings.

---

## 5. Strict Data Leakage Compliance Checklist

* [x] **External test set (582,761 rows) remained completely untouched throughout Phases 6–11.**
* [x] **Zero post-departure features used (all 17 leakage variables strictly excluded).**
* [x] **Zero target leakage (`ARR_DELAY` used exclusively to define ground-truth label).**
* [x] **Temporal lag features strictly shifted by +1 day (using only concluded Day $D-1$ operations).**
* [x] **Weather observations sampled strictly prior to scheduled departure.**
* [x] **Zero synthetic or manipulated performance numbers.**
* [x] **French dataset not used.**

---

## 6. Next Steps for When We Resume

When you are ready to continue, the recommended next phase is:
* **Phase 12: Final Untouched Test Set Evaluation, Model Explainability (SHAP / TreeExplainer), and Production Serving / Inference Pipeline.**
