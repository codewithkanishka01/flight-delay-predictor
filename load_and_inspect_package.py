#!/usr/bin/env python3
"""
Quick Inspection Script for the Consolidated Flight Delay Package (.pkl)
Usage: python load_and_inspect_package.py
"""

import pickle
import os

PKL_PATH = "flight_delay_project_package.pkl"

if not os.path.exists(PKL_PATH):
    PKL_PATH = "flight_delay_all_info.pkl"

print("=" * 80)
print(f"LOADING EXPORT PACKAGE: {PKL_PATH}")
print("=" * 80)

with open(PKL_PATH, "rb") as f:
    pkg = pickle.load(f)

meta = pkg["metadata"]
print(f"Project Title:    {meta['project_title']}")
print(f"Target Variable:  {meta['target_variable']}")
print(f"Dataset Scale:    {meta['total_cleaned_records']:,} flights (BTS National Database)")
print(f"Class Balance:    {meta['class_0_on_time_pct']}% On-Time (<45m) vs {meta['class_1_delayed_pct']}% Severe Delay (>=45m)")
print(f"Version / Date:   {meta['version']} ({meta['export_timestamp']})")

print("\n" + "=" * 80)
print("FEATURE SELECTION: VARIANCE INFLATION FACTOR (VIF <= 5.0) vs ANOVA")
print("=" * 80)
vif_info = pkg["feature_architecture"].get("vif_feature_selection", {})
if vif_info:
    print(f"Applied Threshold: VIF <= {vif_info.get('threshold_applied', 5.0)}")
    print(f"Rationale:         {vif_info.get('threshold_explanation', '')}")
    print("\nFeatures Removed due to Collinearity:")
    for rem in vif_info.get("features_removed_by_vif", []):
        print(f"  [-] {rem['feature']:<20} (Initial VIF: {rem['vif_at_removal']:>5.2f}) -> {rem['reason']}")
    print("\nFinal Selected Numerical Features (All VIF <= 5.0):")
    for feat in vif_info.get("final_selected_features", []):
        print(f"  [+] {feat['feature']:<20} (Final VIF: {feat['final_vif']:>5.2f})")

print("\n" + "-" * 80)
print("MODEL PERFORMANCE: ANOVA SELECTION (Top-30) vs VIF SELECTION (VIF <= 5.0)")
print("-" * 80)
print(f"{'Model':<25} | {'ANOVA Acc':<10} | {'VIF Acc':<10} | {'ANOVA ROC':<10} | {'VIF ROC':<10} | {'Impact / Stability'}")
print("-" * 80)
for row in pkg["benchmarks"].get("anova_vs_vif_comparison", []):
    print(f"{row['model']:<25} | {row['anova_acc']:>8.2f}% | {row['vif_acc']:>8.2f}% | {row['anova_roc']:>8.2f}% | {row['vif_roc']:>8.2f}% | {row['impact']}")

print("\n" + "=" * 80)
print("TIER 1: UNBALANCED BASELINE BENCHMARKS (SECTION 18)")
print("=" * 80)
print(f"{'Model':<30} | {'Accuracy':<10} | {'Recall':<8} | {'ROC-AUC':<8} | {'True Positives':<14}")
print("-" * 80)
for m in pkg["benchmarks"]["unbalanced_baseline_models"]:
    print(f"{m['model']:<30} | {m['accuracy']:>6.2f}%    | {m['recall']:>5.2f}%  | {m['roc_auc']:>5.2f}%  | {m['tp']:>10,}")

print("\n" + "=" * 80)
print("TIER 2: BALANCED TRAINING VIA SMOTE OVERSAMPLING (SECTION 19)")
print("=" * 80)
smote_meta = pkg["feature_architecture"].get("data_balancing", {})
if smote_meta:
    d_bef = smote_meta.get("distribution_before_smote", {})
    d_aft = smote_meta.get("distribution_after_smote", {})
    t_dist = smote_meta.get("untouched_test_distribution", {})
    print(f"Balancing Technique:    {smote_meta.get('technique', 'SMOTE')} (k={smote_meta.get('k_neighbors', 5)})")
    print(f"Applied To:             {smote_meta.get('applied_strictly_to', '')}")
    print(f"Test Partition Status:  {smote_meta.get('test_data_status', '')}")
    print(f"\nTraining Class Distribution BEFORE SMOTE:")
    print(f"  Class 0 (On-Time):   {d_bef.get('class_0_on_time', 0):,} ({d_bef.get('class_0_pct', 0):.2f}%)")
    print(f"  Class 1 (Delayed):   {d_bef.get('class_1_delayed', 0):,} ({d_bef.get('class_1_pct', 0):.2f}%)")
    print(f"  Total Records:       {d_bef.get('total_records', 0):,} (Imbalance Ratio: {d_bef.get('imbalance_ratio', 0):.2f}:1)")
    print(f"\nTraining Class Distribution AFTER SMOTE:")
    print(f"  Class 0 (On-Time):   {d_aft.get('class_0_on_time', 0):,} ({d_aft.get('class_0_pct', 0):.2f}%)")
    print(f"  Class 1 (Delayed):   {d_aft.get('class_1_delayed', 0):,} ({d_aft.get('class_1_pct', 0):.2f}%) [+{d_aft.get('synthetic_instances_added', 0):,} synthetic]")
    print(f"  Total Records:       {d_aft.get('total_records', 0):,} (Imbalance Ratio: 1.00:1 Exact Parity)")
    print(f"\nUntouched Real-World Test Partition:")
    print(f"  Class 0: {t_dist.get('class_0_on_time', 0):,} ({t_dist.get('class_0_pct', 0):.2f}%) vs Class 1: {t_dist.get('class_1_delayed', 0):,} ({t_dist.get('class_1_pct', 0):.2f}%) [Total: {t_dist.get('total_records', 0):,}]")

print("-" * 80)
print(f"{'Model':<30} | {'Accuracy':<10} | {'Recall':<8} | {'ROC-AUC':<8} | {'True Positives':<14}")
print("-" * 80)
for m in pkg["benchmarks"]["balanced_fixed_models"]:
    print(f"{m['model']:<30} | {m['accuracy']:>6.2f}%    | {m['recall']:>5.2f}%  | {m['roc_auc']:>5.2f}%  | {m['tp']:>10,}")

print("\n" + "=" * 80)
print("SECTION 20: UNBALANCED BASELINE VS SMOTE BALANCED COMPARISON")
print("=" * 80)
print(f"{'Model':<25} | {'Unbal Acc':<10} | {'SMOTE Acc':<10} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'TP Caught'}")
print("-" * 80)
for row in pkg["benchmarks"].get("unbalanced_vs_smote_comparison", []):
    print(f"{row['model']:<25} | {row['unbalanced_acc']:>8.2f}% | {row['smote_acc']:>8.2f}% | {row['precision']:>6.2f}% | {row['recall']:>6.2f}% | {row['f1_score']:>6.2f}% | {row['roc_auc']:>6.2f}% | {row['tp_smote']:>9,}")

print("\n" + "=" * 80)
print("TIER 3: SMOTE HYBRID & STACKING MODELS (SECTION 21)")
print("=" * 80)
print(f"{'Architecture':<42} | {'Accuracy':<10} | {'Recall':<8} | {'ROC-AUC':<8} | {'TP Delays Caught':<16}")
print("-" * 80)
for h in pkg["benchmarks"].get("smote_hybrid_ensemble_models", []):
    print(f"{h['architecture']:<42} | {h['accuracy']:>6.2f}%    | {h['recall']:>5.2f}%  | {h['roc_auc']:>5.2f}%  | {h['tp']:>10,}")

print("\n" + "=" * 80)
print("SECTION 21 & 29: GRAND 4-TIER MODEL COMPARISON TABLE")
print("=" * 80)
print(f"{'Category / Label':<22} | {'Model Architecture':<38} | {'Accuracy':<10} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}")
print("-" * 115)
for row in pkg["benchmarks"].get("final_4tier_comparison", []):
    print(f"{row['category']:<22} | {row['model']:<38} | {row['accuracy']:>8.2f}% | {row['precision']:>6.2f}% | {row['recall']:>6.2f}% | {row['f1_score']:>6.2f}% | {row['roc_auc']:>6.2f}%")

print("\n" + "=" * 80)
print("TOP 5 EXPLAINABLE AI PREDICTORS (SHAP GLOBAL IMPORTANCE)")
print("=" * 80)
for feat in pkg["explainable_ai"]["top_20_shap_global_features"][:5]:
    print(f"Rank {feat['rank']}: {feat['feature']:<20} | Mean |SHAP|: {feat['mean_abs_shap']:.4f} | {feat['operational_role']}")

print("\n" + "=" * 80)
print("INTERACTIVE INFERENCE DEMONSTRATION")
print("=" * 80)
loc = {}
exec(pkg["inference_engine"]["predict_function_code"], loc)
predict_fn = loc["predict_flight_delay"]

sample1 = predict_fn("Delta Air Lines Inc.", "ATL", "MCO", 830, 1005, 404, 95)
print("[Test 1 - Morning Flight]:", sample1["airline"], f"{sample1['origin']}->{sample1['destination']}")
print(f"  * Delay Probability: {sample1['severe_delay_probability_pct']}")
print(f"  * Standard Verdict:  {sample1['standard_verdict_T050']}")

sample2 = predict_fn("American Airlines Inc.", "ORD", "LAX", 1845, 2115, 1744, 270)
print("\n[Test 2 - Evening Hub Flight]:", sample2["airline"], f"{sample2['origin']}->{sample2['destination']}")
print(f"  * Delay Probability: {sample2['severe_delay_probability_pct']}")
print(f"  * Operational Alert: {sample2['operational_alert_verdict_T020']}")

print("\n" + "=" * 80)
print("PACKAGE IS FULLY SELF-CONTAINED AND READY TO SHARE!")
print("=" * 80)
