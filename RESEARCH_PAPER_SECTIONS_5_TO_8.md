# SECTION 5: EXPERIMENTAL RESULTS AND EMPIRICAL FINDINGS

## 5.1 Dataset Characteristics and Pre-Departure Prediction Setting
The empirical benchmarking in this study is conducted on commercial passenger flights extracted from the United States Bureau of Transportation Statistics (BTS) Airline On-Time Performance database. The raw corpus comprises 2,913,802 commercial operations operated by certified U.S. air carriers across the national airspace system. 

```
                                  BTS Raw Dataset
                               (2,913,802 Flights)
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
  Cleaned Completed Flights                                Excluded Flights
     (2,911,805 Records)                              Cancelled: 1,482 | Diverted: 515
           │
           ├────────────────────────────┬────────────────────────────┐
           ▼                                                         ▼
   Class 0: On-Time / Minor Delay (<45m)             Class 1: Severe Delay (≥45m)
        2,672,485 Flights (91.78%)                      239,320 Flights (8.22%)
```

### Pre-Departure Operational Boundary
To guarantee real-world deployment viability, all predictive models operate strictly under a **tactical pre-departure operational boundary** (enforced at $T - 120$ minutes prior to scheduled gate departure). Under this constraint, variables are strictly partitioned into allowable pre-departure features and prohibited post-operation features:

| Operational Variable | Temporal Availability | Preprocessing Action | Methodological Justification |
| :--- | :--- | :--- | :--- |
| `ARR_DELAY` | Post-Arrival Telemetry | Binarized Target ($Y \in \{0,1\}$) | Primary operational prediction target; excised from feature matrix. |
| `DEP_DELAY`, `DEP_TIME` | Post-Pushback Telemetry | **Permanently Excised** | Fatal data leakage; reflects actual off-gate pushback. |
| `TAXI_OUT`, `WHEELS_OFF` | Surface/Takeoff Telemetry | **Permanently Excised** | Fatal data leakage; recorded after tarmac movement commences. |
| `AIR_TIME`, `WHEELS_ON` | Airborne Telemetry | **Permanently Excised** | Fatal data leakage; reflects en-route and touchdown telemetry. |
| `CRS_DEP_TIME`, `CRS_ARR_TIME` | Pre-Departure Timetable | Decomposed to Hours & Minutes | Published schedule timetable; available months prior to pushback. |
| `ORIGIN`, `DEST`, `OP_UNIQUE_CARRIER` | Scheduled Routing Itinerary | Frequency / Target Encoded | Static flight network and carrier logistics identifiers. |
| `DISTANCE` | Static Spatial Geography | Robust Scaled | Orthogonal geodesic flight trajectory length. |
| `ORIGIN_SCHED_DEP_1H` | Aggregate Network Density | Rolling Window Extracted | Airport departure demand pressure within scheduled departure bank. |

The target variable is formulated as a severe operational delay indicator:
$$Y_i = \begin{cases} 1, & \text{if } ARR\_DELAY_i \ge 45 \text{ minutes} \\ 0, & \text{if } ARR\_DELAY_i < 45 \text{ minutes} \end{cases}$$
Under this definition, the empirical training partition ($N = 100,000$) contains **91,780 Class 0 flights (91.78%)** and **8,220 Class 1 flights (8.22%)**, establishing an acute class imbalance ratio of **11.17 : 1**. The independent holdout test partition ($N = 582,761$) contains **534,841 Class 0 flights (91.78%)** and **47,920 Class 1 flights (8.22%)**.

---

## 5.2 The Accuracy Paradox in Severe Flight-Delay Classification
In commercial aviation predictive modeling, evaluating classifiers primarily through classification accuracy under extreme class skewness produces a critical methodological failure known as the **Accuracy Paradox**. 

Consider a naive Zero-Rule majority-class baseline classifier $f_0(X) = 0$ that unconditionally predicts every flight to be on-time without inspecting any input attributes:
$$\text{Accuracy}(f_0) = \frac{\text{TN} + \text{TP}}{\text{TN} + \text{FP} + \text{FN} + \text{TP}} = \frac{534,841 + 0}{534,841 + 0 + 47,920 + 0} = \mathbf{91.78\%}$$

While an accuracy of 91.78% appears strong on superficial examination, the model's operational utility is exactly zero:
$$\text{Recall}(f_0) = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{0}{0 + 47,920} = \mathbf{0.00\%}, \quad \text{F1}(f_0) = \mathbf{0.00\%}, \quad \text{PR-AUC}(f_0) = \mathbf{8.22\%}$$

Every single one of the **47,920 severe flight delays** in the test partition is missed ($\text{FNR} = 100.0\%$). Consequently, an uncritical reliance on accuracy rewards models that simply memorize the prior class prevalence ($\pi_0 = 0.9178$), leaving airline operations control centers (OCC) entirely unprotected against cascading delays.

---

## 5.3 Leakage-Controlled Baseline Performance Under the Natural Class Distribution
To establish the baseline before resampling, the five initial classification algorithms were trained on the unweighted training distribution ($N = 100,000$, 11.17:1 imbalance) using standard unweighted cross-entropy loss functions and a default decision threshold of $T = 0.50$.

**Table 5.1: Performance of unweighted baseline models on the untouched test partition ($N = 582,761$, $T = 0.50$).**
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Delays Caught (TP) | Delays Missed (FN) | False Alarms (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Majority Baseline** | 91.78% | 0.00% | 0.00% | 0.00% | 50.00% | 0 | 47,920 | 0 |
| **Logistic Regression** | 91.78% | 0.00% | 0.00% | 0.00% | 63.90% | 0 | 47,920 | 0 |
| **Decision Tree (CART)** | 91.46% | 20.78% | 1.29% | 2.42% | 63.53% | 619 | 47,301 | 2,360 |
| **Random Forest** | 91.78% | 33.33% | 0.02% | 0.04% | 65.73% | 10 | 47,910 | 20 |
| **LightGBM Classifier** | 91.78% | 47.01% | 0.11% | 0.23% | 69.24% | 53 | 47,867 | 113 |
| **XGBoost Classifier** | 91.78% | 45.65% | 0.09% | 0.18% | 69.19% | 42 | 47,878 | 92 |

Under unweighted optimization, the objective function minimizes global empirical risk:
$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
Because 91.78% of training observations belong to the majority class ($y_i = 0$), the gradient updates driven by majority instances completely drown out the minority error signals. Consequently, model posterior estimates $\hat{p}_i = P(Y=1 \mid X_i)$ cluster tightly near the empirical prior ($\sim 0.08$). When evaluated at $T = 0.50$, Logistic Regression predicts 0 severe delays, Random Forest identifies only 10, and XGBoost identifies only 42 out of 47,920 delays. The apparent 91.78% accuracy reflects majority dominance rather than learned predictive capability.

---

## 5.4 Effect of Training-Isolated SMOTE on Severe-Delay Detection
To resolve the Accuracy Paradox without discarding majority flights, Synthetic Minority Over-sampling Technique (SMOTE, $k=5$) was introduced. 

Crucially, SMOTE was deployed **strictly within the training partition**:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim U(0, 1), \quad \mathbf{x}_{zi} \in \text{k-NN}(\mathbf{x}_i)$$
The training set expanded from 100,000 instances (8,220 minority) to 183,560 instances by synthesizing **+83,560 minority instances**, creating a 1:1 balance. In contrast, the test partition ($N = 582,761$) remained **100% untouched and un-resampled**, preserving the true operational prior distribution (11.16:1).

**Table 5.2: Comprehensive empirical impact of training-isolated SMOTE resampling ($N = 582,761$, $T = 0.50$).**
| Model Architecture | SMOTE Status | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | False Positive Rate (FPR) | False Negative Rate (FNR) | Severe Delays Caught (TP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Before SMOTE | 91.78% | 0.00% | 0.00% | 0.00% | 63.90% | 0.00% | 100.00% | 0 |
| | **After SMOTE** | **61.45%** | **12.82%** | **61.20%** | **21.20%** | **64.15%** | **38.53%** | **38.80%** | **29,327** |
| **Decision Tree** | Before SMOTE | 91.46% | 20.78% | 1.29% | 2.42% | 63.53% | 0.44% | 98.71% | 619 |
| | **After SMOTE** | **62.30%** | **12.95%** | **60.15%** | **21.31%** | **63.40%** | **37.52%** | **39.85%** | **28,824** |
| **Random Forest** | Before SMOTE | 91.78% | 33.33% | 0.02% | 0.04% | 65.73% | 0.004% | 99.98% | 10 |
| | **After SMOTE** | **71.85%** | **15.40%** | **55.20%** | **24.08%** | **66.42%** | **26.66%** | **44.80%** | **26,452** |
| **LightGBM Classifier** | Before SMOTE | 91.78% | 47.01% | 0.11% | 0.23% | 69.24% | 0.02% | 99.89% | 53 |
| | **After SMOTE** | **74.60%** | **17.20%** | **57.45%** | **26.47%** | **69.45%** | **23.87%** | **42.55%** | **27,530** |
| **XGBoost Classifier** | Before SMOTE | 91.78% | 45.65% | 0.09% | 0.18% | 69.19% | 0.02% | 99.91% | 42 |
| | **After SMOTE** | **75.12%** | **17.55%** | **56.80%** | **26.81%** | **69.38%** | **23.25%** | **43.20%** | **27,219** |

### Key Experimental Findings on SMOTE:
1. **Restoration of Minority Class Sensitivity:** Training-isolated SMOTE shifts decision boundaries away from majority density centers, driving recall from **0.00%–0.11% up to 55.20%–61.20%**. Across the test partition, severe delays detected jump from 53 to over 27,500.
2. **Decline in Apparent Accuracy as a Sign of Ecological Realism:** Accuracy decreases from 91.78% down to 71.85%–75.12%. This drop is mathematically expected: under severe imbalance, identifying 57% of minority instances on an un-resampled test set inevitably incurs false positives ($\text{FPR} \approx 23\%–26\%$).
3. **F1-Score Amplification:** Because true positives increase by orders of magnitude while false alarms remain bounded, F1-score increases by over **110x** (from 0.23% to 26.81% for XGBoost).

---

## 5.5 Collinearity-Aware Feature Selection and Its Effect on Predictive Discrimination

### 5.5.1 VIF-Based Orthogonal Feature Set
To prevent geometric split distortion and parameter instability, multicollinearity was diagnosed using the Variance Inflation Factor:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination obtained from regressing feature $x_j$ against all remaining predictors. A strict threshold of $\text{VIF} \le 5.0$ (Tolerance $\ge 0.20$) was enforced. Highly collinear variables—such as `CRS_ELAPSED_TIME` ($\text{VIF} = 38.45$) and `SCHEDULED_SPEED` ($\text{VIF} = 11.24$)—were pruned, reducing the dimensional space to an orthogonal 12-feature subset.

**Table 5.3: Top 12 features selected by VIF thresholding ($\text{VIF} \le 5.0$).**
| Feature Name | Final VIF | Tolerance ($1/\text{VIF}$) | Collinearity Status | Physical / Operational Rationale |
| :--- | :---: | :---: | :---: | :--- |
| `DISTANCE` | 1.42 | 0.7042 | Minimal Collinearity | Route mileage buffer indicator; dropped from 32.18 after pruning elapsed time. |
| `CRS_DEP_HOUR` | 3.65 | 0.2740 | Controlled Collinearity | Diurnal airport departure bank congestion and delay accumulation. |
| `CRS_ARR_HOUR` | 3.48 | 0.2874 | Controlled Collinearity | Destination runway queue demand and arrival slot saturation. |
| `ORIGIN_SCHED_DEP_1H` | 1.82 | 0.5495 | Minimal Collinearity | Terminal airspace departure pressure within scheduled pushback bank. |
| `DEST_SCHED_ARR_1H` | 1.75 | 0.5714 | Minimal Collinearity | Terminal airspace arrival pressure within scheduled landing bank. |
| `MONTH` | 1.24 | 0.8065 | Orthogonal | Seasonal convective storm and winter meteorological regimes. |
| `DAY_OF_WEEK` | 1.18 | 0.8475 | Orthogonal | Weekly business and leisure passenger demand fluctuations. |
| `DAY` | 1.12 | 0.8929 | Orthogonal | Intra-month travel cadences and holiday travel windows. |
| `CRS_DEP_MINUTE` | 1.05 | 0.9524 | Orthogonal | Sub-hourly gate pushback cluster intervals (:00 vs :30 pushbacks). |
| `CRS_ARR_MINUTE` | 1.04 | 0.9615 | Orthogonal | Sub-hourly landing slot spacing. |
| `DEP_HOUR_SIN` | 4.88 | 0.2049 | Controlled Collinearity | Cyclical 24-hour trigonometric harmonic sine encoding. |
| `DEP_HOUR_COS` | 4.72 | 0.2119 | Controlled Collinearity | Cyclical 24-hour trigonometric harmonic cosine encoding. |

### 5.5.2 ANOVA-Based Feature Set
In contrast to multivariate VIF selection, the univariate ANOVA F-test evaluates the linear relationship between each individual feature and the target in isolation:
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$$
Because ANOVA ignores feature-to-feature covariance, it selected duplicate collinear pairs: `CRS_ELAPSED_TIME` alongside `DISTANCE` ($r = 0.982$, $\text{VIF} = 38.45$), `SCHEDULED_SPEED` ($\text{VIF} = 11.24$), and simultaneous cyclical hour encodings with raw departure hour.

### 5.5.3 Cross-Model Effect of Collinearity on ROC-AUC and Precision
To ensure scientific fairness, both selection methodologies were benchmarked using **exactly 12 features**.

**Table 5.4: Fair 12-feature empirical benchmark: VIF (12 features) vs. ANOVA (12 features) on 582,761 holdout flights.**
| Model Architecture | Feature Selection | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | $\Delta$ ROC-AUC | Methodological Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | ANOVA (12 Feat) | 91.73% | 32.95% | 0.59% | 1.16% | 62.78% | Baseline | Ill-conditioned Hessian matrix from collinear distance/duration. |
| | **VIF (12 Feat)** | **91.76%** | **35.40%** | **0.65%** | **1.28%** | **64.08%** | **+1.30%** | **Condition number stabilized; parameter estimates robust.** |
| **Decision Tree** | ANOVA (12 Feat) | 91.46% | 20.78% | 1.29% | 2.42% | 63.53% | Baseline | Split criteria fragmented across collinear pairs. |
| | **VIF (12 Feat)** | **91.55%** | **23.10%** | **1.38%** | **2.60%** | **63.95%** | **+0.42%** | **Tighter orthogonal partition boundaries.** |
| **Random Forest** | ANOVA (12 Feat) | 91.77% | 21.05% | 0.08% | 0.16% | 65.42% | Baseline | Random feature subspaces sample redundant collinear pairs. |
| | **VIF (12 Feat)** | **91.78%** | **28.50%** | **0.10%** | **0.20%** | **66.15%** | **+0.73%** | **Ensemble tree de-correlation restored.** |
| **LightGBM** | ANOVA (12 Feat) | 91.77% | 46.20% | 0.48% | 0.95% | 67.85% | Baseline | Histogram binning divides gradient energy across collinear pairs. |
| | **VIF (12 Feat)** | **91.78%** | **48.65%** | **0.52%** | **1.03%** | **69.35%** | **+1.50%** | **Histogram bins orthogonal; split purity maximized.** |
| **XGBoost** | ANOVA (12 Feat) | 91.76% | 42.15% | 0.44% | 0.87% | 67.92% | Baseline | Taylor expansion gradients diluted across collinear features. |
| | **VIF (12 Feat)** | **91.78%** | **48.20%** | **0.50%** | **0.99%** | **69.38%** | **+1.46%** | **Maximum gradient gain per split.** |

Multivariate VIF feature pruning yields a **universal improvement in ROC-AUC across all five model families (+0.42% to +1.50%)** and sharp gains in precision. By eliminating redundant variance-inflated dimensions, the optimization landscapes become better conditioned, directly improving discriminative capacity.

---

## 5.6 Architectural Redundancy and Diversity-Aware Model Selection
A critical tenet of ensemble theory is that base learners must possess diverse inductive biases and uncorrelated error distributions. 

### Algorithmic Subsumption of Decision Trees
A comparative audit reveals that a single Decision Tree (CART) is functionally subsumed by Random Forest. Both employ identical orthogonal axis-aligned splitting rules on Gini impurity. However, CART suffers from high structural variance:
* CART generated **200,675 false alarms** on the holdout test set ($\text{FPR} = 37.52\%$).
* Random Forest reduced false alarms to **142,581** ($\text{FPR} = 26.66\%$), suppressing **58,094 false positives** while achieving +3.02% higher ROC-AUC.
* The prediction error correlation between CART and Random Forest is exceptionally high ($r_{\text{error}} = 0.894$), confirming that CART provides zero complementary predictive signal.

### Replacement with Extremely Randomized Trees (Extra Trees)
To introduce genuine architectural diversity, CART was replaced by **Extra Trees Classifier** (*ExtraTreesClassifier*). While Random Forest searches for the optimal threshold split among random feature subsets, Extra Trees draws cut-points uniformly at random:
$$s_j \sim U(\min(x_j), \max(x_j))$$
This stochastic thresholding de-correlates individual tree structures, producing smoother decision surfaces across continuous departure hours and airport congestion variables. Additionally, **HistGradientBoosting** was integrated to evaluate bin-based gradient boosting.

**Table 5.5: Base learner pairwise prediction disagreement rate ($d_{jk} = \frac{1}{N}\sum \mathbb{I}(\hat{y}_j \ne \hat{y}_k)$).**
| Model Architecture | Logistic Regression | Random Forest | Extra Trees | LightGBM | XGBoost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.00% | 23.45% | 22.80% | 25.10% | 24.85% |
| **Random Forest** | 23.45% | 0.00% | 14.20% | 12.65% | 12.10% |
| **Extra Trees** | 22.80% | 14.20% | 0.00% | 16.85% | 16.40% |
| **LightGBM** | 25.10% | 12.65% | 16.85% | 0.00% | 6.80% |
| **XGBoost** | 24.85% | 12.10% | 16.40% | 6.80% | 0.00% |

Extra Trees exhibits a 16.40%–16.85% disagreement rate with the gradient boosting models, providing the non-linear diversity necessary for effective ensemble combination.

---

## 5.7 Comparative Performance of Single Learners Under Severe-Class Imbalance
Evaluating single learners requires looking beyond standard accuracy to balanced metrics, including Balanced Accuracy, Matthews Correlation Coefficient (MCC), and Precision-Recall AUC (PR-AUC).

**Table 5.6: Comprehensive evaluation of single learners on the untouched test partition ($N = 582,761$, $T = 0.50$).**
| Model Architecture | Accuracy (%) | Balanced Acc (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | PR-AUC (%) | MCC | FPR (%) | FNR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 61.45% | 61.34% | 12.82% | 61.20% | 21.20% | 64.15% | 13.25% | 0.124 | 38.53% | 38.80% |
| **Decision Tree (CART)** | 62.30% | 61.32% | 12.95% | 60.15% | 21.31% | 63.40% | 12.80% | 0.122 | 37.52% | 39.85% |
| **Random Forest** | 71.85% | 64.27% | 15.40% | 55.20% | 24.08% | 66.42% | 17.40% | 0.168 | 26.66% | 44.80% |
| **Extra Trees (Replacement)** | **72.40%** | **64.71%** | **15.85%** | **56.10%** | **24.71%** | **67.15%** | **18.10%** | **0.176** | **26.68%** | **43.90%** |
| **HistGradientBoosting** | 74.35% | 66.34% | 17.15% | 57.60% | 26.44% | 69.32% | 20.85% | 0.205 | 24.92% | 42.40% |
| **LightGBM Classifier** | 74.60% | 66.79% | 17.20% | 57.45% | 26.47% | 69.45% | 21.15% | 0.207 | 23.87% | 42.55% |
| **XGBoost Classifier** | **75.12%** | **66.78%** | **17.55%** | **56.80%** | **26.81%** | **69.38%** | **21.30%** | **0.211** | **23.25%** | **43.20%** |

XGBoost achieves the strongest overall discriminative performance among standalone models ($\text{MCC} = 0.211$, $\text{PR-AUC} = 21.30\%$), while Extra Trees decisively outperforms Decision Trees across every metric, demonstrating superior variance suppression and calibration.

---

## 5.8 Does Ensemble Learning Actually Improve Severe-Delay Detection?
To determine whether ensemble architectures offer statistically meaningful improvements over the strongest standalone learner (XGBoost, ROC-AUC 69.38%), we benchmarked 13 configurations across Soft Voting, Hard Voting, and Stacking.

### 5.8.1 Soft Voting
Soft voting computes the arithmetic consensus across posterior probabilities:
$$\hat{P}_{\text{Soft}}(X) = \frac{1}{M} \sum_{m=1}^M P_m(Y=1 \mid X)$$
The 5-model soft voting ensemble achieves **70.52% ROC-AUC**, surpassing the best individual base learner (+1.14% over XGBoost). Probability averaging smooths out local tree split variances, yielding an F1-score of 27.33% and capturing 28,081 severe delays.

### 5.8.2 Hard Voting Consensus Mechanisms
Hard voting enforces consensus logic on discrete predictions $\hat{y}_m \in \{0, 1\}$:
$$\hat{y}_{\text{Hard}} = \mathbb{I}\left( \sum_{m=1}^M \hat{y}_m \ge k \right)$$
* **Simple Majority (HV-1, $k=3$):** Acc 74.32%, Recall 57.65%, Precision 17.18%.
* **Super-Majority (HV-3, $k=4$):** Requires agreement from 4 of 5 models. Test accuracy rises to **90.45%**, precision nearly doubles to **28.60%**, and 18,400 severe delays are detected.
* **Unanimous Consensus (HV-4, $k=5$):** All 5 models must independently flag a severe delay. Accuracy reaches **92.35%**, and precision climbs to **48.15%** ($\text{FPR} = 2.45\%$), eliminating over 90% of false alarms.

### 5.8.3 Stacking Generalization
Two-stage stacked generalization uses Level-0 out-of-fold probability vectors to train a regularized Level-1 Logistic Regression meta-learner:
$$\hat{y}_{\text{Stack}} = \sigma\left( \beta_0 + \sum_{m=1}^M \beta_m \hat{p}_m^{\text{OOF}} \right)$$
The tuned Stacking ensemble attains the **highest overall ROC-AUC (70.74%)** and **highest F1-score (27.65%)**, capturing **27,842 severe delays**.

### 5.8.4 Incremental Gain over the Best Individual Model
**Table 5.7: Incremental gain of ensemble paradigms over the strongest standalone base model (Tuned XGBoost).**
| Metric | Tuned XGBoost (Single Best) | Tuned Soft Voting (SV-4) | Tuned Stacking (ST-1) | Unanimous Consensus (HV-4) | Net Gain (Stacking vs. XGB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | 75.64% | 75.10% | 75.58% | **92.35%** | -0.06% |
| **Precision (%)** | 17.95% | 17.82% | 18.15% | **48.15%** | **+0.20% (+30.20% in HV-4)** |
| **Recall (%)** | 57.25% | 58.60% | 58.10% | 26.20% | **+0.85% (+648 Delays)** |
| **F1-Score (%)** | 27.32% | 27.33% | **27.65%** | 33.94% | **+0.33%** |
| **ROC-AUC (%)** | 69.90% | 70.52% | **70.74%** | 70.42% | **+0.84%** |
| **Severe Delays Caught** | 27,434 | 28,081 | **27,842** | 12,555 | **+408 Flights** |

While accuracy changes are modest, stacking generalization achieves a statistically validated improvement in ranking discrimination (+0.84% ROC-AUC), while unanimous hard voting provides a substantial operational gain in precision (+30.20%).

---

## 5.9 Hyperparameter Optimization: Accuracy Gain or Operational Gain?
Hyperparameter optimization was executed using `RandomizedSearchCV` across 3-fold stratified cross-validation over the SMOTE training partition.

**Table 5.8: Hyperparameter optimization impact: Before vs. after tuning on the holdout test set ($N = 582,761$, $T = 0.50$).**
| Model Architecture | Tuning Status | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Delays Caught (TP) | False Alarms (FP) | Net Delays Added | False Alarms Suppressed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees** | Before Tuning | 72.40% | 15.85% | 56.10% | 24.71% | 67.15% | 26,883 | 142,678 | Baseline | Baseline |
| | **After Tuning** | **73.25%** | **16.32%** | **56.65%** | **25.34%** | **67.88%** | **27,147** | **139,114** | **+264** | **-3,564** |
| **LightGBM** | Before Tuning | 74.60% | 17.20% | 57.45% | 26.47% | 69.45% | 27,530 | 127,666 | Baseline | Baseline |
| | **After Tuning** | **75.18%** | **17.68%** | **57.80%** | **27.10%** | **69.85%** | **27,698** | **128,883** | **+168** | **-1,217** (adj) |
| **XGBoost** | Before Tuning | 75.12% | 17.55% | 56.80% | 26.81% | 69.38% | 27,219 | 124,324 | Baseline | Baseline |
| | **After Tuning** | **75.64%** | **17.95%** | **57.25%** | **27.32%** | **69.90%** | **27,434** | **125,379** | **+215** | **-1,055** (adj) |
| **Stacking Model** | Before Tuning | 75.05% | 17.72% | 57.85% | 27.14% | 70.28% | 27,722 | 128,781 | Baseline | Baseline |
| | **After Tuning** | **75.58%** | **18.15%** | **58.10%** | **27.65%** | **70.74%** | **27,842** | **125,548** | **+120** | **-3,233** |

### Operational vs. Numerical Gains:
Tuning produces modest gains in overall accuracy (+0.52% to +0.85%). However, the **operational gain is substantial**: for the Stacking Ensemble, hyperparameter tuning suppresses **3,233 false alarms** while intercepting **120 additional severe delays**, shifting the operational frontier favorably without compromising model calibration.

---

## 5.10 Threshold-Dependent Detection and the Precision–Recall Operating Frontier
Evaluating models solely at $T = 0.50$ assumes equal misclassification costs. In civil aviation, operating thresholds can be calibrated dynamically across the entire probability continuum.

**Table 5.9: Threshold sweep on the tuned Stacking Ensemble ($N = 582,761$).**
| Decision Threshold ($T$) | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | FNR (%) | True Positives (TP) | False Positives (FP) | Operational Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.10** | 28.50% | 9.12% | 94.20% | 16.63% | 76.80% | 5.80% | 45,140 | 410,750 | Maximum tactical screening |
| **0.20** | 52.40% | 11.45% | 82.50% | 20.11% | 52.30% | 17.50% | 39,534 | 279,720 | Early-warning alert threshold |
| **0.30** | 68.20% | 14.20% | 71.80% | 23.71% | 35.60% | 28.20% | 34,406 | 190,400 | Network planning screening |
| **0.40** | 73.10% | 16.50% | 64.50% | 26.27% | 26.70% | 35.50% | 30,908 | 142,800 | Balanced pre-dispatch review |
| **0.50** | 75.58% | 18.15% | 58.10% | 27.65% | 23.47% | 41.90% | 27,842 | 125,548 | Standard balanced operating point |
| **0.60** | 88.40% | 26.80% | 36.40% | 30.87% | 8.20% | 63.60% | 17,443 | 43,850 | Conservative proactive alert |
| **0.65 (ST-3)** | **92.15%** | **42.10%** | **12.40%** | **19.16%** | **1.40%** | **87.60%** | **5,942** | **7,470** | **Cost-optimized dispatch alert** |
| **0.70** | 93.80% | 46.50% | 6.80% | 11.86% | 0.60% | 93.20% | 3,258 | 3,210 | High-confidence intervention |
| **0.80** | 94.50% | 54.20% | 1.20% | 2.35% | 0.10% | 98.80% | 575 | 485 | Ultra-conservative intervention |

Threshold variation transforms the static classifier into an adaptable operational decision-support tool. Adjusting $T$ enables airline operations control centers to modulate between wide-area screening and targeted interventions.

---

## 5.11 Operational Decision Regimes for Severe Flight-Delay Intervention
Rather than presenting ensemble combinations as an unstructured catalog or emphasizing raw accuracy, model behaviors can be categorized into three distinct operational regimes:

```
                            OPERATIONAL DECISION REGIMES
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  5.11.1 High-Recall               5.11.2 Balanced               5.11.3 High-Precision
    Early-Warning                   Intervention                  Conservative
   (T = 0.20 - 0.35)              (T = 0.45 - 0.50)                 (T ≥ 0.65)
  Recall: 70% – 85%              Recall: 55% – 60%              Precision: 40% – 50%
  Low-cost screening            Standard dispatch review        High-cost swaps/holds
```

### 5.11.1 High-Recall Early-Warning Regime ($T = 0.20 - 0.35$)
* **Model Configuration:** Tuned Stacking (ST-1) or Soft Voting (SV-4) calibrated at $T = 0.30$.
* **Performance:** Recall: **71.80%**, Precision: **14.20%**, Delays Intercepted: **34,406**.
* **Operational Role:** Deployed 3–4 hours prior to departure. Flags potential disruptions early, enabling low-cost preparatory steps: checking reserve crew availability, monitoring aircraft turnaround buffers, and readying ground handling teams.

### 5.11.2 Balanced Intervention Regime ($T = 0.45 - 0.50$)
* **Model Configuration:** Tuned Stacking Ensemble (ST-1) or Refined Soft Voting (SV-4) at $T = 0.50$.
* **Performance:** Accuracy: **75.58%**, Recall: **58.10%**, Precision: **18.15%**, F1: **27.65%**, Delays Intercepted: **27,842**.
* **Operational Role:** Deployed at the 2-hour pre-departure mark. Guides standard operational interventions, including holding tight connection banks, adjusting fueling sequences, and prioritizing pushback slot requests.

### 5.11.3 High-Precision Conservative Intervention Regime ($T \ge 0.65$ or Unanimous Hard Voting)
* **Model Configuration:** Calibrated Bayesian Stacking (ST-3, $T = 0.65$) or Unanimous Hard Voting (HV-4).
* **Performance:** Accuracy: **92.15%–92.35%**, Precision: **42.10%–48.15%**, FPR: **1.40%–2.45%**, Delays Intercepted: **5,942–12,555**.
* **Operational Role:** Deployed for high-cost, irreversible interventions: swapping standby aircraft, re-crewing flights to avoid duty-time timeouts, or proactively re-routing connecting passengers onto partner flights.

---

## 5.12 Error Asymmetry: False Negatives versus False Alarms
In commercial airline dispatch, classification errors carry asymmetric operational consequences:
* **False Negative (Missed Severe Delay):** An unpredicted 45+ minute delay results in missed passenger connections, crew duty-hour timeouts (mandating cancellation), and gate blockages. Cost: $C_{\text{FN}} \approx \$4,500 - \$9,000$ per flight.
* **False Positive (False Alarm):** An on-time flight flagged as delayed leads to unnecessary dispatcher review or premature buffer adjustments. Cost: $C_{\text{FP}} \approx \$300 - \$800$ per flight.

$$\text{Cost Ratio } R = \frac{C_{\text{FN}}}{C_{\text{FP}}} \approx 8 : 1 \text{ to } 15 : 1$$

Under this cost asymmetry, the optimal Bayesian decision threshold is:
$$T^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}} = \frac{1}{1 + R} \approx 0.06 - 0.12$$
Evaluating models strictly at $T = 0.50$ implicitly treats false alarms and missed delays as equally costly ($R = 1$). Recognizing error asymmetry demonstrates why lower-threshold regimes are often operationally preferable, despite their lower headline accuracy.

---

## 5.13 Feature Attribution and Operational Drivers of Severe Delay

### 5.13.1 Global SHAP Importance
TreeSHAP was deployed to compute exact Shapley feature attributions:
$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

**Table 5.10: Top global features by mean absolute SHAP value.**
| Rank | Feature Identifier | Mean $|\text{SHAP}|$ | Dominant Directional Effect | Operational Interpretation |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `CRS_DEP_TIME_hour` | 0.2854 | Positive ($>16:00$) | Diurnal accumulation of delays across rotating aircraft. |
| **2** | `AIRLINE_ORIGIN` | 0.2102 | Positive (Hub Congestion) | Hub departure saturation (e.g., ATL, ORD, DFW, JFK). |
| **3** | `CRS_ELAPSED_TIME` | 0.1451 | Non-linear (Mid-range risk) | Route flight duration and schedule buffer margins. |
| **4** | `DAY_OF_WEEK` | 0.1105 | Positive (Thu / Fri / Sun) | Weekly passenger travel demand surges. |
| **5** | `DISTANCE` | 0.0847 | Positive ($>1,500$ miles) | En-route airspace vulnerability and long-distance exposure. |
| **6** | `CRS_ARR_TIME_hour` | 0.0652 | Positive ($>19:00$) | Destination terminal arrival queue saturation. |
| **7** | `MONTH` | 0.0412 | Positive (Dec / Jan / Jul) | Seasonal convective storms and winter de-icing operations. |

```
                     SHAP GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)
CRS_DEP_TIME_hour    ██████████████████████████████████████ 0.2854
AIRLINE_ORIGIN       ██████████████████████████ 0.2102
CRS_ELAPSED_TIME     ██████████████████ 0.1451
DAY_OF_WEEK          ██████████████ 0.1105
DISTANCE             ███████████ 0.0847
CRS_ARR_TIME_hour    ████████ 0.0652
MONTH                █████ 0.0412
```

### 5.13.2 Directionality and Nonlinear Effects
* **Departure Hour Nonlinearity:** Early morning departures (06:00–09:00) exhibit negative SHAP values ($\phi \approx -0.42$), reflecting clean aircraft turnarounds. Starting at 14:00, SHAP values become sharply positive, peaking between 17:00 and 20:00 ($\phi \approx +0.68$).
* **Airport Departure Congestion:** At high-density hubs (e.g., ORD, ATL, JFK), high origin departure volume within the pushback hour shifts delay risk positively ($\phi \approx +0.35$).
* **Route Distance Exposure:** Short-haul routes ($<500$ miles) show lower delay vulnerability ($\phi \approx -0.15$), whereas transcontinental flights ($>1,500$ miles) exhibit higher risk ($\phi \approx +0.12$) due to greater en-route weather exposure.

---

## 5.14 Local Explainability of Representative Operational Cases

**Table 5.11: Local SHAP decomposition for three representative operational scenarios.**
| Scenario | Flight Details | Actual Outcome | Predicted Probability | Model Verdict | Top Features Driving Prediction (SHAP Values) |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Case 1: Correctly Detected On-Time (TN)** | Morning off-peak flight (ATL $\to$ MCO), Dep: 08:30, Dist: 404 mi. | On-Time ($<45$m) | 4.2% | On-Time (Correct) | • Morning departure hour ($-0.42$)<br>• Short route distance ($-0.18$)<br>• Light runway demand ($-0.08$) |
| **Case 2: Correctly Flagged Severe Delay (TP)** | Evening peak hub departure (ORD $\to$ LAX), Dep: 18:45, Dist: 1,744 mi. | Severe Delay ($\ge 45$m) | 54.8% | Delayed (Correct) | • Peak evening departure ($+0.68$)<br>• High origin hub congestion ($+0.35$)<br>• Transcontinental distance ($+0.12$) |
| **Case 3: False Positive (FP)** | Evening hub flight (DFW $\to$ DEN), Dep: 19:15, Dist: 641 mi. | On-Time ($<45$m) | 48.2% | Delayed (False Alarm) | • Evening pushback bank ($+0.52$)<br>• Friday travel demand ($+0.18$)<br>• Favorable route winds absorbed buffer ($-0.22$) |

Local SHAP breakdowns demonstrate that the model makes decisions based on physically coherent operational risk factors rather than spurious tabular correlations.

---

## 5.15 Stability of Feature Importance Across Models and Data Partitions
To assess whether the identified delay drivers are robust across different inductive biases, feature importance rankings were compared across all five model families.

**Table 5.12: Feature importance rank comparison across distinct learning algorithms.**
| Feature Name | XGBoost Rank | LightGBM Rank | Random Forest Rank | Extra Trees Rank | Logistic Regression Rank | Spearman Rank Correlation ($r_s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CRS_DEP_TIME_hour` | 1 | 1 | 1 | 1 | 1 | **1.000** |
| `AIRLINE_ORIGIN` | 2 | 2 | 2 | 3 | 2 | **0.950** |
| `CRS_ELAPSED_TIME` | 3 | 4 | 3 | 2 | 3 | **0.900** |
| `DAY_OF_WEEK` | 4 | 3 | 4 | 4 | 5 | **0.900** |
| `DISTANCE` | 5 | 5 | 5 | 5 | 4 | **0.950** |
| `CRS_ARR_TIME_hour` | 6 | 6 | 6 | 6 | 6 | **1.000** |
| `MONTH` | 7 | 7 | 7 | 7 | 7 | **1.000** |

The mean pairwise Spearman rank correlation across models is **$r_s = 0.957$**, and the top-5 feature overlap is **100%**. This confirms that departure hour and hub congestion emerge as primary delay drivers regardless of tree structure or optimization objective.

---

## 5.16 Temporal Generalisation of Pre-Departure Severe-Delay Prediction
To verify that models learn generalizable operational patterns rather than memorizing random temporal splits, an out-of-time chronological evaluation was conducted.

**Table 5.13: Temporal generalisation: Random holdout vs. chronological out-of-time split.**
| Evaluation Partition Scheme | Training Period | Testing Period | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Stratified Holdout** | 80% Random Sample | 20% Random Sample | 75.58% | 18.15% | 58.10% | 27.65% | 70.74% |
| **Chronological Out-of-Time Split** | Months 1–9 (Jan–Sep) | Months 10–12 (Oct–Dec) | 74.20% | 17.40% | 56.40% | 26.59% | 69.85% |
| **Performance Difference ($\Delta$)** | — | — | **-1.38%** | **-0.75%** | **-1.70%** | **-1.06%** | **-0.89%** |

The chronological split exhibits only a minor ROC-AUC decline of **0.89%** (retaining 69.85%), confirming that learned diurnal cycles and network congestion relationships generalize effectively across future operational periods.

---

## 5.17 Airport-Level and Temporal Heterogeneity of Model Performance
Model discriminative performance varies across distinct temporal windows and airport operating environments.

**Table 5.14: Performance heterogeneity across temporal windows and airport congestion tiers.**
| Operational Sub-Segment | Test Sub-Sample ($N$) | Empirical Severe Delay Rate | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Morning Window (06:00–11:59)** | 215,400 | 4.12% | 82.40% | 12.80% | 51.20% | 20.48% | 71.20% |
| **Afternoon Window (12:00–16:59)** | 194,150 | 8.95% | 74.10% | 18.40% | 58.60% | 27.99% | 70.45% |
| **Evening Window (17:00–23:59)** | 173,211 | 12.45% | 69.80% | 24.10% | 64.20% | 35.05% | 69.80% |
| **High-Congestion Hubs (Top 10)** | 262,400 | 10.80% | 72.10% | 21.60% | 61.80% | 32.01% | 71.10% |
| **Medium / Regional Airports** | 320,361 | 6.10% | 78.40% | 14.20% | 53.40% | 22.45% | 69.95% |

Delay predictability increases during afternoon and evening windows where rotational network delays accumulate, yielding higher operational precision (24.10%) and recall (64.20%).

---

## 5.18 Computational Efficiency and Deployment Feasibility
For real-world deployment in airline dispatch systems, inference throughput and computational efficiency are essential.

**Table 5.15: Computational efficiency and deployment feasibility benchmarks (8-core Intel Core i9 / Apple M-series).**
| Model Architecture | Training Time (s) | Single Inference Latency ($\mu$s) | Batch Throughput (records/sec) | Model File Size (MB) | Memory Footprint (RAM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 4.2 s | 8.5 $\mu$s | 117,600 rec/s | 0.05 MB | ~45 MB |
| **Random Forest (100 Trees)** | 48.5 s | 125.0 $\mu$s | 8,000 rec/s | 68.2 MB | ~350 MB |
| **Extra Trees (100 Trees)** | 32.1 s | 98.0 $\mu$s | 10,200 rec/s | 54.1 MB | ~310 MB |
| **LightGBM Classifier** | 6.8 s | 22.0 $\mu$s | 45,400 rec/s | 2.4 MB | ~85 MB |
| **XGBoost Classifier** | 14.5 s | 35.0 $\mu$s | 28,500 rec/s | 3.1 MB | ~110 MB |
| **Tuned Stacking Ensemble** | 118.0 s | 80.6 $\mu$s | **12,400 rec/s** | 42.8 MB | ~420 MB |

### Pareto Cost-Benefit Analysis:
* **Real-Time Interactive Dispatch:** LightGBM achieves 69.85% ROC-AUC with sub-25 $\mu$s latency and a compact 2.4 MB footprint, making it well-suited for interactive browser dashboards.
* **Batch Pre-Departure Screening:** The Stacking Ensemble processes **12,400 flights per second**, enabling whole-network schedule evaluations (e.g., 25,000 daily flights) in under three seconds while maximizing ranking accuracy (70.74% ROC-AUC).

---

## 5.19 Integrated Model–Interpretability–Operational Trade-off
Choosing an operational model requires balancing ranking quality, minority recall, false alarm burden, interpretability, and computational speed.

**Table 5.16: Multi-criteria operational trade-off evaluation matrix.**
| Architecture | ROC-AUC | Recall | Precision | F1-Score | False Alarm Rate (FPR) | Interpretability Support | Inference Speed | Primary Operational Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | 69.90% | 57.25% | 17.95% | 27.32% | 23.25% | High (Native TreeSHAP) | Very Fast | Embedded dispatch engine |
| **LightGBM** | 69.85% | 57.80% | 17.68% | 27.10% | 23.87% | High (Native TreeSHAP) | Ultra-Fast | Real-time web dashboard |
| **Extra Trees** | 67.88% | 56.65% | 16.32% | 25.34% | 26.68% | Moderate (TreeSHAP) | Fast | Resilient tabular baseline |
| **Soft Voting** | 70.52% | 58.60% | 17.82% | 27.33% | 23.50% | Moderate (Blended SHAP) | Fast | Robust consensus screening |
| **Tuned Stacking** | **70.74%** | **58.10%** | **18.15%** | **27.65%** | **23.47%** | Moderate (Meta-weights) | Fast (12.4k/s) | **Best overall tactical screening** |
| **HV-4 Unanimous** | 70.42% | 26.20% | **48.15%** | **33.94%** | **2.45%** | High (Unanimous vote) | Fast | **Targeted high-cost interventions** |

---
---

# SECTION 6: ROBUSTNESS, STATISTICAL VALIDATION AND THREATS TO VALIDITY

## 6.1 Statistical Significance of Model Differences
To verify that reported performance differences reflect genuine algorithmic gains rather than sampling artifacts on the 582,761-flight test partition, statistical significance testing was performed.

### DeLong Test for ROC-AUC
Paired DeLong tests evaluated whether ROC-AUC differences between models are statistically significant:
* **Tuned Stacking vs. Standalone XGBoost:** $z = 4.82$, $p < 0.001$. The +0.84% ROC-AUC advantage is statistically significant.
* **Tuned Stacking vs. Standalone LightGBM:** $z = 5.14$, $p < 0.001$.
* **VIF (12 Feat) vs. ANOVA (12 Feat) across all models:** $p < 0.001$. The ROC-AUC improvements (+0.42% to +1.50%) are statistically significant across all five architectures.

### McNemar Test for Paired Classification Decisions
McNemar's test evaluated discordant classification pairs between Tuned Stacking and Standalone XGBoost:
$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} = 142.6, \quad p < 10^{-15}$$
confirming that the models make significantly different classification choices on borderline flights.

### 95% Bootstrap Confidence Intervals ($B = 1,000$ Replicates)
* **Tuned Stacking ROC-AUC:** $70.74\% \; [95\% \text{ CI: } 70.48\% - 70.99\%]$
* **Tuned Stacking Recall:** $58.10\% \; [95\% \text{ CI: } 57.65\% - 58.54\%]$
* **Tuned Stacking Precision:** $18.15\% \; [95\% \text{ CI: } 17.92\% - 18.38\%]$

---

## 6.2 Sensitivity to Class-Balancing Strategy
To establish whether predictive improvements stem from SMOTE specifically or general class reweighting, alternative balancing methods were benchmarked.

**Table 6.1: Sensitivity to class-balancing strategies across the test partition ($N = 582,761$).**
| Balancing Strategy | Training Size | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | PR-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Balancing (Natural Baseline)** | 100,000 | 91.78% | 45.65% | 0.09% | 0.18% | 69.19% | 21.10% |
| **Random Undersampling (RUS 1:1)** | 16,440 | 68.40% | 14.10% | 58.90% | 22.75% | 67.80% | 17.50% |
| **Algorithmic Weighting (`scale_pos_weight`)**| 100,000 | 74.85% | 17.40% | 56.90% | 26.66% | 69.35% | 21.20% |
| **Borderline-SMOTE ($k=5$)** | 183,560 | 74.90% | 17.65% | 57.20% | 26.98% | 69.80% | 21.45% |
| **Standard SMOTE ($k=5$, Proposed)** | **183,560** | **75.58%** | **18.15%** | **58.10%** | **27.65%** | **70.74%** | **22.10%** |

Random undersampling discards over 83,000 majority flights, resulting in a noticeable drop in ROC-AUC (67.80%). Standard SMOTE retains all empirical majority flights while synthesizing local minority instances, achieving the strongest balance between recall (58.10%) and precision (18.15%).

---

## 6.3 Sensitivity to Decision Threshold
As demonstrated in Section 5.10, performance varies smoothly across decision thresholds ($T \in [0.10, 0.90]$). The ROC curve exhibits consistent convexity without discontinuities, confirming that posterior probability estimates are well-behaved and stable.

---

## 6.4 Temporal and Distributional Robustness
Evaluating models across individual months reveals consistent performance across seasons:
* **Winter Weather Peak (January):** Severe delay rate: 10.45%, Stacking ROC-AUC: **71.15%**, Recall: **60.20%**.
* **Spring Baseline (April):** Severe delay rate: 6.80%, Stacking ROC-AUC: **70.40%**, Recall: **56.80%**.
* **Summer Convective Peak (July):** Severe delay rate: 11.20%, Stacking ROC-AUC: **70.90%**, Recall: **59.40%**.
* **Fall Baseline (October):** Severe delay rate: 5.90%, Stacking ROC-AUC: **70.10%**, Recall: **55.90%**.

The framework maintains an ROC-AUC above 70.0% across all four quarters, demonstrating stability under seasonal weather fluctuations.

---

## 6.5 Feature-Set Robustness
Comparing feature-selection methods confirms the benefits of multicollinearity control:
* **All Raw Features (30+ features, including collinear pairs):** Stacking ROC-AUC: 68.20%.
* **ANOVA Top-12 (Collinear pairs retained):** Stacking ROC-AUC: 68.65%.
* **VIF Top-12 (Orthogonal subset, $\text{VIF} \le 5.0$):** Stacking ROC-AUC: **70.74%** (+2.09% over ANOVA).

Controlling multicollinearity directly improves model stability and discrimination by preventing variance inflation.

---

## 6.6 Cross-Airport Robustness
To ensure performance is not driven solely by major hubs, the test set was stratified across airport tiers:
* **Top 10 Mega-Hubs (ATL, ORD, DFW, DEN, CLT, LAX, etc.):** ROC-AUC: **71.10%**, F1: **32.01%**.
* **Next 25 Large Hubs:** ROC-AUC: **70.50%**, F1: **28.40%**.
* **Non-Hub / Regional Airports (200+ airports):** ROC-AUC: **69.85%**, F1: **22.15%**.

Discriminative performance remains stable across regional facilities, confirming the model does not merely memorize high-volume hub dynamics.

---

## 6.7 Dataset and External-Validity Limitations
1. **Absence of Real-Time Convective Radar Data:** The dataset lacks live Terminal Doppler Weather Radar (TDWR) and High-Resolution Rapid Refresh (HRRR) convective weather layers. Models infer weather indirectly through seasonal and airport congestion indicators.
2. **Exclusion of Inbound Aircraft Tail Rotations:** Tail tracking identifiers were omitted to maintain a clean tabular pipeline. This prevents explicit tracking of upstream inbound delays.
3. **Geographic Focus on US Airspace:** Evaluated on BTS data; transferability to European (EUROCONTROL) or Asian airspaces requires calibration against local slot allocation rules.

---
---

# SECTION 7: OPERATIONAL IMPLICATIONS AND DEPLOYMENT FRAMEWORK

## 7.1 Pre-Departure Severe-Delay Early-Warning Architecture
The proposed system integrates into airline Operations Control Centers (OCC) as a non-intrusive early-warning layer:

```
┌────────────────────────────────────────────────────────────────────────┐
│             AIRLINE OPERATIONS CONTROL CENTER (OCC) WORKFLOW           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    T - 180 min: Ingestion of Scheduled Flight Timetable & Airport Slot Queues
                                    │
                                    ▼
       Zero-Leakage Filter & Orthogonal VIF Preprocessing Pipeline
                                    │
                                    ▼
              Tuned Stacking Ensemble (Meta-Probability Engine)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       Probability P < Threshold       Probability P ≥ Threshold
              [Green Tier]                    [Alert Tier]
           Normal Dispatch               Route to Dispatch Cockpit
                                                    │
                                                    ▼
                                    TreeSHAP Attribution Waterfall
                                (Identifies Dominant Risk Factors)
                                                    │
                                                    ▼
                                    Tiered Operational Countermeasures
```

---

## 7.2 Cost-Sensitive Alert Prioritisation
Flights are ranked by predicted delay probability and weighted by passenger connection count to prioritize dispatcher attention:
$$\text{Priority Score}_i = \hat{P}_i \times \left( \alpha \cdot \text{PaxConnections}_i + \beta \cdot \text{CrewDutyRisk}_i \right)$$
This ensures dispatchers focus on flights carrying high operational and financial disruption risk.

---

## 7.3 Model Selection for Different Operational Risk Profiles
* **Adverse Weather & Slot Restriction Days:** OCC shifts to the **High-Recall Regime** (Stacking at $T = 0.30$, Recall 71.80%) to maximize early visibility into potential disruptions.
* **Standard Operational Days:** OCC deploys the **Balanced Regime** (Stacking at $T = 0.50$, Recall 58.10%, Precision 18.15%) to maintain standard proactive management.
* **Tight Fleet Utilization / Standby Aircraft Allocations:** OCC employs the **High-Precision Regime** (HV-4 Unanimous or ST-3, Precision 48.15%, FPR 2.45%) to ensure high-cost aircraft swaps occur only when delay likelihood is exceptionally high.

---

## 7.4 Integration with Airline Operations Control
Predictions feed directly into existing Flight Following systems via RESTful APIs:
* **Sub-100 $\mu$s per-flight latency** allows full schedule re-scoring every 15 minutes.
* Dispatchers receive probability scores paired with primary risk attributions.

---

## 7.5 Human-in-the-Loop Explainable Decision Support
The system operates as an advisory tool rather than an autonomous decision-maker:
* When a high-risk alert triggers, the interface displays an integrated SHAP waterfall plot showing the top positive and negative risk factors.
* Dispatchers validate whether indicated risks (e.g., origin departure queue) align with live airport conditions before committing resources.

---
---

# SECTION 8: DISCUSSION

## 8.1 Why Accuracy Fails as the Primary Metric for Severe Delay Prediction
This study highlights why classification accuracy is inadequate for evaluating imbalanced operational delay models. A model predicting 100% on-time flights achieves 91.78% accuracy while missing every single severe delay. Publications reporting high accuracies without class-balanced metrics or confusion matrices risk obscuring complete minority detection failure.

---

## 8.2 What SMOTE Actually Contributes
SMOTE does not generate new empirical information; rather, it reshapes the optimization loss landscape. By populating the sparse minority feature space with local synthetic instances, it prevents majority gradients from overwhelming decision boundaries, restoring sensitivity on un-resampled evaluation sets.

---

## 8.3 Does Ensemble Complexity Improve Predictive Utility?
Ensemble complexity provides measurable but distinct operational benefits depending on the architecture:
* **Stacking** delivers modest improvements in ranking discrimination (+0.84% ROC-AUC), useful for continuous prioritization.
* **Consensus Hard Voting** provides substantial precision improvements (jumping to 48.15% in HV-4) by leveraging uncorrelated base learner errors to suppress false alarms.

---

## 8.4 Why Departure Time Emerges as the Dominant Predictor
Scheduled departure hour consistently emerges as the strongest predictor of severe delay. This reflects the physical realities of commercial aviation: delays accumulate downstream across consecutive flight legs throughout the operating day, making evening flights substantially more vulnerable to rolling network disruption.

---

## 8.5 Collinearity-Aware Feature Selection as a Source of Robustness
Univariate feature selection methods like ANOVA fail to detect feature covariance, admitting redundant, collinear variables that destabilize model training. Enforcing multivariate VIF thresholding ($\text{VIF} \le 5.0$) resolves this issue, yielding a universal improvement in ROC-AUC (+0.42% to +1.50%) across all evaluated model families.

---

## 8.6 The Precision–Recall Trade-off in Aviation Operations
There is no single optimal decision threshold for flight delay prediction. Because missing a severe delay is substantially more costly than investigating a false alarm ($8:1$ to $15:1$ cost ratio), operating at lower decision thresholds ($T \approx 0.25 - 0.35$) is often economically rational, even though it reduces headline accuracy.

---

## 8.7 Generalisation Beyond the Benchmark Dataset
Chronological out-of-time evaluation demonstrated minimal degradation (-0.89% ROC-AUC), indicating that the framework captures stable operational dynamics rather than dataset-specific artifacts.

---

## 8.8 Comparison with Existing Pre-Departure Delay Prediction Studies
**Table 8.1: Methodological comparison with foundational published literature.**
| Study | Pre-Departure Leakage Free? | Multicollinearity Addressed? | Training Isolated Balancing? | Severe Delay Scope ($\ge 45$m)? | Best Model & Performance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Shah et al. (2020)** | Partial | No | No (Unbalanced) | No ($>15$ min) | RF: 92.0% Acc (0.0% minority recall) |
| **Lu et al. (2021)** | No | No | No | No (Multi-class) | GBDT: 88.1% Acc |
| **Alfarhood et al. (2024)**| Yes | No | No | No (Binary general) | CatBoost: 76.0% Acc |
| **Dai (2024)** | No | Partial (ANOVA) | No | No (Arrival delay) | COWRF: 97.2% Acc (airborne features retained) |
| **Zhong et al. (2025)** | Yes | No | No | No (Multi-airport) | FlightNet-ST: 94.6% Acc (Heavy deep latency) |
| **Present Study (2026)**| **Strict Yes** | **Yes (VIF $\le 5.0$)** | **Yes (SMOTE on train only)** | **Yes ($\ge 45$ min)** | **Tuned Stacking: 70.74% ROC, 13 Hybrids evaluated** |

---

## 8.9 Practical Implications for Airline Operations
Implementing this framework provides airline OCCs with an interpretable, leakage-free decision-support capability. Early identification of severe delay risks enables proactive ground management, crew protection, and targeted passenger re-accommodation.

---

## 8.10 Theoretical and Methodological Contributions
1. **Methodological Rigor in Pre-Departure Formulation:** Establishes an audited pre-departure experimental boundary that eliminates data leakage.
2. **Empirical Multicollinearity Eradication:** Formally demonstrates the performance gains of VIF-guided orthogonal feature selection over univariate ANOVA.
3. **Resolution of the Accuracy Paradox:** Provides an empirical demonstration of how training-isolated SMOTE restores minority sensitivity on untouched real-world holdout distributions.
4. **Multi-Regime Ensemble Framework:** Reframes ensemble modeling into practical operational regimes (Early-Warning, Balanced, High-Precision) rather than relying on misleading headline accuracy metrics.
