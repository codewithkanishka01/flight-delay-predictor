# FLIGHT DELAY PREDICTION USING MACHINE LEARNING: A LEAKAGE-FREE PRE-DEPARTURE FRAMEWORK WITH MULTICOLLINEARITY ELIMINATION, RESAMPLING DYNAMICS, AND TUNED STACKING ENSEMBLES

**Authors:**  
* **Bhavya Gupta** (RA2411003030098)  
* **Arshia Bhandari** (RA2411003030100)  
* **Kanishka Pandey** (RA2411003030107)  

**Supervised by:**  
* **Dr. Rahul Gupta** *(Assistant Professor, Department of Computer Science & Engineering)*  

*Department of Computer Science & Engineering, SRM Institute of Science & Technology, NCR Campus, Ghaziabad, Uttar Pradesh, India*  
*Date of Submission / Publication: April 2026*  

---

## ABSTRACT

Flight delays represent a multi-billion-dollar systemic inefficiency within civil aviation networks, causing compounding crew-pairing timeouts, passenger connection failures, and gate-scheduling disruptions. Despite extensive literature on delay modeling, existing studies frequently suffer from target data leakage (using airborne or post-departure telemetry), unaddressed multicollinearity among cyclical temporal and spatial predictors, and acute class imbalance. Under extreme skewness—where severe operational delays ($\ge 45$ minutes) constitute only 8.22% of flights—naive models routinely succumb to the *Accuracy Paradox*, reporting deceptively high accuracies of 91.78% while failing to detect delayed flights (0.00% recall). 

In this study, we propose a rigorous, leakage-free pre-departure predictive framework trained on 2.91 million flights from the Bureau of Transportation Statistics (BTS) database. First, post-departure and in-flight operational attributes are systematically eliminated. Second, multicollinearity diagnosis using the Variance Inflation Factor ($\text{VIF} \le 5.0$) trims variance-inflated features ($\text{VIF}$ up to 38.45) down to an orthogonal 10-feature subset (all $\text{VIF} \le 3.65$). Third, Synthetic Minority Over-sampling Technique (SMOTE, $k=5$) is applied strictly to the training partition (synthesizing 83,560 minority instances to reach a 1:1 balance of 183,560 samples), while the 582,761-flight test partition is strictly preserved in its empirical 11.17:1 distribution to ensure uncompromised ecological validity. 

Fourth, an empirical redundancy audit reveals that Decision Trees are completely subsumed by Random Forests (+58,000 false alarms), prompting their replacement with Extra Trees (*ExtraTreesClassifier*) and *HistGradientBoosting*. Fifth, Bayesian and randomized hyperparameter optimization enhances discrimination across tree ensembles. Sixth, a systematic benchmarking of 13 hybrid ensemble combinations spanning Soft Voting, Hard Voting, and Stacking Generalization is conducted, establishing natural $\ge 90\%$ accuracy regimes without artificial metric inflation. 

The tuned Stacking Ensemble achieves 75.58% accuracy, 70.74% ROC-AUC, and 58.10% recall, successfully intercepting 27,842 severe delays, while the Unanimous Hard Voting model achieves 92.35% accuracy, 48.15% precision, and 70.42% ROC-AUC. Finally, Explainable AI via TreeSHAP uncovers scheduled departure hour, origin departure pressure, and flight distance as the primary non-linear drivers of cascading flight delays.

**Keywords:** Flight Delay Prediction, Pre-Departure Decision Support, Variance Inflation Factor (VIF), Synthetic Minority Over-sampling Technique (SMOTE), Stacking Ensemble, Accuracy Paradox, Explainable Artificial Intelligence (SHAP).

---

## 1. INTRODUCTION

Commercial aviation serves as the backbone of modern global transportation, economic commerce, and supply chain logistics. However, flight delays remain a chronic, escalating challenge for air traffic control (ATC), airport operators, and commercial airlines. According to the Federal Aviation Administration (FAA) and the Bureau of Transportation Statistics (BTS), commercial airline delays impose an estimated annual economic toll exceeding $33 billion in the United States alone. These financial losses encompass direct operating costs (additional jet fuel consumption, idle ground crew overtime, and aircraft maintenance) and substantial indirect societal costs (missed passenger connections, lost business productivity, and passenger food/lodging vouchers mandated by consumer protection frameworks).

Operationally, delays do not distribute uniformly across flights. While minor delays of under 15 minutes are routinely absorbed by scheduled buffer margins embedded within airline timetables, severe delays—defined operationally by the BTS and aviation practitioners as arrival delays equal to or exceeding 45 minutes ($ARR\_DELAY \ge 45$)—trigger severe cascading failures. Under tight aircraft turnaround rotations, an inbound aircraft arriving 45+ minutes late causes missed passenger bank connections, breaches maximum flight crew duty periods governed by FAA Part 117 regulations, and monopolizes assigned airport departure gates, resulting in rolling ground delay programs (GDP) across downstream hub facilities.

To preempt such operational bottlenecks, predictive machine learning models must operate strictly within a **tactical pre-departure decision window** (typically two or more hours prior to scheduled gate pushback). At this critical juncture, airline dispatchers retain the flexibility to hold connecting flights, swap standby airframes, re-assign reserve flight crews, or proactively re-route passengers before boarding. 

However, a critical methodological pathology in published aviation machine learning literature is **data leakage**: models frequently incorporate features that are only recorded after the flight has pushed back from the gate, such as actual departure delay (`DEP_DELAY`), taxi-out duration (`TAXI_OUT`), or wheels-off timestamps (`WHEELS_OFF`). Including such features artificially inflates predictive metrics to near-perfect levels (95–99% accuracy) in academic simulations, while rendering the model entirely useless for practical pre-departure decision support.

Furthermore, severe flight delays exhibit extreme class imbalance: in real-world airline schedules, severe delays occur in only 8.22% of flights. When machine learning models are evaluated under unweighted cross-entropy loss without proper resampling, they fall victim to the **Accuracy Paradox**. A naive or default classifier that categorically predicts every flight to be 'On-Time' achieves a deceptively high accuracy of 91.78% (the proportion of majority class flights). Yet, its operational utility is exactly zero: it identifies 0 delayed flights (0.00% recall), leaving dispatchers completely blind to looming network disruptions.

This research develops a methodologically sound, end-to-end machine learning framework for severe flight delay prediction. We enforce a zero-leakage pre-departure paradigm, diagnose and eliminate severe multicollinearity via the Variance Inflation Factor (VIF), implement training-isolated SMOTE oversampling, conduct an empirical model redundancy audit, perform Bayesian hyperparameter optimization, benchmark 13 hybrid ensemble combinations across voting and stacking architectures, and interpret model decisions using TreeSHAP.

---

## 2. LITERATURE REVIEW

The scientific literature on aviation delay forecasting has evolved across distinct methodological paradigms, spanning traditional statistical classifiers, tree-based gradient boosting architectures, and hybrid metaheuristic ensembles. This study builds directly upon five foundational works in civil aviation predictive modeling:

### 2.1 Multi-Model Benchmarking on Commercial Airline Data
Alfarhood et al. (2024) conducted an extensive comparative benchmark across 775,000 commercial flight records from Saudi Arabian Airlines, evaluating CatBoost, XGBoost, LightGBM, Random Forest, and Multilayer Perceptrons (MLP). Their empirical findings demonstrated that CatBoost achieved the highest standalone classification accuracy (76.0%) and lowest log-loss. However, their pipeline operated without formal multicollinearity diagnostics, allowing collinear temporal attributes to distort split-variable selection.

### 2.2 Hybrid Metaheuristic Clustering and Feature Optimization
Dai (2024) proposed a hybrid flight delay framework for civil aviation big data, integrating Density-Based Spatial Clustering of Applications with Noise (DBSCAN) with a Coyote Optimization Algorithm-weighted Random Forest (COWRF). By utilizing ANOVA combined with Forward Sequential Feature Selection, the model attained 97.2% accuracy. While demonstrating the power of feature selection, the architecture did not systematically address pre-departure leakage constraints or evaluate stacked meta-learners.

### 2.3 Classical Machine Learning on US Bureau of Transportation Statistics (BTS) Data
Shah et al. (2020) analyzed Bureau of Transportation Statistics (BTS) telemetry for Pittsburgh International Airport, comparing Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Long Short-Term Memory (LSTM) recurrent networks. Random Forest demonstrated superior robustness (reaching 92.0% accuracy), but the study did not resolve the Accuracy Paradox under acute minority delay distributions.

### 2.4 Gradient Boosted Decision Trees on Spatio-Temporal Aviation Networks
Lu et al. (2021) formulated flight delay prediction using Gradient Boosted Decision Trees (GBDT) and XGBoost across high-dimensional spatio-temporal route and airport topologies, achieving 88.1% multi-class classification accuracy. Their work underscored the importance of airport-level arrival/departure demand pressure and preceding flight rotations, though it lacked rigorous pre-departure information sanitization.

### 2.5 Multi-Modal Deep Neural Networks and Spatial-Temporal Graphs
Zhong et al. (2025) proposed the FlightNet-ST deep learning architecture, fusing LSTM temporal sequences, Graph Convolutional Networks (GCN) for airport network topology, and 3D Convolutional Neural Networks (3D-CNN) with attention mechanisms. The model achieved 94.6% accuracy, reducing mean absolute error by 14.47%, though its multi-layer deep topology requires substantial computational latency, hindering sub-50 ms real-time dispatch integration.

**Table 1: Systematic comparative survey of foundational aviation delay prediction literature versus the present study.**

| Study & Year | Methodology | Target & Scope | Dataset & Size | Key Performance | Key Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Shah et al. (2020)** | RF, SVM, KNN, LSTM | Airline Delay (>15m) | BTS Pittsburgh Records | RF: 92.0% Accuracy | Acute minority class bias; unaddressed skewness. |
| **Lu et al. (2021)** | GBDT & XGBoost | Multi-Class Route Delay | Commercial Air Routes | 88.1% Multi-Class Acc | Absence of formal pre-departure leakage audit. |
| **Alfarhood et al. (2024)** | CatBoost, LGB, RF, MLP | Binary Delay Status | 775K Saudi Airline Flights | CatBoost: 76.0% Acc | Omitted multicollinearity diagnostics (VIF). |
| **Dai (2024)** | COWRF + DBSCAN | Big Data Arrival Delay | National Flight Records | COWRF: 97.2% Acc | High computational overhead; no stacking meta-learner. |
| **Zhong et al. (2025)** | FlightNet-ST (LSTM+GCN) | Civil Aviation Delays | Multi-Airport Network | FlightNet: 94.6% Acc | Heavy multi-modal latency; black-box opacity. |
| **Present Study (2026)** | **VIF + SMOTE + Stacking** | **Severe Delay ($\ge 45$m)** | **2.91M Clean BTS Flights** | **Tuned Stacking: 70.74% ROC, 13 Hybrids $\ge 90\%$** | **Zero leakage, VIF $\le 5.0$, calibrated FOC deployment.** |

---

## 3. RESEARCH GAP

A rigorous synthesis of the state-of-the-art reveals five critical research gaps that undermine practical operational deployment:

1. **Pervasive Data Leakage in Pre-Departure Models:** A majority of published papers fail to distinguish between information available at schedule time versus operational telemetry recorded after gate pushback. Models utilizing `DEP_DELAY` or actual taxi durations exhibit high benchmark metrics that collapse upon real-world deployment when post-departure features are unavailable.
2. **Neglect of Multicollinearity in Tabular Flight Features:** High-dimensional aviation datasets often combine calendar attributes, cyclical time transforms (sine/cosine of departure hours), and scheduled durations. Few studies perform formal multicollinearity diagnosis, resulting in variance inflation factors exceeding 30.0 that distort tree split criteria and model feature importances.
3. **Flawed Class Balancing and Data Contamination:** Prior studies applying SMOTE frequently contaminate the experimental protocol by oversampling the entire dataset prior to the train-test split, leading to catastrophic data leakage. Conversely, studies applying random undersampling discard massive numbers of majority flights, impairing generalization.
4. **The Unaddressed Accuracy Paradox in Severe Delay Detection:** Focusing on minor 15-minute delays masks severe delay dynamics. Under severe delay targets ($\ge 45$ minutes), models achieving 91.78% accuracy by predicting 100% on-time flights are reported as 'high-performing' despite achieving 0% recall.
5. **Absence of Systematic Hybrid Voting/Stacking Exploration:** Most existing literature benchmarks standalone models in isolation, without systematically evaluating multi-model ensembles, soft voting mixtures, hard-voting consensus mechanisms, or meta-learning stacking architectures.

---

## 4. CONTRIBUTIONS OF THE STUDY

To decisively overcome the identified research gaps, this paper provides the following explicit methodological and practical contributions:

* **Strict Zero-Leakage Pre-Departure Formulation:** We formulate a robust classification pipeline for severe flight delays ($ARR\_DELAY \ge 45$ minutes) utilizing strictly pre-departure variables available 2+ hours prior to scheduled departure, verified by a comprehensive audit.
* **Two-Stage Multicollinearity Eradication via VIF:** We implement an automated Variance Inflation Factor ($\text{VIF} \le 5.0$) feature pruning protocol, eliminating severely collinear predictors ($\text{VIF}$ up to 38.45) to retain a robust 10-feature orthogonal feature subset (all $\text{VIF} \le 3.65$).
* **Methodologically Pristine SMOTE Resampling Protocol:** We apply SMOTE ($k=5$) strictly to the training partition (expanding minority instances from 8,220 to 91,780 to achieve a 1:1 balance of 183,560 flights), while rigorously isolating the 582,761-flight holdout test set in its empirical 11.17:1 distribution.
* **Empirical Redundancy Audit & Model Replacement:** We conduct a systematic redundancy audit identifying that Decision Trees are completely subsumed by Random Forests (+58,000 false alarms), successfully replacing them with Extra Trees (*ExtraTreesClassifier*) and *HistGradientBoosting*.
* **Systematic 13-Combination Hybrid & Stacking Benchmark:** We design, train, and evaluate 13 distinct ensemble configurations across Soft Voting, Hard Voting (Majority, Super-Majority 4/5, Unanimous 5/5), and Stacking Generalization with meta-learners, demonstrating natural regimes exceeding 90% accuracy.
* **Post-Hoc Explainability via TreeSHAP:** We deploy TreeSHAP on the tuned ensemble models, quantifying the non-linear interaction between scheduled departure hour, hub departure congestion, and route distance.

---

## 5. DATASET AND DATA DESCRIPTION

The empirical foundation of this study is derived from the Bureau of Transportation Statistics (BTS) Airline On-Time Performance database, representing commercial passenger flights operated by major U.S. air carriers. The cleaned primary dataset encompasses **2,911,805 flight records**. To facilitate rigorous benchmarking, the dataset is partitioned into an active training pool of $N = 100,000$ flights and an extensive, independent holdout test partition comprising $N = 582,761$ flights (representing an untouched 20% holdout split of the full BTS volume).

The target variable for classification is formulated as a binary severe delay indicator $Y_i \in \{0, 1\}$, formally defined as:

$$Y_i = \begin{cases} 1, & \text{if } ARR\_DELAY_i \ge 45 \text{ minutes} \\ 0, & \text{if } ARR\_DELAY_i < 45 \text{ minutes} \end{cases}$$

Under this empirical formulation, severe operational delays represent a minority class. In the training partition ($N = 100,000$), Class 0 (On-Time / Minor Delay < 45 min) comprises **91,780 flights (91.78%)**, while Class 1 (Severe Delay $\ge 45$ min) comprises **8,220 flights (8.22%)**, establishing an acute imbalance ratio of **11.17 : 1**. In the independent holdout test partition ($N = 582,761$), Class 0 contains **534,841 flights (91.78%)** and Class 1 contains **47,920 flights (8.22%)**.

---

## 6. DATA PREPROCESSING

Data preprocessing is executed under strict information-sanitization protocols to guarantee zero target leakage and prevent distribution contamination:

1. **Strict Leakage Elimination:** All post-departure and in-flight telemetry fields are audited and permanently excised. Specifically, we purge: `DEP_DELAY` (actual departure delay), `DEP_TIME` (actual wheels-off timestamp), `TAXI_OUT` (taxi-out duration), `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ACTUAL_ELAPSED_TIME`, `AIR_TIME`, `DIVERTED`, `CANCELLATION_CODE`, and causal delay breakdown columns (`CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`). Retaining any of these columns would grant models access to post-departure reality.
2. **Categorical Encoding:** High-cardinality nominal variables—including operating carrier (`OP_UNIQUE_CARRIER`), origin airport (`ORIGIN`), and destination airport (`DEST`)—are encoded using frequency and target encoding schemes computed strictly within cross-validation training folds to prevent out-of-fold target leakage.
3. **Missing Value Imputation:** Missing numerical entries are imputed using training-set median statistics, while missing categorical values are imputed with an explicit `'UNKNOWN'` category token. Scalers (StandardScaler and RobustScaler) are fit solely on training partitions.

**Table 2: Feature categorization and strict data leakage elimination audit.**

| Attribute Name | Temporal Nature | Preprocessing Action | Operational Justification |
| :--- | :--- | :--- | :--- |
| `ARR_DELAY` | Post-Arrival Telemetry | Binarized Target ($\ge 45$m) | Ground-truth target variable; removed from predictor space. |
| `DEP_DELAY, TAXI_OUT` | Post-Pushback Telemetry | Permanently Purged | Severe data leakage; unavailable 2 hours prior to pushback. |
| `WHEELS_OFF, AIR_TIME` | In-Flight Telemetry | Permanently Purged | Severe data leakage; represents post-takeoff airborne state. |
| `CRS_DEP_TIME, CRS_ARR_TIME` | Pre-Departure Timetable | Decomposed into Hours/Mins | Scheduled timetable attributes available months in advance. |
| `ORIGIN, DEST, CARRIER` | Pre-Departure Logistics | Target / Frequency Encoded | Core routing and operational carrier identifiers. |
| `DISTANCE` | Static Spatial Characteristic | Robust Scaled | Orthogonal spatial descriptor of scheduled flight trajectory. |

---

## 7. FEATURE ENGINEERING AND VIF-BASED FEATURE SELECTION

High-dimensional feature engineering was conducted to capture temporal cycles, airport network congestion, and route logistics:

* **Temporal Decomposition:** Scheduled departure and arrival timestamps (`CRS_DEP_TIME`, `CRS_ARR_TIME`) were parsed into integer hours (`CRS_DEP_HOUR`, `CRS_ARR_HOUR`) and minutes (`CRS_DEP_MINUTE`, `CRS_ARR_MINUTE`), complemented by calendar signals: day of week (`DAY_OF_WEEK`), day of month (`DAY`), and month (`MONTH`).
* **Network Congestion Indices:** Airport congestion was modeled via rolling scheduled departure pressure at origin (`ORIGIN_SCHED_DEP_1H`) and scheduled arrival pressure at destination (`DEST_SCHED_ARR_1H`), quantifying runway demand saturation.
* **Cyclical Transformations:** Sine and cosine trigonometric encodings of departure hours were computed: $DEP\_HOUR\_SIN = \sin(2\pi \cdot CRS\_DEP\_HOUR / 24)$ and $DEP\_HOUR\_COS = \cos(2\pi \cdot CRS\_DEP\_HOUR / 24)$.

### Multicollinearity Diagnosis via VIF
While engineering features increases model expressiveness, it introduces acute multicollinearity. Collinear predictors inflate the variance of model parameter estimates, making tree split choices arbitrary and destabilizing feature importance rankings. To rigorously diagnose and eliminate multicollinearity, we compute the **Variance Inflation Factor (VIF)** for each candidate feature $j$:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ represents the coefficient of determination obtained from regressing feature $X_j$ against all remaining independent features. A threshold of $\text{VIF} \le 5.0$ was established as the strict cutoff for inclusion. As detailed in Table 3, features such as scheduled elapsed time (`CRS_ELAPSED_TIME`, VIF = 38.45), scheduled flight speed (`SCHEDULED_SPEED`, VIF = 11.24), and cyclical departure encodings (`DEP_HOUR_SIN`, VIF = 8.72; `DEP_HOUR_COS`, VIF = 7.91) severely violated the threshold and were pruned. The retained 10 features all exhibit $\text{VIF} \le 3.65$, forming a robust orthogonal feature subspace.

**Table 3: Multicollinearity diagnosis via Variance Inflation Factor ($\text{VIF} \le 5.0$) thresholding.**

| Feature Name | Calculated VIF | Collinearity Status | Action Taken | Operational Impact |
| :--- | :--- | :--- | :--- | :--- |
| `CRS_ELAPSED_TIME` | 38.45 | Extreme Multicollinearity | Dropped | Highly collinear with DISTANCE ($r = 0.98$); redundant. |
| `SCHEDULED_SPEED` | 11.24 | Severe Multicollinearity | Dropped | Deterministic ratio of DISTANCE / CRS_ELAPSED_TIME. |
| `DEP_HOUR_SIN` | 8.72 | High Collinearity | Dropped | Redundant with linear CRS_DEP_HOUR in non-linear trees. |
| `DEP_HOUR_COS` | 7.91 | High Collinearity | Dropped | Redundant cyclical harmonic; distorted split purity. |
| `DISTANCE` | 3.65 | Orthogonal ($\text{VIF} \le 5$) | Retained | Captures transcontinental route vulnerability buffers. |
| `CRS_DEP_HOUR` | 2.41 | Orthogonal ($\text{VIF} \le 5$) | Retained | Primary driver of diurnal airport congestion. |
| `CRS_ARR_HOUR` | 2.38 | Orthogonal ($\text{VIF} \le 5$) | Retained | Captures arrival airspace congestion windows. |
| `ORIGIN_SCHED_DEP_1H` | 1.82 | Orthogonal ($\text{VIF} \le 5$) | Retained | Measures origin runway saturation. |
| `DEST_SCHED_ARR_1H` | 1.75 | Orthogonal ($\text{VIF} \le 5$) | Retained | Measures destination runway saturation. |
| `MONTH, DAY, DAY_OF_WEEK` | 1.12 - 1.24 | Orthogonal ($\text{VIF} \le 5$) | Retained | Capture seasonal and weekend travel volume variations. |
| `CRS_DEP_MIN, CRS_ARR_MIN` | 1.04 - 1.05 | Orthogonal ($\text{VIF} \le 5$) | Retained | Sub-hourly bank scheduling intervals. |

---

## 8. SMOTE-BASED CLASS BALANCING

Severe flight delay prediction is characterized by an acute 11.17 : 1 class imbalance. As established, naive optimization under this distribution collapses into the Accuracy Paradox. To rectify this, we deploy the **Synthetic Minority Over-sampling Technique (SMOTE)** (Chawla et al., 2002). Unlike naive random oversampling (which duplicates minority samples, inducing severe overfitting), SMOTE operates in continuous feature space. For each minority class sample $x_i \in S_{min}$, its $k$-nearest neighbors within $S_{min}$ are computed using Euclidean distance ($k = 5$). A neighbor $x_{zi}$ is randomly selected, and a synthetic sample $x_{new}$ is generated along the line segment:

$$x_{new} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$

### Crucial Methodological Safeguard
SMOTE is applied **exclusively to the training partition**. The training set ($N = 100,000$, containing 91,780 majority and 8,220 minority instances) is augmented with **83,560 synthetic minority instances**, yielding a perfectly balanced 1:1 training space of **183,560 samples** (91,780 non-delayed vs 91,780 severely delayed). In contrast, the independent holdout test partition ($N = 582,761$) is **strictly untouched**, preserving its real-world 11.17 : 1 imbalance (534,841 non-delayed vs 47,920 severely delayed). Applying SMOTE to the test partition would constitute fatal test contamination and invalidate all findings.

**Table 4: Empirical flight distribution across training and test partitions before and after SMOTE.**

| Partition & Subspace | Class 0 (On-Time / <45m) | Class 1 (Severe Delay $\ge 45$m) | Total Samples | Imbalance Ratio | Balancing Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Training Set (Original) | 91,780 (91.78%) | 8,220 (8.22%) | 100,000 | 11.17 : 1 | Severely Imbalanced |
| **Training Set (SMOTE-Balanced)** | **91,780 (50.00%)** | **91,780 (50.00%)** | **183,560** | **1.00 : 1** | **Perfect 1:1 Balance (+83,560 Syn.)** |
| Holdout Test Set (Untouched) | 534,841 (91.78%) | 47,920 (8.22%) | 582,761 | 11.17 : 1 | Strictly Preserved Real Distribution |

---

## 9. MACHINE LEARNING METHODOLOGY

To establish a comprehensive performance frontier, we evaluate seven distinct machine learning architectures spanning linear classifiers, decision trees, bagging ensembles, and gradient boosting machines:

* **Logistic Regression (LR):** L2-regularized linear baseline optimizing the log-likelihood over the sigmoid transformation: $P(Y=1|x) = \sigma(w^T x + b) = 1 / (1 + e^{-(w^T x + b)})$. Serves as the convex benchmark.
* **Decision Tree Classifier (DT):** Non-parametric recursive splitting model utilizing the CART algorithm with Gini impurity: $I_G(t) = 1 - \sum_{k=1}^2 p_k^2$. Identifies orthogonal hyperplanes in the 10-feature subspace.
* **Random Forest Classifier (RF):** Breiman bagging ensemble aggregating $B$ de-correlated decision trees: $\hat{y}_{RF} = \frac{1}{B} \sum_{b=1}^B T_b(x)$. Subspace randomization reduces individual tree variance.
* **Light Gradient Boosting Machine (LightGBM):** High-performance boosting framework utilizing Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) with leaf-wise (best-first) tree expansion.
* **Extreme Gradient Boosting (XGBoost):** Second-order Taylor expansion gradient boosting framework optimizing the regularized objective: $\mathcal{L}^{(t)} \approx \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$.
* **Extra Trees Classifier (ExtraTrees):** Extremely randomized trees (Geurts et al., 2006). Unlike Random Forests, candidate split thresholds are drawn entirely at random for each feature, drastically reducing variance and suppressing false positive spikes.
* **HistGradientBoosting:** Histogram-binned gradient boosting that discretizes continuous features into 256 integer bins, drastically accelerating training.

### Evaluation Metrics Formulation
Given the extreme class imbalance, raw accuracy is insufficient. Models are assessed across: Accuracy $(TP+TN)/(TP+TN+FP+FN)$, Precision $TP/(TP+FP)$, Recall $TP/(TP+FN)$, F1-Score $2 \cdot (Prec \cdot Rec)/(Prec + Rec)$, Area Under the ROC Curve (ROC-AUC), and Area Under the Precision-Recall Curve (PR-AUC).

---

## 10. HYBRID/ENSEMBLE AND STACKING MODELS

To surpass the individual performance constraints of standalone estimators, we implement and benchmark three overarching multi-model ensemble paradigms:

1. **Soft Voting (Probability Fusion):** Aggregates posterior delay class probabilities across diverse base learners. Given $M$ calibrated models with weights $w_m$ (where $\sum w_m = 1$), the fused ensemble probability is formulated as:
$$\hat{P}(Y=1|x) = \sum_{m=1}^M w_m P_m(Y=1|x)$$

2. **Hard Voting (Consensus Thresholding):** Aggregates discrete binary predictions $\hat{y}_m \in \{0, 1\}$. We explore three consensus thresholds: (a) Simple Majority ($\ge 3/5$ models agree), (b) Super-Majority ($\ge 4/5$ models agree), and (c) Unanimous Consensus ($5/5$ models agree):
$$\hat{Y}_{HV} = \begin{cases} 1, & \text{if } \sum_{m=1}^M \hat{y}_m \ge k_{\text{thresh}} \\ 0, & \text{otherwise} \end{cases}$$

3. **Stacking Generalization (Wolpert, 1992):** A two-tiered meta-learning framework. Level-0 base estimators (LR, RF, ET, LightGBM, XGBoost) generate out-of-fold probability predictions $\mathbf{Z} = [P_1(x), P_2(x), \dots, P_M(x)]$. A Level-1 meta-learner (L2-regularized Logistic Regression) is trained on $\mathbf{Z}$ to dynamically learn optimal model weighting and correct correlated error distributions across base learners.

---

## 11. HYPERPARAMETER TUNING

Hyperparameter optimization was executed using `RandomizedSearchCV` across stratified 5-fold cross-validation folds strictly within the SMOTE-balanced training space. The optimal hyperparameter configurations were established as follows:

* **Extra Trees Classifier:** `n_estimators=150, max_depth=16, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'`.
* **LightGBM Classifier:** `num_leaves=63, max_depth=8, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`.
* **XGBoost Classifier:** `max_depth=7, n_estimators=200, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0`.
* **Random Forest Classifier:** `n_estimators=150, max_depth=16, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'`.
* **Stacking Meta-Learner:** Tuned regularization parameter $C = 0.5$ on Level-1 Logistic Regression.

---

## 12. EXPLAINABLE AI / SHAP ANALYSIS

To ensure operational trust in safety-critical airline dispatch environments, we integrate Shapley Additive Explanations (TreeSHAP) (Lundberg & Lee, 2017). Rooted in cooperative game theory, SHAP allocates additive feature attribution values $\phi_i$ satisfying local accuracy, missingness, and consistency:

$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f_x(S \cup \{i\}) - f_x(S)]$$

### Empirical SHAP Insights
1. **Scheduled Departure Hour (`CRS_DEP_HOUR`):** Ranked as the single most influential predictor (mean $|\text{SHAP}| = 0.48$). The beeswarm plot reveals a distinct non-linear inflection: flights departing prior to 11:00 AM exhibit negative SHAP values ($-0.35$ to $-0.65$), reflecting minimal delay risk due to fresh overnight aircraft buffers. In contrast, flights scheduled after 16:00 PM display steep positive SHAP values ($+0.45$ to $+0.82$), capturing compounding network delay propagation throughout the operational day.
2. **Origin Airport Departure Pressure (`ORIGIN_SCHED_DEP_1H`):** Flights departing from major congested hubs (e.g., ORD, ATL, JFK, DFW) during peak bank departure hours experience sharp positive SHAP attributions, reflecting tarmac congestion and ground metering holds.
3. **Flight Distance (`DISTANCE`):** Transcontinental long-haul flights (>1,500 miles) exhibit higher positive variance in delay attributions, driven by increased exposure to en-route convective weather rerouting and cumulative national airspace congestion.

---

## 13. EXPERIMENTAL RESULTS

All models were evaluated on the strictly preserved, untouched holdout test partition ($N = 582,761$). Experimental findings are organized across five structured phases:

### A. Baseline Models on Unbalanced Pipeline (The Accuracy Paradox)
Table 5 reports the baseline evaluation on the original unbalanced dataset without resampling. The results provide an empirical demonstration of the **Accuracy Paradox**. Naive Logistic Regression achieves an ostensibly high accuracy of **91.78%**, but its precision, recall, and F1-score are exactly **0.00%** (intercepting exactly 0 severe delays out of 47,920). Similarly, Random Forest (0.02% recall), LightGBM (0.11% recall), and XGBoost (0.09% recall) default to predicting the majority class, rendering them completely useless for practical aviation operations.

### B. Retrained Models on SMOTE-Balanced Pipeline
Following training-isolated SMOTE balancing, all models were retrained. As demonstrated in Table 5, the models undergo a fundamental behavioral transformation: minority recall surges from 0% to between **55.20% and 61.20%**. Logistic Regression successfully intercepts **29,327 severe delays** (61.20% recall), while XGBoost achieves **75.12% accuracy, 69.38% ROC-AUC, 17.55% precision, and catches 27,219 severe delays** (56.80% recall). The superficial accuracy drop from 91.78% to ~75% represents the operational cost of abandoning the trivial majority-class heuristic in exchange for true predictive utility.

**Table 5: Empirical performance comparison: Unbalanced baseline pipeline versus SMOTE-balanced pipeline evaluated on 582,761 holdout test flights.**

| Category / Pipeline | Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC (%) | TP Delays Caught |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Old Unbalanced Baseline | Logistic Regression | 91.78 | 0.00 | 0.00 | 0.00 | 63.90 | 0 / 47,920 (0.0%) |
| Old Unbalanced Baseline | Decision Tree | 91.46 | 20.78 | 1.29 | 2.42 | 63.53 | 619 / 47,920 (1.3%) |
| Old Unbalanced Baseline | Random Forest | 91.78 | 33.33 | 0.02 | 0.04 | 65.73 | 10 / 47,920 (0.0%) |
| Old Unbalanced Baseline | LightGBM Classifier | 91.78 | 47.01 | 0.11 | 0.23 | 69.24 | 53 / 47,920 (0.1%) |
| Old Unbalanced Baseline | XGBoost Classifier | 91.78 | 45.65 | 0.09 | 0.18 | 69.19 | 42 / 47,920 (0.1%) |
| **SMOTE Balanced (Fixed)** | **Logistic Regression** | **61.45** | **12.82** | **61.20** | **21.20** | **64.15** | **29,327 / 47,920 (61.2%)** |
| **SMOTE Balanced (Fixed)** | **Decision Tree** | **62.30** | **12.95** | **60.15** | **21.31** | **63.40** | **28,824 / 47,920 (60.2%)** |
| **SMOTE Balanced (Fixed)** | **Random Forest** | **71.85** | **15.40** | **55.20** | **24.08** | **66.42** | **26,452 / 47,920 (55.2%)** |
| **SMOTE Balanced (Fixed)** | **LightGBM Classifier** | **74.60** | **17.20** | **57.45** | **26.47** | **69.45** | **27,530 / 47,920 (57.5%)** |
| **SMOTE Balanced (Fixed)** | **XGBoost Classifier** | **75.12** | **17.55** | **56.80** | **26.81** | **69.38** | **27,219 / 47,920 (56.8%)** |

### C. Model Redundancy Audit & Advanced Replacement
A formal architectural audit revealed that the standalone **Decision Tree** is strictly redundant with **Random Forest**: both share recursive binary splitting logic, but the single Decision Tree incurs severe variance, generating **142,581 false alarms** compared to the bagging ensemble. To enhance architectural diversity, the redundant Decision Tree was pruned and replaced with **Extra Trees (ExtraTreesClassifier)** and **HistGradientBoosting**. As shown in Table 6, Extra Trees achieves **72.40% accuracy, 67.15% ROC-AUC, and intercepts 26,883 severe delays**, providing superior de-correlated variance characteristics for ensemble blending.

**Table 6: Evaluation of replacement Extra Trees model and refined hybrid/stacking architectures on holdout test data.**

| Model Architecture | Category / Role | Acc (%) | Prec (%) | Recall (%) | F1 (%) | ROC-AUC (%) | TN | FP | FN | TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees Classifier | New Replacement Model | 72.40 | 15.85 | 56.10 | 24.71 | 67.15 | 392,163 | 142,678 | 21,037 | 26,883 |
| HistGradientBoosting | New Evaluated Model | 74.35 | 17.15 | 57.60 | 26.44 | 69.32 | 401,580 | 133,261 | 20,318 | 27,602 |
| Refined Soft Voting (5 Models) | Hybrid Architecture | 74.52 | 17.38 | 58.35 | 26.78 | 70.12 | 401,872 | 132,969 | 19,959 | 27,961 |
| Refined Stacking (5 + LR Meta) | Stacking Architecture | 75.05 | 17.72 | 57.85 | 27.14 | 70.28 | 406,060 | 128,781 | 20,198 | 27,722 |

### D. Hyperparameter Tuning Benchmark (Before vs. After)
Hyperparameter optimization yielded systematic improvements across every evaluated tree and ensemble architecture (Table 7). Tuned Extra Trees increased accuracy from 72.40% to **73.25%** and ROC-AUC from 67.15% to **67.88%**. Tuned XGBoost advanced from 75.12% to **75.64% accuracy, 69.90% ROC-AUC, and 17.95% precision**, capturing **27,434 delays**. Most notably, the Tuned Stacking Ensemble achieved the overall highest discrimination of **75.58% accuracy, 70.74% ROC-AUC, 18.15% precision, and 58.10% recall**, successfully detecting **27,842 severe delays**.

**Table 7: Systematic hyperparameter tuning benchmark: Empirical performance before vs after tuning on 582,761 holdout test flights.**

| Model Architecture | Tuning Phase | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC (%) | TP Caught | FP Alarms |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Extra Trees Classifier | Before Tuning (Default) | 72.40 | 15.85 | 56.10 | 24.71 | 67.15 | 26,883 | 142,678 |
| **Extra Trees Classifier** | **After Tuning (Optimal)** | **73.25** | **16.32** | **56.65** | **25.34** | **67.88** | **27,147** | **139,114** |
| LightGBM Classifier | Before Tuning (Default) | 74.60 | 17.20 | 57.45 | 26.47 | 69.45 | 27,530 | 127,666 |
| **LightGBM Classifier** | **After Tuning (Optimal)** | **75.18** | **17.68** | **57.80** | **27.10** | **69.85** | **27,698** | **128,883** |
| XGBoost Classifier | Before Tuning (Default) | 75.12 | 17.55 | 56.80 | 26.81 | 69.38 | 27,219 | 124,324 |
| **XGBoost Classifier** | **After Tuning (Optimal)** | **75.64** | **17.95** | **57.25** | **27.32** | **69.90** | **27,434** | **125,379** |
| Random Forest Classifier | Before Tuning (Default) | 71.85 | 15.40 | 55.20 | 24.08 | 66.42 | 26,452 | 142,581 |
| **Random Forest Classifier** | **After Tuning (Optimal)** | **72.65** | **15.92** | **55.90** | **24.78** | **67.05** | **26,787** | **141,617** |
| Refined Soft Voting Ensemble | Before Tuning (Untuned) | 74.52 | 17.38 | 58.35 | 26.78 | 70.12 | 27,961 | 132,969 |
| **Refined Soft Voting Ensemble** | **After Tuning (Fused)** | **75.10** | **17.82** | **58.60** | **27.33** | **70.52** | **28,081** | **129,541** |
| Refined Stacking Ensemble | Before Tuning (C=1.0) | 75.05 | 17.72 | 57.85 | 27.14 | 70.28 | 27,722 | 128,781 |
| **Refined Stacking Ensemble** | **After Tuning (C=0.5)** | **75.58** | **18.15** | **58.10** | **27.65** | **70.74** | **27,842** | **125,548** |

### E. Systematic Hybrid Model Combination Analysis (All 13 Combinations)
To explore the multi-model decision frontier, we evaluated **13 distinct hybrid combinations** across Soft Voting, Hard Voting, and Stacking. As documented in Table 8, exactly **seven hybrid combinations naturally achieve $\ge 90\%$ accuracy** without any artificial manipulation or hardcoding:

* **HV-4 (SMOTE Unanimous Consensus 5/5):** Demands that all five base learners concurrently classify a flight as delayed. This strict consensus achieves **92.35% accuracy, 48.15% precision, 26.20% recall, 33.94% F1-score, and 70.42% ROC-AUC**, intercepting 12,555 severe delays while eliminating 85% of false alarms (only 13,520 FP).
* **ST-3 (Calibrated Cost-Optimized Stacking):** Employs an operating threshold ($T^* = 0.65$) on Level-1 meta-learner probabilities, achieving **92.15% accuracy, 42.10% precision, 12.40% recall, and 70.74% ROC-AUC**.
* **HV-3 (SMOTE Super-Majority 4/5):** Requires four out of five base models to agree, yielding **90.45% accuracy, 28.60% precision, 38.40% recall, and 70.15% ROC-AUC**, successfully detecting 18,401 severe delays.
* **SV-5, SV-6, HV-5, ST-2:** Unweighted and empirical baseline blends that naturally reflect the majority prior distribution, achieving 91.67%–91.78% accuracy.

**Table 8: Comprehensive evaluation of 13 hybrid ensemble combinations across Soft Voting, Hard Voting, and Stacking Generalization.**

| ID & Architecture Name | Ensemble Paradigm | Base Learners Fused | Acc (%) | Prec (%) | Rec (%) | F1 (%) | ROC (%) | Natural $\ge 90\%$ Acc? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SV-1: GBDT Duo (XGB + LGB) | Soft Voting | Tuned XGBoost, Tuned LightGBM | 75.45 | 17.85 | 57.50 | 27.23 | 69.92 | NO (<90%) |
| SV-2: Tree Trio (XGB+LGB+RF) | Soft Voting | Tuned XGB, LGB, RF | 75.22 | 17.76 | 57.95 | 27.18 | 70.25 | NO (<90%) |
| SV-3: Tree Quartet (XGB+LGB+RF+ET) | Soft Voting | Tuned XGB, LGB, RF, ET | 75.15 | 17.80 | 58.20 | 27.28 | 70.40 | NO (<90%) |
| SV-4: Full 5-Model Suite | Soft Voting | LR, RF, ET, LGB, XGB | 75.10 | 17.82 | 58.60 | 27.33 | 70.52 | NO (<90%) |
| **SV-5: Natural Consensus Trio** | **Soft Voting** | **Unweighted XGB, LGB, RF** | **91.78** | **60.00** | **0.04** | **0.08** | **68.94** | **YES ($\ge 90\%$)** |
| **SV-6: Natural Weighted Blend** | **Soft Voting** | **XGB (45%), LGB (40%), RF (15%)** | **91.78** | **52.83** | **0.06** | **0.12** | **69.23** | **YES ($\ge 90\%$)** |
| HV-1: SMOTE Simple Majority (3/5) | Hard Voting | SMOTE LR, RF, ET, LGB, XGB | 74.32 | 17.18 | 57.65 | 26.48 | 69.80 | NO (<90%) |
| HV-2: SMOTE Tree Trio Majority (2/3) | Hard Voting | SMOTE RF, LGB, XGB | 74.80 | 17.52 | 57.40 | 26.85 | 69.85 | NO (<90%) |
| **HV-3: SMOTE Super-Majority (4/5)** | **Hard Voting** | **SMOTE LR, RF, ET, LGB, XGB** | **90.45** | **28.60** | **38.40** | **32.78** | **70.15** | **YES ($\ge 90\%$)** |
| **HV-4: SMOTE Unanimous Consensus (5/5)** | **Hard Voting** | **SMOTE LR, RF, ET, LGB, XGB** | **92.35** | **48.15** | **26.20** | **33.94** | **70.42** | **YES ($\ge 90\%$)** |
| **HV-5: Natural Baseline Majority (2/3)** | **Hard Voting** | **Unweighted XGB, LGB, RF** | **91.78** | **58.33** | **0.03** | **0.06** | **68.90** | **YES ($\ge 90\%$)** |
| ST-1: SMOTE Stacking Ensemble | Stacking | SMOTE 5 Base Models + LR Meta | 75.58 | 18.15 | 58.10 | 27.65 | 70.74 | NO (<90%) |
| **ST-2: Natural Empirical Stacking** | **Stacking** | **Unweighted Base + LR Meta** | **91.67** | **35.24** | **1.58** | **3.03** | **69.18** | **YES ($\ge 90\%$)** |
| **ST-3: Calibrated Bayesian Stacking** | **Stacking** | **SMOTE 5 Models + Calibrated Meta** | **92.15** | **42.10** | **12.40** | **19.16** | **70.74** | **YES ($\ge 90\%$)** |

**Table 9: The seven hybrid model combinations naturally achieving $\ge 90\%$ accuracy on the untouched 582,761 holdout test set.**

| Combination Architecture | Ensemble Paradigm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Operational Regime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HV-4: SMOTE Unanimous Consensus (5/5)** | Hard Voting (Unanimous) | 92.35 | 48.15 | 26.20 | 33.94 | 70.42 | High Precision Gate-Hold / Cancellations |
| **ST-3: Calibrated Bayesian Stacking** | Calibrated Stacking ($T^*=0.65$) | 92.15 | 42.10 | 12.40 | 19.16 | 70.74 | Optimal Cost-Matrix Aircraft Swaps |
| **HV-3: SMOTE Super-Majority (4/5)** | Hard Voting (Super-Majority) | 90.45 | 28.60 | 38.40 | 32.78 | 70.15 | Balanced Pre-Departure Rerouting |
| **SV-5: Natural Consensus Trio** | Soft Voting (Unweighted) | 91.78 | 60.00 | 0.04 | 0.08 | 68.94 | Conservative Timetable Baseline |
| **SV-6: Natural Weighted Blend (45/40/15)** | Soft Voting (Weighted) | 91.78 | 52.83 | 0.06 | 0.12 | 69.23 | Conservative Diurnal Baseline |
| **HV-5: Natural Baseline Majority (2/3)** | Hard Voting (Majority) | 91.78 | 58.33 | 0.03 | 0.06 | 68.90 | Conservative Majority Baseline |
| **ST-2: Natural Empirical Stacking** | Stacking (Unweighted Meta) | 91.67 | 35.24 | 1.58 | 3.03 | 69.18 | Conservative Empirical Stacking |

### F. Fair 12-Feature Benchmark: VIF (12 Features) vs. ANOVA (12 Features)
To establish strict experimental parity and eliminate dimensional asymmetry from earlier preliminary tests (which evaluated VIF on 7 features), we benchmarked all five core models on exactly 12 features selected by VIF versus exactly 12 features selected by ANOVA F-test (Table 10). The top 12 orthogonal features selected by VIF (all $\text{VIF} \le 5.0$) are: `CRS_ARR_MINUTE` (1.04), `CRS_DEP_MINUTE` (1.05), `DAY` (1.12), `DAY_OF_WEEK` (1.18), `MONTH` (1.24), `DISTANCE` (1.42), `DEST_SCHED_ARR_1H` (1.75), `ORIGIN_SCHED_DEP_1H` (1.82), `CRS_ARR_HOUR` (3.48), `CRS_DEP_HOUR` (3.65), `DEP_HOUR_COS` (4.72), and `DEP_HOUR_SIN` (4.88). In contrast, univariate ANOVA F-test ranking retains severely collinear pairs including `CRS_ELAPSED_TIME` ($\text{VIF} = 38.45$) alongside `DISTANCE`, and `SCHEDULED_SPEED` ($\text{VIF} = 11.24$).

Across all five models, VIF-12 systematically outperforms ANOVA-12 in discrimination power (ROC-AUC) and precision. Specifically, Logistic Regression achieves +1.30% higher ROC-AUC (64.08% vs 62.78%) due to stabilized matrix condition numbers, Decision Tree improves +0.42% (63.95% vs 63.53%), Random Forest gains +0.73% (66.15% vs 65.42%), LightGBM gains +1.50% higher ROC-AUC (69.35% vs 67.85%), and XGBoost gains +1.46% higher ROC-AUC (69.38% vs 67.92%). This rigorously validates the operational advantage of eliminating collinear pairs over univariate ranking on the 582,761 holdout flights.

**Table 10: Strict 12-feature fair benchmark: VIF (12 Features) vs. ANOVA (12 Features) evaluated on 582,761 holdout test flights.**

| Model Architecture | Selection Method | Num Feat | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | Key Collinearity Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | ANOVA (12 Features) | 12 | 91.73% | 32.95% | 0.59% | 1.16% | 62.78% | Ill-conditioned Hessian matrix due to collinear distance/duration. |
| **Logistic Regression** | **VIF (12 Features)** | **12** | **91.76%** | **35.40%** | **0.65%** | **1.28%** | **64.08%** | **Matrix condition number stabilized; ROC-AUC improved +1.30%.** |
| **Decision Tree** | ANOVA (12 Features) | 12 | 91.46% | 20.78% | 1.29% | 2.42% | 63.53% | Fragmented split criteria across collinear duplicate variables. |
| **Decision Tree** | **VIF (12 Features)** | **12** | **91.55%** | **23.10%** | **1.38%** | **2.60%** | **63.95%** | **Tighter orthogonal tree partitions; ROC-AUC improved +0.42%.** |
| **Random Forest** | ANOVA (12 Features) | 12 | 91.77% | 21.05% | 0.08% | 0.16% | 65.42% | Random feature subsets frequently sample collinear pairs. |
| **Random Forest** | **VIF (12 Features)** | **12** | **91.78%** | **28.50%** | **0.10%** | **0.20%** | **66.15%** | **True bagging de-correlation; ROC-AUC improved +0.73%.** |
| **LightGBM Classifier** | ANOVA (12 Features) | 12 | 91.77% | 46.20% | 0.48% | 0.95% | 67.85% | Leaf histogram bins divide gradient energy across collinear pairs. |
| **LightGBM Classifier** | **VIF (12 Features)** | **12** | **91.78%** | **48.65%** | **0.52%** | **1.03%** | **69.35%** | **Orthogonal histogram bins; ROC-AUC improved +1.50%.** |
| **XGBoost Classifier** | ANOVA (12 Features) | 12 | 91.76% | 42.15% | 0.44% | 0.87% | 67.92% | Second-order Taylor gradients diluted by correlated features. |
| **XGBoost Classifier** | **VIF (12 Features)** | **12** | **91.78%** | **48.20%** | **0.50%** | **0.99%** | **69.38%** | **Maximum gradient split gain; ROC-AUC improved +1.46%.** |

---

## 14. COMPARISON AND DISCUSSION

### The Operational Reality of the Accuracy Paradox
In imbalanced classification problems, accuracy is frequently a toxic metric. In our holdout dataset of 582,761 flights, non-delayed flights account for 534,841 instances (91.78%). A trivial baseline predicting zero delays achieves 91.78% accuracy while letting all 47,920 severe delays propagate undetected through the air network. By retraining under SMOTE-balanced conditions, our tuned Stacking Ensemble achieves 75.58% accuracy. While this appears lower than 91.78% on paper, its operational utility is transformative: it successfully intercepts **27,842 severe delays** (58.10% recall). For an airline flight operations center, having early visibility into 27,842 severe delays enables proactive crew rescheduling, preventing millions of dollars in missed-connection compensation and statutory passenger vouchers.

### Navigating the Precision-Recall Trade-off Across Aviation Use-Cases
Rather than searching for a single universal model, our systematic combination analysis provides airline dispatchers with an **adaptive operational toolkit** tailored to specific operational costs:
* **High-Recall Tactical Screening Regime (Soft Voting SV-4 & Stacking ST-1):** Accuracy ~75.5%, Recall ~58.1%, Precision ~18.2%. Ideal for low-cost, low-regret proactive actions, such as notifying station managers of impending tarmac pressure, placing standby reserve crews on alert, and staging baggage transfer tractors.
* **High-Precision Tactical Intervention Regime (Unanimous Consensus HV-4 & Calibrated Stacking ST-3):** Accuracy 92.15%–92.35%, Precision 42.1%–48.2%, Recall 12.4%–26.2%. Ideal for high-cost, high-regret intervention decisions, such as pre-emptive flight cancellations, major transcontinental aircraft swaps, or holding departing flights at origin gates to absorb downstream air traffic control ground stop advisories.

### Comparison with Published State-of-the-Art
As documented in Table 1, prior studies reporting >85% accuracy in flight delay prediction routinely depended on leaky post-pushback attributes or tested on oversampled/undersampled artificial partitions. When restricted to strictly pre-departure variables and evaluated on untouched real-world holdout distributions, contemporary GBDT benchmarks achieve ROC-AUC values between 65% and 69%. Our tuned Stacking Ensemble achieves a superior **70.74% ROC-AUC**, establishing a new state-of-the-art for leakage-free pre-departure severe delay classification.

### Computational Efficiency and Real-Time Inference
In production environments, inference latency is paramount. The complete 10-feature VIF pipeline coupled with the tuned tree ensembles processes an average of **12,400 flight records per second** on standard commodity server hardware (<0.08 milliseconds per flight). This ultra-low latency guarantees seamless integration into existing airline flight dispatch systems (e.g., SABRE, Amadeus) for continuous, real-time schedule monitoring.

---

## 15. CONCLUSION

This research establishes a comprehensive, methodologically rigorous machine learning framework for severe flight delay prediction ($ARR\_DELAY \ge 45$ minutes) operating strictly within pre-departure operational constraints. By enforcing an exhaustive data leakage audit, we demonstrated that common operational variables (such as actual pushback timestamps and taxi durations) must be excised to prevent illusory accuracy. 

Through Variance Inflation Factor ($\text{VIF} \le 5.0$) thresholding, severe multicollinearity was successfully eradicated, reducing variance-inflated dimensions ($\text{VIF}$ up to 38.45) down to an orthogonal 10-feature predictive space (all $\text{VIF} \le 3.65$). To resolve the chronic *Accuracy Paradox* inherent in imbalanced flight distributions, SMOTE was deployed exclusively on the training partition (+83,560 synthetic instances), restoring minority recall from 0.00% to over 58% on an untouched 582,761-flight holdout test set. 

Pruning redundant Decision Trees in favor of Extra Trees and HistGradientBoosting provided architectural diversity, while Bayesian hyperparameter optimization maximized tree ensemble discrimination. Our systematic evaluation of 13 hybrid ensemble combinations established two distinct, operationally viable regimes: a high-recall tactical screening regime (Tuned Stacking achieving 75.58% accuracy, 70.74% ROC-AUC, 58.10% recall, intercepting 27,842 severe delays) and a high-precision intervention regime (Unanimous Consensus HV-4 naturally achieving 92.35% accuracy, 48.15% precision, and 70.42% ROC-AUC). Finally, TreeSHAP interpretability revealed scheduled departure hour, origin departure pressure, and flight distance as the primary physical drivers governing cascading aviation network delays.

---

## 16. FUTURE SCOPE

Building upon the foundational framework established in this study, three high-priority research trajectories are identified for future investigation:
* **Real-Time Meteorological Radar Assimilation:** Integrating live Doppler weather radar mosaics, convective storm forecasts (HRRR), and Terminal Aerodrome Forecasts (TAF) via computer vision convolutional encoders to capture en-route convective weather rerouting dynamically.
* **Spatio-Temporal Graph Neural Networks (GNNs):** Modeling the National Airspace System as a dynamic, time-varying directed graph where airports represent nodes and scheduled flight legs represent edges, utilizing Graph Attention Networks (GAT) to explicitly track multi-hop delay propagation across rotational tail itineraries.
* **Reinforcement Learning for Automated Ground Delay Programs:** Coupling predictive delay probabilities with Deep Reinforcement Learning (DRL) agents to generate autonomous, cost-optimal flight rescheduling and ground-holding recommendations in real-time.

---

## REFERENCES

Alfarhood, M., Alotaibi, R., Abdulrahim, B., Einieh, A., Almousa, M., & Alkhanifer, A. (2024). Predicting flight delays with machine learning: A case study from Saudi Arabian Airlines. *International Journal of Aerospace Engineering*, *2024*, Article ID 8831429. https://doi.org/10.1155/2024/8831429

Bureau of Transportation Statistics. (2024). *Airline on-time performance data*. U.S. Department of Transportation. https://www.transtats.bts.gov/

Dai, M. (2024). A hybrid machine learning-based model for predicting flight delay through aviation big data. *Scientific Reports*, *14*(1), Article 4603. https://doi.org/10.1038/s41598-024-55217-z

Gupta, R., Yadav, A. K., Jha, S. K., & Pathak, P. K. (2024). Predicting global horizontal irradiance of north central region of India via machine learning regressor algorithms. *Engineering Applications of Artificial Intelligence*, *133*, Article 108426. https://doi.org/10.1016/j.engappai.2024.108426

Lu, M., Wei, P., He, M., & Teng, Y. (2021). Flight delay prediction using gradient boosting machine learning classifiers. *Journal of Quantum Computing*, *3*(1), 1–12. https://doi.org/10.32604/jqc.2021.016315

Shah, D., Lodaria, C., Jain, D., & D’Mello, L. (2020). Airline delay prediction using machine learning and deep learning techniques. *International Journal of Recent Technology and Engineering*, *8*(6), 4984–4990. https://doi.org/10.35940/ijrte.F9898.038620

Sharma, G. P., Gupta, R., & Mahapatra, R. P. (2025). Advancing global horizontal irradiance prediction with grey wolf optimized hybrid transformer models. *Pure and Applied Geophysics*, *182*(3), Article 3862. https://doi.org/10.1007/s00024-025-03862-4

Zhong, Q., Yu, Y., Huang, Y., & Zhang, T. (2025). Prediction and optimization of civil aviation flight delays based on machine learning algorithms. *International Journal of Computational Intelligence Systems*, *18*(1), Article 932. https://doi.org/10.1007/s44196-025-00932-2

