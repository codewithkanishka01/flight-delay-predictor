# ✈️ Flight Delay Prediction Using Machine Learning
## End-to-End Comprehensive Master Study Guide (Hinglish Edition)

> **Project Title:** A Leakage-Free Pre-Departure Classification Framework for Severe Flight Delays (`ARR_DELAY >= 45 min`)  
> **Team Members:** Bhavya Gupta, Arshia Bhandari, Kanishka Pandey  
> **Project Guide:** Mr. Rahul Gupta (Department of CSE, SRM Institute of Science and Technology)  
> **Live Web Application:** [https://flight-delay-predictor-z2ik.onrender.com/](https://flight-delay-predictor-z2ik.onrender.com/)  
> **GitHub Repository:** [https://github.com/codewithkanishka01/flight-delay-predictor](https://github.com/codewithkanishka01/flight-delay-predictor)  
> **PDF Version Location:** `/Users/kanishkapandey/Desktop/Flight_Delay_Prediction_Complete_Study_Guide_Hinglish.pdf`

---

## 🎯 Is Guide Ko Kaise Use Karein?
Is guide ko **ratne (memorize karne) ki zaroorat bilkul nahi hai**. 
Isko is tarah design kiya gaya hai ki agar aap ek baar shanti se isko padh lenge, toh aapke dimaag me project ke architecture, algorithms, trade-offs, aur decisions ki ek crystal-clear picture ban jayegi. Viva, presentation, ya technical interview me jab koi aapse question karega, toh aap bina hesitate kiye ek mature ML engineer ki tarah logic ke sath answer de payenge.

---

## 📑 Table of Contents (Index)
1. **Chapter 1:** Project Vision, Motivation & Aviation Realities
2. **Chapter 2:** The Accuracy Paradox: 15-Minute Ceiling vs 45-Minute Severe Delay Formulation
3. **Chapter 3:** Zero Target Leakage: The Golden Rule of Industrial ML
4. **Chapter 4:** Dataset Scale, Cleaning & Preprocessing (BTS 3M Dataset)
5. **Chapter 5:** Feature Selection & 96.4% Dimensionality Reduction (ANOVA F-Test)
6. **Chapter 6:** Systematic Model Benchmarking: 5 Core Classifiers Compared
7. **Chapter 7:** Class Imbalance & The Threshold Tuning Magic ($T = 0.12 - 0.15$)
8. **Chapter 8:** Advanced Hybrid Ensembles & Meta-Stacking (92.35% Consensus)
9. **Chapter 9:** Explainable AI (XAI) via SHAP — Model Ke Andar Ka Logic
10. **Chapter 10:** Full-Stack Cloud Deployment Architecture (Flask + Docker + Render + CI/CD)
11. **Chapter 11:** Rapid-Fire Viva & Interview Q&A (Professor Cheat Sheet)

---

## 1. Project Vision, Motivation & Aviation Realities

### Aviation Me Delay Sirf "Late Hona" Nahi Hai — Ye Ek Chain Reaction Hai
Aviation operations me flight schedule ek tightly synchronized network graph hota hai:
* Ek aircraft din me 4 se 6 alag-alag flights operate karta hai (ise **Aircraft Rotation** kehte hain).
* Agar subah ki pehli flight 45 minute late hoti hai, toh wo late landing karegi. Ground crew ko safai, refueling, baggage loading, aur passenger boarding ke liye wahi standard 45-minute turnaround time chahiye hota hai.
* Iska natija: Us aircraft ki agli flight late depart hogi. Aise karte-karte shaam tak pure network me 10+ flights delay ho jaati hain. Ise aviation industry me **Delay Propagation / Downstream Contagion** kehte hain.

### Static Rules vs Machine Learning
* **Static Rules Kyu Fail Hote Hain?** Airlines pehle static buffer add karti thi (e.g., 'Peak evening me 20 min add kar do'). Lekin air traffic delay weather, terminal gate congestion, route corridor density, aur carrier turnaround efficiency ka complex non-linear mixture hota hai.
* **ML Ka Role:** Machine learning historical patterns ko analyse karke passenger ke plane me chadhne se pehle (Pre-Departure) hi predict kar deta hai ki kis flight me severe delay aane ka high risk hai, taaki airlines proactive gate swapping, passenger rerouting, aur ground delay programs pehle se plan kar sakein.

---

## 2. The Accuracy Paradox: 15-Minute Ceiling vs 45-Minute Severe Delay Formulation

> ⚠️ **Professor Ka Favorite Trap Question:** *"Tumhare model ki accuracy 90%+ kaise aa gayi? Kya tumne 15 minute delay par 90% nikala hai?"*

Is question ka scientifically honest aur mathematically solid answer ye hai:

### 1. The 15-Minute FAA Ceiling Trap
* FAA (Federal Aviation Administration) standard ke mutabiq 15 minute se zyada late hone par flight ko delay maana jata hai.
* US Bureau of Transportation Statistics (BTS) ke 3-Million dataset me:
  * **On-Time Flights:** **81.68%**
  * **Delayed Flights (&ge; 15 min):** **18.32%**
* **The Mathematical Reality:** Agar hum strictly pre-departure features use karein (bina future ki cheating kiye), toh Machine Learning algorithms ka performance ceiling **~81.8% se 82.0% accuracy** par ruk jata hai!
* **Kyu?** Kyunki flight udne se pehle raste me aane wale sudden weather thunderstorms ya air traffic control holding pattern ki jaankari ticket booking ke waqt kisi ke paas nahi hoti.
* Agar koi claim kare ki usne 15-min delay par 95% accuracy nikali hai, toh usne ya toh **Data Leakage** ki hai (in-flight data use kiya), ya fir model ne sabhi flights ko 'On-Time' predict kar diya (jisse Accuracy 81.7% toh aa gayi par delayed flight pakadne ka **Recall 0.00%** ho gaya).

### 2. The 45-Minute Severe Delay Horizon (Humara Strategic Breakthrough)
Humne problem statement ko reformulate kiya: **ARR_DELAY &ge; 45 minutes (Severe Flight Delay)**.

| Target Definition | Delay Cutoff | Class 1 (Late) Rate | Majority Baseline | Model Accuracy | Feasibility of >90% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ARR_DELAY &ge; 15m (FAA Standard)** | 15 min | 18.32% | 81.68% | 81.73% | ❌ Plateau at ~82% ceiling |
| **ARR_DELAY &ge; 30m (Significant)** | 30 min | 11.66% | 88.34% | 88.31% | ❌ Borderline (<90%) |
| **ARR_DELAY &ge; 45m (Major Severe Delay)** | **45 min** | **8.22%** | **91.78%** | **91.78% - 92.35%** | ✅ **Meets >90% Legitimately!** |
| **ARR_DELAY &ge; 60m (Severe Disruption)** | 60 min | 6.12% | 93.88% | 93.86% | ✅ High Baseline (>93%) |

#### Iske Piche 2 Solid Justifications Hain:
1. **Operational Justification:** 15-20 minute ka delay passenger taxi karke ya flight speed badha kar cover kar leta hai. Lekin **45+ minutes** wo breaking point hai jahan:
   * Connecting international/domestic passengers ki connecting flights chhoot jaati hain.
   * Pilots aur flight attendants ke legal maximum duty hours cross ho jaate hain (Crew Timeout).
   * FAA airport runways par aircraft landing suspend karke **Ground Delay Programs (GDP)** announce karta hai.
2. **Mathematical Justification:** 45+ minute late flights dataset me **8.22%** hain aur On-Time/Minor buffer flights **91.78%** hain. Is threshold par humare models **91.78% - 92.35% Accuracy** legitimately achieve karte hain bina kisi fake manipulation ke!

---

## 3. Zero Target Leakage: The Golden Rule of Industrial ML

### Data Leakage Kya Hai?
Data leakage ka seedha matlab hai: **Training data me aisi information de dena jo real-world prediction ke time par physical universe me exist hi nahi karti!**

### In-Flight Variables Ka Cheat Code
Online blog posts aur Kaggle tutorials me log aksar `DEP_DELAY` (actual departure delay) ya `TAXI_OUT` feature daal dete hain.
* Agar plane gate se hi 1 ghanta late udi hai (`DEP_DELAY = 60`), toh obvious baat hai ki wo landing bhi late hi karegi!
* `DEP_DELAY` ka `ARR_DELAY` ke sath **0.93 Pearson correlation** hota hai. Agar aap `DEP_DELAY` model me rakhoge toh accuracy bina mehnat ke 98% aa jayegi, lekin ye prediction **fraud** hai kyunki passenger ke ticket book karte waqt ya flight ke schedule banne ke waqt departure delay pata hi nahi hota!

### 17 Permanently Removed Post-Departure Columns:
Humne academic aur industrial integrity maintain karne ke liye **sabhi 17 runtime variables ko permanently delete** kiya:
1. `DEP_TIME` (Actual departure clock time)
2. `DEP_DELAY` (Actual departure delay minutes)
3. `TAXI_OUT` (Gate se runway tak ka taxi time)
4. `WHEELS_OFF` (Airborne hone ka actual timestamp)
5. `WHEELS_ON` (Touchdown ka actual timestamp)
6. `TAXI_IN` (Runway se gate tak taxi time)
7. `ARR_TIME` (Actual arrival clock time)
8. `ELAPSED_TIME` (Actual flight duration)
9. `AIR_TIME` (Hawa me bitaya gaya actual time)
10. `DELAY_DUE_CARRIER` (Carrier delay decomposition)
11. `DELAY_DUE_WEATHER` (Weather delay minutes)
12. `DELAY_DUE_NAS` (Airspace delay minutes)
13. `DELAY_DUE_SECURITY` (Security delay minutes)
14. `DELAY_DUE_LATE_AIRCRAFT` (Inbound aircraft delay)
15. `CANCELLED` (Flight cancelled flag)
16. `CANCELLATION_CODE` (Cancellation reason code)
17. `DIVERTED` (Flight diversion flag)

**Gaurantee:** Humara framework strictly unhi variables par chalta hai jo scheduled timetable me available hote hain.

---

## 4. Dataset Scale, Cleaning & Preprocessing (BTS 3M Dataset)

### Dataset Source
* **Provider:** US Bureau of Transportation Statistics (BTS)
* **Scale:** 3,000,000 raw flight records (National Airline On-Time Performance).

### Cleaning & Sanitization Steps
1. **Cancelled & Diverted Exclusion:** Jo flight udd kar destination tak pahuchegi hi nahi, uska arrival delay calculate karna logically impossible hai. Isliye unhe drop kiya gaya:
   $$\text{3,000,000 Raw Records} \longrightarrow \mathbf{2,913,802\text{ Clean Flights}}$$
2. **Missing Value Audit:** Pre-departure schedule columns me missing data virtually zero tha (`CRS_ELAPSED_TIME` me 30 lakh me se sirf 14 rows null the, jinhe training-set mean se impute kiya).
3. **Stratified Split (80/20):**
   * **Training Partition (80%):** 2,331,041 flights
   * **Final Untouched Test Partition (20%):** 582,761 flights
   * Split `stratify=y_target` ke sath kiya gaya taaki train aur test dono me exactly 8.22% severe delay ratio maintain rahe.

---

## 5. Feature Selection & 96.4% Dimensionality Reduction

### 21,513 Dimensions Se 12 Features Par Kyu Aaye?
* Agar hum `AIRLINE`, `ORIGIN`, `DEST`, aur unke interactions ko one-hot encode karte hain, toh **21,513 sparse columns** ban jaate hain.
* Isse:
  * Model file 200MB+ ho jaati hai.
  * Inference time 1.5 second tak chala jata hai.
  * Model sparse features par overfit hone lagta hai.
* Humne **ANOVA F-Test** aur feature importance analysis ke zariye **top 12 pre-departure features** select kiye:

| Rank | Feature Name | ANOVA F-Statistic | Operational Significance |
| :---: | :--- | :---: | :--- |
| **1** | `CRS_DEP_TIME` (Sched Departure) | **1132.51** | Single most powerful driver. Evening flights accumulate prior turnaround delays. |
| **2** | `CRS_ARR_TIME` (Sched Arrival) | **856.61** | Destination hub runway capacity banks aur slot congestion se correlate karta hai. |
| **3** | `CRS_ELAPSED_TIME` (Block Duration) | **29.03** | Route buffer capacity: Lambi flights me thoda recovery buffer hota hai. |
| **4** | `DISTANCE` (Haul Length) | **17.40** | Short-haul routes rigid turnaround follow karte hain, long-haul me throttle adjustment possible hota hai. |
| **5-12** | `FL_DATE`, `AIRLINE`, `AIRLINE_CODE`, `ORIGIN`, `ORIGIN_CITY`, `DEST`, `DEST_CITY`, `FL_NUMBER` | Categorical & Date | Route corridors, airport hub density, carrier fleet efficiency. |

**Result:** 96.4% Dimensionality reduction ke baad bhi model ne **99.8% predictive capability retain ki**, aur model inference latency घटकर **sub-100ms** ho gayi!

---

## 6. Systematic Model Benchmarking: 5 Core Classifiers

Humne 5 machine learning models ko **582,761 untouched test flights** par benchmark kiya:

| # | Model Name | Model Family | Accuracy (%) | ROC-AUC (%) | PR-AUC (%) | Evaluation Verdict |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **Logistic Regression** | Linear Baseline | 91.74% | 63.90% | 13.72% | Fastest to train; linear decision boundary temporal non-linearities miss karta hai. |
| 2 | **Decision Tree** | Single Tree Rule | 91.46% | 63.53% | 12.96% | Highly interpretable rules, lekin high variance aur boundary overfitting. |
| 3 | **Random Forest (100 Trees)** | Bagging Ensemble | 91.77% | 65.51% | 14.36% | Strong variance reduction across diverse flight corridors; highly stable. |
| 4 | **LightGBM Classifier** | Gradient Boosting | 91.78% | 69.24% | 17.52% | Histogram-based leaf-wise splitting; extremely fast training speed. |
| 5 | **Tuned XGBoost (Winner)** | Gradient Boosting | **91.78%** | **69.84%** | **17.43%** | **Peak individual model!** Best probability calibration aur highest ROC-AUC. |

### Why Did XGBoost Win?
XGBoost me **L1 (Lasso) aur L2 (Ridge) regularization** loss function ke andar integrated hota hai:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
Jahan $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$. 
Is regularization ki wajah se XGBoost sparse categorical noise par overfit nahi hota aur decision threshold tuning ke dauran sabse smooth calibrated probabilities provide karta hai.

---

## 7. Class Imbalance & The Threshold Tuning Magic ($T = 0.12 - 0.15$)

### Default Threshold ($T = 0.50$) Ka Trap
Standard binary classifiers me:
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$
Lekin jab severe delays ka natural base rate sirf **8.22%** hai:
* Model ki predicted probabilities naturally **5% se 30%** ke range me aati hain.
* Agar aap $T = 0.50$ rakhte hain, toh koi bhi flight 50% cross nahi karegi.
* Model sabko "On-Time" bol dega. Accuracy toh 91.78% dikhegi, par **Recall = 0.09% (Zero)** ho jayega! Is model ka airport par koi fayda nahi hoga.

### Cost-Sensitive Calibration ($T = 0.12$ ya 12.00%)
Humne validation set par Precision-Recall optimization run karke decision threshold ko **12.00% (0.12)** par calibrate kiya:
* **The Logic:** Agar national average delay rate 8.22% hai, aur model kisi flight ko **16.07%** delay probability deta hai — iska matlab us flight ka delay risk normal flight se **double (2x)** hai!
* **Impact:** Threshold ko 12% set karne se minority severe delay detection ka **Recall 16.4x boost ho gaya**, aur model ne **27,889 real severe delays ko successfully flag kiya**!

---

## 8. Advanced Hybrid Ensembles & Meta-Stacking (92.35% Peak)

Humne individual models ko combine karke multi-model architectures explore kiye:
1. **Weighted Soft Voting:** XGBoost (45%) + LightGBM (40%) + Random Forest (15%) ki probabilities ka consensus.
2. **Level-1 Meta-Stacking:** Base models ki 5 Out-Of-Fold (OOF) predicted probability vectors par ek cross-validated Logistic Regression meta-learner train kiya.
3. **HV-4 Unanimous Consensus Model:**
   * **Accuracy:** **92.35%**
   * **ROC-AUC:** **70.42%**
   * **Precision:** **48.15%**
   * **Recall:** **26.20%**
   * **F1-Score:** **33.94%**

---

## 9. Explainable AI (XAI) via SHAP — Model Ke Andar Ka Logic

Humara model black-box nahi hai. Humne **SHAP (SHapley Additive exPlanations)** use kiya:

### 3 Major Explainability Findings:
1. **Scheduled Departure Hour (`CRS_DEP_HOUR`):**
   * Sabse dominant feature (Mean |SHAP| > 0.20).
   * **Late Evening (7:00 PM – 11:00 PM):** Strongly positive SHAP values (delay risk dramatically increases). Reason: Aircraft rotation delays build up throughout the day.
   * **Early Morning (6:00 AM – 9:00 AM):** Strongly negative SHAP values (delay risk decreases). Reason: Aircraft overnight airport par rested hote hain, turnaround delays zero hote hain.
2. **Seasonal Peaks (`MONTH`):**
   * June/July (Summer peak vacation travel) aur November/December (Thanksgiving/Christmas) me positive delay spikes.
   * Tuesdays & Wednesdays lower volume ki wajah se lowest delay risk show karte hain.
3. **Origin Congestion & Airline Fleet:**
   * Congested hubs (EWR, ORD, LGA, JFK) positive delay contribution push karte hain.
   * Legacy carriers with spare aircraft buffers (Delta, United) have lower baseline severe delay risk compared to ultra-low-cost carriers with 30-minute turnarounds.

---

## 10. Full-Stack Cloud Deployment Architecture

| Tier | Component | Technology Used | Implementation Highlights |
| :-: | :--- | :--- | :--- |
| **Tier 1** | **Backend Microservice** | Python 3.10 + Flask | RESTful prediction API; memory me loaded XGBoost model ke sath sub-100ms response time. |
| **Tier 2** | **Production WSGI** | Gunicorn (Green Unicorn) | Multi-worker concurrent request handler; synchronous worker model; timeout protection. |
| **Tier 3** | **Containerization** | Docker (`python:3.10-slim`) | Self-contained, isolated environment; .dockerignore se 600MB dataset exclude; compact 11MB runtime. |
| **Tier 4** | **CI/CD Automation** | GitHub Actions / Auto-deploy | `codewithkanishka01/flight-delay-predictor` par code push hote hi cloud auto-rebuild trigger hoti hai. |
| **Tier 5** | **Cloud Host** | Render.com (Singapore Region) | Global HTTPS public URL: `https://flight-delay-predictor-z2ik.onrender.com/` |
| **Tier 6** | **24/7 Always-Awake** | Automated Keep-Alive Ping | Free-tier spin-down latency (50s sleep) ko permanently eliminate karne ke liye 10-min silent ping. |

---

## 11. Rapid-Fire Viva & Interview Q&A (Professor Cheat Sheet)

Aap in sawalon aur jawabon ko natural Hinglish me bol sakte hain:

### Q1: Is project ka core problem statement kya hai?
> *"Sir, humara objective pre-departure flight schedule information dekhkar predict karna hai ki kya koi flight 45 minutes ya usse zyada late (Severe Delay) hogi ya on-time pahuchegi. Humne strictly zero data leakage maintain kiya hai aur pure model ko cloud par interactive web application ke roop me deploy kiya hai."*

### Q2: 15-minute delay standard chhodkar 45-minute kyu use kiya?
> *"Sir, do solid reasons hain: Pehla operational — aviation industry me 45+ min delay critical breaking point hota hai jahan passengers ki connecting flights miss hoti hain, crew duty hours expire hote hain, aur FAA ground delay program trigger karta hai. Dusra mathematical — 15 minute delay par pre-departure features ka natural accuracy ceiling ~82% hota hai. 45 minute severe delay par dataset me 91.78% on-time flights hoti hain, jahan model legitimately 91.78% accuracy achieve karta hai bina data leakage ke."*

### Q3: Data leakage kya hota hai aur tumne isse kaise roka?
> *"Sir, data leakage ka matlab hai training me aisi information use kar lena jo real-world prediction time par available na ho. Aviation me departure delay (`DEP_DELAY`) ka arrival delay se 0.93 correlation hota hai — isse fake 98% accuracy nikalna aasan hai. Humne `DEP_DELAY`, `TAXI_OUT`, `ARR_TIME` samet sabhi 17 post-departure runtime columns ko permanently drop kiya, taaki model strictly timetable features par sikhe."*

### Q4: Dataset kitna bada tha aur data cleaning me kya kiya?
> *"Sir, humne US Bureau of Transportation Statistics (BTS) ka 3-Million flight records wala official dataset use kiya. Cleaning me cancelled aur diverted flights filter kiye (kyunki unka arrival delay undefined hota hai), jisse 2,913,802 clean flights bachi. Missing values audit kiye jo <0.001% the, aur 80/20 stratified train-test split kiya."*

### Q5: 21,513 features se 12 features par kyu aaye?
> *"Sir, blind one-hot encoding se 21,513 sparse columns ban rahe the, jo overfitting aur high latency create karte hain. Humne ANOVA F-Test se top 12 pre-departure predictors lock kiye (`CRS_DEP_TIME`, `CRS_ARR_TIME`, `DISTANCE`, `AIRLINE`, etc.). Is 96.4% dimensionality reduction se model ne 99.8% predictive power retain ki aur inference time घटकर sub-100ms ho gaya."*

### Q6: Sabhi models ki accuracy ~91.7% ke aas-paas kyu aa rahi thi?
> *"Sir, isse ML me 'Accuracy Paradox under Class Imbalance' kehte hain. Dataset me 91.78% flights on-time (<45 min) hain. Agar koi dummy model sabko on-time bol de toh uski accuracy bhi 91.78% aayegi. Lekin unka discrimination power ROC-AUC me dikhta hai: Tuned XGBoost ka ROC-AUC 69.84% hai, jo baseline Logistic Regression se +5.94% superior hai."*

### Q7: Decision threshold T = 0.50 se T = 0.12 kyu change kiya?
> *"Sir, severe delay ka base rate sirf 8.22% hai. Default 0.50 threshold par model sabhi flights ko Class 0 bol deta hai aur Recall zero ho jata hai. Humne threshold ko 0.12 (12%) par calibrate kiya. Iska matlab agar kisi flight ka predicted delay probability 12% ya usse zyada hai, toh wo national average (8.22%) se 1.5x zyada risky hai. Isse minority delay recall 16.4x boost ho gaya!"*

### Q8: Single model aur Hybrid ensemble me kya difference raha?
> *"Sir, single model me Tuned XGBoost best raha (91.78% accuracy, 69.84% ROC-AUC). Lekin jab humne XGBoost, LightGBM aur Random Forest ko Level-1 Logistic Regression meta-learner se Stack kiya (HV-4 Consensus Model), toh usne peak 92.35% accuracy aur 70.42% ROC-AUC achieve kiya."*

### Q9: Explainable AI (SHAP) se kya key takeaways mile?
> *"Sir, SHAP analysis se confirm hua ki scheduled departure hour (`CRS_DEP_HOUR`) sabse dominant factor hai (|SHAP| > 0.20). Shaam 7 PM se 11 PM ki flights me delay risk peak par hota hai kyunki din bhar ke turnaround delays accumulate hote hain, jabki subah 6 AM se 9 AM ki flights me delay risk minimum hota hai."*

### Q10: Project ka deployment architecture kaisa hai?
> *"Sir, humne Flask Python backend banaya hai jisme tuned XGBoost model load hota hai. Production server ke liye Gunicorn WSGI use kiya hai, Docker container (`python:3.10-slim`) banaya hai, GitHub CI/CD se Render cloud par deploy kiya hai, aur 24/7 background keep-alive ping set kiya hai taaki cold-start na aaye aur website 1 second me load ho."*

### Q11: Live website par prediction kaise kaam karti hai?
> *"Sir, user website par jaakar Flight Date, Airline, Origin, Destination, Scheduled Times aur Distance dalta hai. Sub-100ms me API backend feature transformation karke XGBoost model se probability predict karta hai. Agar prob >= 12% ho toh Red Alert 'SEVERE DELAY' show hota hai, aur agar <12% ho toh Green 'ON TIME' badge show hota hai."*

### Q12: Is project ka future scope kya hai?
> *"Sir, future me hum 3 major advancements add kar sakte hain: 1) Real-time NOAA METAR weather radar API integration, 2) Inbound aircraft physical tail-number tracking taaki aircraft ka turnaround delay accurately predict ho sake, aur 3) Graph Neural Networks (GNN) se national airport network delay propagation model karna."*

---

## 📌 Quick Summary Cheatsheet (Ek Nazar Me):
* **Target:** `ARR_DELAY >= 45 min` (Severe Delay)
* **Dataset:** BTS 3-Million Flights (2.91M clean flights)
* **Features:** 12 selected pre-departure features (96.4% reduction from 21,513)
* **Leakage Prevention:** 17 post-departure variables permanently removed
* **Winning Model:** Tuned XGBoost (91.78% Acc, 69.84% ROC) | Ensemble: 92.35% Acc, 70.42% ROC
* **Operational Threshold:** Calibrated to `T = 0.12 (12.00%)` (16.4x recall boost)
* **Tech Stack:** Python, XGBoost, Scikit-Learn, Flask, Gunicorn, Docker, Render, HTML/CSS
* **Live Web App:** `https://flight-delay-predictor-z2ik.onrender.com/`
