# Phase 6 — Stage 6: Validation vs. Test Generalization Analysis

**Research Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Scope:** Comparative Generalization Audit Across Temporal Partitions (October–November 2025 vs. December 2025)  
**Validation Population:** $N = 1,140,909$ (October 1 – November 30, 2025)  
**Test Population:** $N = 562,202$ (December 1 – December 31, 2025)  
**Status:** Frozen Models Evaluated Without Refitting, Adaptation, or Retuning  

---

## 1. Executive Generalization Summary

A central methodological requirement of this study is the strict temporal isolation of the test dataset. In operational aviation machine learning, random cross-validation yields inflated performance because flight pairs sharing identical aircraft rotations, convective storm fronts, or carrier ground-handling crunches appear in both partitions. Evaluating models forward in time tests their true operational utility.

### Generalization Highlights
1. **Stable Classification Discrimination on Independent Test Data:**
   - **ROC-AUC:** $0.7932$ (Validation) $\rightarrow$ **$0.7943$** (Test), a change of only **$+0.0011$** ($+0.14\%$).
   - The ranking discrimination of the classifier is stable across temporal partitions, indicating that turnaround buffer features condition delay propagation consistently across autumn development and winter test periods.
2. **Prevalence-Driven Precision and Recall Shift:**
   - **Average Precision (AP):** $0.6600$ (Validation) $\rightarrow$ **$0.7150$** (Test), an increase of **$+0.0550$** ($+8.33\%$). Because AP is sensitive to positive-class prevalence, this increase reflects December's higher baseline delay rate ($26.48\%$ vs. $20.21\%$) rather than an improvement in model parameters.
   - At the primary operational operating point ($T = 0.35$):
     - **Precision:** $69.20\% \rightarrow \mathbf{72.52\%}$ ($+3.32$ percentage points)
     - **Recall:** $49.03\% \rightarrow \mathbf{53.24\%}$ ($+4.21$ percentage points)
     - **F1-Score:** $0.5740 \rightarrow \mathbf{0.6140}$ ($+0.0400$, $+6.97\%$)
3. **Variance Explained and Error Decomposition:**
   - **$R^2$ Score:** $0.4516$ (Validation) $\rightarrow$ **$0.4851$** (Test), an improvement of **$+0.0335$** ($+7.41\%$).
   - Aggregate **MAE** increased from $17.40$ minutes to **$19.77$ minutes** ($+2.37$ min, $+13.62\%$), and **RMSE** from $38.62$ minutes to **$45.86$ minutes** ($+7.24$ min, $+18.75\%$). Formal decomposition reveals that $94.0\%–98.4\%$ of this error increase is explained by population composition (a $55\%$ higher share of severe delays $\ge 120$ min carrying larger baseline error magnitudes).

---

## 2. Quantitative Metric Comparison Table

Data serialized in [`results/metrics/stage6_validation_test_comparison.csv`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/metrics/stage6_validation_test_comparison.csv):

| Evaluation Category | Metric | Validation (Oct–Nov) | Test (December) | Absolute Change ($\Delta$) | Relative Change (%) | Scientific Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Classification Global** | ROC-AUC | 0.7932 | **0.7943** | +0.0011 | +0.14% | Stable ranking discrimination |
| **Classification Global** | Average Precision (AP)| 0.6600 | **0.7150** | +0.0550 | +8.33% | Lifted by higher positive prevalence |
| **Classification Global** | Log Loss | 0.3748 | **0.4255** | +0.0507 | +13.52% | Slightly higher entropy due to winter variance |
| **Operational ($T = 0.35$)**| F1-Score | 0.5740 | **0.6140** | +0.0400 | +6.97% | Strong out-of-sample operational transfer |
| **Operational ($T = 0.35$)**| Precision | 69.20% | **72.52%** | +3.32 pp | +4.80% | Alert reliability increases with base rate |
| **Operational ($T = 0.35$)**| Recall | 49.03% | **53.24%** | +4.21 pp | +8.59% | Identifies 53.24% of delayed flights |
| **Operational ($T = 0.35$)**| Accuracy | 85.29% | **82.28%** | -3.01 pp | -3.53% | Reflects larger positive population share |
| **Operational ($T = 0.35$)**| Specificity | 94.47% | **92.74%** | -1.73 pp | -1.83% | False alarm rate remains low ($< 7.3\%$) |
| **Default ($T = 0.50$)** | F1-Score | 0.5556 | **0.5950** | +0.0394 | +7.09% | Conservative benchmark |
| **Default ($T = 0.50$)** | Precision | 86.68% | **87.52%** | +0.84 pp | +0.97% | High precision preserved |
| **Default ($T = 0.50$)** | Recall | 40.88% | **45.07%** | +4.19 pp | +10.25% | Misses 54.9% of propagated delays |
| **Regression** | $R^2$ Score | 0.4516 | **0.4851** | +0.0335 | +7.41% | Explains higher portion of total variance |
| **Regression** | MAE (minutes) | 17.40 | **19.77** | +2.37 min | +13.62% | Driven by larger delay magnitudes |
| **Regression** | RMSE (minutes) | 38.62 | **45.86** | +7.24 min | +18.75% | Sensitive to fat-tail severe delay surge |
| **Regression** | MSE ($\text{min}^2$) | 1491.67 | **2103.41** | +611.74 | +41.01% | Quadratic expansion of extreme residuals |

---

## 3. Dissecting the Temporal Distribution Shift

The observed performance shifts between October–November and December reflect documented macroeconomic and meteorological characteristics of the US National Airspace System (NAS):

```
+---------------------------------------------------------------------------------------------------+
| DISTRIBUTION SHIFT: VALIDATION (Oct-Nov) vs. TEST (Dec)                                           |
+------------------------------------+--------------------------+-----------------------------------+
| Characteristic                     | Validation (Oct-Nov)     | Test (December)                   |
+------------------------------------+--------------------------+-----------------------------------+
| Completed Flight Rotations         | 1,140,909                | 562,202                           |
| Positive Delay Prevalence (>= 15m) | 20.21% (230,521 flights) | 26.48% (148,865 flights) [+6.27%] |
| Mean Arrival Delay                 | +5.41 minutes            | +12.23 minutes (+6.82 min)        |
| Standard Deviation of Delay        | 52.15 minutes            | 63.91 minutes (+11.76 min)        |
| Severe Delays (>= 120m) Share      | 2.60% (29,657 flights)   | 4.04% (22,693 flights) [+55.4%]   |
| On-Time / Early Flights (<= 0m)    | 64.40% (734,735 flights) | 57.32% (322,255 flights) [-7.08%] |
+------------------------------------+--------------------------+-----------------------------------+
```

### Analysis of Shift Mechanisms:
1. **Plausible Operational Context (Unobserved Telemetry):**
   - In commercial aviation, December operations frequently involve adverse winter weather fronts, aircraft de-icing queues, and holiday passenger congestion that extend turnaround durations.
   - However, because direct meteorological radar and terminal congestion logs are not present in the BTS database, these factors must be understood as *plausible operational hypotheses* rather than verified causal drivers of the observed distribution shift.
2. **Mathematical Invariance of ROC-AUC:**
   - Because ROC-AUC evaluates pairwise ranking ($P(\hat{p}_{\text{positive}} > \hat{p}_{\text{negative}})$), it is mathematically invariant to monotonic shifts in class prevalence. The close alignment between validation ($0.7932$) and test ($0.7943$) indicates that the model's fundamental ranking discrimination was stable across temporal regimes.
3. **Average Precision Mechanics:**
   - Average Precision summarizes the Precision-Recall curve and is sensitive to positive-class prevalence. As the baseline proportion of delays increases from $20.21\%$ to $26.48\%$, the precision across recall thresholds rises mechanically. Hence, the test AP increase ($0.6600 \rightarrow 0.7150$) reflects the higher positive base rate rather than an improvement in model parameters.

---

## 4. Delay Severity Band Generalization & Formal MAE Decomposition

Comparing regression MAE across actual delay bands between Validation and Test:

| Delay Severity Band | Validation MAE (min) | Test MAE (min) | MAE Delta ($\Delta$) | Validation Bias ($\hat{y} - y$) | Test Bias ($\hat{y} - y$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Early / On-Time ($\le 0$ min)** | 12.19 | **13.01** | +0.82 min | +11.43 min | **+12.27 min** |
| **Minor Delay ($1–14$ min)** | 10.28 | **10.58** | +0.30 min | -4.06 min | **-3.46 min** |
| **Moderate Delay ($15–59$ min)** | 22.44 | **22.27** | -0.17 min | -14.35 min | **-13.64 min** |
| **Significant Delay ($60–119$ min)** | 45.03 | **42.70** | -2.33 min | -33.85 min | **-30.78 min** |
| **Severe Delay ($\ge 120$ min)** | 117.91 | **109.98** | -7.93 min | -99.29 min | **-90.38 min** |

### Formal Mathematical Shift-Share (Oaxaca-Blinder) Decomposition:
To evaluate whether the $+2.37$ minute increase in aggregate test MAE ($17.40 \rightarrow 19.77$ min) reflects algorithmic deterioration or population composition shift, we decomposed the total change:

$$\Delta \text{MAE} = \sum_{i=1}^5 w_i^{(T)} \cdot \text{MAE}_i^{(T)} - \sum_{i=1}^5 w_i^{(V)} \cdot \text{MAE}_i^{(V)} = +2.3674 \text{ minutes}$$

1. **Counterfactual Test MAE (Test Weights Applied to Validation Within-Band Errors):**
   $$\text{MAE}_{\text{counterfactual}} = \sum_{i=1}^5 w_i^{(T)} \cdot \text{MAE}_i^{(V)} = \mathbf{19.7316 \text{ minutes}}$$
   - **Population-Composition Effect:** **$+2.3297$ minutes ($98.41\%$ of the total increase)**.
   - **Within-Band Model Error Effect:** **$+0.0377$ minutes ($1.59\%$ of the total increase)**.
2. **Symmetric Shapley Decomposition:**
   - **Composition Effect:** **$+2.2250$ minutes ($93.99\%$)**.
   - **Within-Band Model Error Effect:** **$+0.1424$ minutes ($6.01\%$)**.

### Scientific Finding:
The formal decomposition rigorously confirms that between **$94.0\%$ and $98.4\%$** of the aggregate MAE increase is attributable to population-composition change (specifically, the $55\%$ surge in severe delays $\ge 120$ min which carry larger baseline error magnitudes). When evaluated conditionally within each specific delay band, model accuracy was remarkably stable across autumn and winter data.

---

## 5. Threshold Operating Policy: Validation vs. Test

The operating point $T = 0.35$ was chosen during Stage 5 validation as a potential operational trade-off:

```
+---------------------------------------------------------------------------------------------------+
| OPERATIONAL BENEFIT OF T = 0.35 OVER DEFAULT T = 0.50 (TEST SET)                                  |
+---------------------------------------------------------------------------------------------------+
| Metric                             | Default Benchmark (T = 0.50) | Selected Operating Point (T = 0.35) |
+------------------------------------+------------------------------+-------------------------------------+
| True Positives (Delays Caught)     | 67,093                       | 79,250 (+12,157 flights / +18.1%)   |
| False Negatives (Delays Missed)    | 81,772                       | 69,615 (-12,157 flights)            |
| False Positives (False Alarms)     | 9,569                        | 30,025 (+20,456 alerts)             |
| False Alarm Rate on On-Time Flights| 2.32%                        | 7.26% (Remains < 7.3%)              |
| Precision (Alert Accuracy)         | 87.52%                       | 72.52% (Nearly 3 out of 4 correct)  |
| Recall (Delay Coverage)            | 45.07%                       | 53.24% (+8.17 percentage points)    |
| F1-Score                           | 0.5950                       | 0.6140 (+0.0190)                    |
+------------------------------------+------------------------------+-------------------------------------+
```

### Operational Takeaway:
Operating at $T = 0.35$ in December successfully identified **$79,250$** delayed flight rotations—capturing an additional **$12,157$ delayed operations** (+18.1% gain in true positives) relative to the default $0.50$ threshold, while maintaining **$72.52\%$ precision** and holding the false alarm rate across on-time operations to **$7.26\%$**.
