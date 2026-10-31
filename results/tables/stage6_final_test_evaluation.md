# Phase 6 — Stage 6: Final Independent Test Evaluation Report

**Research Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Evaluation Phase:** Final Independent Out-of-Sample Test Evaluation  
**Dataset:** US Bureau of Transportation Statistics (BTS) On-Time Performance (December 2025)  
**Evaluation Population:** $N = 562,202$ consecutive aircraft rotation pairs  
**Dataset Integrity:** File SHA-256: `3f0e5494fdb4d934cda7b77f41eaef6b73de41e1cd3927c752dd3e3e90449d1e` | Size: 12.37 MB | Zero missing values | **Strictly isolated until final test inference**

---

## 1. Executive Summary & Core Results

The frozen final machine learning models—selected exclusively on the October–November validation dataset ($N = 1,140,909$) without any hyperparameter tuning or threshold adjustment on December data—were evaluated on the independent December 2025 test dataset ($N = 562,202$).

### 1.1 Final Classification Results (Random Forest Classifier, Group C)
- **Global Discrimination:**
  - **ROC-AUC:** **$0.7943$** (vs. Validation $0.7932$, delta: $+0.0011$)
  - **Average Precision (AP):** **$0.7150$** (vs. Validation $0.6600$, delta: $+0.0550$)
  - **Log Loss:** **$0.4255$** (vs. Validation $0.3748$)
- **Primary Operational Operating Point ($T = 0.35$):**
  - **Precision:** **$72.52\%$** (vs. Validation $69.20\%$, $+3.32$ percentage points)
  - **Recall:** **$53.24\%$** (vs. Validation $49.03\%$, $+4.21$ percentage points)
  - **F1-Score:** **$0.6140$** (vs. Validation $0.5740$, $+0.0400$)
  - **Specificity:** **$92.74\%$** (vs. Validation $94.47\%$)
  - **Accuracy:** **$82.28\%$** (vs. Validation $85.29\%$)
  - **Confusion Matrix:** True Positives = $79,250$ | False Positives = $30,025$ | False Negatives = $69,615$ | True Negatives = $383,312$
  - **Operational Coverage:** Identifies **$53.24\%$** of all delayed flight operations ($79,250$ of $148,865$) at the selected operating threshold ($T = 0.35$) while maintaining a false alarm rate on on-time flights of **$7.26\%$** ($30,025$ FP out of $413,337$ on-time flights).
- **Benchmark Comparisons:**
  - **Validation F1 Reference ($T = 0.39$):** Precision = $75.19\%$, Recall = $51.90\%$, F1 = **$0.6141$**, Specificity = $93.83\%$, Accuracy = $82.73\%$.
  - **Default Benchmark ($T = 0.50$):** Precision = $87.52\%$, Recall = **$45.07\%$**, F1 = **$0.5950$**, Specificity = $97.68\%$, Accuracy = $83.75\%$. Misses $81,772$ delays ($54.93\%$ false negative rate).

### 1.2 Final Regression Results (Random Forest Regressor, Group C)
- **Variance Explained ($R^2$):** **$0.4851$** (accounts for **$48.51\%$** of total arrival delay variance relative to a naive mean baseline; vs. Validation $0.4516$, delta: $+0.0335$).
- **Mean Absolute Error (MAE):** **$19.77$ minutes** (vs. Validation $17.40$ min).
- **Root Mean Squared Error (RMSE):** **$45.86$ minutes** (vs. Validation $38.62$ min).
- **Mean Squared Error (MSE):** **$2103.41 \text{ min}^2$** (vs. Validation $1491.67 \text{ min}^2$).
- **Residual Distribution ($\text{residual} = \text{actual} - \text{predicted}$):**
  - Mean Residual: **$+1.21$ minutes** (near-zero unconditional aggregate bias)
  - Median Residual: **$-5.33$ minutes**
  - Standard Deviation: **$45.85$ minutes**
  - Interquartile Range (IQR): $[-13.2\text{ min}, +10.8\text{ min}]$

---

## 2. Dataset Provenance & Target Distributions

### 2.1 Independent Temporal Test Set
The test dataset covers all completed domestic rotation pairs operated between December 1, 2025 and December 31, 2025 ($N = 562,202$).

| Dataset Partition | Time Period | Total Records | Delay Prevalence ($\ge 15$m) | Mean Delay | Median Delay | Std Delay |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | Jan 1 – Sep 30, 2025 | 5,065,376 | 21.84% | +7.21 min | -6.0 min | 56.42 min |
| **Validation** | Oct 1 – Nov 30, 2025 | 1,140,909 | 20.21% | +5.41 min | -7.0 min | 52.15 min |
| **Independent Test** | Dec 1 – Dec 31, 2025 | 562,202 | **26.48%** | **+12.23 min** | -4.0 min | **63.91 min** |

### 2.2 Observed Seasonal Distribution Shift
December exhibits an empirical seasonal distribution shift across the national network:
1. **Prevalence Increase:** Delays $\ge 15$ minutes rose by **$+6.27$ percentage points** (from $20.21\%$ in validation to $26.48\%$ in test).
2. **Delay Magnitude Expansion:** The average arrival delay more than doubled from $+5.41$ minutes to $+12.23$ minutes ($+6.82$ min increase), with variance expanding by $50.3\%$ (std from $52.15$ to $63.91$ min).
3. **Severe Delay Concentration:** Flights delayed by $\ge 120$ minutes increased from $2.60\%$ ($29,657$ flights) in validation to **$4.04\%$** ($22,693$ flights) in December—a **$+55.4\%$ relative increase** in severe delay frequency.
*Methodological Note:* While adverse winter weather, de-icing queues, and holiday travel congestion are plausible operational explanations for this shift, direct meteorological telemetry and airport ground congestion logs are unobserved in the BTS dataset and cannot be established as definitive causal factors.

---

## 3. Classification Evaluation: Decision Threshold Spectrum

Evaluated across candidate decision thresholds on the complete test population ($N = 562,202$, Actual Positives = $148,865$, Actual Negatives = $413,337$):

| Threshold ($T$) | TP | FP | FN | TN | Predicted Positives | Precision | Recall | F1-Score | Specificity | Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 148,220 | 396,558 | 645 | 16,779 | 544,778 | 27.21% | **99.57%** | 0.4271 | 4.06% | 29.35% | Exact (100%) |
| 0.15 | 114,818 | 138,590 | 34,047 | 274,747 | 253,408 | 45.31% | 77.13% | 0.5709 | 66.47% | 69.29% | Exact (100%) |
| 0.20 | 89,355 | 51,911 | 59,510 | 361,426 | 141,266 | 63.25% | 60.02% | 0.6160 | 87.44% | 80.18% | Exact (100%) |
| 0.25 | 83,071 | 38,981 | 65,794 | 374,356 | 122,052 | 68.06% | 55.80% | 0.6133 | 90.57% | 81.36% | Exact (100%) |
| 0.30 | 81,173 | 34,449 | 67,692 | 378,888 | 115,622 | 70.21% | 54.53% | 0.6138 | 91.67% | 81.83% | Exact (100%) |
| **0.35 (Primary Operating)** | **79,250** | **30,025** | **69,615** | **383,312** | **109,275** | **72.52%** | **53.24%** | **0.6140** | **92.74%** | **82.28%** | **Exact (100%)** |
| **0.39 (Val F1 Reference)** | **77,267** | **25,501** | **71,598** | **387,836** | **102,768** | **75.19%** | **51.90%** | **0.6141** | **93.83%** | **82.73%** | **Exact (100%)** |
| 0.40 (Coarse Grid Optimum)| 76,577 | 24,082 | 72,288 | 389,255 | 100,659 | 76.08% | 51.44% | 0.6138 | 94.17% | 82.86% | Exact (100%) |
| 0.45 | 72,698 | 17,211 | 76,167 | 396,126 | 89,909 | 80.86% | 48.83% | 0.6089 | 95.84% | 83.39% | Exact (100%) |
| **0.50 (Default Benchmark)** | **67,093** | **9,569** | **81,772** | **403,768** | **76,662** | **87.52%** | **45.07%** | **0.5950** | **97.68%** | **83.75%** | **Exact (100%)** |
| 0.55 | 62,357 | 6,009 | 86,508 | 407,328 | 68,366 | 91.21% | 41.89% | 0.5739 | 98.55% | 83.54% | Exact (100%) |
| 0.60 | 58,454 | 3,923 | 90,411 | 409,414 | 62,377 | 93.71% | 39.27% | 0.5535 | 99.05% | 83.22% | Exact (100%) |
| 0.65 | 53,995 | 2,176 | 94,870 | 411,161 | 56,171 | 96.13% | 36.27% | 0.5267 | 99.47% | 82.74% | Exact (100%) |
| 0.70 | 46,756 | 973 | 102,109 | 412,364 | 47,729 | 97.96% | 31.41% | 0.4757 | 99.76% | 81.67% | Exact (100%) |

*Arithmetic Verification:*
- $TP + FN = 148,865$ across all thresholds ($100.0\%$ matches December test positive delay count).
- $TN + FP = 413,337$ across all thresholds ($100.0\%$ matches December test negative count).
- Total observations $= 562,202$ ($100.0\%$ matches test row count).

---

## 4. Regression Evaluation & Residual Diagnostics

### 4.1 Overall Regression Performance
- **Coefficient of Determination ($R^2$):** **$0.4851$**  
  The Random Forest regressor explains **$48.51\%$** of total variance in December arrival delays.
- **Mean Absolute Error (MAE):** **$19.77$ minutes**
- **Root Mean Squared Error (RMSE):** **$45.86$ minutes**
- **Mean Squared Error (MSE):** **$2103.41 \text{ min}^2$**

### 4.2 Residual Analysis ($\text{residual} = \text{actual} - \text{predicted}$)
- **Mean Residual:** **$+1.21$ minutes** (unconditional mean prediction bias is exceptionally small, $< 1.3$ minutes).
- **Median Residual:** **$-5.33$ minutes** (the model predicts slight positive delays on flights that arrive slightly early or on-time).
- **Standard Deviation of Residuals:** **$45.85$ minutes**.
- **Quantile Distribution of Residuals:**
  - $1^{\text{st}}$ Percentile ($p_1$): $-48.1$ minutes
  - $5^{\text{th}}$ Percentile ($p_5$): $-30.1$ minutes
  - $10^{\text{th}}$ Percentile ($p_{10}$): $-23.5$ minutes
  - $25^{\text{th}}$ Percentile ($p_{25}$): $-13.2$ minutes
  - $50^{\text{th}}$ Percentile ($p_{50}$): $-5.33$ minutes
  - $75^{\text{th}}$ Percentile ($p_{75}$): $+10.8$ minutes
  - $90^{\text{th}}$ Percentile ($p_{90}$): $+47.4$ minutes
  - $95^{\text{th}}$ Percentile ($p_{95}$): $+89.2$ minutes
  - $99^{\text{th}}$ Percentile ($p_{99}$): $+222.0$ minutes

---

## 5. Regression Error Breakdown Across Delay Severity Bands

Residual behavior stratified across the five operational delay regimes on the test dataset ($N = 562,202$):

| Delay Severity Band | Actual Range | Test Observations ($N$) | Population Share | MAE (min) | RMSE (min) | Mean Prediction Bias ($\hat{y} - y$) | Primary Error Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Early / On-Time** | $\le 0$ min | 322,255 | 57.32% | 13.01 | 16.69 | **+12.27 min** | Conservative over-prediction (safe cushion) |
| **Minor Delay** | $1–14$ min | 91,082 | 16.20% | 10.58 | 12.64 | **-3.46 min** | Well-calibrated (minimal bias) |
| **Moderate Delay** | $15–59$ min | 94,233 | 16.76% | 22.27 | 26.57 | **-13.64 min** | Moderate compression toward center |
| **Significant Delay** | $60–119$ min | 31,939 | 5.68% | 42.70 | 54.03 | **-30.78 min** | Under-prediction of extended turnaround holds |
| **Severe Delay** | $\ge 120$ min | 22,693 | 4.04% | 109.98 | 201.18 | **-90.38 min** | Severe under-prediction (fat-tail limitation) |

### Key Observations:
1. **Low-Delay Regime ($\le 14$ min, $73.5\%$ of operations):** The regressor exhibits strong accuracy, with MAE between $10.6$ and $13.0$ minutes.
2. **Severe Delay Regime ($\ge 120$ min, $4.0\%$ of operations):** The model systematically underpredicts extreme delays ($\text{bias} = -90.38$ min). This reflects the inherent mathematical limit of tree-based regression ensembles: leaf predictions are bounded by sample averages observed in terminal nodes during training and cannot extrapolate into unobserved extreme tails. Severe delays in December likely involve compound external operational constraints (e.g., air traffic ground delay programs, de-icing queues, crew duty-time limitations), which are unobserved in the BTS feature set and represent plausible contextual hypotheses rather than proven causal drivers.

---

## 6. Generated Visualizations & Artifacts

All figures generated at 300 DPI publication resolution:
- **Confusion Matrix:** [`results/figures/stage6_test_confusion_matrix.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_confusion_matrix.png)
- **ROC Curve:** [`results/figures/stage6_test_roc_curve.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_roc_curve.png)
- **Precision-Recall Curve:** [`results/figures/stage6_test_precision_recall_curve.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_precision_recall_curve.png)
- **Threshold Curves:** [`results/figures/stage6_test_threshold_comparison.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_threshold_comparison.png)
- **Calibration Curve:** [`results/figures/stage6_test_calibration_curve.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_calibration_curve.png)
- **Actual vs. Predicted Delay:** [`results/figures/stage6_test_actual_vs_predicted.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_actual_vs_predicted.png)
- **Residual Distribution:** [`results/figures/stage6_test_residual_distribution.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_residual_distribution.png)
- **Residuals vs. Predicted:** [`results/figures/stage6_test_residual_vs_predicted.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_residual_vs_predicted.png)
- **Delay Band Performance:** [`results/figures/stage6_test_delay_band_performance.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_delay_band_performance.png)
- **Validation vs. Test Comparison:** [`results/figures/stage6_validation_test_comparison.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_validation_test_comparison.png)
- **Test Feature Importance:** [`results/figures/stage6_test_feature_importance.png`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/figures/stage6_test_feature_importance.png)
- **Serialized Predictions:** [`results/predictions/stage6_test_predictions.parquet`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/predictions/stage6_test_predictions.parquet) ($N = 562,202$, 6.55 MB).
