# Phase 6 — Stage 5.1: Final Model Selection Integrity Audit Report

**Research Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Audit Date:** 2026-10-31 13:36:56 UTC  
**Auditor Role:** Senior ML Engineer & Research Methodology Auditor  
**Scope:** Forensic Verification of Stage 5 Commit (`16c9dd6`) Results, Artifacts, and Threshold Dynamics  
**Validation Population:** Full Validation Set ($N = 1,140,909$, October 1 – November 30, 2025; Positive Delay Cases = 230,521 [20.21%], Negative Cases = 910,388 [79.79%])  
**Test Set Integrity:** **Strictly Isolated & Sealed** (`data/processed/modeling_test.parquet`, $N = 562,202$, December 2025 — **Zero Access**)

---

## 1. Executive Audit Summary & Discrepancy Resolution

This audit investigated and resolved metric, threshold, and textual inconsistencies identified in the Stage 5 deliverables:

### 1.1 Root Cause of Classification Threshold Inconsistencies
1. **Markdown Report Table Mismatch:** The table in Section 4 of `stage5_model_selection.md` contained hardcoded template values from early draft planning (e.g., claiming $T = 0.35$ yielded Precision = 62.43%, Recall = 57.10%, F1 = 0.5964, FP = 74,621, FN = 98,901). These figures were typed as prose strings rather than pulled dynamically from the actual evaluation dataframe.
2. **JSON Key Labeling Inconsistency:** In `stage5_final_model_selection.json`, the dictionary key `"validation_metrics_selected_threshold_0_35"` was populated with `best_f1_row`, which corresponded to threshold **$T = 0.40$** (Precision = 73.49%, Recall = 47.23%, F1 = 0.5751, TP = 108,884, FP = 39,284, FN = 121,637, TN = 871,104).
3. **F1-Optimality Mischaracterization:** The report described $T = 0.35$ as the "F1-optimal" threshold, whereas the actual calculated data in `stage5_threshold_analysis.csv` demonstrated that $T = 0.40$ had higher F1 (0.5751 vs 0.5740), and on a fine grid (step 0.01), the true global maximum occurs at **$T = 0.39$** ($F_1 = 0.57512$). Furthermore, the report's own table listed $T = 0.25$ with F1 = 0.5974, contradicting the claim that 0.35 was optimal.

### 1.2 Resolution & Forensic Verification
- All metrics have been recomputed from the serialized validation predictions and actual binary labels.
- Arithmetic integrity has been verified across all thresholds:
  - $TP + FN = 230,521$ (exact positive validation population)
  - $TN + FP = 910,388$ (exact negative validation population)
  - $TP + TN + FP + FN = 1,140,909$ (exact total validation population)
  - Sklearn metric calculations agree with formulaic confusion-matrix ratios to $< 10^{-9}$.
- Serialized final models in `models/final/` reproduce all audited metrics with zero discrepancy.

---

## 2. Threshold Reconciliation & Confusion Matrix Audit

Evaluated across candidate thresholds for the winning **Random Forest Classifier (Group C)** on the complete validation population ($N = 1,140,909$):

| Threshold ($T$) | TP | FP | FN | TN | Predicted Positives | Precision | Recall | F1-Score | Specificity | Accuracy | Reconciliation Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 229,036 | 877,954 | 1,485 | 32,434 | 1,106,990 | 20.69% | 99.36% | 0.3425 | 3.56% | 22.92% | Reconciled (100%) |
| 0.15 | 169,038 | 289,757 | 61,483 | 620,631 | 458,795 | 36.84% | 73.33% | 0.4905 | 68.17% | 69.21% | Reconciled (100%) |
| 0.20 | 126,865 | 91,704 | 103,656 | 818,684 | 218,569 | 58.04% | 55.03% | 0.5650 | 89.93% | 82.88% | Reconciled (100%) |
| 0.25 | 117,801 | 62,740 | 112,720 | 847,648 | 180,541 | 65.25% | 51.10% | 0.5732 | 93.11% | 84.62% | Reconciled (100%) |
| 0.30 | 115,158 | 56,520 | 115,363 | 853,868 | 171,678 | 67.08% | 49.96% | 0.5726 | 93.79% | 84.93% | Reconciled (100%) |
| **0.35 (Selected Operating)** | **113,035** | **50,305** | **117,486** | **860,083** | **163,340** | **69.20%** | **49.03%** | **0.5740** | **94.47%** | **85.29%** | **Reconciled (100%)** |
| **0.39 (Fine F1 Maximum)** | **109,835** | **41,599** | **120,686** | **868,789** | **151,434** | **72.53%** | **47.65%** | **0.5751** | **95.43%** | **85.78%** | **Reconciled (100%)** |
| 0.40 (Coarse F1 Maximum) | 108,884 | 39,284 | 121,637 | 871,104 | 148,168 | 73.49% | 47.23% | 0.5751 | 95.68% | 85.90% | Reconciled (100%) |
| 0.45 | 102,865 | 27,400 | 127,656 | 882,988 | 130,265 | 78.97% | 44.62% | 0.5702 | 96.99% | 86.41% | Reconciled (100%) |
| **0.50 (Default)** | **94,230** | **14,476** | **136,291** | **895,912** | **108,706** | **86.68%** | **40.88%** | **0.5556** | **98.41%** | **86.79%** | **Reconciled (100%)** |
| 0.55 | 86,852 | 8,677 | 143,669 | 901,711 | 95,529 | 90.92% | 37.68% | 0.5328 | 99.05% | 86.65% | Reconciled (100%) |
| 0.60 | 81,050 | 5,359 | 149,471 | 905,029 | 86,409 | 93.80% | 35.16% | 0.5115 | 99.41% | 86.43% | Reconciled (100%) |
| 0.65 | 74,170 | 2,851 | 156,351 | 907,537 | 77,021 | 96.30% | 32.17% | 0.4823 | 99.69% | 86.05% | Reconciled (100%) |
| 0.70 | 63,241 | 1,138 | 167,280 | 909,250 | 64,379 | 98.23% | 27.43% | 0.4289 | 99.87% | 85.24% | Reconciled (100%) |

*Audit Verification:*
- $TP + FN = 230,521$ for every threshold ($100\%$ verified against validation positive count).
- $TN + FP = 910,388$ for every threshold ($100\%$ verified against validation negative count).
- Total observations $= 1,140,909$ ($100\%$ verified against validation row count).

*Complete threshold audit data:* [`results/metrics/stage5_threshold_integrity_audit.csv`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/metrics/stage5_threshold_integrity_audit.csv) and [`results/metrics/stage5_threshold_confusion_matrices.csv`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/metrics/stage5_threshold_confusion_matrices.csv).

---

## 3. Threshold Policy & Selection Criteria

### 3.1 Characterization of Decision Thresholds
1. **Mathematical F1-Maximizing Threshold ($T = 0.39$ or $0.40$):**
   - On a fine search ($0.01$ step), $T = 0.39$ yields $F_1 = 0.57512$, with Precision = 72.53% and Recall = 47.65%.
   - On the standard coarse grid, $T = 0.40$ yields $F_1 = 0.57506$, with Precision = 73.49% and Recall = 47.23%.
2. **Selected Operational Threshold ($T = 0.35$):**
   - Yields Precision = 69.20%, Recall = 49.03%, and $F_1 = 0.5740$.
   - **Operational Justification:** In airline delay propagation dispatching, false alarms carry a lower marginal operational cost than missed delays (which cascade into downstream misconnections). Retaining $T = 0.35$ provides an operationally balanced dispatch warning rule that captures **49.03%** of delayed legs (+8.15 percentage points higher coverage than default $T = 0.50$) while maintaining nearly 70% precision ($69.20\%$).
   - **Correction:** $T = 0.35$ is strictly designated as a **selected operational operating threshold**, NOT the mathematical F1 optimum.
3. **Default Threshold ($T = 0.50$):**
   - Yields Precision = 86.68%, Recall = 40.88%, and $F_1 = 0.5556$.
   - While highly precise (only 14,476 false positives), it misses 136,291 delayed flights (59.12% false negative rate).

---

## 4. Final Estimator Reproducibility Audit

The serialized artifacts in `models/final/` were loaded and evaluated against the full validation population ($N = 1,140,909$):

### 4.1 Final Classifier
- **Artifact File:** [`models/final/final_selected_classifier.joblib`](file:///Users/arnavraj/Downloads/flight-delay-propagation/models/final/final_selected_classifier.joblib)
- **Preprocessor File:** [`models/final/final_preprocessor_group_c.joblib`](file:///Users/arnavraj/Downloads/flight-delay-propagation/models/final/final_preprocessor_group_c.joblib)
- **Estimator Class:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameters:** `n_estimators = 100`, `max_depth = 14`, `min_samples_leaf = 50`, `random_state = 42`, `n_jobs = 2`
- **Feature Matrix Dimension:** $1,140,909 \times 725$
- **Positive Class Indexing:** `classes_ = array([0, 1])`; positive class ($1$) index $= 1$
- **Reproduced Metrics:**
  - **ROC-AUC:** $0.793201$
  - **Average Precision (AP):** $0.660010$
  - **Log Loss:** $0.374772$
  - **Metrics at $T = 0.50$:** Accuracy = 0.8679, Precision = 0.8668, Recall = 0.4088, F1 = 0.5556
  - **Metrics at $T = 0.35$:** Accuracy = 0.8529, Precision = 0.6920, Recall = 0.4903, F1 = 0.5740

### 4.2 Final Regressor
- **Artifact File:** [`models/final/final_selected_regressor.joblib`](file:///Users/arnavraj/Downloads/flight-delay-propagation/models/final/final_selected_regressor.joblib)
- **Estimator Class:** `sklearn.ensemble.RandomForestRegressor`
- **Hyperparameters:** `n_estimators = 100`, `max_depth = 14`, `min_samples_leaf = 50`, `random_state = 42`, `n_jobs = 2`
- **Reproduced Metrics:**
  - **Mean Absolute Error (MAE):** $17.3997$ minutes (17.40 min)
  - **Root Mean Squared Error (RMSE):** $38.6221$ minutes (38.62 min)
  - **Coefficient of Determination ($R^2$):** $0.451617$ ($45.16\%$ variance explained relative to mean baseline)

Both models match the Stage 4 and Stage 5 reported benchmarks with exact fidelity. Complete reproducibility schema serialized in [`results/metrics/stage5_final_artifact_reproducibility.json`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/metrics/stage5_final_artifact_reproducibility.json).

---

## 5. Hyperparameter Search Methodology Clarification

In `train_stage5_tuning.py`, the hyperparameter tuning procedure was **NOT** an exhaustive grid search or randomized parameter search.

- **Accurate Characterization:** It was a **staged candidate comparison across discrete hypotheses**:
  - **Decision Trees:** 4 manual variants exploring tree depth ($10, 14, 18$) and leaf size ($50, 100$).
  - **Random Forests:** 4 manual variants exploring depth ($12, 14, 16$), leaf size ($30, 50, 100$), tree counts ($50, 100$), and class weighting (`balanced_subsample`).
  - **Multi-Layer Perceptrons:** 3 manual network topologies (`(64,)`, `(64, 32)`, `(128, 64)`) with early stopping on validation loss, strictly enforcing $\alpha = 0.0$.
  - **Linear Baselines:** Unregularized Logistic Regression and OLS evaluated under identical sample conditions ($N = 500,000$).
- **Selection Criteria:**
  - Classification: Maximization of validation Average Precision (AP) and ROC-AUC.
  - Regression: Minimization of validation Root Mean Squared Error (RMSE) and maximization of $R^2$.

---

## 6. Corrected Terminology & Scientific Interpretations

1. **Average Precision (AP) vs. PR-AUC:**
   - Metric reported is **Average Precision (AP)** computed via `average_precision_score()`.
   - In scikit-learn, AP summarizes the precision-recall curve as the weighted mean of precisions at each threshold, avoiding trapezoidal over-estimation inherent in generic AUC-PR calculations on sparse operational data.
2. **Feature Group Progression (Group A vs Group C):**
   - Group A (7 features): Turnaround buffer mechanics (`PREV_ARR_DELAY`, `SCHEDULED_TURNAROUND_BUFFER`, `ESTIMATED_REMAINING_TURNAROUND_MINUTES`, `IS_TURNAROUND_DEFICIT`, etc.).
   - Group B (14 features): Group A + scheduled route distance, origin/destination air traffic hubs, carrier, and time-of-day scheduled context.
   - Group C (18 features): Group B + same-day airframe operational delay history (`CONSECUTIVE_DEL15`, `CUMULATIVE_ARR_DELAY_TODAY`, `DELAY_RATE_TODAY`).
   - **Correction:** The performance gain from Group A to Group C (+1.08% ROC-AUC, -0.30m MAE) reflects the **combined contribution of route/carrier context AND historical airframe stress**, rather than aircraft history alone.
3. **Interpretation of $R^2 = 0.4516$:**
   - $R^2 = 0.4516$ denotes that the Random Forest regressor accounts for **45.16% of the variance in subsequent flight arrival delays relative to a naive mean-prediction baseline** ($1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$). It does NOT mean that 45.16% of flights or delays are predicted accurately.
4. **No Causal Claims:**
   - Predictive importance (e.g., turnaround deficit importance) reflects statistical association and operational constraints, NOT structural counterfactual causality. Interventions such as padding turnaround schedules may induce operational responses not captured by observational models.
5. **Delay Band Nuance:**
   - Random Forest is **not** strictly best across every single delay band:
     - On-time / early flights ($\le 0$ min): MLP achieved lower MAE ($9.36$ min vs RF $9.79$ min).
     - Moderate delays ($15–60$ min): Decision Trees and Random Forests performed similarly (MAE $\approx 17.7$ min).
     - Severe delays ($\ge 120$ min): Random Forest dominated (MAE $117.91$ min vs OLS $143.01$ min and MLP $122.99$ min).
     - Random Forest was selected because it achieves the lowest aggregate RMSE ($38.62$ min) and highest variance explained ($R^2 = 0.4516$).
6. **Validation Optimism Warning:**
   - Because candidate models and decision thresholds were selected based on validation set performance, the validation metrics reported here may carry a minor optimistic bias. The final, unbiased generalization benchmark will be established exclusively on the held-out December 2025 test dataset in Stage 6.

---

## 7. Audit Compliance Verification

- [x] **Test Set Isolation:** `data/processed/modeling_test.parquet` was **never accessed, loaded, or inspected**.
- [x] **Zero Regularization Integrity:** Both `MLPClassifier` and `MLPRegressor` models strictly maintained `alpha = 0.0`.
- [x] **No Prohibited Estimators:** Exclusively syllabus-compliant models were evaluated (Logistic Regression, Decision Trees, Random Forests, MLP, OLS).
- [x] **Exact Population Reconciliation:** All confusion matrix totals equal $1,140,909$, with positive sums equal to $230,521$ and negative sums equal to $910,388$.
- [x] **Versioned Preservation:** Original Stage 5 report preserved at [`results/tables/stage5_model_selection_v1_backup.md`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/tables/stage5_model_selection_v1_backup.md).
