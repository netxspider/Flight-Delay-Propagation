# Phase 6 — Stage 5: Validation-Based Model Selection Report (Reconciled & Audited)

**Research Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Audit Status:** Fully Reconciled against Serialized Artifacts (Stage 5.1 Audit)  
**Validation Evaluation Population:** $N = 1,140,909$ flight pairs (October 1 – November 30, 2025; Positive Rate = 20.21%)  
**Test Set Integrity:** **Strictly Isolated ($N = 562,202$, December 2025 — Zero Access)**

---

## 1. Executive Summary & Selection Decision

Following systematic candidate comparisons across tree-based ensembles, unregularized multi-layer perceptrons (MLP with $\alpha = 0.0$), and linear references on the validation dataset:

1. **Winning Classification Model:**
   - **Estimator:** **Random Forest Classifier (Group C)**
   - **Hyperparameters:** `n_estimators = 100`, `max_depth = 14`, `min_samples_leaf = 50`, `n_jobs = 2`, `random_state = 42`.
   - **Key Discrimination Performance:** **ROC-AUC = 0.7932**, **Average Precision (AP) = 0.6600**, **Log Loss = 0.3748**.
   - **Decision Threshold Policy:**
     - **Default Threshold ($T = 0.50$):** Precision = 86.68%, Recall = 40.88%, F1 = 0.5556 (High precision, but misses 59.12% of delays).
     - **Mathematical F1-Optimum ($T = 0.39$):** Precision = 72.53%, Recall = 47.65%, **F1 = 0.5751**.
     - **Selected Operational Threshold ($T = 0.35$):** Precision = 69.20%, Recall = 49.03%, **F1 = 0.5740** (Captures nearly 50% of delays while preserving ~70% precision).

2. **Winning Regression Model:**
   - **Estimator:** **Random Forest Regressor (Group C)**
   - **Hyperparameters:** `n_estimators = 100`, `max_depth = 14`, `min_samples_leaf = 50`, `n_jobs = 2`, `random_state = 42`.
   - **Key Performance:** **$R^2 = 0.4516$** (explains 45.16% of total arrival delay variance relative to a mean baseline), **RMSE = 38.62 minutes**, **MAE = 17.40 minutes**.

---

## 2. Classification Hyperparameter Candidate Comparison

Evaluated across candidate architectures on the full validation population ($N = 1,140,909$):

| Candidate Name | Model Family | Key Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision (AP) | Fit Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF_Candidate_1 (Selected)** | Random Forest | depth=14, leaf=50, trees=100 | **86.79%** | 86.68% | 40.88% | 0.5556 | **0.7932** | **0.6600** | 13.8s |
| **RF_Candidate_3 (Deeper)** | Random Forest | depth=16, leaf=30, trees=100 | 86.68% | 85.34% | 41.36% | 0.5571 | 0.7925 | 0.6588 | 15.2s |
| **RF_Candidate_2 (Shallower)** | Random Forest | depth=12, leaf=100, trees=100 | 86.81% | 87.12% | 40.35% | 0.5515 | 0.7921 | 0.6582 | 12.1s |
| **RF_Candidate_4 (Balanced)** | Random Forest | depth=14, leaf=50, balanced_subsample | 77.30% | 45.47% | **69.83%** | 0.5508 | 0.7909 | 0.6508 | 14.1s |
| **DT_Candidate_1 (Default)** | Decision Tree | depth=14, leaf=100 | 86.75% | 82.30% | 43.82% | **0.5719** | 0.7887 | 0.6610 | 8.1s |
| **DT_Candidate_4 (Finer)** | Decision Tree | depth=14, leaf=50 | 86.61% | 81.18% | 44.02% | 0.5706 | 0.7872 | 0.6589 | 8.4s |
| **DT_Candidate_3 (Deeper)** | Decision Tree | depth=18, leaf=50 | 86.23% | 76.54% | 46.12% | 0.5756 | 0.7801 | 0.6482 | 9.7s |
| **DT_Candidate_2 (Shallower)**| Decision Tree | depth=10, leaf=100 | 86.82% | 84.81% | 41.76% | 0.5597 | 0.7820 | 0.6521 | 6.2s |
| **MLP_Candidate_1 (Default)** | MLP ($\alpha = 0.0$) | (64, 32), batch=256 | 86.02% | **91.89%** | 33.77% | 0.4939 | 0.7877 | 0.6458 | 9.9s |
| **MLP_Candidate_3 (Wider)** | MLP ($\alpha = 0.0$) | (128, 64), batch=256 | 85.98% | 91.12% | 34.01% | 0.4952 | 0.7865 | 0.6441 | 14.6s |
| **MLP_Candidate_2 (Single)**| MLP ($\alpha = 0.0$) | (64,), batch=256 | 85.81% | 90.45% | 33.10% | 0.4846 | 0.7831 | 0.6402 | 7.8s |
| *Logistic Regression Ref* | Logistic Regression | penalty=None, solver=lbfgs | 85.12% | 77.27% | 37.32% | 0.5033 | 0.7765 | 0.6218 | 4.7s |

---

## 3. Regression Hyperparameter Candidate Comparison

Evaluated across candidate architectures on the full validation population ($N = 1,140,909$):

| Candidate Name | Model Family | Key Hyperparameters | MAE (min) | RMSE (min) | $R^2$ Score | Fit Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **RF_Candidate_1 (Selected)** | Random Forest | depth=14, leaf=50, trees=100 | 17.40 | **38.62** | **0.4516** | 411.9s |
| **RF_Candidate_2 (Faster)** | Random Forest | depth=14, leaf=50, trees=50 | 17.42 | 38.66 | 0.4506 | 173.8s |
| **DT_Candidate_1 (Default)** | Decision Tree | depth=14, leaf=100 | 17.70 | 39.10 | 0.4378 | 9.3s |
| **DT_Candidate_4 (Finer)** | Decision Tree | depth=14, leaf=50 | 17.79 | 39.24 | 0.4340 | 9.4s |
| **DT_Candidate_3 (Deeper)** | Decision Tree | depth=18, leaf=50 | 17.91 | 39.39 | 0.4297 | 13.8s |
| **DT_Candidate_2 (Shallower)**| Decision Tree | depth=10, leaf=100 | 17.84 | 39.05 | 0.4393 | 4.5s |
| **MLP_Candidate_1 (Default)** | MLP ($\alpha = 0.0$) | (64, 32), batch=256 | **17.06** | 39.34 | 0.4311 | 10.1s |
| **MLP_Candidate_2 (Single)**| MLP ($\alpha = 0.0$) | (64,), batch=256 | 17.07 | 38.94 | 0.4426 | 19.9s |
| **MLP_Candidate_3 (Wider)** | MLP ($\alpha = 0.0$) | (128, 64), batch=256 | 17.38 | 39.79 | 0.4178 | 17.5s |
| *OLS Linear Reference* | Linear Regression | OLS Ordinary | 21.20 | 44.43 | 0.2744 | 3.5s |

---

## 4. Reconciled Decision Threshold & Confusion Matrix Analysis

All counts below are calculated directly from the serialized validation predictions ($N = 1,140,909$, True Positives = 230,521, True Negatives = 910,388):

| Threshold ($T$) | TP | FP | FN | TN | Predicted Positives | Precision | Recall | F1-Score | Specificity | Accuracy | Operational Trade-off |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.10 | 229,036 | 877,954 | 1,485 | 32,434 | 1,106,990 | 20.69% | **99.36%** | 0.3425 | 3.56% | 22.92% | Near-universal alarm |
| 0.15 | 169,038 | 289,757 | 61,483 | 620,631 | 458,795 | 36.84% | 73.33% | 0.4905 | 68.17% | 69.21% | High-sensitivity screening |
| 0.20 | 126,865 | 91,704 | 103,656 | 818,684 | 218,569 | 58.04% | 55.03% | 0.5650 | 89.93% | 82.88% | Balanced sensitivity |
| 0.25 | 117,801 | 62,740 | 112,720 | 847,648 | 180,541 | 65.25% | 51.10% | 0.5732 | 93.11% | 84.62% | Moderate precision focus |
| 0.30 | 115,158 | 56,520 | 115,363 | 853,868 | 171,678 | 67.08% | 49.96% | 0.5726 | 93.79% | 84.93% | Conservative warning |
| **0.35 (Selected Operating)** | **113,035** | **50,305** | **117,486** | **860,083** | **163,340** | **69.20%** | **49.03%** | **0.5740** | **94.47%** | **85.29%** | **Selected practical dispatch rule** |
| **0.39 (Fine F1 Maximum)** | **109,835** | **41,599** | **120,686** | **868,789** | **151,434** | **72.53%** | **47.65%** | **0.5751** | **95.43%** | **85.78%** | **Global mathematical F1 optimum** |
| 0.40 (Coarse F1 Maximum) | 108,884 | 39,284 | 121,637 | 871,104 | 148,168 | 73.49% | 47.23% | 0.5751 | 95.68% | 85.90% | Discrete grid F1 optimum |
| 0.45 | 102,865 | 27,400 | 127,656 | 882,988 | 130,265 | 78.97% | 44.62% | 0.5702 | 96.99% | 86.41% | High precision focus |
| **0.50 (Default)** | **94,230** | **14,476** | **136,291** | **895,912** | **108,706** | **86.68%** | **40.88%** | **0.5556** | **98.41%** | **86.79%** | **Default rule / misses 59% delays** |

*Verification Check:* For all rows, $TP + FN = 230,521$, $TN + FP = 910,388$, and $TP + FP + FN + TN = 1,140,909$.

---

## 5. Incremental Contribution: Group A vs Group C

Direct empirical ablation on the exact same validation population ($N = 1,140,909$):

| Selected Model | Feature Group | Primary Performance Metric | ROC-AUC / $R^2$ | Delta Contribution (Context + Aircraft History) |
| :--- | :---: | :---: | :---: | :--- |
| **RF Classifier** | **Group A** (Turnaround Only, 7 feats) | AP = 0.6595 | ROC-AUC = 0.7824 | Baseline turnaround physics |
| **RF Classifier** | **Group C** (+ Route Context & Aircraft History, 18 feats) | **AP = 0.6600** | **ROC-AUC = 0.7932** | **+0.0108 ROC-AUC** (+1.08 percentage points) |
| **RF Regressor** | **Group A** (Turnaround Only, 7 feats) | MAE = 17.70m, RMSE = 38.89m | $R^2 = 0.4440$ | Baseline turnaround physics |
| **RF Regressor** | **Group C** (+ Route Context & Aircraft History, 18 feats) | **MAE = 17.40m, RMSE = 38.62m** | **$R^2 = 0.4516$** | **-0.30m MAE, -0.27m RMSE, +0.0076 $R^2$** |

**Empirical Interpretation:** Group C incorporates BOTH scheduled route/carrier context (Group B) AND same-day aircraft operational history. While turnaround mechanics (`ESTIMATED_REMAINING_TURNAROUND_MINUTES` and `IS_TURNAROUND_DEFICIT`) provide the primary physical propagation bottleneck, the combined operational context and historical airframe stress account for an additional +1.08% ROC-AUC discrimination and explain 0.76% more delay variance across 1.14M operations.

---

## 6. Audit Compliance & Methodological Caveats

1. **Test Set Isolation:** `data/processed/modeling_test.parquet` was **never accessed, loaded, or inspected**.
2. **Zero Regularization Integrity:** Both `MLPClassifier` and `MLPRegressor` models strictly maintained `alpha = 0.0`.
3. **No Prohibited Estimators:** Exclusively syllabus-compliant models were evaluated (Logistic Regression, Decision Trees, Random Forests, MLP, OLS).
4. **Validation Optimism Note:** Selecting models and tuning decision thresholds on the validation set can yield slightly optimistic estimates. Unbiased out-of-sample generalization performance will be determined exclusively on the held-out December 2025 test set.
5. **Reproducibility:** All serialized final models and preprocessors in `models/final/` reproduce these audited metrics exactly.
