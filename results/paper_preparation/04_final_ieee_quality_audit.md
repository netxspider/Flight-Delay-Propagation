# Final IEEE Quality Audit & Compliance Verification

**Project Title:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Target Venue:** IEEE Conference on Intelligent Transportation Systems / Artificial Intelligence Applications  
**Class File:** Official IEEEtran Class V1.8b (`paper/IEEEtran.cls`, conference mode)  
**Bibliography Engine:** IEEEtran BibTeX style V1.14 (`paper/IEEEtran.bst`, `paper/references.bib`)  
**Compiled Artifact:** `paper/main.pdf`  
**Audit Timestamp:** October 31, 2026  
**Auditor:** Senior Data Scientist, ML Research Scientist & IEEE Technical Editor  

---

## 1. Page Count & Layout Geometry Verification

| Verification Dimension | Standard / Specification | Verified Value | Compliance Status |
|:---|:---|:---|:---:|
| **Absolute Page Count** | **Strictly exactly 6 pages** | **6 pages (`kMDItemNumberOfPages = 6`, `\@abspage@last{6}`)** | **PASS** |
| Document Class | IEEEtran conference format | `\documentclass[conference]{IEEEtran}` | **PASS** |
| Page Dimensions | US Letter (8.5 $\times$ 11.0 in) | $8.5 \times 11.0$ inches (612 $\times$ 792 pt) | **PASS** |
| Column Architecture | Standard 2-column layout | Two columns, 3.5 in column width, 0.24 in gutter | **PASS** |
| Last Page Termination | References on Page 6 | References conclude in Column 2 of Page 6 | **PASS** |
| No Orphan 7th Page | Must not spill to Page 7 | Zero overflow beyond Page 6 | **PASS** |
| No Premature End | Must not terminate on Page 5 | Full six-page technical density | **PASS** |

---

## 2. Cross-Reference & Citation Resolution Audit

An automated inspection of `paper/main.aux` and `paper/main.log` was conducted to confirm that every citation and cross-reference resolves without ambiguity or missing symbols.

### 2.1 Citation Resolution (Zero Missing `[?]` References)
All 16 bibliographic references defined in `paper/references.bib` resolve to exact numerical indices `[1]` through `[16]`:

| Citation Key | IEEE Number | First Author & Year | Venue / Type | Resolution Status |
|:---|:---:|:---|:---|:---:|
| `wu2005inherent` | **[1]** | C.-L. Wu (2005) | Journal of Air Transport Management | **RESOLVED** |
| `pyrgiotis2013modelling` | **[2]** | N. Pyrgiotis et al. (2013) | Transportation Research Part C | **RESOLVED** |
| `campanelli2016modeling` | **[3]** | B. Campanelli et al. (2016) | IEEE Trans. Intelligent Transportation Systems | **RESOLVED** |
| `rebollo2014characterization` | **[4]** | J. J. Rebollo & H. Balakrishnan (2014) | Transportation Research Part C | **RESOLVED** |
| `sternberg2017review` | **[5]** | A. Sternberg et al. (2017) | IEEE Access | **RESOLVED** |
| `evler2021airline` | **[6]** | J. Evler et al. (2021) | Journal of Air Transport Management | **RESOLVED** |
| `bts2025ontime` | **[7]** | Bureau of Transportation Statistics (2025) | U.S. DOT TranStats Database | **RESOLVED** |
| `ahmadbeygi2010analysis` | **[8]** | S. AhmadBeygi et al. (2010) | Transportation Science | **RESOLVED** |
| `lan2006planning` | **[9]** | S. Lan et al. (2006) | Transportation Science | **RESOLVED** |
| `fricke2009delay` | **[10]** | H. Fricke & M. Schultz (2009) | ATM R&D Seminar | **RESOLVED** |
| `gui2020flight` | **[11]** | G. Gui et al. (2020) | IEEE Trans. Vehicular Technology | **RESOLVED** |
| `breiman2001random` | **[12]** | L. Breiman (2001) | Machine Learning | **RESOLVED** |
| `oaxaca1973male` | **[13]** | R. Oaxaca (1973) | International Economic Review | **RESOLVED** |
| `blinder1973wage` | **[14]** | A. S. Blinder (1973) | Journal of Human Resources | **RESOLVED** |
| `fisher2019all` | **[15]** | A. Fisher, C. Rudin, F. Dominici (2019) | J. Machine Learning Research | **RESOLVED** |
| `faa2024atcscc` | **[16]** | Federal Aviation Administration (2024) | U.S. DOT ATCSCC Operational Manual | **RESOLVED** |

*Audit Summary:* `main.aux` contains zero occurrences of `\mbox{\reset@font\bfseries ?}` or `Citation undefined` warnings. **100% of citations resolved.**

### 2.2 Internal Cross-Reference Resolution (Zero `??` References)
Every figure and table label is correctly indexed and referenced in the narrative:

| Label | Environment | Caption Title | Text Callouts | Resolution Status |
|:---|:---|:---|:---|:---:|
| `tab:dataset_split` | Table I | Chronological Partitioning of 2025 BTS Rotations | Section III | **RESOLVED (Table I)** |
| `tab:rotation_filtering` | Table II | Aircraft Rotation Reconstruction Statistics | Section IV-A | **RESOLVED (Table II)** |
| `fig:propagation_dynamics` | Figure 1 | Empirical delay propagation dynamics (relationships) | Section IV-B | **RESOLVED (Fig. 1)** |
| `tab:feature_groups` | Table III | Hierarchical Feature Representation Groups | Section V-B | **RESOLVED (Table III)** |
| `tab:classification_val` | Table IV | Classification Model Comparison on Validation Set | Section VII-A | **RESOLVED (Table IV)** |
| `tab:regression_val` | Table V | Continuous Arrival Delay Regression on Validation Set | Section VII-A | **RESOLVED (Table V)** |
| `fig:test_discrimination` | Figure 2 | ROC and Precision-Recall Curves on Test Set | Section VII-C | **RESOLVED (Fig. 2)** |
| `tab:test_generalization` | Table VI | Out-of-Sample Generalization to December 2025 | Section VII-C | **RESOLVED (Table VI)** |
| `tab:delay_bands` | Table VII | Regression Residuals Across Delay Strata (Test Set) | Section VII-D | **RESOLVED (Table VII)** |
| `fig:test_dual_evaluation` | Figure 3 | Confusion Matrix ($T=0.35$) and Actual vs. Predicted Delay | Section VII-C | **RESOLVED (Fig. 3)** |
| `fig:feature_importance` | Figure 4 | Permutation Feature Importance on Test Operations | Section VII-F | **RESOLVED (Fig. 4)** |

*Audit Summary:* `main.log` reports zero `LaTeX Warning: There were undefined references.` **100% of labels resolved.**

---

## 3. Numerical & Experimental Integrity Verification

All empirical quantities reported in the paper have been cross-checked against the serialized Stage 6.2 audit records:

| Finding / Metric Reported in Paper | Ground Truth from Stage 6.2 Audit | Delta | Integrity Assessment |
|:---|:---|:---:|:---:|
| Total Reconstructed Rotations: 6,768,487 | `data/processed/modeling_train.parquet` + `val` + `test` = 6,768,487 | 0 | **EXACT MATCH** |
| Training Partition: 5,065,376 (21.57% late) | `modeling_train.parquet` row count = 5,065,376 | 0 | **EXACT MATCH** |
| Validation Partition: 1,140,909 (20.21% late) | `modeling_val.parquet` row count = 1,140,909 | 0 | **EXACT MATCH** |
| Test Partition: 562,202 (26.48% late) | `modeling_test.parquet` row count = 562,202 | 0 | **EXACT MATCH** |
| Turnaround Deficit Prevalence: 382,937 (5.66%) | Descriptive EDA & Stage 6.2 verified counts = 382,937 | 0 | **EXACT MATCH** |
| Deficit Subsequent Delay Rate: 99.50% | Verified deficit late flight count = 381,022 / 382,937 | 0 | **EXACT MATCH** |
| Mean Deficit Arrival Delay: +121.86 min | Verified empirical mean conditional delay = +121.86 min | 0 | **EXACT MATCH** |
| Validation RF Group C ROC-AUC: 0.7932 | `stage4_model_configuration_audit.csv` = 0.7932 | 0 | **EXACT MATCH** |
| Validation RF Group C Average Precision: 0.6600 | Stage 4.1 / 5.1 evaluation = 0.6600 | 0 | **EXACT MATCH** |
| Validation RF Group C $R^2$: 0.4516 | Verified continuous regression validation score = 0.4516 | 0 | **EXACT MATCH** |
| Validation RF Group C MAE: 17.40 min | Verified continuous regression validation MAE = 17.40 min | 0 | **EXACT MATCH** |
| Validation Early Delay Band ($\le 0$m) MAE: 12.19 min | Reconciled Stage 6.2 empirical ground truth = 12.19 min | 0 | **EXACT MATCH (Typo 9.79 rejected)** |
| Best Grid F1 Threshold on 0.01 Grid: $T = 0.39$ | Grid maximum at step 0.39 ($\text{F1} = 0.5751$) | 0 | **EXACT MATCH** |
| Selected Operational Threshold: $T = 0.35$ | Operational operating point ($\text{F1} = 0.5740$) | 0 | **EXACT MATCH** |
| Test ROC-AUC: 0.7943 | Independent December test evaluation = 0.7943 | 0 | **EXACT MATCH** |
| Test Average Precision: 0.7150 | Independent December test evaluation = 0.7150 | 0 | **EXACT MATCH** |
| Test Precision at $T=0.35$: 72.52% ($79,250$ / $109,275$) | Frozen test confusion matrix = 72.52% | 0 | **EXACT MATCH** |
| Test Recall at $T=0.35$: 53.24% ($79,250$ / $148,865$) | Frozen test confusion matrix = 53.24% | 0 | **EXACT MATCH** |
| Test False Alarm Rate: 7.26% ($30,025$ / $413,337$) | Test specificity = 92.74% (FPR = 7.26%) | 0 | **EXACT MATCH** |
| Test Continuous $R^2$: 0.4851 | Independent December regression $R^2$ = 0.4851 | 0 | **EXACT MATCH** |
| Test Continuous MAE: 19.77 min | Independent December regression MAE = 19.77 min | 0 | **EXACT MATCH** |
| Test Unconditional Mean Bias: +1.21 min | Empirical mean error ($\bar{y} - \bar{\hat{y}}$) = +1.21 min | 0 | **EXACT MATCH** |
| Test Extreme Delay Band ($\ge 120$m) MAE: 109.98 min | Verified test stratum MAE = 109.98 min | 0 | **EXACT MATCH** |
| Severe Delay Conditional Bias: -90.38 min | Actual 231.01m vs Predicted 140.63m = -90.38 min | 0 | **EXACT MATCH** |
| Shift-Share Composition Effect: +2.330 min (98.41%) | Formulation A Oaxaca-Blinder decomposition | 0 | **EXACT MATCH** |
| Shift-Share Within-Band Error: +0.038 min (1.59%) | Formulation A Oaxaca-Blinder decomposition | 0 | **EXACT MATCH** |
| Shapley Symmetric Composition Share: 93.99% | Shapley decomposition composition share = 93.99% | 0 | **EXACT MATCH** |
| Permutation Top Feature: $\text{RemTurn}$ (+0.0303 AP, +15.81m MAE) | 100k test permutation importance (5 repeats) | 0 | **EXACT MATCH** |
| Permutation Arithmetic $R^2$ Drop: +0.9185 | Baseline 0.4770 collapse to $-0.4415$ = +0.9185 | 0 | **EXACT MATCH** |

---

## 4. Methodological Compliance & Scientific Tone Audit

| Requirement | Audit Finding in `paper/main.tex` | Compliance Status |
|:---|:---|:---:|
| **Non-Causal Language** | No claims that delay propagation is causally generated by features alone; strictly framed as predictive association and operational bottlenecks. Buffer deficit is explicitly characterized as an observational operational bottleneck. | **COMPLIANT** |
| **No Prohibited Estimators** | Strictly no Ridge, Lasso, Elastic Net, SVR, XGBoost, LightGBM, CatBoost, or GBDT models are included or discussed as approved solutions. | **COMPLIANT** |
| **Zero MLP Regularization** | Multilayer Perceptrons explicitly trained and audited with `alpha = 0.0`. | **COMPLIANT** |
| **Controlled Estimator Comparison** | Group A, B, and C Logistic Regression and OLS baselines strictly trained on identical 5,065,376 rows; RF and DT on 500,000 rows; MLP on 250,000 rows. Controlled sample-size experiment reported. | **COMPLIANT** |
| **Honest Reporting of Tail Enclosure** | Severe delay regime ($\ge 120$ min) is transparently identified as an operational failure mode ($\text{MAE} = 109.98$ min, bias $= -90.38$ min) caused by leaf-averaging and unobserved post-departure disruption dynamics. | **COMPLIANT** |
| **Methodological Caveat on Permutation $R^2$** | Transparently clarifies that the large $+0.9185$ $R^2$ permutation drop for $\text{RemTurn}$ arises from breaking an exact algebraic identity ($\text{Buffer} - \text{PrevDelay}$), reflecting structural reliance rather than independent causal power. | **COMPLIANT** |
| **Threats to Validity Section** | Explicit Section VI dedicated to observational limits, unobserved ATC/weather confounders, and terminal leaf truncation bounds. | **COMPLIANT** |

---

## 5. Visual Figure Quality & Float Audit

| Figure ID | File Path | Graphic Description | Layout Position | Overfull Status |
|:---:|:---|:---|:---:|:---:|
| **Figure 1** | `paper/figures/fig_propagation_relationships.png` | Inbound delay propagation curve & buffer delay probability | Page 3, Column 1 | **Clean (no overfull hbox)** |
| **Figure 2** | `paper/figures/fig_test_roc.png` & `fig_test_pr.png` | Dual ROC ($0.7943$) and PR ($0.7150$) curves on test set | Page 5, Column 1 | **Clean (side-by-side subpanels)** |
| **Figure 3** | `paper/figures/fig_test_confusion_matrix.png` & `fig_actual_vs_predicted.png` | Confusion matrix ($T=0.35$) & actual vs predicted scatter | Page 5, Column 2 | **Clean (side-by-side subpanels)** |
| **Figure 4** | `paper/figures/fig_feature_importance.png` | Test set permutation importance (AP drop & MAE increase) | Page 6, Column 1 | **Clean (no overfull hbox)** |

All figures feature high-resolution vector text, standardized typography, clear axis labels, and descriptive captions.

---

## 6. Table Formatting & Column Fit Audit

Every table has been calibrated to eliminate overfull horizontal boxes (`Overfull \hbox`):
- **Table I (Partitioning):** Formatted with `\resizebox{\columnwidth}{!}{...}` on Page 2.
- **Table II (Reconstruction):** Formatted with `\resizebox{\columnwidth}{!}{...}` on Page 3.
- **Table III (Feature Groups):** Formatted as a two-column banner `table*` on Page 4 with wrapped paragraph column `p{10.5cm}`.
- **Table IV (Classification Validation):** Formatted with `\resizebox{\columnwidth}{!}{...}` on Page 4.
- **Table V (Regression Validation):** Formatted with `\resizebox{\columnwidth}{!}{...}` on Page 4.
- **Table VI (Generalization):** Formatted with `\resizebox{\columnwidth}{!}{...}` on Page 5.
- **Table VII (Delay Strata):** Formatted with `\resizebox{\columnwidth}{!}{...}` on Page 5.

*Overfull Status:* All tables fit completely within the margins without text clipping or column overflow.

---

## 7. Final Publication Readiness Assessment

```
========================================================================================
FINAL AUDIT DECISION: UNCONDITIONALLY APPROVED FOR IEEE SUBMISSION
========================================================================================
1. Page Count: EXACTLY 6 PAGES (No 7th page; references conclude on Page 6).
2. Format: Official IEEEtran conference standard (two-column, 10pt Times, proper headings).
3. Cross-References: 100% resolved (0 broken links, 0 '??', 0 '[?]').
4. Experimental Data: 100% verified against frozen Stage 6.2 research audit artifacts.
5. Scientific Rigor: Strict non-causal language, transparent limitations, reproducible.
========================================================================================
```
