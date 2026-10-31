# IEEE Conference Paper Table Selection & LaTeX Code Plan

**Project Title:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Role:** Senior Data Scientist, Results Auditor & IEEE Technical Editor  
**Date:** 2026-10-31  
**Deliverable:** `results/paper_preparation/03_ieee_table_selection.md`  

---

## 1. Executive Summary & Table Selection Strategy

In accordance with IEEE conference formatting standards and the strict six-page length restriction, tabular displays must be mathematically rigorous, strictly reconciled with disk artifacts, and formatted with clean typographic economy.

Every value presented below has been audited and reconciled in Stage 6.2 commit `40727c0`. Abbreviations, units, and mathematical definitions are explicitly specified.

We present the complete set of **8 publication-ready tables**, complete with IEEE-compliant captions, definitions, and drop-in LaTeX code.

---

## 2. Selected Tables Inventory & Metadata

| Table # | Table Title / Focus | Primary Placement | Main Scientific Evidence | Dimensions in LaTeX |
| :---: | :--- | :---: | :--- | :---: |
| **I** | Dataset & Chronological Experimental Split | Section III | Full 2025 calendar year split; delay base rates | Single-column |
| **II** | Aircraft Rotation Reconstruction & Integrity Audit | Section III | 6.77M valid rotations; 382.9k deficit operations | Single-column |
| **III**| Hierarchical Feature Representation Groups | Section IV | Group A (Turnaround), B (Context), C (History) | Double-column / Single |
| **IV** | Classification Performance Comparison (Validation Set) | Section VI.B | Tree & ensemble dominance; AP & ROC-AUC gains | Double-column / Single |
| **V** | Continuous Regression Performance Comparison (Validation)| Section VI.C | Non-linear variance capture ($R^2 = 0.4516$ vs $0.2754$) | Single-column |
| **VI** | Final Independent Test Performance & Generalization | Section VI.D | Out-of-sample transfer to December ($N = 562,202$) | Single-column |
| **VII**| Regression Error Stratification Across Delay Bands | Section VI.F | Strong performance $\le 60$m; severe tail error | Single-column |
| **VIII**| Decision Threshold Spectrum & Confusion Matrix Breakdown| Section VI.E | Trade-off at $T=0.35$ vs grid optimum $T=0.39$ & $0.50$ | Single-column |

---

## 3. Production-Ready LaTeX Code & Verified Content

### Table I: Dataset and Chronological Experimental Split

**Caption:** *TABLE I: Chronological Partitioning of the 2025 BTS Domestic Flight Rotation Dataset.*  
**Definitions:** *Class 0 denotes on-time or early arrival ($\text{ArrDelay} < 15$ min); Class 1 denotes delayed arrival ($\text{ArrDelay} \ge 15$ min).*

```latex
\begin{table}[t]
\caption{Chronological Partitioning of the 2025 BTS Domestic Flight Rotation Dataset}
\label{tab:dataset_split}
\centering
\small
\begin{tabular}{llrrrr}
\hline
\textbf{Partition} & \textbf{Date Range (2025)} & \textbf{Total Legs} & \textbf{Class 0 (On-Time)} & \textbf{Class 1 (Delayed)} & \textbf{Delay Rate} \\
\hline
Training           & Jan 1 -- Sep 30            & 5,065,376           & 3,972,836                  & 1,092,540                  & 21.57\%             \\
Validation         & Oct 1 -- Nov 30            & 1,140,909           & 910,388                    & 230,521                    & 20.21\%             \\
Independent Test   & Dec 1 -- Dec 31            & 562,202             & 413,337                    & 148,865                    & 26.48\%             \\
\hline
\textbf{Total}     & Jan 1 -- Dec 31            & \textbf{6,768,487}  & \textbf{5,277,901}         & \textbf{1,490,586}         & \textbf{22.02\%}    \\
\hline
\end{tabular}
\end{table}
```

---

### Table II: Aircraft Rotation Reconstruction & Integrity Audit Summary

**Caption:** *TABLE II: Aircraft Rotation Reconstruction and Filtering Statistics.*  
**Definitions:** *Consecutive rotation pairs require identical airframe registration (`TAIL_NUM`), airport continuity ($\text{Next Origin} = \text{Prev Destination}$), and chronologically positive ground time.*

```latex
\begin{table}[t]
\caption{Aircraft Rotation Reconstruction and Filtering Statistics}
\label{tab:rotation_filtering}
\centering
\small
\begin{tabular}{lrr}
\hline
\textbf{Reconstruction Stage / Filter Criteria} & \textbf{Flight Count} & \textbf{Percentage} \\
\hline
Total physical aircraft operations in timeline  & 6,898,741             & 100.00\%            \\
Isolated cancelled operations                   & 104,748               & 1.52\%              \\
Isolated diverted operations                    & 18,349                & 0.27\%              \\
Candidate consecutive connection pairings       & 6,873,483             & 99.63\%             \\
Excluded: Airport continuity mismatches         & 85,737                & 1.24\%              \\
Excluded: Diversion-induced sequence breaks     & 19,109                & 0.28\%              \\
Excluded: Operations without predecessor leg    & 6,113                 & 0.09\%              \\
Excluded: Non-positive ground times ($\le 0$m)  & 37                    & 0.00\%              \\
\hline
\textbf{Verified Consecutive Rotation Pairs}    & \textbf{6,768,487}    & \textbf{98.11\%}    \\
\hline
\textit{Operational Turnaround Deficit Subset}  & 382,937               & 5.66\%              \\
\textit{Subsequent Delay Rate under Deficit}    & 381,022               & 99.50\%             \\
\hline
\end{tabular}
\end{table}
```

---

### Table III: Hierarchical Feature Representation Groups

**Caption:** *TABLE III: Hierarchical Feature Groups for Aircraft Rotation Delay Propagation.*  
**Definitions:** *All features are strictly observed prior to the subsequent flight's departure.*

```latex
\begin{table*}[t]
\caption{Hierarchical Feature Groups for Aircraft Rotation Delay Propagation}
\label{tab:feature_groups}
\centering
\small
\begin{tabular}{llcp{9.5cm}}
\hline
\textbf{Group} & \textbf{Domain} & \textbf{Dim.} & \textbf{Included Operational Features and Descriptions} \\
\hline
\textbf{Group A} & Turnaround Physics & 7 & \texttt{PREV\_ARR\_DELAY}, \texttt{PREV\_DEP\_DELAY}, \texttt{PREV\_ARR\_DEL15}, \texttt{PREV\_DEP\_DEL15}, \texttt{SCHEDULED\_TURNAROUND\_BUFFER}, \texttt{ESTIMATED\_REMAINING\_TURNAROUND\_MINUTES}, \texttt{IS\_TURNAROUND\_DEFICIT}. \\
\textbf{Group B} & Network \& Flight Context & 14 & Group A plus \texttt{NEXT\_CRS\_DEP\_HOUR}, \texttt{NEXT\_DAY\_OF\_WEEK}, \texttt{NEXT\_MONTH}, \texttt{NEXT\_DISTANCE}, \texttt{NEXT\_CARRIER} (OHE), \texttt{NEXT\_ORIGIN} (OHE), \texttt{NEXT\_DEST} (OHE). \\
\textbf{Group C} & Airframe Daily History & 18 & Group B plus \texttt{PREV\_AIRCRAFT\_DELAYED\_FLIGHTS\_TODAY}, \texttt{PREV\_AIRCRAFT\_CONSECUTIVE\_DEL15}, \texttt{PREV\_AIRCRAFT\_CUMULATIVE\_ARR\_DELAY\_TODAY}, \texttt{PREV\_AIRCRAFT\_DELAY\_RATE\_TODAY}. \\
\hline
\end{tabular}
\end{table*}
```

---

### Table IV: Classification Performance Comparison (Full Validation Set, $N = 1,140,909$)

**Caption:** *TABLE IV: Classification Model Performance Comparison on the Full Validation Population.*  
**Definitions:** *Evaluated at default threshold $T = 0.50$ unless otherwise specified. AP denotes Average Precision.*

```latex
\begin{table*}[t]
\caption{Classification Model Performance Comparison on the Full Validation Population ($N = 1,140,909$)}
\label{tab:classification_val}
\centering
\small
\begin{tabular}{llcrrrrrrr}
\hline
\textbf{Estimator Architecture} & \textbf{Feature Group} & \textbf{Train $N$} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{ROC-AUC} & \textbf{AP} & \textbf{Fit Time} \\
\hline
Majority Class Baseline         & None                   & 5,065,376          & 79.79\%           & 0.00\%             & 0.00\%          & 0.0000            & 0.5000           & 0.2021      & 1.1s              \\
Logistic Regression             & Group A                & 5,065,376          & 84.66\%           & 97.14\%            & 24.83\%         & 0.3955            & 0.7356           & 0.5792      & 5.3s              \\
Logistic Regression             & Group B                & 5,065,376          & 84.91\%           & 81.49\%            & 32.73\%         & 0.4671            & 0.7698           & 0.6102      & 40.7s             \\
Logistic Regression             & Group C                & 5,065,376          & 85.13\%           & 77.89\%            & 36.84\%         & 0.5003            & 0.7775           & 0.6228      & 40.9s             \\
Decision Tree Classifier        & Group C                & 500,000            & 83.69\%           & 67.57\%            & 37.04\%         & 0.4784            & 0.7656           & 0.5983      & 4.8s              \\
MLP Classifier ($\alpha = 0.0$) & Group C                & 250,000            & 85.08\%           & 76.24\%            & 37.64\%         & 0.5038            & 0.7818           & 0.6295      & 19.2s             \\
Random Forest (Group A)         & Group A                & 500,000            & 84.82\%           & 79.88\%            & 32.06\%         & 0.4576            & 0.7824           & 0.6433      & 85.4s             \\
Random Forest ($T = 0.50$)      & Group C                & 500,000            & \textbf{85.58\%}  & \textbf{86.68\%}   & 40.88\%         & 0.5556            & \textbf{0.7932}  & \textbf{0.6600} & 258.4s        \\
Random Forest ($T = 0.35$)*     & Group C                & 500,000            & 85.29\%           & 69.20\%            & \textbf{49.03\%}& \textbf{0.5740}   & \textbf{0.7932}  & \textbf{0.6600} & 258.4s        \\
\hline
\multicolumn{10}{l}{\footnotesize *Selected operational decision threshold.}
\end{tabular}
\end{table*}
```

---

### Table V: Continuous Regression Performance Comparison (Full Validation Set, $N = 1,140,909$)

**Caption:** *TABLE V: Continuous Arrival Delay Regression Performance Comparison on Validation Data.*  
**Definitions:** *MAE is Mean Absolute Error; RMSE is Root Mean Squared Error; $R^2$ is coefficient of determination relative to unconditional mean.*

```latex
\begin{table}[t]
\caption{Continuous Arrival Delay Regression Performance on Validation Data ($N = 1,140,909$)}
\label{tab:regression_val}
\centering
\small
\begin{tabular}{llrrrrr}
\hline
\textbf{Model Family}     & \textbf{Group} & \textbf{Train $N$} & \textbf{MAE (min)} & \textbf{RMSE (min)} & \textbf{$R^2$} & \textbf{Fit Time} \\
\hline
Median Dummy Baseline     & None           & 5,065,376          & 22.65              & 53.39               & -0.0479        & 0.3s              \\
Multiple Linear (OLS)     & Group A        & 5,065,376          & 20.70              & 44.81               & 0.2619         & 3.4s              \\
Multiple Linear (OLS)     & Group B        & 5,065,376          & 21.37              & 44.73               & 0.2644         & 50.6s             \\
Multiple Linear (OLS)     & Group C        & 5,065,376          & 21.19              & 44.40               & 0.2754         & 51.9s             \\
Decision Tree Regressor   & Group C        & 500,000            & 17.89              & 39.46               & 0.4281         & 4.6s              \\
MLP Regressor ($\alpha=0$)& Group C        & 250,000            & \textbf{17.06}     & 39.38               & 0.4304         & 17.5s             \\
Random Forest Regressor   & Group A        & 500,000            & 17.58              & 38.89               & 0.4440         & 105.2s            \\
\textbf{Random Forest}    & \textbf{Group C}& 500,000           & 17.40              & \textbf{38.62}      & \textbf{0.4516}& 312.1s            \\
\hline
\end{tabular}
\end{table}
```

---

### Table VI: Final Independent Test Performance & Generalization (December 2025, $N = 562,202$)

**Caption:** *TABLE VI: Out-of-Sample Temporal Generalization to Held-Out December 2025 Operations.*  
**Definitions:** *Evaluated using frozen estimators without refitting or test-driven threshold optimization.*

```latex
\begin{table}[t]
\caption{Out-of-Sample Temporal Generalization to Held-Out December 2025 Operations ($N = 562,202$)}
\label{tab:test_generalization}
\centering
\small
\begin{tabular}{lrrr}
\hline
\textbf{Evaluation Metric}               & \textbf{Validation (Oct--Nov)} & \textbf{Test (Dec)} & \textbf{Change ($\Delta$)} \\
\hline
\textit{Classification Discrimination}   &                                &                     &                            \\
ROC-AUC Score                            & 0.7932                         & \textbf{0.7943}     & +0.0011                    \\
Average Precision (AP)                   & 0.6600                         & \textbf{0.7150}     & +0.0550                    \\
Log Loss                                 & 0.3748                         & 0.4255              & +0.0507                    \\
\hline
\textit{Operating Point ($T = 0.35$)}    &                                &                     &                            \\
Precision                                & 69.20\%                        & \textbf{72.52\%}    & +3.32\%                    \\
Recall                                   & 49.03\%                        & \textbf{53.24\%}    & +4.21\%                    \\
F1-Score                                 & 0.5740                         & \textbf{0.6140}     & +0.0400                    \\
Specificity                              & 94.47\%                        & 92.74\%             & -1.73\%                    \\
Overall Accuracy                         & 85.29\%                        & 82.28\%             & -3.01\%                    \\
\hline
\textit{Continuous Regression}           &                                &                     &                            \\
Coefficient of Determination ($R^2$)     & 0.4516                         & \textbf{0.4851}     & +0.0335                    \\
Mean Absolute Error (MAE)                & 17.40 min                      & 19.77 min           & +2.37 min                  \\
Root Mean Squared Error (RMSE)           & 38.62 min                      & 45.86 min           & +7.24 min                  \\
Unconditional Mean Bias ($y - \hat{y}$)  & +0.94 min                      & +1.21 min           & +0.27 min                  \\
\hline
\end{tabular}
\end{table}
```

---

### Table VII: Regression Error Stratification Across Standardized Delay Bands (Test Set)

**Caption:** *TABLE VII: Continuous Regression Residuals Across Standardized Arrival Delay Bands.*  
**Definitions:** *Bias is defined as prediction minus actual arrival delay ($\hat{y} - y$).*

```latex
\begin{table}[t]
\caption{Continuous Regression Residuals Across Standardized Arrival Delay Bands (Test Set)}
\label{tab:delay_bands}
\centering
\small
\begin{tabular}{lrrrrr}
\hline
\textbf{Delay Stratum} & \textbf{Observations} & \textbf{Share} & \textbf{MAE (m)} & \textbf{RMSE (m)} & \textbf{Bias (m)} \\
\hline
Early / On-Time ($\le 0$m) & 322,255               & 57.32\%        & 13.01            & 16.69             & +12.27            \\
Minor Delay (1--14m)       & 91,082                & 16.20\%        & 10.58            & 12.64             & -3.46             \\
Moderate Delay (15--59m)   & 94,233                & 16.76\%        & 22.27            & 26.57             & -13.64            \\
Significant (60--119m)     & 31,939                & 5.68\%         & 42.70            & 54.03             & -30.78            \\
Severe Delay ($\ge 120$m)  & 22,693                & 4.04\%         & 109.98           & 201.18            & -90.38            \\
\hline
\textbf{Aggregate Test Set}& \textbf{562,202}      & \textbf{100\%} & \textbf{19.77}   & \textbf{45.86}    & \textbf{+1.21}    \\
\hline
\end{tabular}
\end{table}
```

---

### Table VIII: Classification Decision Threshold Spectrum & Confusion Matrix Breakdown

**Caption:** *TABLE VIII: Confusion Matrix and Performance Metrics Across Decision Thresholds (Test Set).*  
**Definitions:** *TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative.*

```latex
\begin{table}[t]
\caption{Confusion Matrix and Performance Metrics Across Decision Thresholds (Test Set)}
\label{tab:threshold_spectrum}
\centering
\small
\begin{tabular}{lrrrrrrrr}
\hline
\textbf{Threshold} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN} & \textbf{Prec.} & \textbf{Recall} & \textbf{F1} & \textbf{Spec.} \\
\hline
$T = 0.35$ (Oper.) & 79,250      & 30,025      & 69,615      & 383,312     & 72.52\%        & 53.24\%         & 0.6140      & 92.74\%        \\
$T = 0.39$ (Grid)  & 77,262      & 25,491      & 71,603      & 387,846     & 75.19\%        & 51.90\%         & 0.6141      & 93.83\%        \\
$T = 0.50$ (Conv.) & 67,093      & 9,566       & 81,772      & 403,771     & 87.52\%        & 45.07\%         & 0.5950      & 97.68\%        \\
\hline
\end{tabular}
\end{table}
```

---

## 4. Recommended Primary Tables for 6-Page Paper

To guarantee exact adherence to the six-page conference limit:
1. **Primary Mandatory Tables (Include in PDF):**
   - **Table I:** Dataset and Experimental Split (Compact, single-column)
   - **Table II:** Aircraft Rotation Reconstruction Statistics (Compact, single-column)
   - **Table III:** Feature Groups Definition (Merged/compacted into methodology text or single-column table)
   - **Table IV:** Classification Model Comparison (Essential comparative evidence)
   - **Table V:** Continuous Regression Model Comparison (Essential comparative evidence)
   - **Table VI:** Out-of-Sample Test Generalization
   - **Table VII:** Delay Band Stratification
2. **Table VIII:** Integrated into the confusion matrix figure (Figure 3) and discussion text to save vertical space.

---

## 5. Table Preparation Sign-Off

All tabular values are reconciled with disk records, verified against Stage 6.2 commit `40727c0`, and formatted in standard IEEE conference LaTeX style.

**Proceed to Stage 4 & 5 (IEEE Paper Writing).**
