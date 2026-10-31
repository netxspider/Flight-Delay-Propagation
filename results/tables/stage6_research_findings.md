# Phase 6 — Stage 6: Final Research Findings & Methodology Synthesis

**Research Paper Title:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Research Question:** *To what extent can delays from previous flight operations predict delays in subsequent flights operated by the same aircraft?*  
**Empirical Foundation:** 6.77 Million Completed Consecutive Rotation Pairs (BTS 2025 On-Time Performance Data)  
**Final Test Population:** $N = 562,202$ Independent Operations (December 2025)  
**Winning Model Family:** Random Forest (Group C: Turnaround Physics + Flight Context + Daily Aircraft History)  

---

## 1. Executive Research Synthesis

This study investigated the degree to which inbound delays from an aircraft's previous leg propagate into its subsequent flight operations. Using 6,768,487 domestic commercial rotation pairs reconstructed from the 2025 BTS database across 6,124 physical aircraft, we trained, validated, and independently tested machine learning architectures across three hierarchical feature representations (Group A: turnaround mechanics; Group B: route and carrier context; Group C: cumulative same-day airframe history).

### Primary Research Answer
**Delays from previous flight operations possess strong, statistically reliable predictive power for subsequent flight delays, accounting for nearly half ($R^2 = 0.4851$) of the total variance in subsequent arrival delays and enabling high-discrimination classification ($\text{ROC-AUC} = 0.7943$, $\text{AP} = 0.7150$) on out-of-sample temporal test data.**

However, the nature of this predictability is subject to three essential scientific qualifications:
1. **Asymmetric Physical Bottleneck:** The predictive association is non-linear and operates primarily through the turnaround buffer deficit ($\text{ESTIMATED\_REMAINING\_TURNAROUND\_MINUTES} < 0$). When an inbound delay exceeds the scheduled ground cushion, propagation risk jumps abruptly; when buffer remains positive, inbound delay has minimal predictive impact.
2. **Context and Aircraft History Combined Signal:** Incorporating scheduled network context (Group B) alongside same-day airframe operational stress (Group C) provides an incremental boost over turnaround mechanics alone (Group A) for Random Forest ($+1.08\%$ ROC-AUC to $0.7932$ on validation; $0.7943$ on test). For linear models where all three groups were evaluated independently, isolated airframe history (Group B $\rightarrow$ Group C) contributed $+0.0077$ ROC-AUC and $+0.0125$ AP, and permutation feature importance confirms that rotational history features carry distinct non-redundant predictive signal.
3. **Severe Delay Degradation:** While the model accurately estimates on-time, minor, and moderate delays ($MAE \approx 10–22$ min across $90\%$ of flights), it systematically underpredicts severe delays ($\ge 120$ min, $MAE = 109.98$ min, bias = $-90.38$ min), reflecting unobserved external compound events (e.g. FAA ground stop programs, winter convective weather) that exceed rotational turnaround mechanics.

---

## 2. Detailed Findings Addressing Core Research Questions

### Question 1: How well does the classifier identify subsequent arrival delays?
On the completely held-out December 2025 test dataset ($N = 562,202$):
- **Overall Discrimination:** The Random Forest classifier achieved an out-of-sample **ROC-AUC of $0.7943$** and an **Average Precision (AP) of $0.7150$** (relative to a $26.48\%$ baseline delay prevalence).
- **Operational Performance at $T = 0.35$:**
  - **Precision:** **$72.52\%$** (nearly 3 out of every 4 positive predictions correctly identify a propagated delay $\ge 15$ minutes).
  - **Recall:** **$53.24\%$** (identifies more than half of all actual propagated delays: $79,250$ of $148,865$).
  - **Specificity:** **$92.74\%$** (false alarm rate on on-time flights is held to $7.26\%$).
  - **F1-Score:** **$0.6140$**.
- **Comparison to Default Benchmark ($T = 0.50$):**
  - Operating at the standard threshold ($T = 0.50$) produces higher precision ($87.52\%$) but suffers severe recall collapse, missing **$54.93\%$** of all delays ($81,772$ false negatives). Calibrating the threshold to $0.35$ recovers an additional **$12,157$ delayed flights** (+18.1% gain in true positives) with minimal erosion of specificity ($97.68\% \rightarrow 92.74\%$).

### Question 2: How well does the regression model estimate delay magnitude?
- **Variance Explained:** The Random Forest regressor achieved **$R^2 = 0.4851$**, demonstrating that approximately **$48.5\%$ of the variation in subsequent arrival delays is accounted for by the combination of prior delay, turnaround cushion, network context, and airframe history.**
- **Absolute Error:** Out-of-sample **MAE is $19.77$ minutes** and **RMSE is $45.86$ minutes**.
- **Residual Distribution:** The unconditional mean residual is **$+1.21$ minutes**, indicating minimal aggregate global bias. Half of all predictions fall within an error margin between $-13.2$ and $+10.8$ minutes (IQR $\approx 24$ min).

### Question 3: Does the model generalize from development to December?
- **Yes, with remarkable stability.**
  - Classification ROC-AUC transferred from $0.7932$ (validation) to **$0.7943$** (test), a difference of only $+0.0011$.
  - F1-score improved from $0.5740$ to **$0.6140$** due to the higher prevalence of delays in December.
  - Regression $R^2$ improved from $0.4516$ to **$0.4851$**.
- **Distribution Shift Impact:** December experienced a significant temporal distribution shift (delay prevalence rose from $20.21\%$ to $26.48\%$, and severe delays $\ge 120$ min rose by $+55.4\%$). While overall MAE increased from $17.40$ to $19.77$ minutes, error within each individual delay band remained virtually constant (e.g. moderate delay MAE was $22.44$ min on validation and $22.27$ min on test). A formal shift-share decomposition proves that $94.0\%$ to $98.4\%$ ($+2.23$ to $+2.33$ minutes) of the $+2.37$ minute test MAE increase is mathematically accounted for by population composition shift (a higher proportion of severe delays carrying higher baseline errors), while within-band error remained virtually flat ($+0.04$ to $+0.25$ min net within-band effect). While winter convective storms and holiday travel volumes are plausible contextual hypotheses for this shift, they are unobserved in the BTS flight records and cannot be asserted as proven causal drivers.

### Question 4: How does performance vary with delay severity?
The empirical results reveal a clear dichotomy across operational regimes:
1. **On-Time & Minor Delays ($\le 14$ min, $73.5\%$ of operations):** The model is highly accurate, with MAE between **$10.6$ and $13.0$ minutes**. The model displays a mild positive bias ($+12.27$ min on early flights), safely erring on the side of caution.
2. **Moderate Delays ($15–59$ min, $16.8\%$ of operations):** MAE is **$22.27$ minutes**, with an average bias of $-13.64$ minutes.
3. **Severe Delays ($\ge 120$ min, $4.0\%$ of operations):** Performance degrades sharply, with MAE reaching **$109.98$ minutes** and a large negative prediction bias (**$-90.38$ minutes**).
- **Physical Explanation:** When delays exceed 2 hours, ground operations are no longer governed by normal gate turnaround buffering. Instead, compound external constraints dominate—such as flight crew legal duty-time expirations, gate occupancy gridlocks, aircraft equipment substitutions, and FAA Ground Delay Programs. Observational turn features cannot predict the duration of these administrative and regulatory interventions.

### Question 5: Does aircraft operational history contribute useful predictive information?
- **Yes.** Permutation feature importance on the independent test dataset ($N = 100,000$, 5 repeats) and baseline ablation studies confirm that Group C features provide distinct predictive value:
  - `PREV_AIRCRAFT_CONSECUTIVE_DEL15` ranked as the **$5^{\text{th}}$ most important predictor** out of 18 features (Mean AP drop: $+0.0082$, Mean MAE increase: $+0.84$ min).
  - `PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY` ranked as the **$8^{\text{th}}$ most important predictor** (Mean AP drop: $+0.0036$, Mean MAE increase: $+0.22$ min).
  - In feature group ablations, Group C (turnaround + route context + airframe history) delivered a statistically verified $+1.08\%$ gain in validation ROC-AUC ($0.7824 \rightarrow 0.7932$) and $+0.0076$ gain in $R^2$ ($0.4440 \rightarrow 0.4516$) over turnaround mechanics alone (Group A).
  - Because Random Forest was not trained on Group B in Stage 4/5, this comparison reflects the *combined* addition of route context (Group B) and airframe history (Group C). However, linear baseline models (Logistic Regression and OLS) trained across all three groups establish the isolated marginal contribution of airframe history (Group B $\rightarrow$ Group C): adding airframe history improved Logistic Regression ROC-AUC by $+0.0077$ ($0.7698 \rightarrow 0.7775$) and AP by $+0.0125$ ($0.6102 \rightarrow 0.6228$), and improved OLS $R^2$ by $+0.0110$ ($0.2644 \rightarrow 0.2754$). Furthermore, permutation feature importance directly isolates individual Group C features within the Random Forest model, confirming that rotational history provides non-redundant predictive signal.
  - **Permutation Importance $R^2$ Drop Clarification:** For `ESTIMATED_REMAINING_TURNAROUND_MINUTES`, the baseline evaluation sample $R^2$ was $+0.4770$. When this feature was permuted, the resulting $R^2$ collapsed to $-0.4415$, producing an arithmetic $R^2$ drop of $+0.9185$. Because remaining turnaround is an exact algebraic identity ($SCHEDULED\_TURNAROUND\_BUFFER - PREV\_ARR\_DELAY$), permuting it in isolation breaks the physical dependency with prior delay and scheduled buffer, creating contradictory feature combinations (e.g., negative remaining buffer alongside zero prior delay) that disrupt the tree split logic. This dramatic drop demonstrates the ensemble's structural reliance on this derived feature, rather than a causal effect.
- **Operational Interpretation:** When an airframe accumulates consecutive delays across multiple daily legs, rotational slack in downstream schedules is eroded, and maintenance or crew fatigue compounds, increasing propagation sensitivity.

---

## 3. Scientific Limitations & What Cannot Be Concluded

### 3.1 Observational vs. Causal Attribution
- **Crucial Distinction:** **Predictive association does NOT prove causality.**
- While `ESTIMATED_REMAINING_TURNAROUND_MINUTES` is the strongest predictor (accounting for a $+15.81$ minute increase in MAE under permutation), it is an observational mathematical construct derived from $SCHEDULED\_TURNAROUND\_BUFFER - PREV\_ARR\_DELAY$.
- Finding that a negative turnaround buffer predicts a subsequent delay does **not** prove that adding buffer minutes will causally eliminate downstream delays in a 1:1 counterfactual intervention. In practice, schedule padding induces behavioral adjustments by airlines (e.g. relaxed ground crew pacing, altered taxi procedures, or tighter aircraft rotations elsewhere in the network).

### 3.2 Unobserved Confounding Variables in BTS Data
The BTS On-Time Performance dataset contains only scheduled and actual operational timestamps. It lacks several critical operational variables:
1. **Weather Intensity:** Terminal Doppler Weather Radar (TDWR), convective storm positions, and de-icing queue wait times are unobserved.
2. **Air Traffic Control (ATC) Restrictions:** FAA Air Traffic Control System Command Center (ATCSCC) Ground Delay Programs, Ground Stops, and Miles-in-Trail restrictions are not directly recorded in flight-pair records.
3. **Crew Schedule Mechanics:** Crew duty limits (FAR Part 117), crew swaps, and crew misconnections sharing the same flight are unrecorded.
4. **Airport Gate Congestion:** Gate hold times and taxiway congestion are unobserved at prediction time.

### 3.3 Extrapolation Limits of Tree Ensembles
Random Forest regressors produce piecewise constant predictions bounded by the training target values within terminal leaf nodes. Consequently, the model cannot extrapolate to unprecedented delay durations ($> 300$ minutes) that occur during catastrophic blizzard shutdowns or nationwide ground stops.

---

## 4. Methodological Conclusions for IEEE Publication

1. **Validity of the Research Hypothesis:** The empirical results strongly support the hypothesis that flight delays propagate systematically across consecutive aircraft operations. Turnaround cushion mechanics represent the single most important operational vulnerability in commercial airline networks.
2. **Reproducibility & Integrity:** All test evaluations were executed on frozen models loaded from disk, without refitting, hyperparameter tuning, or threshold search on December data. All confusion matrix totals reconcile to $562,202$ observations with zero arithmetic discrepancy.
3. **Deployment Implications:** An operational threshold of $T = 0.35$ provides an actionable dispatch alert tool for airline operations control centers (AOCCs), identifying $53.24\%$ of propagated delays with $72.52\%$ reliability ($79,250$ of $148,865$ delayed flights), providing dispatchers with 45–90 minutes of advance notice before scheduled departure. However, this threshold represents an operational trade-off balancing alert precision and coverage, rather than a proven cost-optimal policy, as specific financial costs of delay mitigation interventions versus false alarms are not parameterized in BTS data.
