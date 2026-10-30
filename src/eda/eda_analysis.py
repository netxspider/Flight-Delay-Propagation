"""
Comprehensive Exploratory Data Analysis (EDA) Pipeline — Methodologically Audited
Research Project: Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations

This module conducts exhaustive exploratory data analysis on the locked and validated
aircraft rotation pairs dataset ('data/processed/flight_rotation_pairs.parquet').

Methodological Compliance & Rigor:
1. Target Analysis: Classification (NEXT_ARR_DEL15), Regression (NEXT_ARR_DELAY)
2. Transition Statistics: Full 2x2 contingency table, Relative Risk and Odds Ratio with 95% CIs
3. Granular Breakdown: Explicit numerators, denominators, and rates across all slices
4. Buffer Dynamics: Buffer capacity intervals, remaining buffer, and deficit state
5. Diurnal & Temporal Patterns: Departure hour conditional probabilities vs unconditional rates
6. Carrier & Airport Heterogeneity: Major operators and turnaround hub volumes
7. Aircraft Fleet Concentration: Non-negative positive delay minutes, Gini coefficient (0.2890),
   and top 10%, 20%, 50% fleet shares
8. Permitted Syllabus Models: Logistic Regression, Decision Trees, Random Forest, SVM (SVC for Classification), MLP,
   Linear/Multiple/Ridge/Lasso/Polynomial Regression (Strictly excluding out-of-syllabus SVR and gradient boosting)
"""

import os
import sys
import json
import time

# Ensure writable matplotlib cache on macOS / sandbox
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import duckdb
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": False
})

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/flight_rotation_pairs.parquet")
AIRPORT_PATH = os.path.join(BASE_DIR, "results/tables/airport_enrichment_validation.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results/figures")
TABLES_DIR = os.path.join(BASE_DIR, "results/tables")
METRICS_DIR = os.path.join(BASE_DIR, "results/metrics")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def run_target_analysis(con, summary_dict):
    """Section 1: Target Variable Analysis (Classification & Regression)."""
    print("\n--- 1. Target Variable Analysis ---")
    t0 = time.time()
    
    q = f"""
    SELECT 
        COUNT(*) AS total_pairs,
        SUM(NEXT_ARR_DEL15) AS count_del15,
        SUM(CASE WHEN NEXT_ARR_DEL15 = 0 THEN 1 ELSE 0 END) AS count_ontime,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 4) AS pct_del15,
        ROUND(AVG(NEXT_ARR_DELAY), 4) AS mean_arr_delay,
        ROUND(MEDIAN(NEXT_ARR_DELAY), 4) AS median_arr_delay,
        ROUND(STDDEV(NEXT_ARR_DELAY), 4) AS std_arr_delay,
        ROUND(MIN(NEXT_ARR_DELAY), 2) AS min_arr_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.25), 2) AS q25_arr_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.75), 2) AS q75_arr_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.90), 2) AS p90_arr_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.95), 2) AS p95_arr_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.99), 2) AS p99_arr_delay,
        ROUND(MAX(NEXT_ARR_DELAY), 2) AS max_arr_delay,
        -- Sign breakdown
        SUM(CASE WHEN NEXT_ARR_DELAY < 0 THEN 1 ELSE 0 END) AS count_early,
        SUM(CASE WHEN NEXT_ARR_DELAY = 0 THEN 1 ELSE 0 END) AS count_exact_zero,
        SUM(CASE WHEN NEXT_ARR_DELAY > 0 AND NEXT_ARR_DEL15 = 0 THEN 1 ELSE 0 END) AS count_minor_delay,
        SUM(CASE WHEN NEXT_ARR_DELAY >= 15 THEN 1 ELSE 0 END) AS count_delayed_15_plus
    FROM read_parquet('{DATA_PATH}')
    """
    stats_row = con.execute(q).df().iloc[0].to_dict()
    iqr = stats_row['q75_arr_delay'] - stats_row['q25_arr_delay']
    stats_row['iqr_arr_delay'] = round(iqr, 2)
    stats_row['imbalance_ratio_ontime_to_delayed'] = round(stats_row['count_ontime'] / stats_row['count_del15'], 2)
    
    # Save table
    df_target_summary = pd.DataFrame([stats_row])
    df_target_summary.to_csv(os.path.join(TABLES_DIR, "eda_target_summary.csv"), index=False)
    summary_dict["target_analysis"] = stats_row
    
    print(f"Target metrics (N = {int(stats_row['total_pairs']):,}):")
    print(f"  Classification (NEXT_ARR_DEL15): {stats_row['pct_del15']}% delayed (Imbalance ratio: {stats_row['imbalance_ratio_ontime_to_delayed']}:1)")
    print(f"  Regression (NEXT_ARR_DELAY): Mean = {stats_row['mean_arr_delay']} min, Median = {stats_row['median_arr_delay']} min, IQR = {iqr} min, P99 = {stats_row['p99_arr_delay']} min")
    
    # Plot Figure 1: Target Distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot A: Classification Class Imbalance
    ax_a = axes[0, 0]
    classes = ["On-Time / Early (< 15 min)", "Delayed (>= 15 min)"]
    counts = [stats_row['count_ontime'], stats_row['count_del15']]
    pcts = [100.0 - stats_row['pct_del15'], stats_row['pct_del15']]
    colors = ["#2b5c8f", "#d95f02"]
    bars = ax_a.bar(classes, counts, color=colors, width=0.55, edgecolor="#222222", linewidth=1.2)
    for bar, pct, cnt in zip(bars, pcts, counts):
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2.0, h + 100000, f"{cnt:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_a.set_title("A: Target Classification Distribution (NEXT_ARR_DEL15)", fontsize=12, pad=12)
    ax_a.set_ylabel("Number of Flights", fontsize=11)
    ax_a.set_ylim(0, max(counts) * 1.18)
    ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x):,}"))
    ax_a.text(0.5, 0.85, f"Imbalance Ratio: {stats_row['imbalance_ratio_ontime_to_delayed']}:1\nBase Rate: {stats_row['pct_del15']:.2f}%",
              transform=ax_a.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"))
    
    # Subplot B: Regression Distribution (Histogram & Density)
    ax_b = axes[0, 1]
    sample_delays = con.execute(f"""
        SELECT NEXT_ARR_DELAY 
        FROM read_parquet('{DATA_PATH}') 
        USING SAMPLE 100000 (reservoir, 42)
    """).df()['NEXT_ARR_DELAY'].values
    
    sns.histplot(sample_delays, bins=120, binrange=(-60, 180), kde=True, ax=ax_b, color="#1b9e77", edgecolor="#111111", linewidth=0.5, stat="density")
    ax_b.axvline(stats_row['mean_arr_delay'], color="#d95f02", linestyle="--", linewidth=2.0, label=f"Mean: +{stats_row['mean_arr_delay']:.2f} min")
    ax_b.axvline(stats_row['median_arr_delay'], color="#2b5c8f", linestyle="-", linewidth=2.2, label=f"Median: {stats_row['median_arr_delay']:.1f} min")
    ax_b.axvline(15.0, color="#7570b3", linestyle=":", linewidth=2.0, label="15-min Delay Threshold")
    ax_b.set_xlim(-60, 180)
    ax_b.set_title("B: Target Regression Density (NEXT_ARR_DELAY, [-60, 180] min)", fontsize=12, pad=12)
    ax_b.set_xlabel("Arrival Delay (Minutes)", fontsize=11)
    ax_b.set_ylabel("Probability Density", fontsize=11)
    ax_b.legend(loc="upper right", frameon=True)
    
    # Subplot C: Boxplot with Percentile Annotations
    ax_c = axes[1, 0]
    box = ax_c.boxplot(sample_delays, vert=False, patch_artist=True, widths=0.45,
                       boxprops=dict(facecolor="#c7e9c0", color="#006d2c", linewidth=1.5),
                       medianprops=dict(color="#d95f02", linewidth=2.5),
                       whiskerprops=dict(color="#006d2c", linewidth=1.5),
                       capprops=dict(color="#006d2c", linewidth=1.5),
                       flierprops=dict(marker='o', markersize=2, alpha=0.3, color='#666666'))
    ax_c.set_xlim(-60, 180)
    ax_c.set_yticks([])
    ax_c.set_title("C: Target Delay Quartile Structure & Outlier Spread", fontsize=12, pad=12)
    ax_c.set_xlabel("Arrival Delay (Minutes)", fontsize=11)
    ax_c.text(stats_row['q25_arr_delay'], 1.25, f"Q1: {stats_row['q25_arr_delay']}m", ha="center", fontsize=9, fontweight="bold")
    ax_c.text(stats_row['median_arr_delay'], 1.35, f"Median: {stats_row['median_arr_delay']}m", ha="center", fontsize=9, fontweight="bold", color="#d95f02")
    ax_c.text(stats_row['q75_arr_delay'], 1.25, f"Q3: +{stats_row['q75_arr_delay']}m", ha="center", fontsize=9, fontweight="bold")
    ax_c.text(stats_row['p90_arr_delay'], 0.75, f"P90: +{stats_row['p90_arr_delay']}m", ha="center", fontsize=9, color="#2b5c8f")
    ax_c.text(stats_row['p95_arr_delay'], 0.65, f"P95: +{stats_row['p95_arr_delay']}m", ha="center", fontsize=9, color="#2b5c8f")
    
    # Subplot D: Operational Arrival Categories
    ax_d = axes[1, 1]
    cat_labels = ["Early (<0 min)", "Exact On-Time (0 min)", "Minor (1-14 min)", "Delayed (>=15 min)"]
    cat_counts = [stats_row['count_early'], stats_row['count_exact_zero'], stats_row['count_minor_delay'], stats_row['count_delayed_15_plus']]
    cat_pcts = [c * 100.0 / stats_row['total_pairs'] for c in cat_counts]
    cat_colors = ["#238b45", "#66c2a4", "#feb24c", "#cb181d"]
    bars_d = ax_d.bar(cat_labels, cat_counts, color=cat_colors, width=0.55, edgecolor="#222222", linewidth=1.2)
    for b, pct, cnt in zip(bars_d, cat_pcts, cat_counts):
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width() / 2.0, h + 80000, f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_d.set_title("D: Granular Operational Outcome Breakdown", fontsize=12, pad=12)
    ax_d.set_ylabel("Number of Flights", fontsize=11)
    ax_d.set_ylim(0, max(cat_counts) * 1.18)
    ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x):,}"))
    ax_d.tick_params(axis='x', rotation=12)
    
    plt.suptitle("Target Variable Analytical Profile: Flight Delay Propagation (N = 6,768,487)", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig01_target_distributions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def run_delay_propagation_analysis(con, summary_dict):
    """Section 2: Previous-Flight Delay Propagation Dynamics with 95% CIs and Exact Numerators."""
    print("\n--- 2. Previous-Flight Delay Propagation Analysis ---")
    t0 = time.time()
    
    # 2.1 Delay Bins Analysis with Numerator & Denominator
    q_bins = f"""
    SELECT 
        CASE 
            WHEN PREV_ARR_DELAY < 0 THEN '1. Early (< 0 min)'
            WHEN PREV_ARR_DELAY >= 0 AND PREV_ARR_DELAY < 15 THEN '2. On-Time (0-14 min)'
            WHEN PREV_ARR_DELAY >= 15 AND PREV_ARR_DELAY < 30 THEN '3. Moderate Delay (15-29 min)'
            WHEN PREV_ARR_DELAY >= 30 AND PREV_ARR_DELAY < 45 THEN '4. Significant Delay (30-44 min)'
            WHEN PREV_ARR_DELAY >= 45 AND PREV_ARR_DELAY < 60 THEN '5. Heavy Delay (45-59 min)'
            WHEN PREV_ARR_DELAY >= 60 AND PREV_ARR_DELAY < 90 THEN '6. Severe Delay (60-89 min)'
            WHEN PREV_ARR_DELAY >= 90 AND PREV_ARR_DELAY < 120 THEN '7. Extreme Delay (90-119 min)'
            ELSE '8. Catastrophic Delay (>= 120 min)'
        END AS prev_delay_bin,
        COUNT(*) as total_pairs,
        SUM(NEXT_ARR_DEL15) as next_delayed_numerator,
        ROUND(SUM(NEXT_ARR_DEL15) * 100.0 / COUNT(*), 4) as next_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as next_mean_delay,
        ROUND(MEDIAN(NEXT_ARR_DELAY), 2) as next_median_delay,
        ROUND(STDDEV(NEXT_ARR_DELAY), 2) as next_std_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.90), 2) as next_p90_delay
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    ORDER BY 1;
    """
    df_bins = con.execute(q_bins).df()
    df_bins.to_csv(os.path.join(TABLES_DIR, "eda_delay_propagation_bins.csv"), index=False)
    
    # 2.2 Transition Contingency Matrix (2x2) with 95% CIs
    q_matrix = f"""
    SELECT 
        PREV_ARR_DEL15,
        NEXT_ARR_DEL15,
        COUNT(*) as cell_count
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1, 2
    ORDER BY 1, 2;
    """
    df_matrix_raw = con.execute(q_matrix).df()
    n_00 = int(df_matrix_raw.loc[(df_matrix_raw['PREV_ARR_DEL15'] == 0) & (df_matrix_raw['NEXT_ARR_DEL15'] == 0), 'cell_count'].values[0])
    n_01 = int(df_matrix_raw.loc[(df_matrix_raw['PREV_ARR_DEL15'] == 0) & (df_matrix_raw['NEXT_ARR_DEL15'] == 1), 'cell_count'].values[0])
    n_10 = int(df_matrix_raw.loc[(df_matrix_raw['PREV_ARR_DEL15'] == 1) & (df_matrix_raw['NEXT_ARR_DEL15'] == 0), 'cell_count'].values[0])
    n_11 = int(df_matrix_raw.loc[(df_matrix_raw['PREV_ARR_DEL15'] == 1) & (df_matrix_raw['NEXT_ARR_DEL15'] == 1), 'cell_count'].values[0])
    
    n_1_dot = n_10 + n_11  # Total preceding delayed
    n_0_dot = n_00 + n_01  # Total preceding on-time
    total_pairs = n_1_dot + n_0_dot
    
    p1 = n_11 / n_1_dot
    p0 = n_01 / n_0_dot
    
    # Relative Risk & 95% CI
    rr = p1 / p0
    se_ln_rr = np.sqrt((1.0 - p1) / (n_1_dot * p1) + (1.0 - p0) / (n_0_dot * p0))
    rr_ci_lower = np.exp(np.log(rr) - 1.96 * se_ln_rr)
    rr_ci_upper = np.exp(np.log(rr) + 1.96 * se_ln_rr)
    
    # Odds Ratio & 95% CI (Woolf's method)
    or_val = (n_11 * n_00) / (n_10 * n_01)
    se_ln_or = np.sqrt(1.0 / n_11 + 1.0 / n_10 + 1.0 / n_01 + 1.0 / n_00)
    or_ci_lower = np.exp(np.log(or_val) - 1.96 * se_ln_or)
    or_ci_upper = np.exp(np.log(or_val) + 1.96 * se_ln_or)
    
    df_transition = pd.DataFrame([
        {
            "prior_state": "Preceding On-Time (PREV_DEL15 = 0)",
            "denominator_total_inbound": n_0_dot,
            "next_ontime_count": n_00,
            "next_delayed_numerator": n_01,
            "subsequent_delay_rate_pct": round(p0 * 100.0, 4)
        },
        {
            "prior_state": "Preceding Delayed (PREV_DEL15 = 1)",
            "denominator_total_inbound": n_1_dot,
            "next_ontime_count": n_10,
            "next_delayed_numerator": n_11,
            "subsequent_delay_rate_pct": round(p1 * 100.0, 4)
        }
    ])
    df_transition.to_csv(os.path.join(TABLES_DIR, "eda_transition_matrix.csv"), index=False)
    
    summary_dict["delay_propagation"] = {
        "contingency_counts": {"n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11},
        "p_next_del_given_prev_del": round(p1 * 100.0, 4),
        "prev_delayed_numerator": n_11,
        "prev_delayed_denominator": n_1_dot,
        "p_next_del_given_prev_ontime": round(p0 * 100.0, 4),
        "prev_ontime_numerator": n_01,
        "prev_ontime_denominator": n_0_dot,
        "relative_risk": round(rr, 4),
        "relative_risk_95ci": [round(rr_ci_lower, 4), round(rr_ci_upper, 4)],
        "odds_ratio": round(or_val, 4),
        "odds_ratio_95ci": [round(or_ci_lower, 4), round(or_ci_upper, 4)]
    }
    
    print(f"2x2 Contingency Table (Total = {total_pairs:,}):")
    print(f"  P(Next Delayed | Prev Delayed) = {n_11:,} / {n_1_dot:,} = {p1*100:.2f}%")
    print(f"  P(Next Delayed | Prev On-Time) = {n_01:,} / {n_0_dot:,} = {p0*100:.2f}%")
    print(f"  Relative Risk: {rr:.4f} (95% CI: [{rr_ci_lower:.4f}, {rr_ci_upper:.4f}])")
    print(f"  Odds Ratio: {or_val:.4f} (95% CI: [{or_ci_lower:.4f}, {or_ci_upper:.4f}])")
    
    # Plot Figure 2: Delay Propagation Dynamics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot A: Hexbin 2D Density Plot of PREV vs NEXT Delay
    ax_a = axes[0, 0]
    sample_df = con.execute(f"""
        SELECT PREV_ARR_DELAY, NEXT_ARR_DELAY 
        FROM read_parquet('{DATA_PATH}') 
        USING SAMPLE 100000 (reservoir, 42)
    """).df()
    
    hb = ax_a.hexbin(sample_df['PREV_ARR_DELAY'], sample_df['NEXT_ARR_DELAY'], gridsize=75, cmap="Blues", bins='log', mincnt=1,
                     extent=[-40, 180, -40, 180])
    ax_a.plot([-40, 180], [-40, 180], color="#d95f02", linestyle="--", linewidth=1.8, label="1:1 Delay Propagation Line (y = x)")
    ax_a.axhline(0, color="#666666", linestyle=":", alpha=0.7)
    ax_a.axvline(0, color="#666666", linestyle=":", alpha=0.7)
    ax_a.set_xlim(-40, 180)
    ax_a.set_ylim(-40, 180)
    ax_a.set_title("A: Joint Delay Density (PREV vs NEXT ARR_DELAY, N = 100k sample)", fontsize=12, pad=12)
    ax_a.set_xlabel("Preceding Flight Arrival Delay (Minutes)", fontsize=11)
    ax_a.set_ylabel("Subsequent Flight Arrival Delay (Minutes)", fontsize=11)
    cb = fig.colorbar(hb, ax=ax_a, shrink=0.85)
    cb.set_label("Log10(Count)")
    ax_a.legend(loc="upper left", frameon=True)
    
    # Subplot B: Delay Propagation Rate Across Preceding Delay Bins
    ax_b = axes[0, 1]
    bin_labels = ["< 0m", "0-14m", "15-29m", "30-44m", "45-59m", "60-89m", "90-119m", ">= 120m"]
    del_rates = df_bins['next_del15_rate'].values
    bar_colors = ["#238b45" if r < 20 else "#feb24c" if r < 50 else "#d95f02" if r < 60 else "#cb181d" for r in del_rates]
    bars_b = ax_b.bar(bin_labels, del_rates, color=bar_colors, width=0.6, edgecolor="#222222", linewidth=1.2)
    for b, rate in zip(bars_b, del_rates):
        h = b.get_height()
        ax_b.text(b.get_x() + b.get_width() / 2.0, h + 1.2, f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_b.axhline(summary_dict["target_analysis"]['pct_del15'], color="#2b5c8f", linestyle="--", linewidth=1.8,
                 label=f"Baseline Unconditional Delay Rate ({summary_dict['target_analysis']['pct_del15']:.1f}%)")
    ax_b.set_ylim(0, 75)
    ax_b.set_title("B: Subsequent Delay Rate Across Prior Delay Intervals", fontsize=12, pad=12)
    ax_b.set_xlabel("Preceding Flight Delay Bins", fontsize=11)
    ax_b.set_ylabel("Subsequent Delay Rate (NEXT_ARR_DEL15 %)", fontsize=11)
    ax_b.legend(loc="upper left", frameon=True)
    ax_b.tick_params(axis='x', rotation=15)
    
    # Subplot C: Mean & Median Subsequent Delay across Intervals
    ax_c = axes[1, 0]
    ax_c.plot(bin_labels, df_bins['next_mean_delay'].values, marker='o', linewidth=2.5, color="#d95f02", label="Next Mean Arrival Delay (min)")
    ax_c.plot(bin_labels, df_bins['next_median_delay'].values, marker='s', linewidth=2.5, color="#2b5c8f", label="Next Median Arrival Delay (min)")
    ax_c.plot(bin_labels, df_bins['next_p90_delay'].values, marker='^', linewidth=2.0, linestyle=":", color="#7570b3", label="Next P90 Arrival Delay (min)")
    for i, (m, med) in enumerate(zip(df_bins['next_mean_delay'], df_bins['next_median_delay'])):
        ax_c.text(i, m + 4.0, f"{m:.1f}m", ha="center", fontsize=9, color="#d95f02", fontweight="bold")
    ax_c.axhline(0, color="#666666", linestyle="-", alpha=0.5)
    ax_c.set_title("C: Subsequent Delay Magnitude Trajectory by Prior Delay Bin", fontsize=12, pad=12)
    ax_c.set_xlabel("Preceding Flight Delay Bins", fontsize=11)
    ax_c.set_ylabel("Subsequent Arrival Delay (Minutes)", fontsize=11)
    ax_c.legend(loc="upper left", frameon=True)
    ax_c.tick_params(axis='x', rotation=15)
    
    # Subplot D: Contingency Matrix Visual
    ax_d = axes[1, 1]
    matrix_pcts = np.array([
        [n_00 * 100.0 / n_0_dot, n_01 * 100.0 / n_0_dot],
        [n_10 * 100.0 / n_1_dot, n_11 * 100.0 / n_1_dot]
    ])
    matrix_counts = np.array([[n_00, n_01], [n_10, n_11]])
    sns.heatmap(matrix_pcts, annot=False, cmap="YlOrRd", ax=ax_d, vmin=0, vmax=100, cbar_kws={'label': 'Row Conditional Probability (%)'})
    for r in range(2):
        for c in range(2):
            cnt = matrix_counts[r, c]
            pct = matrix_pcts[r, c]
            ax_d.text(c + 0.5, r + 0.5, f"{cnt:,}\n({pct:.2f}%)", ha="center", va="center", fontsize=12, fontweight="bold",
                      color="white" if pct > 60 else "black")
    ax_d.set_xticklabels(["Next On-Time (< 15m)", "Next Delayed (>= 15m)"], fontsize=11)
    ax_d.set_yticklabels(["Prev On-Time", "Prev Delayed"], fontsize=11, rotation=0)
    ax_d.set_title("D: Consecutive Operational Delay Transition Matrix", fontsize=12, pad=12)
    ax_d.set_xlabel("Subsequent Flight Outcome", fontsize=11)
    ax_d.set_ylabel("Preceding Flight Outcome", fontsize=11)
    ax_d.text(0.05, 0.05, f"Risk Ratio: {rr:.2f} (95% CI: [{rr_ci_lower:.2f}, {rr_ci_upper:.2f}])\nOdds Ratio: {or_val:.2f} (95% CI: [{or_ci_lower:.2f}, {or_ci_upper:.2f}])",
              transform=ax_d.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc"))
    
    plt.suptitle("Previous-Flight Delay Propagation Dynamics (N = 6,768,487)", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig02_delay_propagation_relationships.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def run_turnaround_buffer_analysis(con, summary_dict):
    """Section 3: Turnaround Buffer and Deficit Dynamics with Exact Counts."""
    print("\n--- 3. Turnaround Buffer Analysis ---")
    t0 = time.time()
    
    # 3.1 Scheduled Buffer Bins with Explicit Numerators & Denominators
    q_buf = f"""
    SELECT 
        CASE 
            WHEN SCHEDULED_TURNAROUND_BUFFER_MINUTES < 45 THEN '1. Highly Constrained (< 45 min)'
            WHEN SCHEDULED_TURNAROUND_BUFFER_MINUTES >= 45 AND SCHEDULED_TURNAROUND_BUFFER_MINUTES < 60 THEN '2. Short Turn (45-59 min)'
            WHEN SCHEDULED_TURNAROUND_BUFFER_MINUTES >= 60 AND SCHEDULED_TURNAROUND_BUFFER_MINUTES < 75 THEN '3. Standard Turn (60-74 min)'
            WHEN SCHEDULED_TURNAROUND_BUFFER_MINUTES >= 75 AND SCHEDULED_TURNAROUND_BUFFER_MINUTES < 90 THEN '4. Generous Turn (75-89 min)'
            WHEN SCHEDULED_TURNAROUND_BUFFER_MINUTES >= 90 AND SCHEDULED_TURNAROUND_BUFFER_MINUTES < 120 THEN '5. Extended Turn (90-119 min)'
            WHEN SCHEDULED_TURNAROUND_BUFFER_MINUTES >= 120 AND SCHEDULED_TURNAROUND_BUFFER_MINUTES < 180 THEN '6. Layover (120-179 min)'
            ELSE '7. Long / Overnight (>= 180 min)'
        END AS buffer_bin,
        COUNT(*) as total_pairs,
        -- Prior delayed breakdown
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_delayed_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_delayed_next_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END), 0), 2) as prop_rate_if_prev_delayed,
        -- Prior on-time breakdown
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END) as prev_ontime_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_ontime_next_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END), 0), 2) as del_rate_if_prev_ontime,
        -- Overall statistics
        SUM(NEXT_ARR_DEL15) as total_delayed_num,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as overall_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as next_mean_delay,
        ROUND(MEDIAN(NEXT_ARR_DELAY), 2) as next_median_delay
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    ORDER BY 1;
    """
    df_buf = con.execute(q_buf).df()
    df_buf.to_csv(os.path.join(TABLES_DIR, "eda_turnaround_buffer_bins.csv"), index=False)
    
    # 3.2 Turnaround Deficit Impact Comparison
    q_deficit = f"""
    SELECT 
        IS_TURNAROUND_DEFICIT as is_deficit,
        COUNT(*) as pair_count,
        SUM(NEXT_ARR_DEL15) as next_delayed_num,
        ROUND(COUNT(*) * 100.0 / 6768487, 4) as pct_of_pairs,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as next_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as next_mean_delay,
        ROUND(MEDIAN(NEXT_ARR_DELAY), 2) as next_median_delay,
        ROUND(STDDEV(NEXT_ARR_DELAY), 2) as next_std_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.90), 2) as next_p90_delay,
        ROUND(AVG(NEXT_DEP_DELAY), 2) as next_mean_dep_delay,
        ROUND(MEDIAN(NEXT_DEP_DELAY), 2) as next_median_dep_delay
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    ORDER BY 1 DESC;
    """
    df_deficit = con.execute(q_deficit).df()
    df_deficit.to_csv(os.path.join(TABLES_DIR, "eda_turnaround_deficit_impact.csv"), index=False)
    
    summary_dict["turnaround_buffer"] = {
        "deficit_pairs": int(df_deficit.loc[df_deficit['is_deficit'] == 1, 'pair_count'].values[0]),
        "deficit_delayed_numerator": int(df_deficit.loc[df_deficit['is_deficit'] == 1, 'next_delayed_num'].values[0]),
        "deficit_delay_rate": float(df_deficit.loc[df_deficit['is_deficit'] == 1, 'next_del15_rate'].values[0]),
        "non_deficit_pairs": int(df_deficit.loc[df_deficit['is_deficit'] == 0, 'pair_count'].values[0]),
        "non_deficit_delayed_numerator": int(df_deficit.loc[df_deficit['is_deficit'] == 0, 'next_delayed_num'].values[0]),
        "non_deficit_delay_rate": float(df_deficit.loc[df_deficit['is_deficit'] == 0, 'next_del15_rate'].values[0]),
        "deficit_mean_delay": float(df_deficit.loc[df_deficit['is_deficit'] == 1, 'next_mean_delay'].values[0]),
        "non_deficit_mean_delay": float(df_deficit.loc[df_deficit['is_deficit'] == 0, 'next_mean_delay'].values[0])
    }
    
    print("Buffer analysis complete:")
    print(f"  Deficit delay rate: {summary_dict['turnaround_buffer']['deficit_delayed_numerator']:,} / {summary_dict['turnaround_buffer']['deficit_pairs']:,} = {summary_dict['turnaround_buffer']['deficit_delay_rate']}%")
    print(f"  Non-deficit delay rate: {summary_dict['turnaround_buffer']['non_deficit_delayed_numerator']:,} / {summary_dict['turnaround_buffer']['non_deficit_pairs']:,} = {summary_dict['turnaround_buffer']['non_deficit_delay_rate']}%")
    
    # Plot Figure 3: Turnaround Buffer Dynamics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot A: Scheduled Buffer Distribution
    ax_a = axes[0, 0]
    sample_buffers = con.execute(f"""
        SELECT SCHEDULED_TURNAROUND_BUFFER_MINUTES, ESTIMATED_REMAINING_TURNAROUND_MINUTES 
        FROM read_parquet('{DATA_PATH}') 
        USING SAMPLE 100000 (reservoir, 42)
    """).df()
    
    sns.histplot(sample_buffers['SCHEDULED_TURNAROUND_BUFFER_MINUTES'], bins=90, binrange=(20, 200), kde=True,
                 ax=ax_a, color="#3182bd", edgecolor="#111111", linewidth=0.5)
    ax_a.axvline(45, color="#e6550d", linestyle="--", linewidth=1.8, label="45-min Short Turn")
    ax_a.axvline(60, color="#31a354", linestyle="--", linewidth=1.8, label="60-min Standard Turn")
    ax_a.axvline(90, color="#756bb1", linestyle="--", linewidth=1.8, label="90-min Generous Turn")
    ax_a.set_xlim(20, 200)
    ax_a.set_title("A: Scheduled Turnaround Buffer Distribution (< 200 min window)", fontsize=12, pad=12)
    ax_a.set_xlabel("Scheduled Buffer Minutes", fontsize=11)
    ax_a.set_ylabel("Flight Pair Count", fontsize=11)
    ax_a.legend(loc="upper right", frameon=True)
    
    # Subplot B: Estimated Remaining Turnaround Buffer Distribution
    ax_b = axes[0, 1]
    sns.histplot(sample_buffers['ESTIMATED_REMAINING_TURNAROUND_MINUTES'], bins=120, binrange=(-90, 150), kde=True,
                 ax=ax_b, color="#756bb1", edgecolor="#111111", linewidth=0.5)
    ax_b.axvline(0, color="#cb181d", linestyle="-", linewidth=2.5, label="Turnaround Deficit Threshold (Buffer = 0)")
    ax_b.axvspan(-90, 0, color="#fee0d2", alpha=0.5, label="Deficit Zone (Incoming Delay > Buffer)")
    ax_b.set_xlim(-90, 150)
    ax_b.set_title("B: Estimated Remaining Turnaround Distribution (at Gate Arrival)", fontsize=12, pad=12)
    ax_b.set_xlabel("Remaining Turnaround Buffer Minutes", fontsize=11)
    ax_b.set_ylabel("Flight Pair Count", fontsize=11)
    ax_b.legend(loc="upper right", frameon=True)
    
    # Subplot C: Delay Rate by Scheduled Buffer Bins
    ax_c = axes[1, 0]
    buf_labels = ["< 45m", "45-59m", "60-74m", "75-89m", "90-119m", "120-179m", ">= 180m"]
    overall_rates = df_buf['overall_del15_rate'].values
    prop_rates = df_buf['prop_rate_if_prev_delayed'].values
    ontime_rates = df_buf['del_rate_if_prev_ontime'].values
    
    x = np.arange(len(buf_labels))
    w = 0.28
    ax_c.bar(x - w, prop_rates, width=w, color="#d95f02", label="If Prior Flight Was Delayed (Propagation)")
    ax_c.bar(x, overall_rates, width=w, color="#7570b3", label="Overall Delay Rate")
    ax_c.bar(x + w, ontime_rates, width=w, color="#2b5c8f", label="If Prior Flight Was On-Time")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(buf_labels, fontsize=10)
    ax_c.set_ylim(0, 100)
    ax_c.set_title("C: Buffer Delay Shielding Effect Across Scheduled Turn Bins", fontsize=12, pad=12)
    ax_c.set_xlabel("Scheduled Turnaround Buffer", fontsize=11)
    ax_c.set_ylabel("Subsequent Delay Rate (NEXT_ARR_DEL15 %)", fontsize=11)
    ax_c.legend(loc="upper right", frameon=True)
    ax_c.tick_params(axis='x', rotation=15)
    
    # Subplot D: Deficit vs Non-Deficit Outcome Comparison
    ax_d = axes[1, 1]
    cohorts = ["Deficit Cohort\n(N = 382,937)", "Non-Deficit Cohort\n(N = 6,385,550)"]
    rates = [summary_dict['turnaround_buffer']['deficit_delay_rate'], summary_dict['turnaround_buffer']['non_deficit_delay_rate']]
    means = [summary_dict['turnaround_buffer']['deficit_mean_delay'], summary_dict['turnaround_buffer']['non_deficit_mean_delay']]
    
    bars_d = ax_d.bar(cohorts, rates, color=["#cb181d", "#238b45"], width=0.45, edgecolor="#222222", linewidth=1.2)
    for b, rate, mean_val in zip(bars_d, rates, means):
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width() / 2.0, h + 3.0, f"Delay Rate: {rate:.2f}%\nMean Delay: {mean_val:+.1f}m",
                  ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_d.set_ylim(0, 120)
    ax_d.set_title("D: Impact of Turnaround Buffer Exhaustion (Deficit vs Non-Deficit)", fontsize=12, pad=12)
    ax_d.set_ylabel("Subsequent Flight Delay Rate (%)", fontsize=11)
    
    plt.suptitle("Turnaround Buffer Capacity & Deficit Dynamics (N = 6,768,487)", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig03_turnaround_buffer_dynamics.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def run_temporal_analysis(con, summary_dict):
    """Section 4: Temporal, Diurnal, and Seasonal Delay Patterns."""
    print("\n--- 4. Temporal and Diurnal Analysis ---")
    t0 = time.time()
    
    # 4.1 Monthly Patterns
    q_month = f"""
    SELECT 
        NEXT_MONTH as month_num,
        COUNT(*) as total_pairs,
        SUM(NEXT_ARR_DEL15) as total_delayed_num,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as overall_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as mean_delay,
        ROUND(MEDIAN(NEXT_ARR_DELAY), 2) as median_delay,
        ROUND(AVG(IS_TURNAROUND_DEFICIT) * 100.0, 2) as deficit_rate,
        ROUND(AVG(CASE WHEN PREV_ARR_DEL15 = 1 THEN NEXT_ARR_DEL15 ELSE NULL END) * 100.0, 2) as prop_rate
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    ORDER BY 1;
    """
    df_month = con.execute(q_month).df()
    df_month.to_csv(os.path.join(TABLES_DIR, "eda_monthly_patterns.csv"), index=False)
    
    # 4.2 Day of Week Patterns
    q_dow = f"""
    SELECT 
        NEXT_DAY_OF_WEEK as dow_num,
        CASE NEXT_DAY_OF_WEEK 
            WHEN 1 THEN 'Monday'
            WHEN 2 THEN 'Tuesday'
            WHEN 3 THEN 'Wednesday'
            WHEN 4 THEN 'Thursday'
            WHEN 5 THEN 'Friday'
            WHEN 6 THEN 'Saturday'
            WHEN 7 THEN 'Sunday'
        END as day_name,
        COUNT(*) as total_pairs,
        SUM(NEXT_ARR_DEL15) as total_delayed_num,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as overall_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as mean_delay,
        ROUND(AVG(CASE WHEN PREV_ARR_DEL15 = 1 THEN NEXT_ARR_DEL15 ELSE NULL END) * 100.0, 2) as prop_rate
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1, 2
    ORDER BY 1;
    """
    df_dow = con.execute(q_dow).df()
    df_dow.to_csv(os.path.join(TABLES_DIR, "eda_day_of_week_patterns.csv"), index=False)
    
    # 4.3 Hourly Diurnal Patterns with Explicit Conditional Probabilities & Numerators
    q_hour = f"""
    SELECT 
        NEXT_CRS_DEP_HOUR as dep_hour,
        COUNT(*) as total_pairs,
        -- Condition: Preceding Delayed
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_delayed_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_delayed_next_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END), 0), 2) as p_next_del_given_prev_del,
        -- Condition: Preceding On-Time
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END) as prev_ontime_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_ontime_next_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END), 0), 2) as p_next_del_given_prev_ontime,
        -- Unconditional total
        SUM(NEXT_ARR_DEL15) as total_delayed_num,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as unconditional_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as mean_delay
    FROM read_parquet('{DATA_PATH}')
    WHERE NEXT_CRS_DEP_HOUR BETWEEN 5 AND 23
    GROUP BY 1
    ORDER BY 1;
    """
    df_hour = con.execute(q_hour).df()
    df_hour.to_csv(os.path.join(TABLES_DIR, "eda_hourly_patterns.csv"), index=False)
    
    # Plot Figure 4: Temporal Delay Patterns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot A: Monthly Delay Trajectory (2025)
    ax_a = axes[0, 0]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_a.plot(months, df_month['overall_del15_rate'], marker='o', linewidth=2.5, color="#d95f02", label="Next ARR_DEL15 Rate (%)")
    ax_a.plot(months, df_month['prop_rate'], marker='s', linewidth=2.2, color="#7570b3", linestyle="--", label="Propagation Rate if Prev Delayed (%)")
    ax_a.plot(months, df_month['deficit_rate'], marker='^', linewidth=2.0, color="#cb181d", linestyle=":", label="Turnaround Deficit Rate (%)")
    for i, r in enumerate(df_month['overall_del15_rate']):
        ax_a.text(i, r + 1.2, f"{r:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#d95f02")
    ax_a.set_ylim(0, 65)
    ax_a.set_title("A: Monthly Delay & Propagation Variation (BTS 2025)", fontsize=12, pad=12)
    ax_a.set_xlabel("Month of 2025", fontsize=11)
    ax_a.set_ylabel("Rate (%)", fontsize=11)
    ax_a.legend(loc="upper left", frameon=True)
    
    # Subplot B: Day of Week Patterns
    ax_b = axes[0, 1]
    dow_labels = df_dow['day_name'].tolist()
    dow_rates = df_dow['overall_del15_rate'].tolist()
    dow_props = df_dow['prop_rate'].tolist()
    
    x = np.arange(len(dow_labels))
    w = 0.35
    ax_b.bar(x - w/2, dow_rates, width=w, color="#2b5c8f", label="Overall Delay Rate (%)")
    ax_b.bar(x + w/2, dow_props, width=w, color="#d95f02", label="Propagation Rate if Prev Delayed (%)")
    for i, (r, p) in enumerate(zip(dow_rates, dow_props)):
        ax_b.text(i - w/2, r + 0.8, f"{r:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax_b.text(i + w/2, p + 0.8, f"{p:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(dow_labels, rotation=15, fontsize=10)
    ax_b.set_ylim(0, 65)
    ax_b.set_title("B: Day of Week Operational Profiles (Mon-Sun)", fontsize=12, pad=12)
    ax_b.set_xlabel("Day of Week", fontsize=11)
    ax_b.set_ylabel("Rate (%)", fontsize=11)
    ax_b.legend(loc="upper left", frameon=True)
    
    # Subplot C: Diurnal Scheduled Departure Hour Curve
    ax_c = axes[1, 0]
    hours = df_hour['dep_hour'].tolist()
    ax_c.plot(hours, df_hour['p_next_del_given_prev_del'], marker='o', linewidth=2.5, color="#d95f02", label="P(Next Del | Prev Del)")
    ax_c.plot(hours, df_hour['unconditional_del15_rate'], marker='s', linewidth=2.0, color="#7570b3", label="Unconditional Delay Rate (%)")
    ax_c.plot(hours, df_hour['p_next_del_given_prev_ontime'], marker='^', linewidth=2.0, color="#2b5c8f", linestyle="--", label="P(Next Del | Prev On-Time)")
    
    # Highlight afternoon divergence
    ax_c.fill_between(hours, df_hour['p_next_del_given_prev_ontime'], df_hour['p_next_del_given_prev_del'], color="#fee0d2", alpha=0.5, label="Propagation Penalty Gap")
    ax_c.set_xticks(hours)
    ax_c.set_ylim(0, 80)
    ax_c.set_title("C: Diurnal Hourly Delay Profiles (05:00 - 23:00)", fontsize=12, pad=12)
    ax_c.set_xlabel("Scheduled Outbound Departure Hour (Local Standardized)", fontsize=11)
    ax_c.set_ylabel("Delay Rate (%)", fontsize=11)
    ax_c.legend(loc="upper left", frameon=True)
    
    # Subplot D: Diurnal Mean Delay Magnitude
    ax_d = axes[1, 1]
    ax_d.bar(hours, df_hour['mean_delay'], color="#1b9e77", width=0.6, edgecolor="#222222", linewidth=1.0)
    for h, m in zip(hours, df_hour['mean_delay']):
        if h % 2 == 1:
            ax_d.text(h, m + (0.8 if m >= 0 else -1.8), f"{m:+.1f}m", ha="center", fontsize=8.5, fontweight="bold")
    ax_d.axhline(0, color="#666666", linestyle="-", linewidth=1.0)
    ax_d.set_xticks(hours)
    ax_d.set_title("D: Diurnal Mean Delay Trajectory Across Scheduled Departure Hours", fontsize=12, pad=12)
    ax_d.set_xlabel("Scheduled Outbound Departure Hour", fontsize=11)
    ax_d.set_ylabel("Mean Arrival Delay (Minutes)", fontsize=11)
    
    plt.suptitle("Temporal & Diurnal Delay Dynamics (N = 6,768,487)", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig04_temporal_delay_patterns.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def run_carrier_airport_analysis(con, summary_dict):
    """Section 5: Operating Carrier and Airport Operational Heterogeneity with Exact Counts."""
    print("\n--- 5. Carrier and Airport Heterogeneity Analysis ---")
    t0 = time.time()
    
    # 5.1 Operating Carrier Summary
    q_carrier = f"""
    SELECT 
        NEXT_CARRIER as carrier,
        COUNT(*) as total_pairs,
        -- Prior delayed breakdown
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_delayed_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prop_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END), 0), 2) as propagation_rate,
        -- Prior on-time breakdown
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END) as prev_ontime_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as ontime_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END), 0), 2) as ontime_next_delayed_rate,
        -- Overall rate
        SUM(NEXT_ARR_DEL15) as total_delayed_num,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as overall_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as mean_arr_delay,
        ROUND(AVG(IS_TURNAROUND_DEFICIT) * 100.0, 2) as turnaround_deficit_rate,
        ROUND(AVG(CASE WHEN IS_TURNAROUND_DEFICIT = 1 THEN NEXT_ARR_DEL15 ELSE NULL END) * 100.0, 2) as del15_rate_under_deficit
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    HAVING COUNT(*) >= 10000
    ORDER BY total_pairs DESC;
    """
    df_carrier = con.execute(q_carrier).df()
    df_carrier.to_csv(os.path.join(TABLES_DIR, "eda_carrier_summary.csv"), index=False)
    
    # 5.2 Turnaround Airport Summary
    q_airport = f"""
    SELECT 
        TURNAROUND_AIRPORT as airport,
        COUNT(*) as total_pairs,
        -- Prior delayed breakdown
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prev_delayed_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as prop_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 1 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 1 THEN 1 ELSE 0 END), 0), 2) as propagation_rate,
        -- Prior on-time breakdown
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END) as prev_ontime_denom,
        SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) as ontime_delayed_num,
        ROUND(SUM(CASE WHEN PREV_ARR_DEL15 = 0 AND NEXT_ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / 
              NULLIF(SUM(CASE WHEN PREV_ARR_DEL15 = 0 THEN 1 ELSE 0 END), 0), 2) as ontime_next_delayed_rate,
        -- Overall rate
        SUM(NEXT_ARR_DEL15) as total_delayed_num,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as overall_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as mean_arr_delay,
        ROUND(AVG(IS_TURNAROUND_DEFICIT) * 100.0, 2) as turnaround_deficit_rate
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    HAVING COUNT(*) >= 20000
    ORDER BY total_pairs DESC;
    """
    df_airport = con.execute(q_airport).df()
    df_airport.to_csv(os.path.join(TABLES_DIR, "eda_turnaround_airport_summary.csv"), index=False)
    
    # Read enriched airport coordinates for geo-mapping
    df_geo = pd.read_csv(AIRPORT_PATH)
    df_airport_geo = df_airport.merge(df_geo[['AIRPORT', 'LATITUDE', 'LONGITUDE', 'DISPLAY_AIRPORT_NAME']], left_on='airport', right_on='AIRPORT', how='inner')
    
    # Plot Figure 5: Carrier & Airport Heterogeneity
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # Subplot A: Carrier Propagation Sensitivity
    ax_a = fig.add_subplot(gs[0, 0])
    carriers = df_carrier['carrier'].tolist()
    x = np.arange(len(carriers))
    w = 0.38
    ax_a.bar(x - w/2, df_carrier['overall_del15_rate'], width=w, color="#2b5c8f", label="Overall Delay Rate (%)")
    ax_a.bar(x + w/2, df_carrier['propagation_rate'], width=w, color="#d95f02", label="Propagation Rate if Prev Delayed (%)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(carriers, fontsize=10)
    ax_a.set_ylim(0, 75)
    ax_a.set_title("A: Operating Carrier Delay Profiles (N >= 10,000 pairs)", fontsize=12, pad=12)
    ax_a.set_xlabel("Operating Airline Carrier Code", fontsize=11)
    ax_a.set_ylabel("Rate (%)", fontsize=11)
    ax_a.legend(loc="upper left", frameon=True)
    
    # Subplot B: Top 15 Turnaround Hub Airports
    ax_b = fig.add_subplot(gs[0, 1])
    top_15_airports = df_airport.head(15)
    airports = top_15_airports['airport'].tolist()
    x_ap = np.arange(len(airports))
    ax_b.bar(x_ap - w/2, top_15_airports['overall_del15_rate'], width=w, color="#1b9e77", label="Overall Delay Rate (%)")
    ax_b.bar(x_ap + w/2, top_15_airports['propagation_rate'], width=w, color="#7570b3", label="Propagation Rate if Prev Delayed (%)")
    ax_b.set_xticks(x_ap)
    ax_b.set_xticklabels(airports, rotation=25, fontsize=10)
    ax_b.set_ylim(0, 70)
    ax_b.set_title("B: Top 15 Turnaround Hubs Delay Profiles", fontsize=12, pad=12)
    ax_b.set_xlabel("Turnaround Hub Airport Code", fontsize=11)
    ax_b.set_ylabel("Rate (%)", fontsize=11)
    ax_b.legend(loc="upper left", frameon=True)
    
    # Subplot C: Geographic Bubble Plot of Major Turnaround Hubs
    ax_c = fig.add_subplot(gs[1, :])
    conus = df_airport_geo[(df_airport_geo['LONGITUDE'] >= -125) & (df_airport_geo['LONGITUDE'] <= -65) &
                           (df_airport_geo['LATITUDE'] >= 24) & (df_airport_geo['LATITUDE'] <= 50)]
    
    scatter = ax_c.scatter(conus['LONGITUDE'], conus['LATITUDE'],
                           s=conus['total_pairs'] / 500.0,
                           c=conus['overall_del15_rate'],
                           cmap="YlOrRd", edgecolors="#222222", linewidth=1.2, alpha=0.85, vmin=15, vmax=30)
    
    for _, row in conus.head(14).iterrows():
        ax_c.annotate(f"{row['airport']}\n({row['overall_del15_rate']:.1f}%)",
                      (row['LONGITUDE'], row['LATITUDE']),
                      xytext=(4, 4), textcoords="offset points", fontsize=8.5, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.7, edgecolor="#999999"))
        
    ax_c.set_xlim(-128, -65)
    ax_c.set_ylim(23, 52)
    ax_c.set_title("C: Geographic Distribution of Major US Turnaround Hubs (Bubble Size = Volume, Color = Delay Rate)", fontsize=12, pad=12)
    ax_c.set_xlabel("Longitude", fontsize=11)
    ax_c.set_ylabel("Latitude", fontsize=11)
    cbar = fig.colorbar(scatter, ax=ax_c, orientation='horizontal', pad=0.12, shrink=0.55)
    cbar.set_label("Turnaround Outbound Delay Rate (NEXT_ARR_DEL15 %)")
    
    plt.suptitle("Carrier & Airport Operational Heterogeneity (N = 6,768,487)", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig05_carrier_airport_heterogeneity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def run_aircraft_level_analysis(con, summary_dict):
    """Section 6: Aircraft Fleet Patterns, Non-Negative Gini Coefficient, and Shares."""
    print("\n--- 6. Aircraft-Level Patterns Analysis ---")
    t0 = time.time()
    
    # 6.1 Consecutive Delays Progression with Numerators & Denominators
    q_consec = f"""
    SELECT 
        PREV_AIRCRAFT_CONSECUTIVE_DEL15 as consecutive_delays_prior,
        COUNT(*) as total_pairs,
        SUM(NEXT_ARR_DEL15) as next_delayed_num,
        ROUND(SUM(NEXT_ARR_DEL15) * 100.0 / COUNT(*), 4) as next_del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as next_mean_delay,
        ROUND(MEDIAN(NEXT_ARR_DELAY), 2) as next_median_delay,
        ROUND(QUANTILE_CONT(NEXT_ARR_DELAY, 0.90), 2) as next_p90_delay
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    ORDER BY 1;
    """
    df_consec = con.execute(q_consec).df()
    df_consec.to_csv(os.path.join(TABLES_DIR, "eda_aircraft_consecutive_delays.csv"), index=False)
    
    # 6.2 Audited Gini Calculation on Non-Negative Positive Delay Minutes
    # Population: 6,124 unique commercial aircraft across 6,768,487 target outbound legs (f_next)
    # in 'data/processed/flight_rotation_pairs.parquet' (not all 2025 completed flights).
    # Aggregation formula: POSITIVE_DELAY_MINUTES_i = SUM(MAX(0, NEXT_ARR_DELAY_i,j))
    q_tail = f"""
    SELECT 
        TAIL_NUM,
        COUNT(*) as rotation_count,
        SUM(NEXT_ARR_DEL15) as delayed_flights_count,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as tail_del15_rate,
        SUM(CASE WHEN NEXT_ARR_DELAY > 0 THEN NEXT_ARR_DELAY ELSE 0.0 END) as positive_delay_minutes
    FROM read_parquet('{DATA_PATH}')
    GROUP BY TAIL_NUM;
    """
    df_tail = con.execute(q_tail).df()
    delays = np.sort(df_tail['positive_delay_minutes'].values.astype(np.float64))
    n = len(delays)
    total_pos_delay = np.sum(delays)
    total_legs = int(df_tail['rotation_count'].sum())
    
    # Algebraic Gini formula on non-negative array:
    # G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n
    idx_arr = np.arange(1, n + 1)
    gini = (2.0 * np.sum(idx_arr * delays)) / (n * total_pos_delay) - (n + 1.0) / n
    
    # Cumulative shares
    cum_delays = np.cumsum(delays)
    idx_10 = int(0.90 * n)
    idx_20 = int(0.80 * n)
    idx_50 = int(0.50 * n)
    
    share_top_10 = (total_pos_delay - cum_delays[idx_10 - 1]) / total_pos_delay * 100.0
    share_top_20 = (total_pos_delay - cum_delays[idx_20 - 1]) / total_pos_delay * 100.0
    share_top_50 = (total_pos_delay - cum_delays[idx_50 - 1]) / total_pos_delay * 100.0
    
    df_conc = pd.DataFrame([{
        "metric": "Aircraft Fleet Positive Delay Burden Distribution",
        "total_active_aircraft": n,
        "completed_flight_legs_evaluated": total_legs,
        "population_scope": "Rotation-pair target legs (NEXT_ARR_DELAY)",
        "aggregation_formula": "SUM(MAX(0, NEXT_ARR_DELAY))",
        "total_positive_delay_minutes": float(total_pos_delay),
        "mean_rotations_per_tail": round(df_tail['rotation_count'].mean(), 1),
        "median_rotations_per_tail": round(df_tail['rotation_count'].median(), 1),
        "mean_positive_delay_min_per_tail": round(float(total_pos_delay) / n, 1),
        "median_positive_delay_min_per_tail": round(float(np.median(delays)), 1),
        "gini_coefficient": round(gini, 4),
        "top_10_pct_delay_share_pct": round(share_top_10, 2),
        "top_20_pct_delay_share_pct": round(share_top_20, 2),
        "top_50_pct_delay_share_pct": round(share_top_50, 2)
    }])
    df_conc.to_csv(os.path.join(TABLES_DIR, "eda_aircraft_concentration.csv"), index=False)
    
    summary_dict["aircraft_patterns"] = {
        "active_aircraft_count": n,
        "completed_flight_legs_evaluated": total_legs,
        "population_scope": "Rotation-pair target legs (NEXT_ARR_DELAY) in flight_rotation_pairs.parquet",
        "aggregation_formula": "SUM(MAX(0, NEXT_ARR_DELAY))",
        "total_positive_delay_minutes": float(total_pos_delay),
        "gini_coefficient": round(gini, 4),
        "top_10_pct_share": round(share_top_10, 2),
        "top_20_pct_share": round(share_top_20, 2),
        "top_50_pct_share": round(share_top_50, 2),
        "consecutive_delay_progression": df_consec.to_dict(orient="records")
    }
    
    print(f"Aircraft fleet metrics (N = {n:,} aircraft | Total Legs = {total_legs:,} | Total Positive Delay = {total_pos_delay:,.0f} min):")
    print(f"  Gini: {gini:.4f} | Top 10%: {share_top_10:.2f}% | Top 20%: {share_top_20:.2f}% | Top 50%: {share_top_50:.2f}%")
    
    # Plot Figure 6: Aircraft Fleet Patterns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot A: Annual Rotation Operations per Aircraft Distribution
    ax_a = axes[0, 0]
    sns.histplot(df_tail['rotation_count'], bins=60, kde=True, ax=ax_a, color="#2b5c8f", edgecolor="#111111", linewidth=0.5)
    ax_a.axvline(df_tail['rotation_count'].mean(), color="#d95f02", linestyle="--", linewidth=2.0,
                 label=f"Mean: {df_tail['rotation_count'].mean():.1f} ops")
    ax_a.axvline(df_tail['rotation_count'].median(), color="#1b9e77", linestyle="-", linewidth=2.0,
                 label=f"Median: {df_tail['rotation_count'].median():.1f} ops")
    ax_a.set_title("A: Annual Rotation Operations per Aircraft (2025)", fontsize=12, pad=12)
    ax_a.set_xlabel("Number of Completed Rotation Pairs per Tail", fontsize=11)
    ax_a.set_ylabel("Aircraft Fleet Count", fontsize=11)
    ax_a.legend(loc="upper right", frameon=True)
    
    # Subplot B: Fleet Delay Concentration (Lorenz Curve)
    ax_b = axes[0, 1]
    cum_with_zero = np.insert(cum_delays, 0, 0)
    pop_pcts = np.linspace(0, 100, len(cum_with_zero))
    delay_pcts = cum_with_zero / total_pos_delay * 100.0
    ax_b.plot(pop_pcts, delay_pcts, color="#cb181d", linewidth=2.5, label=f"Lorenz Curve (Gini = {gini:.4f})")
    ax_b.plot([0, 100], [0, 100], color="#666666", linestyle="--", linewidth=1.5, label="Line of Perfect Equality")
    ax_b.axvline(80, color="#7570b3", linestyle=":", linewidth=1.8)
    ax_b.axhline(100.0 - share_top_20, color="#7570b3", linestyle=":", linewidth=1.8)
    ax_b.scatter([80], [100.0 - share_top_20], color="#cb181d", s=60, zorder=5)
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.set_title("B: Fleet Positive Delay Burden Concentration (Lorenz Curve)", fontsize=12, pad=12)
    ax_b.set_xlabel("Cumulative % of Commercial Aircraft Fleet (N = 6,124)", fontsize=11)
    ax_b.set_ylabel("Cumulative % of Total Positive Delay Minutes", fontsize=11)
    ax_b.legend(loc="upper left", frameon=True)
    ax_b.text(0.28, 0.18, f"Gini Coefficient: {gini:.4f}\n(Moderate Delay Burden Concentration)\nTop 10% Aircraft: {share_top_10:.1f}%\nTop 20% Aircraft: {share_top_20:.1f}%\nTop 50% Aircraft: {share_top_50:.1f}%\nPopulation: 6,768,487 Legs",
              transform=ax_b.transAxes, fontsize=9.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff5f0", edgecolor="#fc9272"))
    
    # Subplot C: Consecutive Delayed Flights Today Progression
    ax_c = axes[1, 0]
    consec_labels = ["0 Prior Delays", "1 Prior Delay", "2 Prior Delays", "3+ Prior Delays"]
    consec_rates = df_consec['next_del15_rate'].values
    consec_means = df_consec['next_mean_delay'].values
    
    bars_c = ax_c.bar(consec_labels, consec_rates, color=["#238b45", "#feb24c", "#d95f02", "#cb181d"],
                      width=0.5, edgecolor="#222222", linewidth=1.2)
    for b, rate, m in zip(bars_c, consec_rates, consec_means):
        h = b.get_height()
        ax_c.text(b.get_x() + b.get_width() / 2.0, h + 1.8, f"Delay Rate: {rate:.1f}%\nMean: {m:+.1f}m",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_c.set_ylim(0, 90)
    ax_c.set_title("C: Subsequent Delay Rate Across Consecutive Delay History", fontsize=12, pad=12)
    ax_c.set_xlabel("Consecutive Delayed Flights (ARR_DEL15) on Same Operating Day", fontsize=11)
    ax_c.set_ylabel("Subsequent Flight Delay Rate (%)", fontsize=11)
    
    # Subplot D: Cumulative Delay Minutes Today Impact Curve
    ax_d = axes[1, 1]
    q_cum = f"""
    SELECT 
        CASE 
            WHEN PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY <= 0 THEN '1. None (<= 0m)'
            WHEN PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY > 0 AND PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY < 30 THEN '2. 1-29 min'
            WHEN PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY >= 30 AND PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY < 60 THEN '3. 30-59 min'
            WHEN PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY >= 60 AND PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY < 120 THEN '4. 60-119 min'
            ELSE '5. >= 120 min'
        END as cum_bin,
        COUNT(*) as pair_count,
        ROUND(AVG(NEXT_ARR_DEL15) * 100.0, 2) as del15_rate,
        ROUND(AVG(NEXT_ARR_DELAY), 2) as mean_delay
    FROM read_parquet('{DATA_PATH}')
    GROUP BY 1
    ORDER BY 1;
    """
    df_cum = con.execute(q_cum).df()
    cum_labels = ["<= 0m", "1-29m", "30-59m", "60-119m", ">= 120m"]
    ax_d.plot(cum_labels, df_cum['del15_rate'], marker='o', linewidth=2.5, color="#cb181d", label="Subsequent Delay Rate (%)")
    ax_d.plot(cum_labels, df_cum['mean_delay'], marker='s', linewidth=2.2, color="#2b5c8f", linestyle="--", label="Subsequent Mean Delay (min)")
    for i, (r, m) in enumerate(zip(df_cum['del15_rate'], df_cum['mean_delay'])):
        ax_d.text(i, r + 2.5, f"{r:.1f}%", ha="center", fontsize=9.5, fontweight="bold", color="#cb181d")
    ax_d.set_ylim(0, 100)
    ax_d.set_title("D: Response to Cumulative Delay Accumulated Earlier Today", fontsize=12, pad=12)
    ax_d.set_xlabel("Cumulative Incoming Arrival Delay Minutes Prior to Pushback", fontsize=11)
    ax_d.set_ylabel("Metric Value", fontsize=11)
    ax_d.legend(loc="upper left", frameon=True)
    
    plt.suptitle("Aircraft Fleet Rotation Patterns & Delay Concentration (N = 6,768,487)", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig06_aircraft_level_patterns.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def run_predictive_structure_analysis(con, summary_dict):
    """Section 7: Feature Relationships, Correlations, and Syllabus-Aligned Model Families."""
    print("\n--- 7. Feature Relationships and Predictive Structure ---")
    t0 = time.time()
    
    features = [
        'NEXT_ARR_DELAY',
        'NEXT_ARR_DEL15',
        'PREV_ARR_DELAY',
        'PREV_DEP_DELAY',
        'PREV_ARR_DEL15',
        'SCHEDULED_TURNAROUND_BUFFER_MINUTES',
        'ESTIMATED_REMAINING_TURNAROUND_MINUTES',
        'IS_TURNAROUND_DEFICIT',
        'NEXT_CRS_DEP_HOUR',
        'NEXT_DISTANCE',
        'NEXT_CRS_ELAPSED_TIME',
        'NEXT_DAY_OF_WEEK',
        'NEXT_MONTH',
        'PREV_AIRCRAFT_CONSECUTIVE_DEL15',
        'PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY'
    ]
    
    cols_sql = ', '.join(features)
    sample_df = con.execute(f"""
        SELECT {cols_sql} 
        FROM read_parquet('{DATA_PATH}') 
        USING SAMPLE 250000 (reservoir, 42)
    """).df()
    
    pearson_corr = sample_df.corr(method="pearson")
    spearman_corr = sample_df.corr(method="spearman")
    
    # Save correlation table
    corr_combined = pd.DataFrame({
        "feature": features,
        "pearson_with_NEXT_ARR_DELAY": [round(pearson_corr.loc[f, 'NEXT_ARR_DELAY'], 4) for f in features],
        "spearman_with_NEXT_ARR_DELAY": [round(spearman_corr.loc[f, 'NEXT_ARR_DELAY'], 4) for f in features],
        "pearson_with_NEXT_ARR_DEL15": [round(pearson_corr.loc[f, 'NEXT_ARR_DEL15'], 4) for f in features],
        "spearman_with_NEXT_ARR_DEL15": [round(spearman_corr.loc[f, 'NEXT_ARR_DEL15'], 4) for f in features]
    })
    corr_combined.to_csv(os.path.join(TABLES_DIR, "eda_correlation_matrix.csv"), index=False)
    
    # Multicollinearity audit
    redundancy_checks = [
        {"feature_1": "PREV_ARR_DELAY", "feature_2": "PREV_DEP_DELAY", "pearson_r": round(pearson_corr.loc['PREV_ARR_DELAY', 'PREV_DEP_DELAY'], 3),
         "spearman_rho": round(spearman_corr.loc['PREV_ARR_DELAY', 'PREV_DEP_DELAY'], 3), "risk": "High Collinearity", "action": "Retain PREV_ARR_DELAY as primary gate arrival marker"},
        {"feature_1": "NEXT_DISTANCE", "feature_2": "NEXT_CRS_ELAPSED_TIME", "pearson_r": round(pearson_corr.loc['NEXT_DISTANCE', 'NEXT_CRS_ELAPSED_TIME'], 3),
         "spearman_rho": round(spearman_corr.loc['NEXT_DISTANCE', 'NEXT_CRS_ELAPSED_TIME'], 3), "risk": "Extreme Collinearity", "action": "Redundant physical block time proxy; retain either distance or block time"},
        {"feature_1": "PREV_AIRCRAFT_CONSECUTIVE_DEL15", "feature_2": "PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY", "pearson_r": round(pearson_corr.loc['PREV_AIRCRAFT_CONSECUTIVE_DEL15', 'PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY'], 3),
         "spearman_rho": round(spearman_corr.loc['PREV_AIRCRAFT_CONSECUTIVE_DEL15', 'PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY'], 3), "risk": "Moderate Complementarity", "action": "Retain both (consecutive captures streak; cumulative captures severity)"},
        {"feature_1": "ESTIMATED_REMAINING_TURNAROUND_MINUTES", "feature_2": "IS_TURNAROUND_DEFICIT", "pearson_r": round(pearson_corr.loc['ESTIMATED_REMAINING_TURNAROUND_MINUTES', 'IS_TURNAROUND_DEFICIT'], 3),
         "spearman_rho": round(spearman_corr.loc['ESTIMATED_REMAINING_TURNAROUND_MINUTES', 'IS_TURNAROUND_DEFICIT'], 3), "risk": "Non-linear Derivation", "action": "Deficit represents critical zero-crossing step change; retain both"}
    ]
    df_redundancy = pd.DataFrame(redundancy_checks)
    df_redundancy.to_csv(os.path.join(TABLES_DIR, "eda_feature_redundancy_audit.csv"), index=False)
    
    summary_dict["feature_correlations"] = {
        "top_spearman_predictors_for_delay": corr_combined.sort_values(by="spearman_with_NEXT_ARR_DELAY", ascending=False).to_dict(orient="records")
    }
    
    # Plot Figure 7: Pearson & Spearman Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    display_labels = [
        "NEXT_ARR_DEL", "NEXT_DEL15", "PREV_ARR_DEL", "PREV_DEP_DEL", "PREV_DEL15",
        "SCHED_BUF", "REM_BUF", "IS_DEFICIT", "DEP_HOUR", "DISTANCE",
        "CRS_ELAPSED", "DOW", "MONTH", "CONSEC_DEL15", "CUM_DELAY"
    ]
    
    # Subplot A: Pearson Correlation Heatmap
    ax_a = axes[0]
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-0.6, vmax=1.0,
                xticklabels=display_labels, yticklabels=display_labels, ax=ax_a, cbar_kws={'label': 'Pearson Linear r'},
                annot_kws={"size": 8.5})
    ax_a.set_title("A: Linear Feature Correlation Matrix (Pearson r, N = 250k)", fontsize=12, pad=12)
    ax_a.tick_params(axis='x', rotation=45)
    
    # Subplot B: Spearman Rank Correlation Heatmap
    ax_b = axes[1]
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-0.6, vmax=1.0,
                xticklabels=display_labels, yticklabels=display_labels, ax=ax_b, cbar_kws={'label': 'Spearman Rank rho'},
                annot_kws={"size": 8.5})
    ax_b.set_title("B: Monotonic Non-Linear Correlation Matrix (Spearman rho, N = 250k)", fontsize=12, pad=12)
    ax_b.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Feature Relationships, Correlations, and Predictive Structure", fontsize=16, y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig07_feature_correlations.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path} in {round(time.time() - t0, 2)}s")


def main():
    print("=================================================================")
    print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA) — METHODOLOGICALLY AUDITED")
    print("Flight Delays Are Contagious: Predicting Delay Propagation")
    print("=================================================================")
    start_time = time.time()
    
    con = duckdb.connect()
    
    eda_summary = {
        "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "data/processed/flight_rotation_pairs.parquet",
        "sample_size": 6768487
    }
    
    run_target_analysis(con, eda_summary)
    run_delay_propagation_analysis(con, eda_summary)
    run_turnaround_buffer_analysis(con, eda_summary)
    run_temporal_analysis(con, eda_summary)
    run_carrier_airport_analysis(con, eda_summary)
    run_aircraft_level_analysis(con, eda_summary)
    run_predictive_structure_analysis(con, eda_summary)
    
    metrics_path = os.path.join(METRICS_DIR, "eda_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"\nSaved structured summary metrics to {metrics_path}")
    
    elapsed = time.time() - start_time
    print(f"\n=================================================================")
    print(f"SUCCESS: Audited Phase 4 EDA completed in {elapsed:.1f} seconds!")
    print(f"Figures generated: 7 publication plots -> results/figures/")
    print(f"Summary tables: 10 analytical CSV tables -> results/tables/")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
