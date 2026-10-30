"""
Generate Publication-Quality Visualizations for Phase 1 Audit
"""

import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_audit_figures(results_dir="results"):
    figures_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    
    con = duckdb.connect()

    # 1. Monthly Flight Volumes & Cancellations
    df_monthly = pd.read_csv(os.path.join(tables_dir, "monthly_summary.csv"))
    
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    color = '#1f77b4'
    ax1.set_xlabel('Month (2025)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Total Scheduled Flights', color=color, fontsize=12, fontweight='bold')
    bars = ax1.bar(df_monthly['month_name'], df_monthly['total_rows'], color=color, alpha=0.75, width=0.55, label='Scheduled Flights')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(400000, 700000)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add second axis for cancellation rate
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Cancellation Rate (%)', color=color2, fontsize=12, fontweight='bold')
    line = ax2.plot(df_monthly['month_name'], df_monthly['cancel_rate_pct'], color=color2, marker='o', linewidth=2.2, label='Cancellation Rate (%)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 4.0)

    # Title & grid
    plt.title('2025 US BTS Flight Operations: Monthly Volume & Cancellation Rates', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    fig1_path = os.path.join(figures_dir, "monthly_flight_volume_cancellations.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Monthly Delay Rate Trend (ARR_DEL15)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    ax.plot(df_monthly['month_name'], df_monthly['arr_del15_rate_pct'], color='#2ca02c', marker='s', linewidth=2.5, markersize=7)
    for i, txt in enumerate(df_monthly['arr_del15_rate_pct']):
        ax.annotate(f"{txt:.1f}%", (df_monthly['month_name'][i], df_monthly['arr_del15_rate_pct'][i] + 0.6), ha='center', fontsize=9.5, fontweight='bold')
    ax.set_ylim(12, 33)
    ax.set_xlabel('Month (2025)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Flights Delayed ≥ 15 Minutes (%)', fontsize=12, fontweight='bold')
    ax.set_title('Seasonal Arrival Delay Rate Trend (2025 US DOT BTS)', fontsize=14, fontweight='bold', pad=15)
    ax.axhline(df_monthly['arr_del15_rate_pct'].mean(), color='#7f7f7f', linestyle='--', linewidth=1.2, label=f"Annual Mean ({df_monthly['arr_del15_rate_pct'].mean():.1f}%)")
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    fig.tight_layout()
    fig2_path = os.path.join(figures_dir, "monthly_delay_rate_trend.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Arrival Delay Distribution (Completed flights)
    print("Sampling completed flight arrival delays for distribution plot...")
    delays_sample = con.execute("""
        SELECT ARR_DELAY FROM read_csv_auto('data/raw/2025/*.csv')
        WHERE CANCELLED = 0 AND DIVERTED = 0 AND ARR_DELAY IS NOT NULL
        USING SAMPLE 500000
    """).df()["ARR_DELAY"]

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Full distribution box/percentile note
    sns.histplot(delays_sample[(delays_sample >= -60) & (delays_sample <= 180)], bins=60, kde=True, ax=ax_main, color='#1f77b4', edgecolor='none')
    ax_main.axvline(0, color='black', linestyle='-', linewidth=1.2, label='On-Time (0 min)')
    ax_main.axvline(15, color='#d62728', linestyle='--', linewidth=1.5, label='BTS Delay Threshold (15 min)')
    ax_main.axvline(delays_sample.median(), color='#ff7f0e', linestyle=':', linewidth=1.8, label=f'Median ({delays_sample.median():.0f} min)')
    ax_main.set_title('Arrival Delay Density (-60 to +180 min)', fontsize=12, fontweight='bold')
    ax_main.set_xlabel('Arrival Delay (Minutes)', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Sampled Flight Count', fontsize=11, fontweight='bold')
    ax_main.legend(frameon=True, facecolor='white')

    # Cumulative distribution
    sorted_delays = np.sort(delays_sample)
    cdf = np.arange(len(sorted_delays)) / float(len(sorted_delays))
    ax_zoom.plot(sorted_delays, cdf * 100, color='#9467bd', linewidth=2)
    ax_zoom.set_xlim(-60, 180)
    ax_zoom.axvline(0, color='black', linestyle='-', linewidth=1.0)
    ax_zoom.axvline(15, color='#d62728', linestyle='--', linewidth=1.2)
    ax_zoom.axhline(62.43, color='grey', linestyle=':', label='62.4% On-time/Early')
    ax_zoom.axhline(77.69, color='red', linestyle=':', label='77.7% Delay < 15 min')
    ax_zoom.set_title('Empirical Cumulative Distribution Function (ECDF)', fontsize=12, fontweight='bold')
    ax_zoom.set_xlabel('Arrival Delay (Minutes)', fontsize=11, fontweight='bold')
    ax_zoom.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    ax_zoom.legend(loc='lower right', frameon=True, facecolor='white')

    plt.suptitle('2025 Arrival Delay Distribution: Asymmetry and Long-Tail Characteristics', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig3_path = os.path.join(figures_dir, "annual_arrival_delay_distribution.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Geographic Distribution of BTS Reporting Airports
    df_airports = pd.read_csv(os.path.join(tables_dir, "airport_coverage_report.csv"))
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    
    # Filter contiguous US + AK/HI coordinates
    valid_coords = df_airports.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    scatter = ax.scatter(
        valid_coords['LONGITUDE'], 
        valid_coords['LATITUDE'],
        c='#1f77b4', 
        alpha=0.6, 
        edgecolors='none', 
        s=35
    )
    
    # Highlight top 10 busy airports
    top_airports = con.execute("""
        SELECT ORIGIN AS code, COUNT(*) AS flight_count
        FROM read_csv_auto('data/raw/2025/*.csv')
        GROUP BY ORIGIN
        ORDER BY flight_count DESC
        LIMIT 10
    """).df()
    
    top_merged = pd.merge(top_airports, valid_coords, left_on='code', right_on='flight_iata_code')
    for _, row in top_merged.iterrows():
        ax.scatter(row['LONGITUDE'], row['LATITUDE'], c='#d62728', s=80, edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(row['code'], (row['LONGITUDE'] + 0.6, row['LATITUDE'] + 0.5), fontsize=9, fontweight='bold', color='#333333', zorder=6)

    ax.set_xlim(-170, -65)
    ax.set_ylim(15, 72)
    ax.set_xlabel('Longitude (°)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=11, fontweight='bold')
    ax.set_title('Geographic Distribution of 352 Active BTS Airports (2025) with Top 10 Hubs', fontsize=13, fontweight='bold', pad=12)
    
    fig.tight_layout()
    fig4_path = os.path.join(figures_dir, "airport_geographic_distribution.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

if __name__ == "__main__":
    generate_audit_figures()

