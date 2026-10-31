# Flight Delay Propagation — Interactive Research Dashboard & Model Server

This directory contains the production-grade interactive web dashboard and live machine learning model inference backend for the research study:

> **Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations**  
> *Author:* Arnav Raj (Vellore Institute of Technology, Vellore, India)  
> *Publication Target:* IEEE Conference (Strict 6-Page Paper Included)

---

## 🚀 Quick Start Guide

### Prerequisites
The dashboard uses the project's existing virtual environment (`.venv`), which already includes `duckdb`, `scikit-learn`, `joblib`, and `pandas`. No additional global dependencies or external packages are required.

### 1. Launching the Web Server

From the repository root:

```bash
./.venv/bin/python web/server.py 8000
```

Or using an active virtual environment:

```bash
source .venv/bin/activate
python web/server.py 8000
```

### 2. Accessing the Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🧭 Dashboard Architecture & Layout

The dashboard features a **fixed left-side navigation bar** with 5 primary views, built with high-end dark aviation operations styling (zinc/slate theme, frosted glass cards, and status badges):

```
┌─────────────────┬──────────────────────────────────────────────────────────────────┐
│  NAVIGATION     │  CONTENT AREA                                                    │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ 🛫 Overview     │  • Problem statement, physical turnaround equations, pipeline   │
│ 📊 Dataset      │  • DuckDB Parquet explorer, filters, sorting, missing values     │
│ 📈 Performance  │  • Model comparisons, threshold simulator, confusion matrix     │
│ 🎯 Prediction   │  • Real-time model cockpit, turnaround physics, factor drivers   │
│ 📝 Findings     │  • Empirical conclusions, buffer thresholds, IEEE paper download │
└─────────────────┴──────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Features & Capabilities

### 1. Project Overview (`#overview`)
- **Research Question & Executive Summary:** Background on airline hub networks, rotational aircraft scheduling, and delay propagation across 6.77 million verified flight pairs.
- **Physical Turnaround Mechanics:** Mathematical formulations with KaTeX LaTeX rendering:
  - $\text{Buffer} = \text{CRS\_DEP\_TIME}_{k+1} - \text{CRS\_ARR\_TIME}_k$
  - $\text{RemTurn} = \text{Buffer} - \text{PREV\_ARR\_DELAY}$
  - $\text{IS\_DEFICIT} = \mathbb{I}(\text{RemTurn} < 0)$ (Observed in 5.66% of operations; results in a 99.50% delay propagation rate).
- **End-to-End Pipeline Architecture:** Visual flow chart tracking BTS raw records $\to$ Rotation pair construction $\to$ Temporal partitioning $\to$ Feature engineering $\to$ Frozen model evaluation.

### 2. Dataset Explorer & Missing Values (`#dataset`)
- **High-Performance DuckDB Query Engine:** Queries `data/processed/modeling_test.parquet` (562k rows), `modeling_validation.parquet` (1.14M rows), or `modeling_train.parquet` (5.06M rows) with sub-second response times.
- **Interactive Multi-Column Filtering:** Filter by Carrier (AA, DL, UA, WN, etc.), Origin, Destination, Delay Status ($\ge 15$ min vs. on-time), and Turnaround Cushion (Deficit vs. Sufficient).
- **Live Server-Side Sorting & Pagination:** Sort by any column ascending or descending with responsive pagination controls.
- **Missing Value Audit:** Complete 18-feature audit showing 0.00% missing values across all partitions.
- **Rotation Reconstruction Funnel:** Complete breakdown of the 6,898,741 raw BTS flights to 6,768,487 valid consecutive pairs.

### 3. Model Performance & Benchmarks (`#performance`)
- **Model Family Benchmarks:** Validation comparisons across Group A, B, and C features for Majority Baseline, Logistic Regression, Decision Tree, MLP ($\alpha=0$), and Random Forest.
- **Independent December Test Generalization:** Full evaluation table comparing Validation vs. Test ($R^2 = 0.4851$, $\text{MAE} = 19.77$ min, $\text{ROC-AUC} = 0.7943$, $\text{AP} = 0.7150$).
- **Interactive Classification Threshold Simulator:** Explore operating trade-offs across $T \in [0.10, 0.90]$. Dynamically computes Precision, Recall, F1, TP, FP, FN, TN, and highlights:
  - $T = 0.35$ (Selected Operational Operating Point: 72.52% Precision, 53.24% Recall, 0.6140 F1)
  - $T = 0.39$ (Grid F1-Maximizing Threshold: 75.76% Precision, 48.91% Recall, 0.5942 F1)
  - $T = 0.50$ (Standard Default: 86.68% Precision, 40.88% Recall, 0.5556 F1)
- **Error Stratification & Shift-Share Analysis:** 5 delay strata MAE decomposition table and Oaxaca-Blinder decomposition.
- **Publication Figures Gallery:** High-resolution view and zoom modal for all 12 research figures.

### 4. Live Prediction Playground (`#prediction`)
- **Trained Model Backend:** Directly calls the frozen `final_selected_classifier.joblib`, `final_selected_regressor.joblib`, and `final_preprocessor_group_c.joblib` loaded into server memory.
- **Scenario Presets:** One-click presets for Severe Turnaround Deficit, Tight Turnaround Buffer, Normal On-Time Dwell, Evening Hub Congestion, and Weather Outflow Spillover.
- **Real-Time Physics Engine:** Dynamically calculates remaining turnaround cushion and deficit minutes.
- **Inference Gauge & Attribution:** Displays predicted delay probability, discrete operational verdict ($\ge 15$ min delay vs. on-time), continuous delay estimate, risk banner, and natural language factor driver tags.

### 5. Findings & IEEE Paper Preview (`#findings`)
- **Core Empirical Conclusions:** 4 key takeaways regarding delay propagation dynamics, the nonlinear deficit cliff, temporal generalization, and rotational fatigue accumulation.
- **Limitations & Disclaimers:** Explicit non-causal language and threats to validity.
- **IEEE Paper Integration:** Built-in PDF viewer and download button serving the strictly 6-page compiled conference paper (`/api/paper`).

---

## ☁️ Google Drive Dataset Hosting Instructions

Because the processed Parquet files total **~1.5 GB** (6.77 million rows), they should be hosted on Google Drive for external sharing, cloud notebook execution (e.g., Google Colab), or team distribution.

### Step 1: Upload Files to Google Drive
1. Go to [Google Drive](https://drive.google.com) and sign in.
2. Click **New** $\to$ **New folder** $\to$ Name it `Flight-Delay-Propagation-2025`.
3. Open the folder and upload the following files from your local `data/processed/` directory:
   - `modeling_train.parquet` (137 MB, 5,065,376 rows)
   - `modeling_validation.parquet` (28 MB, 1,140,909 rows)
   - `modeling_test.parquet` (12 MB, 562,202 rows)
   - `flight_rotation_pairs.parquet` (267 MB, 6,768,487 rows)

### Step 2: Configure Public Sharing Access
1. Right-click on the uploaded file (or the parent folder) $\to$ Select **Share** $\to$ **Share**.
2. Under **General access**, change from **Restricted** to **Anyone with the link**.
3. Set the role to **Viewer**.
4. Click **Copy link** $\to$ **Done**.

The link format looks like:
```
https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing
```
Here, the `FILE_ID` is `1A2B3C4D5E6F7G8H9I0J`.

### Step 3: Loading Data Directly in Python / Colab

Use `gdown` or direct DuckDB streaming to load the dataset without manual browser downloads:

```python
import gdown
import pandas as pd
import duckdb

# Replace with your actual Google Drive File ID
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"
url = f"https://drive.google.com/uc?id={FILE_ID}"
output_path = "modeling_test.parquet"

# Download via gdown
gdown.download(url, output_path, quiet=False)

# Query immediately via DuckDB
con = duckdb.connect()
df = con.execute("SELECT * FROM 'modeling_test.parquet' WHERE NEXT_ARR_DEL15 = 1 LIMIT 10").df()
print(df.head())
```

---

## 📡 REST API Documentation

The backend exposes the following RESTful endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the main dashboard application (`index.html`). |
| `/api/health` | `GET` | Returns server health status and model load state. |
| `/api/overview` | `GET` | Returns project metadata, author info, equations, and high-level statistics. |
| `/api/dataset/summary` | `GET` | Returns dataset partitions, pipeline filter counts, carrier shares, and Drive guide. |
| `/api/dataset/sample` | `GET` | Queries Parquet files with `partition`, `carrier`, `origin`, `dest`, `status`, `deficit`, `search`, `page`, `page_size`, `sort_by`, `order`. |
| `/api/models` | `GET` | Returns all classification/regression benchmarks, test generalization, confusion matrix, and strata metrics. |
| `/api/predict` | `POST` | Accepts JSON flight rotation parameters; returns probability, verdict, continuous delay, cushion, and drivers. |
| `/api/paper` | `GET` | Serves and downloads the compiled 6-page IEEE conference paper PDF. |

---

## 🧪 Testing the Server

To verify that the server and models run without starting a browser:

```bash
./.venv/bin/python -c "
import io, json
from web.server import init_services, DashboardHandler

init_services()
print('Server backend and models initialized successfully!')
"
```
