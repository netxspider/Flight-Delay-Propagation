"""
Stage 4: Main Model Training and Evaluation Pipeline.
Research Project: Flight Delays Are Contagious
BTech CSE CA2 / IEEE Research Paper Implementation

This module implements the 10-run main experimental suite:
1. Classification:
   - Run 1: Decision Tree Classifier — Group A (max_depth=14, min_samples_leaf=100)
   - Run 2: Decision Tree Classifier — Group C (max_depth=14, min_samples_leaf=100)
   - Run 3: Random Forest Classifier — Group A (n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2)
   - Run 4: Random Forest Classifier — Group C (n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2)
   - Run 5: MLP Classifier — Group C (hidden_layer_sizes=(64, 32), alpha=0.0, batch_size=256, early_stopping=True)

2. Regression:
   - Run 6: Decision Tree Regressor — Group A (max_depth=14, min_samples_leaf=100)
   - Run 7: Decision Tree Regressor — Group C (max_depth=14, min_samples_leaf=100)
   - Run 8: Random Forest Regressor — Group A (n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2)
   - Run 9: Random Forest Regressor — Group C (n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2)
   - Run 10: MLP Regressor — Group C (hidden_layer_sizes=(64, 32), alpha=0.0, batch_size=256, early_stopping=True)

Syllabus Constraints & Rules:
- Strictly alpha=0.0 on all MLP estimators (bans L1/L2 regularization).
- Conservative parallelism (n_jobs=2) on Random Forest.
- Preservation of sparse matrices (scipy.sparse.csr_matrix) for high-cardinality one-hot encoding.
- Evaluated on the FULL validation set (1,140,909 rows).
- Test set (modeling_test.parquet) strictly sealed and untouched.
"""

import os
import sys
import time
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TRAIN_PARQUET = os.path.join(DATA_DIR, "modeling_train.parquet")
VAL_PARQUET = os.path.join(DATA_DIR, "modeling_validation.parquet")

# Feature group definitions
NUM_COLS_A = [
    'PREV_ARR_DELAY', 'PREV_DEP_DELAY', 'PREV_ARR_DEL15', 'PREV_DEP_DEL15',
    'SCHEDULED_TURNAROUND_BUFFER', 'ESTIMATED_REMAINING_TURNAROUND_MINUTES', 'IS_TURNAROUND_DEFICIT'
]

CAT_COLS = ['NEXT_CARRIER', 'NEXT_ORIGIN', 'NEXT_DEST']

NUM_COLS_C = NUM_COLS_A + [
    'NEXT_DISTANCE', 'NEXT_CRS_DEP_HOUR', 'NEXT_DAY_OF_WEEK', 'NEXT_MONTH',
    'PREV_AIRCRAFT_DELAYED_FLIGHTS_TODAY', 'PREV_AIRCRAFT_CONSECUTIVE_DEL15',
    'PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY', 'PREV_AIRCRAFT_DELAY_RATE_TODAY'
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_directories():
    os.makedirs(os.path.join(MODELS_DIR, "classification"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "regression"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "classification"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "regression"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)


def evaluate_classifier(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }


def evaluate_regressor(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 6)
    }


def run_stage4_experiments():
    ensure_directories()
    con = duckdb.connect()

    print("================================================================================")
    print("PHASE 6 — STAGE 4: MAIN MODEL EXPERIMENTS (10 RUNS)")
    print("================================================================================")

    # 1. Load Validation Data (Full 1,140,909 rows)
    print("\n--- 1. Loading Full Validation Dataset (1,140,909 rows) ---")
    t0 = time.time()
    df_val = con.execute(f"SELECT * FROM read_parquet('{VAL_PARQUET}')").df()
    print(f"Validation dataset loaded: {len(df_val):,} rows in {time.time()-t0:.2f}s")
    y_val_cls = df_val['NEXT_ARR_DEL15'].values
    y_val_reg = df_val['NEXT_ARR_DELAY'].values

    # 2. Sample Training Data
    # Decision Trees & Random Forests: N = 500,000 (Seed 42)
    # MLP: N = 250,000 (Seed 42)
    print("\n--- 2. Sampling Training Datasets (Strictly Seed 42) ---")
    t0 = time.time()
    df_train_500k = con.execute(f"""
        SELECT * FROM read_parquet('{TRAIN_PARQUET}')
        USING SAMPLE 500000 (reservoir, 42)
    """).df()
    print(f"500k sample loaded: {len(df_train_500k):,} rows in {time.time()-t0:.2f}s")

    t0 = time.time()
    df_train_250k = df_train_500k.sample(n=250000, random_state=42).copy()
    print(f"250k sample created: {len(df_train_250k):,} rows in {time.time()-t0:.2f}s")

    y_train_500k_cls = df_train_500k['NEXT_ARR_DEL15'].values
    y_train_500k_reg = df_train_500k['NEXT_ARR_DELAY'].values

    y_train_250k_cls = df_train_250k['NEXT_ARR_DEL15'].values
    y_train_250k_reg = df_train_250k['NEXT_ARR_DELAY'].values

    # 3. Fit Preprocessing Pipelines
    print("\n--- 3. Fitting Preprocessors on 500k Training Data Strictly ---")
    prep_A = ColumnTransformer(
        transformers=[('num', StandardScaler(), NUM_COLS_A)],
        remainder='drop'
    )
    prep_C = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS_C),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CAT_COLS)
        ],
        remainder='drop'
    )

    t0 = time.time()
    X_train_500k_A = prep_A.fit_transform(df_train_500k)
    X_val_A = prep_A.transform(df_val)
    print(f"Preprocessed Group A: Train shape {X_train_500k_A.shape}, Val shape {X_val_A.shape} in {time.time()-t0:.2f}s")

    t0 = time.time()
    X_train_500k_C = prep_C.fit_transform(df_train_500k)
    X_val_C = prep_C.transform(df_val)
    X_train_250k_C = prep_C.transform(df_train_250k)
    print(f"Preprocessed Group C: Train 500k {X_train_500k_C.shape}, Train 250k {X_train_250k_C.shape}, Val {X_val_C.shape} in {time.time()-t0:.2f}s")

    feature_names_A = prep_A.get_feature_names_out()
    feature_names_C = prep_C.get_feature_names_out()

    cls_results = []
    reg_results = []
    prob_dict = {}
    pred_reg_dict = {}

    # =========================================================================
    # CLASSIFICATION EXPERIMENTS (RUNS 1 to 5)
    # =========================================================================
    print("\n================================================================================")
    print("CLASSIFICATION EXPERIMENTS (RUNS 1 - 5)")
    print("================================================================================")

    # Run 1: Decision Tree Classifier — Group A
    print("\n[Run 1/10] Training Decision Tree Classifier (Group A)...")
    t0 = time.time()
    dt_cls_a = DecisionTreeClassifier(max_depth=14, min_samples_leaf=100, random_state=42)
    dt_cls_a.fit(X_train_500k_A, y_train_500k_cls)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = dt_cls_a.predict(X_val_A)
    y_prob = dt_cls_a.predict_proba(X_val_A)[:, 1]
    t_pred = time.time() - t0

    m1 = evaluate_classifier(y_val_cls, y_pred, y_prob)
    m1.update({
        "run_id": "Run_1",
        "model": "Decision Tree Classifier",
        "feature_group": "Group_A",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2),
        "depth": dt_cls_a.get_depth(),
        "n_leaves": dt_cls_a.get_n_leaves()
    })
    cls_results.append(m1)
    prob_dict["DT (Group A)"] = y_prob
    joblib.dump(dt_cls_a, os.path.join(MODELS_DIR, "classification/decision_tree_group_a.joblib"))
    print(f"  -> Run 1 Done: Acc={m1['accuracy']:.4f}, F1={m1['f1']:.4f}, ROC-AUC={m1['roc_auc']:.4f}, PR-AUC={m1['average_precision_pr_auc']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 1
    df_imp_dt_a = pd.DataFrame({'feature': feature_names_A, 'importance': dt_cls_a.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_dt_a.to_csv(os.path.join(RESULTS_DIR, "classification/feature_importances_dt_group_a.csv"), index=False)

    # Run 2: Decision Tree Classifier — Group C
    print("\n[Run 2/10] Training Decision Tree Classifier (Group C)...")
    t0 = time.time()
    dt_cls_c = DecisionTreeClassifier(max_depth=14, min_samples_leaf=100, random_state=42)
    dt_cls_c.fit(X_train_500k_C, y_train_500k_cls)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = dt_cls_c.predict(X_val_C)
    y_prob = dt_cls_c.predict_proba(X_val_C)[:, 1]
    t_pred = time.time() - t0

    m2 = evaluate_classifier(y_val_cls, y_pred, y_prob)
    m2.update({
        "run_id": "Run_2",
        "model": "Decision Tree Classifier",
        "feature_group": "Group_C",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2),
        "depth": dt_cls_c.get_depth(),
        "n_leaves": dt_cls_c.get_n_leaves()
    })
    cls_results.append(m2)
    prob_dict["DT (Group C)"] = y_prob
    joblib.dump(dt_cls_c, os.path.join(MODELS_DIR, "classification/decision_tree_group_c.joblib"))
    print(f"  -> Run 2 Done: Acc={m2['accuracy']:.4f}, F1={m2['f1']:.4f}, ROC-AUC={m2['roc_auc']:.4f}, PR-AUC={m2['average_precision_pr_auc']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 2
    df_imp_dt_c = pd.DataFrame({'feature': feature_names_C, 'importance': dt_cls_c.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_dt_c.to_csv(os.path.join(RESULTS_DIR, "classification/feature_importances_dt_group_c.csv"), index=False)

    # Run 3: Random Forest Classifier — Group A
    print("\n[Run 3/10] Training Random Forest Classifier (Group A, n_jobs=2)...")
    t0 = time.time()
    rf_cls_a = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2, random_state=42)
    rf_cls_a.fit(X_train_500k_A, y_train_500k_cls)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = rf_cls_a.predict(X_val_A)
    y_prob = rf_cls_a.predict_proba(X_val_A)[:, 1]
    t_pred = time.time() - t0

    m3 = evaluate_classifier(y_val_cls, y_pred, y_prob)
    m3.update({
        "run_id": "Run_3",
        "model": "Random Forest Classifier",
        "feature_group": "Group_A",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2)
    })
    cls_results.append(m3)
    prob_dict["RF (Group A)"] = y_prob
    joblib.dump(rf_cls_a, os.path.join(MODELS_DIR, "classification/random_forest_group_a.joblib"))
    print(f"  -> Run 3 Done: Acc={m3['accuracy']:.4f}, F1={m3['f1']:.4f}, ROC-AUC={m3['roc_auc']:.4f}, PR-AUC={m3['average_precision_pr_auc']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 3
    df_imp_rf_a = pd.DataFrame({'feature': feature_names_A, 'importance': rf_cls_a.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_rf_a.to_csv(os.path.join(RESULTS_DIR, "classification/feature_importances_rf_group_a.csv"), index=False)

    # Run 4: Random Forest Classifier — Group C
    print("\n[Run 4/10] Training Random Forest Classifier (Group C, n_jobs=2)...")
    t0 = time.time()
    rf_cls_c = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2, random_state=42)
    rf_cls_c.fit(X_train_500k_C, y_train_500k_cls)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = rf_cls_c.predict(X_val_C)
    y_prob = rf_cls_c.predict_proba(X_val_C)[:, 1]
    t_pred = time.time() - t0

    m4 = evaluate_classifier(y_val_cls, y_pred, y_prob)
    m4.update({
        "run_id": "Run_4",
        "model": "Random Forest Classifier",
        "feature_group": "Group_C",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2)
    })
    cls_results.append(m4)
    prob_dict["RF (Group C)"] = y_prob
    joblib.dump(rf_cls_c, os.path.join(MODELS_DIR, "classification/random_forest_group_c.joblib"))
    print(f"  -> Run 4 Done: Acc={m4['accuracy']:.4f}, F1={m4['f1']:.4f}, ROC-AUC={m4['roc_auc']:.4f}, PR-AUC={m4['average_precision_pr_auc']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 4
    df_imp_rf_c = pd.DataFrame({'feature': feature_names_C, 'importance': rf_cls_c.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_rf_c.to_csv(os.path.join(RESULTS_DIR, "classification/feature_importances_rf_group_c.csv"), index=False)

    # Run 5: MLP Classifier — Group C (alpha=0.0 strictly)
    print("\n[Run 5/10] Training MLP Classifier (Group C, alpha=0.0, batch_size=256)...")
    t0 = time.time()
    mlp_cls = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.0,
        batch_size=256,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42
    )
    mlp_cls.fit(X_train_250k_C, y_train_250k_cls)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = mlp_cls.predict(X_val_C)
    y_prob = mlp_cls.predict_proba(X_val_C)[:, 1]
    t_pred = time.time() - t0

    m5 = evaluate_classifier(y_val_cls, y_pred, y_prob)
    m5.update({
        "run_id": "Run_5",
        "model": "MLP Classifier",
        "feature_group": "Group_C",
        "sample_size": len(df_train_250k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2),
        "n_iter": mlp_cls.n_iter_,
        "loss": round(mlp_cls.loss_, 4)
    })
    cls_results.append(m5)
    prob_dict["MLP (Group C)"] = y_prob
    joblib.dump(mlp_cls, os.path.join(MODELS_DIR, "classification/mlp_classifier_group_c.joblib"))
    print(f"  -> Run 5 Done: Acc={m5['accuracy']:.4f}, F1={m5['f1']:.4f}, ROC-AUC={m5['roc_auc']:.4f}, PR-AUC={m5['average_precision_pr_auc']:.4f} (Fit {t_train:.1f}s, Iters {mlp_cls.n_iter_})")

    # =========================================================================
    # REGRESSION EXPERIMENTS (RUNS 6 to 10)
    # =========================================================================
    print("\n================================================================================")
    print("REGRESSION EXPERIMENTS (RUNS 6 - 10)")
    print("================================================================================")

    # Run 6: Decision Tree Regressor — Group A
    print("\n[Run 6/10] Training Decision Tree Regressor (Group A)...")
    t0 = time.time()
    dt_reg_a = DecisionTreeRegressor(max_depth=14, min_samples_leaf=100, random_state=42)
    dt_reg_a.fit(X_train_500k_A, y_train_500k_reg)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_reg = dt_reg_a.predict(X_val_A)
    t_pred = time.time() - t0

    m6 = evaluate_regressor(y_val_reg, y_pred_reg)
    m6.update({
        "run_id": "Run_6",
        "model": "Decision Tree Regressor",
        "feature_group": "Group_A",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2),
        "depth": dt_reg_a.get_depth(),
        "n_leaves": dt_reg_a.get_n_leaves()
    })
    reg_results.append(m6)
    pred_reg_dict["DT (Group A)"] = y_pred_reg
    joblib.dump(dt_reg_a, os.path.join(MODELS_DIR, "regression/decision_tree_group_a.joblib"))
    print(f"  -> Run 6 Done: MAE={m6['mae']:.2f} min, RMSE={m6['rmse']:.2f} min, R2={m6['r2']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 6
    df_imp_dt_reg_a = pd.DataFrame({'feature': feature_names_A, 'importance': dt_reg_a.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_dt_reg_a.to_csv(os.path.join(RESULTS_DIR, "regression/feature_importances_dt_group_a.csv"), index=False)

    # Run 7: Decision Tree Regressor — Group C
    print("\n[Run 7/10] Training Decision Tree Regressor (Group C)...")
    t0 = time.time()
    dt_reg_c = DecisionTreeRegressor(max_depth=14, min_samples_leaf=100, random_state=42)
    dt_reg_c.fit(X_train_500k_C, y_train_500k_reg)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_reg = dt_reg_c.predict(X_val_C)
    t_pred = time.time() - t0

    m7 = evaluate_regressor(y_val_reg, y_pred_reg)
    m7.update({
        "run_id": "Run_7",
        "model": "Decision Tree Regressor",
        "feature_group": "Group_C",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2),
        "depth": dt_reg_c.get_depth(),
        "n_leaves": dt_reg_c.get_n_leaves()
    })
    reg_results.append(m7)
    pred_reg_dict["DT (Group C)"] = y_pred_reg
    joblib.dump(dt_reg_c, os.path.join(MODELS_DIR, "regression/decision_tree_group_c.joblib"))
    print(f"  -> Run 7 Done: MAE={m7['mae']:.2f} min, RMSE={m7['rmse']:.2f} min, R2={m7['r2']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 7
    df_imp_dt_reg_c = pd.DataFrame({'feature': feature_names_C, 'importance': dt_reg_c.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_dt_reg_c.to_csv(os.path.join(RESULTS_DIR, "regression/feature_importances_dt_group_c.csv"), index=False)

    # Run 8: Random Forest Regressor — Group A
    print("\n[Run 8/10] Training Random Forest Regressor (Group A, n_jobs=2)...")
    t0 = time.time()
    rf_reg_a = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2, random_state=42)
    rf_reg_a.fit(X_train_500k_A, y_train_500k_reg)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_reg = rf_reg_a.predict(X_val_A)
    t_pred = time.time() - t0

    m8 = evaluate_regressor(y_val_reg, y_pred_reg)
    m8.update({
        "run_id": "Run_8",
        "model": "Random Forest Regressor",
        "feature_group": "Group_A",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2)
    })
    reg_results.append(m8)
    pred_reg_dict["RF (Group A)"] = y_pred_reg
    joblib.dump(rf_reg_a, os.path.join(MODELS_DIR, "regression/random_forest_group_a.joblib"))
    print(f"  -> Run 8 Done: MAE={m8['mae']:.2f} min, RMSE={m8['rmse']:.2f} min, R2={m8['r2']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 8
    df_imp_rf_reg_a = pd.DataFrame({'feature': feature_names_A, 'importance': rf_reg_a.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_rf_reg_a.to_csv(os.path.join(RESULTS_DIR, "regression/feature_importances_rf_group_a.csv"), index=False)

    # Run 9: Random Forest Regressor — Group C
    print("\n[Run 9/10] Training Random Forest Regressor (Group C, n_jobs=2)...")
    t0 = time.time()
    rf_reg_c = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=50, n_jobs=2, random_state=42)
    rf_reg_c.fit(X_train_500k_C, y_train_500k_reg)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_reg = rf_reg_c.predict(X_val_C)
    t_pred = time.time() - t0

    m9 = evaluate_regressor(y_val_reg, y_pred_reg)
    m9.update({
        "run_id": "Run_9",
        "model": "Random Forest Regressor",
        "feature_group": "Group_C",
        "sample_size": len(df_train_500k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2)
    })
    reg_results.append(m9)
    pred_reg_dict["RF (Group C)"] = y_pred_reg
    joblib.dump(rf_reg_c, os.path.join(MODELS_DIR, "regression/random_forest_group_c.joblib"))
    print(f"  -> Run 9 Done: MAE={m9['mae']:.2f} min, RMSE={m9['rmse']:.2f} min, R2={m9['r2']:.4f} (Fit {t_train:.1f}s)")

    # Feature Importance Run 9
    df_imp_rf_reg_c = pd.DataFrame({'feature': feature_names_C, 'importance': rf_reg_c.feature_importances_}).sort_values('importance', ascending=False)
    df_imp_rf_reg_c.to_csv(os.path.join(RESULTS_DIR, "regression/feature_importances_rf_group_c.csv"), index=False)

    # Run 10: MLP Regressor — Group C (alpha=0.0 strictly)
    print("\n[Run 10/10] Training MLP Regressor (Group C, alpha=0.0, batch_size=256)...")
    t0 = time.time()
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.0,
        batch_size=256,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42
    )
    mlp_reg.fit(X_train_250k_C, y_train_250k_reg)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_reg = mlp_reg.predict(X_val_C)
    t_pred = time.time() - t0

    m10 = evaluate_regressor(y_val_reg, y_pred_reg)
    m10.update({
        "run_id": "Run_10",
        "model": "MLP Regressor",
        "feature_group": "Group_C",
        "sample_size": len(df_train_250k),
        "val_sample_size": len(df_val),
        "train_sec": round(t_train, 2),
        "pred_sec": round(t_pred, 2),
        "n_iter": mlp_reg.n_iter_,
        "loss": round(mlp_reg.loss_, 4)
    })
    reg_results.append(m10)
    pred_reg_dict["MLP (Group C)"] = y_pred_reg
    joblib.dump(mlp_reg, os.path.join(MODELS_DIR, "regression/mlp_regressor_group_c.joblib"))
    print(f"  -> Run 10 Done: MAE={m10['mae']:.2f} min, RMSE={m10['rmse']:.2f} min, R2={m10['r2']:.4f} (Fit {t_train:.1f}s, Iters {mlp_reg.n_iter_})")

    # =========================================================================
    # MERGE WITH STAGE 3 BASELINES & SAVE FULL COMPARISON TABLES
    # =========================================================================
    print("\n--- 4. Consolidating Baseline and Stage 4 Comparative Tables ---")
    stage3_path = os.path.join(RESULTS_DIR, "metrics/stage3_baseline_metrics.json")
    with open(stage3_path, 'r') as f:
        stage3_data = json.load(f)

    # Classification consolidated
    all_cls = []
    for b in stage3_data["classification_baselines"]:
        all_cls.append({
            "stage": "Stage 3 Baseline",
            "run_id": "Baseline",
            "model": b["model"],
            "feature_group": b["feature_group"],
            "sample_size": b["train_sample_size"],
            "val_sample_size": b["val_sample_size"],
            "accuracy": b["accuracy"],
            "precision": b["precision"],
            "recall": b["recall"],
            "f1": b["f1"],
            "roc_auc": b["roc_auc"],
            "average_precision_pr_auc": b["average_precision_pr_auc"],
            "log_loss": b["log_loss"],
            "train_sec": b.get("train_duration_sec", 0.0),
            "pred_sec": b.get("pred_duration_sec", 0.0)
        })
    for r in cls_results:
        all_cls.append({
            "stage": "Stage 4 Main",
            "run_id": r["run_id"],
            "model": r["model"],
            "feature_group": r["feature_group"],
            "sample_size": r["sample_size"],
            "val_sample_size": r["val_sample_size"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
            "average_precision_pr_auc": r["average_precision_pr_auc"],
            "log_loss": r["log_loss"],
            "train_sec": r["train_sec"],
            "pred_sec": r["pred_sec"]
        })

    df_all_cls = pd.DataFrame(all_cls)
    df_all_cls.to_csv(os.path.join(RESULTS_DIR, "classification/stage4_model_comparison.csv"), index=False)
    print("Saved consolidated classification comparison: results/classification/stage4_model_comparison.csv")

    # Regression consolidated
    all_reg = []
    for b in stage3_data["regression_baselines"]:
        all_reg.append({
            "stage": "Stage 3 Baseline",
            "run_id": "Baseline",
            "model": b["model"],
            "feature_group": b["feature_group"],
            "sample_size": b["train_sample_size"],
            "val_sample_size": b["val_sample_size"],
            "mae": b["mae"],
            "mse": b["mse"],
            "rmse": b["rmse"],
            "r2": b["r2"],
            "train_sec": b.get("train_duration_sec", 0.0),
            "pred_sec": b.get("pred_duration_sec", 0.0)
        })
    for r in reg_results:
        all_reg.append({
            "stage": "Stage 4 Main",
            "run_id": r["run_id"],
            "model": r["model"],
            "feature_group": r["feature_group"],
            "sample_size": r["sample_size"],
            "val_sample_size": r["val_sample_size"],
            "mae": r["mae"],
            "mse": r["mse"],
            "rmse": r["rmse"],
            "r2": r["r2"],
            "train_sec": r["train_sec"],
            "pred_sec": r["pred_sec"]
        })

    df_all_reg = pd.DataFrame(all_reg)
    df_all_reg.to_csv(os.path.join(RESULTS_DIR, "regression/stage4_model_comparison.csv"), index=False)
    print("Saved consolidated regression comparison: results/regression/stage4_model_comparison.csv")

    # Save detailed JSON metrics
    metrics_payload = {
        "metadata": {
            "experiment": "Phase 6 Stage 4: Main Model Suite (10 Runs)",
            "validation_rows": len(df_val),
            "random_seed": 42,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        },
        "classification_runs": cls_results,
        "regression_runs": reg_results
    }
    with open(os.path.join(RESULTS_DIR, "metrics/stage4_metrics.json"), 'w') as f:
        json.dump(metrics_payload, f, indent=2, cls=NpEncoder)
    print("Saved detailed metrics JSON: results/metrics/stage4_metrics.json")

    # =========================================================================
    # GENERATE PUBLICATION-GRADE DIAGNOSTIC FIGURES
    # =========================================================================
    print("\n--- 5. Generating Diagnostic Figures ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Classification ROC Curve
    plt.figure(figsize=(9, 7), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5000)')

    # Load baseline Logistic Reg Group C for benchmark curve
    lr_c = joblib.load(os.path.join(MODELS_DIR, "classification/logistic_regression_group_c.joblib"))
    y_prob_lr_c = lr_c.predict_proba(X_val_C)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_val_cls, y_prob_lr_c)
    auc_lr = roc_auc_score(y_val_cls, y_prob_lr_c)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Reg Group C (AUC = {auc_lr:.4f})', color='gray', linestyle=':')

    colors = {
        "DT (Group A)": "#2b5c8f",
        "DT (Group C)": "#3b82f6",
        "RF (Group A)": "#15803d",
        "RF (Group C)": "#22c55e",
        "MLP (Group C)": "#a855f7"
    }
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_val_cls, probs)
        auc = roc_auc_score(y_val_cls, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=colors.get(name, 'black'), linewidth=2)

    plt.title("ROC Curves — Delay Propagation Classification (Validation Set)", fontsize=14, weight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures/classification_roc_curve.png"))
    plt.close()
    print("Saved: results/figures/classification_roc_curve.png")

    # 2. Precision-Recall Curve
    plt.figure(figsize=(9, 7), dpi=300)
    baseline_prior = float(np.mean(y_val_cls))
    plt.axhline(y=baseline_prior, color='black', linestyle='--', label=f'Baseline Class Balance (AP = {baseline_prior:.4f})')

    p_lr, r_lr, _ = precision_recall_curve(y_val_cls, y_prob_lr_c)
    ap_lr = average_precision_score(y_val_cls, y_prob_lr_c)
    plt.plot(r_lr, p_lr, label=f'Logistic Reg Group C (AP = {ap_lr:.4f})', color='gray', linestyle=':')

    for name, probs in prob_dict.items():
        p, r, _ = precision_recall_curve(y_val_cls, probs)
        ap = average_precision_score(y_val_cls, probs)
        plt.plot(r, p, label=f'{name} (AP = {ap:.4f})', color=colors.get(name, 'black'), linewidth=2)

    plt.title("Precision-Recall Curves — Delay Propagation (Validation Set)", fontsize=14, weight='bold')
    plt.xlabel("Recall (Coverage of Propagated Delays)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures/classification_precision_recall_curve.png"))
    plt.close()
    print("Saved: results/figures/classification_precision_recall_curve.png")

    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    key_models = [
        ("Logistic Reg (Group C)", (y_prob_lr_c >= 0.5).astype(int)),
        ("Random Forest (Group C)", (prob_dict["RF (Group C)"] >= 0.5).astype(int)),
        ("MLP Classifier (Group C)", (prob_dict["MLP (Group C)"] >= 0.5).astype(int))
    ]
    for ax, (title, preds) in zip(axes, key_models):
        cm = confusion_matrix(y_val_cls, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['On-Time (<15m)', 'Delayed (>=15m)'],
                    yticklabels=['On-Time (<15m)', 'Delayed (>=15m)'])
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
    plt.suptitle("Validation Set Confusion Matrices (Threshold = 0.5)", fontsize=15, weight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures/classification_confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/classification_confusion_matrix.png")

    # 4. Actual vs Predicted Delay (Hexbin / Scatter Sample)
    plt.figure(figsize=(9, 7), dpi=300)
    # Take 50,000 validation sample for smooth hexbin plotting
    np.random.seed(42)
    sample_idx = np.random.choice(len(y_val_reg), size=50000, replace=False)
    y_val_sub = y_val_reg[sample_idx]
    y_pred_rf = pred_reg_dict["RF (Group C)"][sample_idx]

    hb = plt.hexbin(y_val_sub, y_pred_rf, gridsize=60, cmap='viridis', mincnt=1, bins='log')
    plt.plot([-50, 250], [-50, 250], 'r--', linewidth=2, label='Identity (y = x)')
    plt.colorbar(hb, label='log10(Density)')
    plt.xlim(-50, 250)
    plt.ylim(-50, 250)
    plt.title("Actual vs. Predicted Arrival Delay — Random Forest Group C", fontsize=14, weight='bold')
    plt.xlabel("Actual Arrival Delay (minutes)", fontsize=12)
    plt.ylabel("Predicted Arrival Delay (minutes)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures/regression_actual_vs_predicted.png"))
    plt.close()
    print("Saved: results/figures/regression_actual_vs_predicted.png")

    # 5. Regression Residual Distribution
    plt.figure(figsize=(9, 6), dpi=300)
    residuals_rf = y_val_sub - y_pred_rf
    residuals_mlp = y_val_sub - pred_reg_dict["MLP (Group C)"][sample_idx]
    
    # Load linear reg C for residual comparison
    lr_reg_c = joblib.load(os.path.join(MODELS_DIR, "regression/linear_regression_group_c.joblib"))
    y_pred_ols_sub = lr_reg_c.predict(X_val_C[sample_idx])
    residuals_ols = y_val_sub - y_pred_ols_sub

    sns.kdeplot(residuals_ols, label='OLS Linear Reg Group C', color='gray', linestyle=':', clip=(-80, 80))
    sns.kdeplot(residuals_rf, label='Random Forest Group C', color='#22c55e', linewidth=2, clip=(-80, 80))
    sns.kdeplot(residuals_mlp, label='MLP Regressor Group C', color='#a855f7', linewidth=2, clip=(-80, 80))

    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.title("Residual Distribution (y_true - y_pred) on Validation Data", fontsize=14, weight='bold')
    plt.xlabel("Residual Error (minutes)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures/regression_residual_distribution.png"))
    plt.close()
    print("Saved: results/figures/regression_residual_distribution.png")

    print("\n================================================================================")
    print("STAGE 4 COMPLETE: All 10 Runs Executed, Evaluated, and Persisted Successfully!")
    print("================================================================================")


if __name__ == "__main__":
    run_stage4_experiments()

