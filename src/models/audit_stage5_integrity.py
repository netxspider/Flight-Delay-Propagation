"""
Stage 5.1: Final Model Selection Integrity Audit
Forensic verification of classification thresholds, confusion matrices,
artifact reproducibility, and methodological terminology.
Strictly sealed: NO access to data/processed/modeling_test.parquet.
"""

import os
import sys
import time
import json
import duckdb
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

BASE_DIR = "/Users/arnavraj/Downloads/flight-delay-propagation"
VAL_PARQUET = os.path.join(BASE_DIR, "data/processed/modeling_validation.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FINAL_MODELS_DIR = os.path.join(MODELS_DIR, "final")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure test set is never accessed
for arg in sys.argv:
    if "test" in arg.lower():
        raise ValueError("Strictly prohibited: test set referenced!")

print("=" * 80)
print("PHASE 6 — STAGE 5.1: FINAL MODEL SELECTION INTEGRITY AUDIT")
print("=" * 80)

# 1. Load Validation Data
print("\n--- 1. Loading Validation Data ---")
con = duckdb.connect()
df_val = con.execute(f"SELECT * FROM read_parquet('{VAL_PARQUET}')").df()
n_val = len(df_val)
y_val_cls = df_val['NEXT_ARR_DEL15'].values.astype(int)
y_val_reg = df_val['NEXT_ARR_DELAY'].values.astype(float)
n_pos = int(y_val_cls.sum())
n_neg = int(n_val - n_pos)

print(f"Validation population: N = {n_val:,}")
print(f"Actual positive count (NEXT_ARR_DEL15 = 1): {n_pos:,} ({n_pos / n_val * 100:.2f}%)")
print(f"Actual negative count (NEXT_ARR_DEL15 = 0): {n_neg:,} ({n_neg / n_val * 100:.2f}%)")

# 2. Load Serialized Final Artifacts
print("\n--- 2. Loading Serialized Final Artifacts ---")
prep_c = joblib.load(os.path.join(FINAL_MODELS_DIR, "final_preprocessor_group_c.joblib"))
clf_c = joblib.load(os.path.join(FINAL_MODELS_DIR, "final_selected_classifier.joblib"))
reg_c = joblib.load(os.path.join(FINAL_MODELS_DIR, "final_selected_regressor.joblib"))

print("Transforming validation set with Group C preprocessor...")
t0 = time.time()
X_val_c = prep_c.transform(df_val)
t_prep = time.time() - t0
print(f"Group C feature matrix transformed: {X_val_c.shape} in {t_prep:.2f}s")

# 3. Model Classes and Positive Probability Indexing
print("\n--- 3. Verifying Estimators and Positive Probability Indexing ---")
print(f"Classifier class: {clf_c.__class__.__name__}")
print(f"Classifier hyperparameters: {json.dumps({k: v for k, v in clf_c.get_params().items() if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'random_state', 'n_jobs']})}")
print(f"Classifier classes: {clf_c.classes_}")

pos_idx = int(np.where(clf_c.classes_ == 1)[0][0])
print(f"Positive class (1) column index in predict_proba: {pos_idx}")

t0 = time.time()
val_probs_rf = clf_c.predict_proba(X_val_c)[:, pos_idx]
t_prob = time.time() - t0
print(f"Validation probabilities generated in {t_prob:.2f}s. Min: {val_probs_rf.min():.4f}, Max: {val_probs_rf.max():.4f}, Mean: {val_probs_rf.mean():.4f}")

# Overall discrimination metrics
rf_auc = float(roc_auc_score(y_val_cls, val_probs_rf))
rf_ap = float(average_precision_score(y_val_cls, val_probs_rf))
rf_ll = float(log_loss(y_val_cls, val_probs_rf))
print(f"Random Forest Discrimination: ROC-AUC = {rf_auc:.4f}, AP = {rf_ap:.4f}, LogLoss = {rf_ll:.4f}")

# Also load Decision Tree and fit Logistic Regression Reference for complete threshold audit
dt_clf = joblib.load(os.path.join(MODELS_DIR, "classification/decision_tree_group_c.joblib"))
dt_pos_idx = int(np.where(dt_clf.classes_ == 1)[0][0])
val_probs_dt = dt_clf.predict_proba(X_val_c)[:, dt_pos_idx]

print("Training reference Logistic Regression on N=500k training sample to reproduce Stage 5 reference probabilities...")
TRAIN_PARQUET = os.path.join(BASE_DIR, "data/processed/modeling_train.parquet")
df_tr_500k = con.execute(f"SELECT * FROM read_parquet('{TRAIN_PARQUET}') USING SAMPLE 500000 (reservoir, 42)").df()
X_tr_500k_c = prep_c.transform(df_tr_500k)
y_tr_500k_cls = df_tr_500k['NEXT_ARR_DEL15'].values.astype(int)

from sklearn.linear_model import LogisticRegression
lr_ref = LogisticRegression(penalty=None, solver='lbfgs', max_iter=200, random_state=42)
lr_ref.fit(X_tr_500k_c, y_tr_500k_cls)
lr_pos_idx = int(np.where(lr_ref.classes_ == 1)[0][0])
val_probs_lr = lr_ref.predict_proba(X_val_c)[:, lr_pos_idx]
print(f"Reference Logistic Regression fitted. Val AP: {average_precision_score(y_val_cls, val_probs_lr):.4f}")

# 4. Forensic Audit of Thresholds
print("\n--- 4. Full Audit of Threshold Calculations ---")
threshold_list = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
models_to_audit = [
    ("Random Forest Classifier (Group C)", val_probs_rf),
    ("Decision Tree Classifier (Group C)", val_probs_dt),
    ("Logistic Regression Reference (Group C)", val_probs_lr)
]

audit_rows = []
cm_rows = []

for model_name, probs in models_to_audit:
    for t in threshold_list:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_val_cls, preds)
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        
        # Exact mathematical definitions
        pos_preds = tp + fp
        total_obs = tn + fp + fn + tp
        
        acc = (tp + tn) / total_obs
        prec = tp / pos_preds if pos_preds > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Sklearn verification
        acc_skl = float(accuracy_score(y_val_cls, preds))
        prec_skl = float(precision_score(y_val_cls, preds, zero_division=0))
        rec_skl = float(recall_score(y_val_cls, preds, zero_division=0))
        f1_skl = float(f1_score(y_val_cls, preds, zero_division=0))
        
        # Verify perfect identity
        assert abs(acc - acc_skl) < 1e-9, f"Accuracy mismatch at t={t}"
        assert abs(prec - prec_skl) < 1e-9, f"Precision mismatch at t={t}"
        assert abs(rec - rec_skl) < 1e-9, f"Recall mismatch at t={t}"
        assert abs(f1 - f1_skl) < 1e-9, f"F1 mismatch at t={t}"
        
        # Verification checks
        pos_check = (tp + fn == n_pos)
        neg_check = (tn + fp == n_neg)
        tot_check = (total_obs == n_val)
        
        audit_rows.append({
            "model": model_name,
            "threshold": t,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "specificity": round(spec, 4),
            "positive_prediction_count": pos_preds,
            "confusion_matrix_total": total_obs,
            "reconciles_positives": pos_check,
            "reconciles_negatives": neg_check,
            "reconciles_total": tot_check
        })
        
        cm_rows.append({
            "model": model_name,
            "threshold": t,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "TP_plus_FN": tp + fn,
            "TN_plus_FP": tn + fp,
            "total": total_obs,
            "precision_pct": round(prec * 100, 2),
            "recall_pct": round(rec * 100, 2),
            "f1_score": round(f1, 4),
            "specificity_pct": round(spec * 100, 2)
        })

df_audit = pd.DataFrame(audit_rows)
df_cm = pd.DataFrame(cm_rows)

audit_csv = os.path.join(RESULTS_DIR, "metrics/stage5_threshold_integrity_audit.csv")
cm_csv = os.path.join(RESULTS_DIR, "metrics/stage5_threshold_confusion_matrices.csv")
df_audit.to_csv(audit_csv, index=False)
df_cm.to_csv(cm_csv, index=False)
print(f"Saved: {audit_csv}")
print(f"Saved: {cm_csv}")

# 5. Fine Scan for Optimal Threshold
print("\n--- 5. Fine Scan of Thresholds for Random Forest (Group C) ---")
fine_thresholds = np.linspace(0.05, 0.75, 71)
rf_fine_results = []

for t in fine_thresholds:
    preds = (val_probs_rf >= t).astype(int)
    cm = confusion_matrix(y_val_cls, preds)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    
    pos_preds = tp + fp
    prec = tp / pos_preds if pos_preds > 0 else 0.0
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp)
    acc = (tp + tn) / n_val
    
    rf_fine_results.append({
        "threshold": round(t, 2),
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "accuracy": acc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

df_rf_fine = pd.DataFrame(rf_fine_results)

# Global F1 Maximum
idx_max_f1 = df_rf_fine['f1_score'].idxmax()
row_max_f1 = df_rf_fine.iloc[idx_max_f1]
print(f"Global F1 Maximum: T = {row_max_f1['threshold']:.2f}")
print(f"  -> F1: {row_max_f1['f1_score']:.4f}, Precision: {row_max_f1['precision']*100:.2f}%, Recall: {row_max_f1['recall']*100:.2f}%")
print(f"  -> TP: {row_max_f1['TP']:,}, FP: {row_max_f1['FP']:,}, FN: {row_max_f1['FN']:,}, TN: {row_max_f1['TN']:,}")

# Precision-Constrained Maximum (Precision >= 70%)
df_prec70 = df_rf_fine[df_rf_fine['precision'] >= 0.70]
idx_p70_max_rec = df_prec70['recall'].idxmax()
row_p70 = df_prec70.loc[idx_p70_max_rec]
print(f"\nPrecision-Constrained Optimum (Precision >= 70%): T = {row_p70['threshold']:.2f}")
print(f"  -> Precision: {row_p70['precision']*100:.2f}%, Recall: {row_p70['recall']*100:.2f}%, F1: {row_p70['f1_score']:.4f}")

# Recall-Constrained Optimum (Recall >= 50%)
df_rec50 = df_rf_fine[df_rf_fine['recall'] >= 0.50]
idx_r50_max_prec = df_rec50['precision'].idxmax()
row_r50 = df_rec50.loc[idx_r50_max_prec]
print(f"\nRecall-Constrained Optimum (Recall >= 50%): T = {row_r50['threshold']:.2f}")
print(f"  -> Precision: {row_r50['precision']*100:.2f}%, Recall: {row_r50['recall']*100:.2f}%, F1: {row_r50['f1_score']:.4f}")

# Compare Operating Thresholds: T = 0.35 vs T = 0.40 vs T = 0.50
row_035 = df_rf_fine[df_rf_fine['threshold'] == 0.35].iloc[0]
row_040 = df_rf_fine[df_rf_fine['threshold'] == 0.40].iloc[0]
row_050 = df_rf_fine[df_rf_fine['threshold'] == 0.50].iloc[0]

print("\n--- Key Operating Points Comparison ---")
print(f"Threshold T = 0.50 (Default): Prec={row_050['precision']*100:.2f}%, Rec={row_050['recall']*100:.2f}%, F1={row_050['f1_score']:.4f}, FP={row_050['FP']:,}, FN={row_050['FN']:,}")
print(f"Threshold T = 0.40 (Coarse F1-Optimum): Prec={row_040['precision']*100:.2f}%, Rec={row_040['recall']*100:.2f}%, F1={row_040['f1_score']:.4f}, FP={row_040['FP']:,}, FN={row_040['FN']:,}")
print(f"Threshold T = 0.39 (Fine F1-Optimum): Prec={row_max_f1['precision']*100:.2f}%, Rec={row_max_f1['recall']*100:.2f}%, F1={row_max_f1['f1_score']:.4f}, FP={row_max_f1['FP']:,}, FN={row_max_f1['FN']:,}")
print(f"Threshold T = 0.35 (Selected Operating Threshold): Prec={row_035['precision']*100:.2f}%, Rec={row_035['recall']*100:.2f}%, F1={row_035['f1_score']:.4f}, FP={row_035['FP']:,}, FN={row_035['FN']:,}")

# 6. Audit Regression Estimator Reproducibility
print("\n--- 6. Auditing Regression Estimator Reproducibility ---")
print(f"Regressor class: {reg_c.__class__.__name__}")
print(f"Regressor hyperparameters: {json.dumps({k: v for k, v in reg_c.get_params().items() if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'random_state', 'n_jobs']})}")

t0 = time.time()
val_preds_reg = reg_c.predict(X_val_c)
t_pred_reg = time.time() - t0

reg_mae = float(mean_absolute_error(y_val_reg, val_preds_reg))
reg_mse = float(mean_squared_error(y_val_reg, val_preds_reg))
reg_rmse = float(np.sqrt(reg_mse))
reg_r2 = float(r2_score(y_val_reg, val_preds_reg))

print(f"Regression Predictions generated in {t_pred_reg:.2f}s:")
print(f"  MAE: {reg_mae:.4f} min")
print(f"  RMSE: {reg_rmse:.4f} min")
print(f"  R2: {reg_r2:.6f}")

# Also check Group A ablation reproducibility
prep_a = joblib.load(os.path.join(FINAL_MODELS_DIR, "final_preprocessor_group_a.joblib"))
clf_a = joblib.load(os.path.join(MODELS_DIR, "classification/random_forest_group_a.joblib"))
reg_a = joblib.load(os.path.join(MODELS_DIR, "regression/random_forest_group_a.joblib"))

X_val_a = prep_a.transform(df_val)
pos_idx_a = int(np.where(clf_a.classes_ == 1)[0][0])
val_probs_rf_a = clf_a.predict_proba(X_val_a)[:, pos_idx_a]
val_preds_reg_a = reg_a.predict(X_val_a)

rf_a_auc = float(roc_auc_score(y_val_cls, val_probs_rf_a))
rf_a_ap = float(average_precision_score(y_val_cls, val_probs_rf_a))
reg_a_mae = float(mean_absolute_error(y_val_reg, val_preds_reg_a))
reg_a_rmse = float(np.sqrt(mean_squared_error(y_val_reg, val_preds_reg_a)))
reg_a_r2 = float(r2_score(y_val_reg, val_preds_reg_a))

print("\n--- Group A vs Group C Ablation Verification ---")
print(f"Classification RF: Group A AP={rf_a_ap:.4f}, AUC={rf_a_auc:.4f} -> Group C AP={rf_ap:.4f}, AUC={rf_auc:.4f} (Delta AP: {rf_ap - rf_a_ap:+.4f}, Delta AUC: {rf_auc - rf_a_auc:+.4f})")
print(f"Regression RF: Group A MAE={reg_a_mae:.2f}m, RMSE={reg_a_rmse:.2f}m, R2={reg_a_r2:.4f} -> Group C MAE={reg_mae:.2f}m, RMSE={reg_rmse:.2f}m, R2={reg_r2:.4f} (Delta MAE: {reg_mae - reg_a_mae:+.2f}m, Delta R2: {reg_r2 - reg_a_r2:+.4f})")

# 7. Write Reproducibility JSON
reproducibility_payload = {
    "metadata": {
        "audit_phase": "Phase 6 Stage 5.1: Final Model Selection Integrity Audit",
        "audit_timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "validation_population": n_val,
        "positive_observations": n_pos,
        "negative_observations": n_neg,
        "test_set_status": "Strictly Sealed (Unaccessed)"
    },
    "classifier_reproducibility": {
        "saved_model_path": "models/final/final_selected_classifier.joblib",
        "saved_preprocessor_path": "models/final/final_preprocessor_group_c.joblib",
        "model_class": clf_c.__class__.__name__,
        "classes": clf_c.classes_.tolist(),
        "positive_class_index": pos_idx,
        "feature_count_transformed": int(X_val_c.shape[1]),
        "validation_metrics": {
            "roc_auc": round(rf_auc, 6),
            "average_precision_ap": round(rf_ap, 6),
            "log_loss": round(rf_ll, 6)
        },
        "threshold_metrics": {
            "default_threshold_0_50": {
                "accuracy": round(float(row_050['accuracy']), 4),
                "precision": round(float(row_050['precision']), 4),
                "recall": round(float(row_050['recall']), 4),
                "f1_score": round(float(row_050['f1_score']), 4),
                "specificity": round(float(row_050['specificity']), 4),
                "confusion_matrix": {
                    "TP": int(row_050['TP']),
                    "FP": int(row_050['FP']),
                    "FN": int(row_050['FN']),
                    "TN": int(row_050['TN'])
                }
            },
            "f1_maximizing_threshold_0_39": {
                "accuracy": round(float(row_max_f1['accuracy']), 4),
                "precision": round(float(row_max_f1['precision']), 4),
                "recall": round(float(row_max_f1['recall']), 4),
                "f1_score": round(float(row_max_f1['f1_score']), 4),
                "specificity": round(float(row_max_f1['specificity']), 4),
                "confusion_matrix": {
                    "TP": int(row_max_f1['TP']),
                    "FP": int(row_max_f1['FP']),
                    "FN": int(row_max_f1['FN']),
                    "TN": int(row_max_f1['TN'])
                }
            },
            "selected_operational_threshold_0_35": {
                "accuracy": round(float(row_035['accuracy']), 4),
                "precision": round(float(row_035['precision']), 4),
                "recall": round(float(row_035['recall']), 4),
                "f1_score": round(float(row_035['f1_score']), 4),
                "specificity": round(float(row_035['specificity']), 4),
                "confusion_matrix": {
                    "TP": int(row_035['TP']),
                    "FP": int(row_035['FP']),
                    "FN": int(row_035['FN']),
                    "TN": int(row_035['TN'])
                }
            }
        }
    },
    "regressor_reproducibility": {
        "saved_model_path": "models/final/final_selected_regressor.joblib",
        "saved_preprocessor_path": "models/final/final_preprocessor_group_c.joblib",
        "model_class": reg_c.__class__.__name__,
        "hyperparameters": {k: v for k, v in reg_c.get_params().items() if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'random_state', 'n_jobs']},
        "validation_metrics": {
            "mae_minutes": round(reg_mae, 4),
            "mse_minutes2": round(reg_mse, 4),
            "rmse_minutes": round(reg_rmse, 4),
            "r2_score": round(reg_r2, 6)
        }
    },
    "reconciliation_verifications": {
        "all_confusion_matrices_reconcile_exactly": bool((df_audit['reconciles_positives'] & df_audit['reconciles_negatives'] & df_audit['reconciles_total']).all()),
        "positive_sum_exact": n_pos,
        "negative_sum_exact": n_neg,
        "total_sum_exact": n_val
    }
}

repro_json = os.path.join(RESULTS_DIR, "metrics/stage5_final_artifact_reproducibility.json")
with open(repro_json, "w") as f:
    json.dump(reproducibility_payload, f, indent=2)
print(f"Saved: {repro_json}")

# 8. Re-generate Figures with Accurate Threshold Curves
print("\n--- 8. Re-generating Reconciled Threshold Curves ---")
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(df_rf_fine['threshold'], df_rf_fine['f1_score'], color='#2563eb', lw=2.5, label='F1-Score')
plt.plot(df_rf_fine['threshold'], df_rf_fine['precision'], color='#16a34a', lw=2, linestyle='--', label='Precision')
plt.plot(df_rf_fine['threshold'], df_rf_fine['recall'], color='#dc2626', lw=2, linestyle=':', label='Recall (Coverage)')
plt.plot(df_rf_fine['threshold'], df_rf_fine['specificity'], color='#9333ea', lw=1.5, linestyle='-.', label='Specificity')

# Vertical markers
plt.axvline(row_max_f1['threshold'], color='#2563eb', linestyle=':', lw=1.8, label=f'Max F1 Threshold (T = {row_max_f1["threshold"]:.2f}, F1 = {row_max_f1["f1_score"]:.4f})')
plt.axvline(0.35, color='#d97706', linestyle='-.', lw=1.8, label='Selected Operating Threshold (T = 0.35)')
plt.axvline(0.50, color='#64748b', linestyle='--', lw=1.5, label='Default Threshold (T = 0.50)')

plt.title("Classification Threshold Trade-offs & Audit Reconciliation\n(Random Forest Group C, Full Validation N = 1,140,909)", fontsize=12, weight='bold')
plt.xlabel("Decision Probability Threshold", fontsize=11, weight='bold')
plt.ylabel("Validation Metric Value", fontsize=11, weight='bold')
plt.legend(loc='center left', frameon=True, fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0.10, 0.70)
plt.ylim(0.20, 1.00)
plt.tight_layout()
fig_thresh_path = os.path.join(RESULTS_DIR, "figures/stage5_threshold_curves.png")
plt.savefig(fig_thresh_path)
plt.close()
print(f"Saved reconciled figure: {fig_thresh_path}")

# Also update final model diagnostics
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Confusion matrix at T = 0.35
cm_035 = np.array([[int(row_035['TN']), int(row_035['FP'])], [int(row_035['FN']), int(row_035['TP'])]], dtype=int)
sns.heatmap(cm_035, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Pred On-Time', 'Pred Delayed'],
            yticklabels=['True On-Time', 'True Delayed'])
axes[0].set_title(f"Final Classifier: Confusion Matrix (T = 0.35)\nPrecision = {row_035['precision']*100:.2f}%, Recall = {row_035['recall']*100:.2f}%, F1 = {row_035['f1_score']:.4f}", fontsize=12, weight='bold')
axes[0].set_xlabel("Predicted Label", fontsize=11)
axes[0].set_ylabel("True Label", fontsize=11)

# Regression hexbin
np.random.seed(42)
sample_50k = np.random.choice(len(y_val_reg), size=50000, replace=False)
hb = axes[1].hexbin(y_val_reg[sample_50k], val_preds_reg[sample_50k], gridsize=50, cmap='viridis', mincnt=1, bins='log')
axes[1].plot([-40, 200], [-40, 200], 'r--', lw=2, label='Identity (y = x)')
fig.colorbar(hb, ax=axes[1], label='log10(Observations)')
axes[1].set_title(f"Final Regressor: Actual vs. Predicted Arrival Delay\n(Validation N = 1,140,909, R2 = {reg_r2:.4f}, MAE = {reg_mae:.2f}m)", fontsize=12, weight='bold')
axes[1].set_xlabel("Actual Arrival Delay (minutes)", fontsize=11)
axes[1].set_ylabel("Predicted Arrival Delay (minutes)", fontsize=11)
axes[1].set_xlim(-40, 200)
axes[1].set_ylim(-40, 200)
axes[1].legend(loc='upper left')

plt.tight_layout()
fig_diag_path = os.path.join(RESULTS_DIR, "figures/stage5_final_model_diagnostics.png")
plt.savefig(fig_diag_path)
plt.close()
print(f"Saved reconciled figure: {fig_diag_path}")

# 9. Backup original Stage 5 report
orig_report_path = os.path.join(RESULTS_DIR, "tables/stage5_model_selection.md")
backup_report_path = os.path.join(RESULTS_DIR, "tables/stage5_model_selection_v1_backup.md")

if not os.path.exists(backup_report_path):
    shutil.copyfile(orig_report_path, backup_report_path)
    print(f"Backed up original Stage 5 report to: {backup_report_path}")

print("\n" + "=" * 80)
print("STAGE 5.1 AUDIT SCRIPT FINISHED SUCCESSFULLY!")
print("=" * 80)
