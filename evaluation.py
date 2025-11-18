import pandas as pd
from sklearn.metrics import precision_score, recall_score

file_path = "/Users/tajju/Desktop/Assignments/COMP7707/ahme0423_A3/a3_prototype/output/2025-11-17--11/.part-cc73fb00-b66a-4c1f-95ba-7347d04844d8-0.inprogress.9f6d692d-4caf-44b5-90e6-5097957cab41"
headers = [
    "timestamp", "vehicle_id", "latitude", "longitude", "rule_anomaly", "rule_anomaly_types",
    "half_space_score", "half_space_compute_ms",
    "iforest_score", "iforest_compute_ms",
    "lof_score", "lof_compute_ms",
    "dbscan_label", "dbscan_compute_ms"
]

df = pd.read_csv(file_path, names=headers)

# Helper functions for anomaly detection


def is_hs_anomaly(row): return row['half_space_score'] > 0.0
def is_iforest_anomaly(row): return row['iforest_score'] > 0.0
def is_lof_anomaly(row): return row['lof_score'] > 1.0
def is_dbscan_anomaly(row): return row['dbscan_label'] in [1, 2]


# ---- 1. Ground Truth ----
ground_truth_anomalies = df[df['rule_anomaly'] == 1]
ground_truth_count = ground_truth_anomalies.shape[0]
ground_truth_types = ground_truth_anomalies['rule_anomaly_types'].value_counts(
)
print(f"#1. Total ground-truth anomalies: {ground_truth_count}")
print("Ground truth anomaly types:\n", ground_truth_types, "\n")

# ---- 2. Algorithm–Ground Truth Agreement ----


def algo_agree_count(row):
    return sum([
        is_hs_anomaly(row),
        is_iforest_anomaly(row),
        is_lof_anomaly(row),
        is_dbscan_anomaly(row)
    ])


ground_truth_anomalies['algos_agree_with_truth'] = ground_truth_anomalies.apply(
    algo_agree_count, axis=1)
agree_with_truth_summary = ground_truth_anomalies['algos_agree_with_truth'].value_counts(
).sort_index()
print("#2. (Algorithm–Ground Truth Agreement)")
print("Counts for how many algorithms agree on each ground-truth anomaly (0 = none, 4 = all):")
print(agree_with_truth_summary, "\n")

# ---- 3. Algorithms–Algorithms Agreement (all rows) ----
df['algos_agree'] = df.apply(algo_agree_count, axis=1)
algos_agree_summary = df['algos_agree'].value_counts().sort_index()
print("#3. (Algorithms–Algorithms Agreement for ALL rows):")
print("For every row, how many algorithms independently detected anomaly (0–4):")
print(algos_agree_summary, "\n")

# ---- 4. Per-algorithm anomaly counts ----
df['hs_anomaly'] = df.apply(is_hs_anomaly, axis=1)
df['iforest_anomaly'] = df.apply(is_iforest_anomaly, axis=1)
df['lof_anomaly'] = df.apply(is_lof_anomaly, axis=1)
df['dbscan_anomaly'] = df.apply(is_dbscan_anomaly, axis=1)
print("#4. Algorithm anomaly counts:")
print("Half-space:", df['hs_anomaly'].sum())
print("Isolation Forest:", df['iforest_anomaly'].sum())
print("LOF:", df['lof_anomaly'].sum())
print("DBSCAN:", df['dbscan_anomaly'].sum(), "\n")

# ---- 5. Precision and Recall per Algorithm ----
y_true = df['rule_anomaly'].astype(int)
metrics = []
for col, name in [('hs_anomaly', 'Half-space'), ('iforest_anomaly', 'Isolation Forest'), ('lof_anomaly', 'LOF'), ('dbscan_anomaly', 'DBSCAN')]:
    y_pred = df[col].astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    metrics.append((name, prec, rec))

print("#5. Precision and Recall vs Ground Truth:")
print("(Precision: % of detected anomalies that were correct, Recall: % of ground-truth anomalies detected)")
for name, prec, rec in metrics:
    print(f"{name:15s} Precision: {prec:.2f}, Recall: {rec:.2f}")

# ---- 6. Average Compute Time for Each Algorithm ----
avg_hs_time = df['half_space_compute_ms'].mean()
avg_iforest_time = df['iforest_compute_ms'].mean()
avg_lof_time = df['lof_compute_ms'].mean()
avg_dbscan_time = df['dbscan_compute_ms'].mean()

print("\n#6. Average Compute Times (milliseconds):")
print(f"Half-space:         {avg_hs_time:.4f} ms")
print(f"Isolation Forest:   {avg_iforest_time:.4f} ms")
print(f"LOF:                {avg_lof_time:.4f} ms")
print(f"DBSCAN:             {avg_dbscan_time:.4f} ms")

# ---- Optional: Save summary to CSV ----
summary = {
    "ground_truth_count": ground_truth_count,
    "ground_truth_types": ground_truth_types.to_dict(),
    "algos_agree_with_truth": agree_with_truth_summary.to_dict(),
    "algos_agree_summary": algos_agree_summary.to_dict(),
    "avg_hs_time_ms": avg_hs_time,
    "avg_iforest_time_ms": avg_iforest_time,
    "avg_lof_time_ms": avg_lof_time,
    "avg_dbscan_time_ms": avg_dbscan_time
}
for name, prec, rec in metrics:
    summary[f"{name}_precision"] = prec
    summary[f"{name}_recall"] = rec
pd.DataFrame.from_dict(summary, orient='index').to_csv(
    "/Users/tajju/Desktop/Assignments/COMP7707/ahme0423_A3/a3_prototype/output/2025-11-16--13/summary.csv"
)
print("\nSummary saved to summary.csv")

# ---- Clarification of 'agreement': ----
print("""
AGREEMENT DEFINITIONS:
- 'algos_agree_with_truth': For each ground-truth anomaly, number of algorithms that also detected anomaly (0–4).
- 'algos_agree_summary': For every row, how many algorithms independently detected anomaly (0–4), regardless of ground-truth.
- Precision/Recall: For each algorithm, how closely results match ground-truth anomaly labels.
- Average compute times show typical algorithm runtime per row (ms).
""")
