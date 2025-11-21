# simple_evaluation.py - Binary Anomaly Detection Evaluation for Adelaide Metro
# Compares multiple unsupervised anomaly detection algorithms against rule-based ground truth
# Focuses on anomaly-centric metrics due to severe class imbalance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from constants import CSV_HEADERS
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


class AnomalyDetectionEvaluator:
    """
    Evaluates multiple anomaly detection algorithms on Adelaide Metro transit data.

    Compares unsupervised methods (Isolation Forest, LOF, DBSCAN, Half-Space Trees)
    against rule-based ground truth anomalies using anomaly-focused metrics.

    Due to severe class imbalance (~94% normal, ~6% anomalies), accuracy is not used.
    Instead, focuses on recall (detection rate) and precision (flag accuracy).
    """

    def __init__(self, output_dir: str):
        """
        Initialize evaluator with output directory containing CSV results.

        Args:
            output_dir: Path to directory containing algorithm output CSVs
        """
        self.output_dir = Path(output_dir)
        self.df = None
        self.anomaly_df = None
        self.algorithms = ['half_space', 'iforest', 'lof', 'dbscan']

    def load_data(self):
        """
        Load and concatenate all CSV files from output directory.

        Returns:
            Combined DataFrame with all samples
        """
        print("=" * 80)
        print("📁 LOADING DATA")
        print("=" * 80)

        all_dfs = []
        folders = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])

        print(f"\nFound {len(folders)} data folders")

        for folder in folders:
            files = list(folder.glob("*"))
            if not files:
                continue

            for f in files:
                try:
                    df = pd.read_csv(f, names=CSV_HEADERS,
                                     header=None, index_col=False)
                    all_dfs.append(df)
                    print(f"  ✓ Loaded: {f.name} ({len(df):,} rows)")
                except Exception as e:
                    print(f"  ✗ Failed to load {f.name}: {e}")

        self.df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n{'─' * 80}")
        print(
            f"✅ LOADED: {len(self.df):,} total samples from {len(all_dfs)} files")
        print(f"   Unique vehicles: {self.df['vehicle_id'].nunique()}")
        print(f"   Unique routes: {self.df['route_id'].nunique()}")
        print(f"{'─' * 80}\n")

        return self.df

    def clean_dataframe(self):
        """
        Remove rows with missing values and report cleaning statistics.

        Returns:
            Cleaned DataFrame
        """
        print("🧹 CLEANING DATA")
        print("─" * 80)

        initial_shape = self.df.shape
        print(
            f"Before cleaning: {initial_shape[0]:,} rows × {initial_shape[1]} columns")

        nan_counts = self.df.isnull().sum()
        if nan_counts.sum() > 0:
            print("\nColumns with missing values:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"  • {col}: {count:,} ({count/len(self.df)*100:.2f}%)")

        self.df = self.df.dropna(axis=0)

        final_shape = self.df.shape
        rows_removed = initial_shape[0] - final_shape[0]

        print(
            f"\nAfter cleaning: {final_shape[0]:,} rows × {final_shape[1]} columns")
        print(
            f"Removed: {rows_removed:,} rows ({rows_removed/initial_shape[0]*100:.2f}%)")
        print("─" * 80 + "\n")

        return self.df

    def report_on_ground_truth(self):
        """
        Analyze and report ground truth anomaly statistics.

        Provides breakdown of total anomalies and their types to understand
        the baseline distribution and expected detection rates.

        Returns:
            Percentage of data that are anomalies according to ground truth
        """
        print("📊 GROUND TRUTH ANALYSIS")
        print("─" * 80)

        total_samples = len(self.df)
        total_anomalies = self.df['rule_anomaly'].sum()
        pct_anomaly = (total_anomalies / total_samples) * 100

        print(f"Total samples: {total_samples:,}")
        print(
            f"Rule-based anomalies: {total_anomalies:,} ({pct_anomaly:.2f}%)")
        print(
            f"Normal samples: {total_samples - total_anomalies:,} ({100 - pct_anomaly:.2f}%)")

        if 'rule_anomaly_types' in self.df.columns:
            anomaly_types = self.df[self.df['rule_anomaly']
                                    == 1]['rule_anomaly_types'].value_counts()
            if len(anomaly_types) > 0:
                print("\nAnomaly type breakdown:")
                for anom_type, count in anomaly_types.items():
                    print(
                        f"  • {anom_type}: {count:,} ({count/total_anomalies*100:.1f}%)")

        print("─" * 80 + "\n")

        return pct_anomaly

    def visualize_anomaly_score_histograms(self, percentile, save_path='evaluation/anomaly_score_histograms.png'):
        """
        Create histogram visualizations for all algorithm anomaly scores with threshold lines.

        Helps understand score distributions and verify that thresholds are reasonable.
        The threshold line shows where the binary classification cutoff will be applied.

        Args:
            percentile: Percentile to use for threshold (e.g., 5 for bottom 5%)
            save_path: Where to save the figure
        """
        print(
            f"📈 VISUALIZING SCORE DISTRIBUTIONS (Threshold: Top {percentile:.2f}th percentile)")
        print("─" * 80)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        score_columns = ['half_space_score',
                         'iforest_score', 'lof_score', 'dbscan_score']
        algo_names = {
            'half_space_score': 'Half-Space Trees',
            'iforest_score': 'Isolation Forest',
            'lof_score': 'Local Outlier Factor',
            'dbscan_score': 'DBSCAN'
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, col in enumerate(score_columns):
            scores = self.df[col].values
            threshold = np.percentile(scores, 100-percentile)

            axes[idx].hist(scores, bins=100, alpha=0.7,
                           color='steelblue', edgecolor='black')
            axes[idx].axvline(threshold, color='red', linestyle='--', linewidth=2,
                              label=f'{(100-percentile):.1f}th percentile')

            axes[idx].set_title(
                algo_names[col], fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Anomaly Score', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)

            stats_text = f"Mean: {scores.mean():.4f}\nStd: {scores.std():.4f}"
            axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Anomaly Score Distributions\n(Adelaide Metro - {len(self.df):,} samples)',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Saved histogram to: {save_path}")
        print("─" * 80 + "\n")

    def apply_thresholds_and_classify_anomalies(self, pct_threshold):
        """
        Apply percentile-based thresholds to convert continuous scores to binary classifications.

        Uses the ground truth anomaly rate as the percentile threshold, following the approach
        recommended in SAS anomaly detection documentation. This ensures all algorithms flag
        the same proportion of data for fair comparison.

        Args:
            pct_threshold: Percentile threshold matching expected anomaly rate

        Returns:
            DataFrame with binary anomaly flags for each algorithm
        """
        print(f"🎯 APPLYING THRESHOLDS ({pct_threshold:.2f}th percentile)")
        print("─" * 80)

        self.anomaly_df = pd.DataFrame()
        self.anomaly_df['ground_truth'] = self.df['rule_anomaly'].astype(int)

        # Isolation Forest & LOF: more negative = more anomalous
        iforest_threshold = np.percentile(
            self.df['iforest_score'], 100 - pct_threshold)
        lof_threshold = np.percentile(self.df['lof_score'], 100 - pct_threshold)

        # Half-Space & DBSCAN: higher values = more anomalous
        half_space_threshold = np.percentile(
            self.df['half_space_score'], 100- pct_threshold)
        dbscan_threshold = np.percentile(
            self.df['dbscan_score'], 100 - pct_threshold)

        self.anomaly_df['iforest'] = (
            self.df['iforest_score'] >= iforest_threshold).astype(int)
        self.anomaly_df['lof'] = (
            self.df['lof_score'] >= lof_threshold).astype(int)
        self.anomaly_df['half_space'] = (
            self.df['half_space_score'] >= half_space_threshold).astype(int)
        self.anomaly_df['dbscan'] = (
            self.df['dbscan_score'] >= dbscan_threshold).astype(int)

        print("Threshold values:")
        print(f"  • Isolation Forest: ≥ {iforest_threshold:.6f}")
        print(f"  • LOF: ≥ {lof_threshold:.6f}")
        print(f"  • Half-Space Trees: ≥ {half_space_threshold:.6f}")
        print(f"  • DBSCAN: ≥ {dbscan_threshold:.6f}")

        print("\nDetected anomalies per algorithm:")
        for algo in ['iforest', 'lof', 'half_space', 'dbscan']:
            count = self.anomaly_df[algo].sum()
            pct = (count / len(self.anomaly_df)) * 100
            print(
                f"  • {algo.replace('_', ' ').title()}: {count:,} ({pct:.2f}%)")

        print("─" * 80 + "\n")

        return self.anomaly_df

    def compute_anomaly_focused_metrics(self, anomaly_df):
        """
        Compute anomaly-focused performance metrics comparing each algorithm to ground truth.

        Due to severe class imbalance (94% normal, 6% anomalies), accuracy is meaningless
        and is excluded. Instead, focuses on:
        - Recall (Detection Rate): What % of true anomalies did we catch?
        - Precision (Flag Accuracy): What % of our flags were correct?
        - F1-Score: Harmonic mean balancing precision and recall
        - False Negative Rate: What % of anomalies did we miss?
        - False Positive Rate: What % of normal data was falsely flagged?

        Args:
            anomaly_df: DataFrame with binary anomaly classifications

        Returns:
            DataFrame with performance metrics for each algorithm
        """
        print("📈 ANOMALY-FOCUSED PERFORMANCE METRICS")
        print("=" * 80)
        print("⚠️  Note: Accuracy omitted due to severe class imbalance (93.54% normal)")
        print("    Focus on Recall (detection rate) and Precision (flag accuracy)")
        print("=" * 80)

        results = []
        y_true = anomaly_df['ground_truth']
        total_true_anomalies = y_true.sum()
        total_normal = len(y_true) - total_true_anomalies

        for algo in self.algorithms:
            y_pred = anomaly_df[algo]

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            results.append({
                'Algorithm': algo.replace('_', ' ').title(),
                'Recall': recall,
                'Precision': precision,
                'F1-Score': f1,
                'FNR': fnr,
                'FPR': fpr,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            })

            print(f"\n{algo.replace('_', ' ').upper()}")
            print("─" * 80)
            print(
                f"  🎯 Recall (Detection Rate):     {recall:.4f} ({recall*100:.2f}% of anomalies caught)")
            print(
                f"  ✓  Precision (Flag Accuracy):   {precision:.4f} ({precision*100:.2f}% of flags correct)")
            print(f"  📊 F1-Score:                    {f1:.4f}")
            print(
                f"  ❌ False Negative Rate (Miss):  {fnr:.4f} ({fn:,}/{total_true_anomalies:,} anomalies missed)")
            print(
                f"  ⚠️  False Positive Rate:         {fpr:.4f} ({fp:,}/{total_normal:,} false alarms)")
            print(f"\n  Confusion Matrix:")
            print(f"    TP: {tp:,}  |  FP: {fp:,}")
            print(f"    FN: {fn:,}  |  TN: {tn:,}")

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80 + "\n")

        results_df.to_csv('evaluation/performance_metrics.csv', index=False)
        print("✓ Saved metrics to: evaluation/performance_metrics.csv\n")

        self._plot_confusion_matrices(anomaly_df)
        self._plot_performance_comparison(results_df)

        return results_df

    def _plot_confusion_matrices(self, anomaly_df):
        """
        Create heatmap visualizations of confusion matrices for all algorithms.

        Args:
            anomaly_df: DataFrame with binary anomaly classifications
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        y_true = anomaly_df['ground_truth']
        algo_names = {
            'half_space': 'Half-Space Trees',
            'iforest': 'Isolation Forest',
            'lof': 'Local Outlier Factor',
            'dbscan': 'DBSCAN'
        }

        for idx, algo in enumerate(self.algorithms):
            y_pred = anomaly_df[algo]
            cm = confusion_matrix(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Anomaly'],
                        yticklabels=['Normal', 'Anomaly'],
                        ax=axes[idx], cbar_kws={'label': 'Count'})

            axes[idx].set_title(f'{algo_names[algo]}\nConfusion Matrix',
                                fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Ground Truth', fontsize=11)
            axes[idx].set_xlabel('Predicted', fontsize=11)

        plt.suptitle('Confusion Matrices - All Algorithms vs Ground Truth',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('evaluation/confusion_matrices.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        print("✓ Saved confusion matrices to: evaluation/confusion_matrices.png\n")

    def _plot_performance_comparison(self, results_df):
        """
        Create bar chart comparing key metrics across algorithms.

        Args:
            results_df: DataFrame with performance metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['Recall', 'Precision', 'F1-Score', 'FNR']
        titles = ['Recall (Detection Rate)', 'Precision (Flag Accuracy)',
                  'F1-Score', 'False Negative Rate (Miss Rate)']
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx // 2, idx % 2]

            bars = ax.bar(results_df['Algorithm'],
                          results_df[metric], color=color, alpha=0.7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

        plt.suptitle('Algorithm Performance Comparison',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('evaluation/performance_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        print("✓ Saved performance comparison to: evaluation/performance_comparison.png\n")

    def analyze_by_anomaly_type(self, anomaly_df):
        """
        Analyze algorithm performance separately for each type of anomaly.

        This is critical because algorithms may be better at detecting certain
        anomaly types (e.g., STATIONARY_TOO_LONG vs SPEED_ANOMALY).

        Args:
            anomaly_df: DataFrame with binary anomaly classifications
        """
        print("📊 PERFORMANCE BY ANOMALY TYPE")
        print("=" * 80)

        if 'rule_anomaly_types' not in self.df.columns:
            print("⚠️  Anomaly type information not available")
            print("─" * 80 + "\n")
            return

        anomaly_types = self.df[self.df['rule_anomaly']
                                == 1]['rule_anomaly_types'].unique()
        type_results = []

        for anom_type in anomaly_types:
            if pd.isna(anom_type):
                continue

            mask = self.df['rule_anomaly_types'] == anom_type
            subset_anomaly_df = anomaly_df[mask]

            total_of_type = mask.sum()
            true_anomalies_of_type = subset_anomaly_df['ground_truth'].sum()

            print(f"\n{anom_type}")
            print("─" * 80)
            print(f"Total samples of this type: {total_of_type:,}")
            print(f"True anomalies: {true_anomalies_of_type:,}\n")

            for algo in self.algorithms:
                y_true = subset_anomaly_df['ground_truth']
                y_pred = subset_anomaly_df[algo]

                recall = recall_score(y_true, y_pred, zero_division=0)
                precision = precision_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                type_results.append({
                    'Anomaly Type': anom_type,
                    'Algorithm': algo.replace('_', ' ').title(),
                    'Recall': recall,
                    'Precision': precision,
                    'F1-Score': f1
                })

                print(
                    f"  {algo.replace('_', ' ').title():20s}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")

        type_results_df = pd.DataFrame(type_results)
        type_results_df.to_csv(
            'evaluation/performance_by_anomaly_type.csv', index=False)

        print("\n" + "=" * 80)
        print(
            "✓ Saved type-specific metrics to: evaluation/performance_by_anomaly_type.csv")
        print("=" * 80 + "\n")

        return type_results_df

    def analyze_disagreements(self, anomaly_df):
        """
        Analyze cases where algorithms disagree with ground truth and with each other.

        This helps identify:
        - Potential false positives that might actually be real anomalies
        - Systematic blind spots where all algorithms miss anomalies
        - Algorithm consensus patterns

        Args:
            anomaly_df: DataFrame with binary anomaly classifications
        """
        print("🔍 DISAGREEMENT ANALYSIS")
        print("=" * 80)

        algo_cols = ['iforest', 'lof', 'half_space', 'dbscan']
        anomaly_df['algo_consensus'] = anomaly_df[algo_cols].sum(axis=1)

        print("\nALGORITHM CONSENSUS DISTRIBUTION")
        print("─" * 80)
        for i in range(5):
            count = (anomaly_df['algo_consensus'] == i).sum()
            pct = (count / len(anomaly_df)) * 100
            print(f"  {i}/4 algorithms agree: {count:,} samples ({pct:.2f}%)")

        print("\n\nUNANIMOUS ALGORITHM DISAGREEMENTS WITH GROUND TRUTH")
        print("─" * 80)

        unanimous_anomaly = (anomaly_df['algo_consensus'] == 4) & (
            anomaly_df['ground_truth'] == 0)
        unanimous_anomaly_count = unanimous_anomaly.sum()

        unanimous_normal = (anomaly_df['algo_consensus'] == 0) & (
            anomaly_df['ground_truth'] == 1)
        unanimous_normal_count = unanimous_normal.sum()

        print(
            f"  All algorithms flag anomaly, ground truth says normal: {unanimous_anomaly_count:,}")
        print(f"    → Potential false positives OR real anomalies missed by rules")
        print(
            f"\n  All algorithms say normal, ground truth flags anomaly: {unanimous_normal_count:,}")
        print(f"    → Systematic blind spot - all algorithms missed these")

        print("\n\nMAJORITY VOTE (≥3/4 algorithms) VS GROUND TRUTH")
        print("─" * 80)

        majority_anomaly = (anomaly_df['algo_consensus'] >= 3) & (
            anomaly_df['ground_truth'] == 0)
        majority_normal = (anomaly_df['algo_consensus'] <= 1) & (
            anomaly_df['ground_truth'] == 1)

        print(
            f"  Majority says anomaly, ground truth disagrees: {majority_anomaly.sum():,}")
        print(
            f"  Majority says normal, ground truth disagrees: {majority_normal.sum():,}")

        print("\n\nPAIRWISE ALGORITHM AGREEMENT (on true anomalies only)")
        print("─" * 80)
        print("Shows what % of ground truth anomalies were detected by BOTH algorithms")

        true_anomalies = anomaly_df[anomaly_df['ground_truth'] == 1]
        total_true_anomalies = len(true_anomalies)

        agreement_matrix = np.zeros((4, 4))
        for i, algo1 in enumerate(algo_cols):
            for j, algo2 in enumerate(algo_cols):
                if i <= j:
                    agreement = ((true_anomalies[algo1] == 1) & (
                        true_anomalies[algo2] == 1)).sum()
                    agreement_pct = (agreement / total_true_anomalies) * 100
                    agreement_matrix[i, j] = agreement_pct
                    agreement_matrix[j, i] = agreement_pct

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                    xticklabels=[a.replace('_', ' ').title()
                                 for a in algo_cols],
                    yticklabels=[a.replace('_', ' ').title()
                                 for a in algo_cols],
                    ax=ax, cbar_kws={'label': '% of True Anomalies Detected by Both'})
        ax.set_title('Algorithm Agreement on True Anomalies\n(% of ground truth anomalies detected by both algorithms)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('evaluation/algorithm_agreement.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        print("\n✓ Saved agreement heatmap to: evaluation/algorithm_agreement.png")

        interesting_samples = self.df[unanimous_anomaly |
                                      unanimous_normal].copy()
        interesting_samples['disagreement_type'] = ''
        interesting_samples.loc[unanimous_anomaly,
                                'disagreement_type'] = 'All_algos_flag_GT_normal'
        interesting_samples.loc[unanimous_normal,
                                'disagreement_type'] = 'All_algos_normal_GT_flag'

        interesting_samples.to_csv(
            'evaluation/unanimous_disagreements.csv', index=False)
        print(
            f"✓ Saved {len(interesting_samples):,} unanimous disagreement samples to: evaluation/unanimous_disagreements.csv")
        print("─" * 80 + "\n")

    def report_algorithm_computation_time(self):
        """
        Analyze and report computational performance statistics for each algorithm.

        Real-time anomaly detection requires fast inference. This analysis shows
        which algorithms are practical for streaming data with 15-second updates.

        Reports mean, median, std deviation, min, and max computation times.
        """
        print("⏱️  ALGORITHM COMPUTATION TIME ANALYSIS")
        print("=" * 80)

        time_columns = {
            'half_space_compute_ms': 'Half-Space Trees',
            'iforest_compute_ms': 'Isolation Forest',
            'lof_compute_ms': 'Local Outlier Factor',
            'dbscan_compute_ms': 'DBSCAN'
        }

        time_stats = []

        for col, name in time_columns.items():
            times = self.df[col].dropna()

            stats = {
                'Algorithm': name,
                'Mean (ms)': times.mean(),
                'Median (ms)': times.median(),
                'Std Dev (ms)': times.std(),
                'Min (ms)': times.min(),
                'Max (ms)': times.max(),
                'Total Time (s)': times.sum() / 1000
            }
            time_stats.append(stats)

            print(f"\n{name.upper()}")
            print("─" * 80)
            print(f"  Mean:   {stats['Mean (ms)']:.4f} ms")
            print(f"  Median: {stats['Median (ms)']:.4f} ms")
            print(f"  Std:    {stats['Std Dev (ms)']:.4f} ms")
            print(
                f"  Range:  [{stats['Min (ms)']:.4f}, {stats['Max (ms)']:.4f}] ms")
            print(f"  Total:  {stats['Total Time (s)']:.2f} seconds")

        time_df = pd.DataFrame(time_stats)

        print("\n" + "=" * 80)
        print("COMPUTATION TIME SUMMARY")
        print("=" * 80)
        print(time_df.to_string(index=False))
        print("=" * 80 + "\n")

        time_df.to_csv('evaluation/computation_times.csv', index=False)
        print("✓ Saved computation times to: evaluation/computation_times.csv\n")

        fig, ax = plt.subplots(figsize=(12, 6))

        time_data = [self.df[col].dropna() for col in time_columns.keys()]
        labels = [name for name in time_columns.values()]

        bp = ax.boxplot(time_data, labels=labels, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_ylabel('Computation Time (ms)', fontsize=12)
        ax.set_title('Algorithm Computation Time Distribution',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig('evaluation/computation_time_boxplot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(
            "✓ Saved computation time boxplot to: evaluation/computation_time_boxplot.png")
        print("─" * 80 + "\n")

        return time_df


def main():
    """
    Main execution pipeline for Adelaide Metro anomaly detection evaluation.

    Loads real-time streaming data from CSV files, applies percentile-based thresholds
    to convert continuous anomaly scores to binary classifications, and evaluates
    performance using anomaly-focused metrics.
    """
    OUTPUT_DIR = "output"

    print("\n" + "=" * 80)
    print("🚊 ADELAIDE METRO ANOMALY DETECTION EVALUATION")
    print("=" * 80)
    print("\nComparing unsupervised algorithms against rule-based ground truth")
    print("Using percentile-based thresholds for binary classification")
    print("Focusing on anomaly-centric metrics due to class imbalance")
    print("=" * 80 + "\n")

    evaluator = AnomalyDetectionEvaluator(OUTPUT_DIR)

    evaluator.load_data()

    evaluator.clean_dataframe()

    pct_anomaly = evaluator.report_on_ground_truth()

    evaluator.visualize_anomaly_score_histograms(percentile=pct_anomaly)

    anomaly_df = evaluator.apply_thresholds_and_classify_anomalies(
        pct_threshold=pct_anomaly)

    evaluator.compute_anomaly_focused_metrics(anomaly_df)

    evaluator.analyze_by_anomaly_type(anomaly_df)

    evaluator.analyze_disagreements(anomaly_df)

    evaluator.report_algorithm_computation_time()

    print("💾 SAVING FINAL RESULTS")
    print("─" * 80)
    combined_output = evaluator.df.copy()
    for algo in ['iforest', 'lof', 'half_space', 'dbscan']:
        combined_output[f'{algo}_anomaly_flag'] = anomaly_df[algo]

    combined_output.to_csv(
        'evaluation/full_results_with_flags.csv', index=False)
    print(f"✓ Saved complete dataset with anomaly flags to: evaluation/full_results_with_flags.csv")
    print(f"  ({len(combined_output):,} samples)")
    print("─" * 80 + "\n")

    print("=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print("\n📁 Generated files in evaluation/:")
    print("  • performance_metrics.csv - Recall, precision, F1 scores (NO accuracy)")
    print("  • performance_by_anomaly_type.csv - Metrics split by anomaly type")
    print("  • confusion_matrices.png - Visual confusion matrices")
    print("  • performance_comparison.png - Bar charts comparing algorithms")
    print("  • algorithm_agreement.png - Pairwise agreement heatmap")
    print("  • unanimous_disagreements.csv - Samples where all algorithms disagree with GT")
    print("  • computation_times.csv - Algorithm speed statistics")
    print("  • computation_time_boxplot.png - Computation time visualization")
    print("  • anomaly_score_histograms.png - Score distributions with thresholds")
    print("  • full_results_with_flags.csv - Complete dataset with binary flags")
    print("\n📊 KEY METRICS TO REVIEW:")
    print("  1. RECALL - Most important: Are we catching the anomalies?")
    print("  2. PRECISION - Are our flags reliable or too many false alarms?")
    print("  3. F1-SCORE - Balanced view of overall performance")
    print("  4. PERFORMANCE BY TYPE - Which anomaly types are harder to detect?")
    print("\n💡 Remember: Accuracy is meaningless with 94% normal data!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
