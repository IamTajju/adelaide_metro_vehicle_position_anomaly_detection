# dbscan_detector_with_speed.py (CORRECTED)

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
from scipy.spatial.distance import cdist

from constants import (
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_WINDOW_SIZE,
    DBSCAN_UPDATE_INTERVAL,
)


class DBSCANAnomalyDetector(MapFunction):
    """
    Detects anomalies by identifying points labeled as 'noise' by the DBSCAN 
    clustering algorithm using lat, lon, and speed features.
    """

    def __init__(self):
        # Store recent (lat, lon, speed) tuples for clustering
        self.history_data: List[List[float]] = []
        self.window_size = DBSCAN_WINDOW_SIZE
        self.update_interval = DBSCAN_UPDATE_INTERVAL
        self.sample_count = 0
        self.last_cluster_count = 0

        # DBSCAN parameters
        self.eps = DBSCAN_EPS
        self.min_samples = DBSCAN_MIN_SAMPLES

        # Cluster labels from last clustering run
        self.current_labels = None
        self.ready = False

        # StandardScaler for feature normalization
        # Need to refit periodically for concept drift
        self.scaler = StandardScaler()
        self.scaler_update_interval = DBSCAN_UPDATE_INTERVAL
        self.last_scaler_update = 0

    def _update_scaler(self, data: np.ndarray):
        """Refit scaler to adapt to concept drift."""
        if len(data) >= self.min_samples:
            self.scaler.fit(data)
            self.last_scaler_update = self.sample_count

    def _cluster_data(self, data: np.ndarray) -> np.ndarray:
        """
        Runs DBSCAN on standardized 3D features (lat, lon, speed).
        Returns cluster labels for all points in the window.
        """
        if len(data) < self.min_samples:
            return np.array([-1] * len(data))

        # Standardize features
        data_scaled = self.scaler.transform(data)

        # Run DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(data_scaled)
        return labels

    def _get_anomaly_score(self, label: int, all_labels: np.ndarray) -> float:
        """
        Convert DBSCAN label to continuous anomaly score for fair comparison.

        Score interpretation:
        - label == -1 (noise): High anomaly score (1.0)
        - label in small cluster: Medium anomaly score (0.5 to 0.9)
        - label in large cluster: Low anomaly score (0.0 to 0.5)

        This makes DBSCAN scores comparable to IForest and HST.
        """
        if label == -1:
            # Noise point = definite anomaly
            return 1.0

        # Calculate cluster size
        cluster_size = np.sum(all_labels == label)
        total_points = len(all_labels)

        # Smaller clusters = more anomalous
        # Normalize: [0, total_points] -> [0.9, 0.0]
        normalized_size = cluster_size / total_points
        anomaly_score = max(0.0, 0.9 - (normalized_size * 0.9))

        return float(anomaly_score)

    def map(self, json_str: str) -> str:
        try:
            vehicle: Dict[str, Any] = json.loads(json_str)
            lat = vehicle.get('latitude')
            lon = vehicle.get('longitude')
            speed_kmh = vehicle.get('speed_kmh')

            # Check valid data
            if lat is None or lon is None or speed_kmh is None:
                vehicle['dbscan_score'] = None
                vehicle['dbscan_compute_ms'] = None
                return json.dumps(vehicle)

            # Feature vector
            current_point = [float(lat), float(lon), float(speed_kmh)]

            self.sample_count += 1

            # -----------------------------
            # 1. Warm-up Phase
            # -----------------------------
            if not self.ready:
                # Collect initial data
                self.history_data.append(current_point)

                # Maintain window size
                if len(self.history_data) > self.window_size:
                    self.history_data.pop(0)

                # Initialize when we have enough data
                if len(self.history_data) >= self.window_size:
                    # Fit scaler on initial window
                    coords = np.array(self.history_data)
                    self._update_scaler(coords)

                    # Run initial clustering
                    self.current_labels = self._cluster_data(coords)
                    self.ready = True
                    self.last_cluster_count = self.sample_count

                # Return None during warm-up
                vehicle['dbscan_score'] = None
                vehicle['dbscan_compute_ms'] = None
                return json.dumps(vehicle)

            # -----------------------------
            # 2. Scoring Phase (model ready)
            # -----------------------------

            start_compute = time.perf_counter()

            # Transform current point using existing scaler
            current_scaled = self.scaler.transform(
                np.array([current_point])
            )

            # Find which cluster this point would belong to
            # Use existing cluster labels to determine score
            if self.current_labels is not None and len(self.current_labels) > 0:
                # Calculate distance to all points in history
                history_scaled = self.scaler.transform(
                    np.array(self.history_data)
                )

                # Find nearest neighbors within eps radius
                distances = cdist(
                    current_scaled, history_scaled, metric='euclidean')[0]
                neighbors = np.sum(distances <= self.eps)

                if neighbors < self.min_samples:
                    # Not enough neighbors = noise = anomaly
                    predicted_label = -1
                else:
                    # Assign to most common label among neighbors
                    neighbor_indices = np.where(distances <= self.eps)[0]
                    neighbor_labels = self.current_labels[neighbor_indices]
                    # Filter out noise labels
                    valid_labels = neighbor_labels[neighbor_labels != -1]
                    if len(valid_labels) > 0:
                        predicted_label = np.bincount(valid_labels).argmax()
                    else:
                        predicted_label = -1

                # Convert label to continuous score
                anomaly_score = self._get_anomaly_score(
                    predicted_label,
                    self.current_labels
                )
            else:
                anomaly_score = None

            end_compute = time.perf_counter()
            compute_ms = (end_compute - start_compute) * 1000.0

            # Add current point to history (after scoring)
            self.history_data.append(current_point)

            # Maintain window size
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)

            # Periodic reclustering (after scoring)
            samples_since_cluster = self.sample_count - self.last_cluster_count
            if samples_since_cluster >= self.update_interval:
                coords = np.array(self.history_data)

                # Update scaler for concept drift
                self._update_scaler(coords)

                # Recluster
                self.current_labels = self._cluster_data(coords)
                self.last_cluster_count = self.sample_count

            vehicle['dbscan_score'] = anomaly_score
            vehicle['dbscan_compute_ms'] = compute_ms

            return json.dumps(vehicle)

        except Exception as e:
            # Fallback behavior
            vehicle = json.loads(json_str)
            vehicle['dbscan_score'] = None
            vehicle['dbscan_compute_ms'] = None
            print(f"Error in DBSCANAnomalyDetector: {e}")
            return json.dumps(vehicle)
