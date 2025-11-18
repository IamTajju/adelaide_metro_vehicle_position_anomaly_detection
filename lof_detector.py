# lof_detector.py

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
import time
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Dict, Any
import warnings

from constants import (
    LOF_MIN_SAMPLES,
    LOF_N_NEIGHBORS,
    LOF_WINDOW_SIZE,
    LOF_UPDATE_INTERVAL,
    RANDOM_STATE,
)

# Suppress early-sample warnings
warnings.filterwarnings(
    "ignore",
    message="n_neighbors.*is greater than the total number of samples",
    category=UserWarning
)


class LOFAnomalyDetector(MapFunction):
    """
    Detects anomalies using the Local Outlier Factor (LOF) algorithm
    using (latitude, longitude, speed_kmh).

    """

    def __init__(self):
        # Store recent history: [lat, lon, speed_kmh]
        self.history_data: List[List[float]] = []

        self.window_size: int = LOF_WINDOW_SIZE
        self.update_interval: int = LOF_UPDATE_INTERVAL
        self.sample_count: int = 0
        self.last_update_count: int = 0

        # LOF Hyperparameters
        self.n_neighbors: int = LOF_N_NEIGHBORS
        self.min_samples: int = LOF_MIN_SAMPLES

        # LOF model (trained on history, predicts on new points)
        self.lof_model: LocalOutlierFactor = None
        self.ready: bool = False

    def _train_lof(self, data: np.ndarray):
        """
        Train LOF model on historical data using novelty=True.
        This allows scoring of new points without train-test leakage.
        """
        if len(data) < self.n_neighbors:
            return

        # novelty=True allows us to use decision_function on new points
        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,  # Critical: enables prediction on new data
            contamination='auto'
        )
        self.lof_model.fit(data)

    def _normalize_lof_score(self, lof_score: float) -> float:
        """
        Normalize LOF score to [0, ~1] range for fair comparison.

        LOF interpretation:
        - score < 0: Inlier (denser than neighbors)
        - score ≈ 0: Similar density to neighbors  
        - score > 0: Outlier (less dense than neighbors)

        We transform to: higher values = more anomalous
        """
        # Shift and scale: map [-inf, +inf] to [0, 1+]
        # Use sigmoid-like transformation
        if lof_score <= 0:
            return 0.0  # Clear inlier
        else:
            # Map positive scores to [0.5, 1.0]
            # Using: 1 - 1/(1 + score)
            return min(1.0, 1.0 - 1.0/(1.0 + lof_score))

    def map(self, json_str: str) -> str:
        try:
            vehicle: Dict[str, Any] = json.loads(json_str)

            lat = vehicle.get('latitude')
            lon = vehicle.get('longitude')
            speed_kmh = vehicle.get('speed_kmh', 0.0)

            # Check valid data
            if lat is None or lon is None:
                vehicle['lof_score'] = None
                vehicle['lof_compute_ms'] = None
                return json.dumps(vehicle)

            # Current point features
            current_point = np.array(
                [[float(lat), float(lon), float(speed_kmh)]])

            self.sample_count += 1

            # -----------------------------
            # 1. Warm-up Phase
            # -----------------------------
            if not self.ready:
                # Collect initial data
                self.history_data.append(
                    [float(lat), float(lon), float(speed_kmh)])

                # Maintain window size during warm-up
                if len(self.history_data) > self.window_size:
                    self.history_data.pop(0)

                # Train initial model when we have enough data
                if len(self.history_data) >= self.min_samples:
                    data = np.array(self.history_data)
                    self._train_lof(data)
                    self.ready = True
                    self.last_update_count = self.sample_count

                # Return None during warm-up
                vehicle['lof_score'] = None
                vehicle['lof_compute_ms'] = None
                vehicle['lof_warmup'] = True
                return json.dumps(vehicle)

            # -----------------------------
            # 2. Scoring Phase (model ready)
            # -----------------------------

            start_compute = time.perf_counter()

            # Use decision_function to score new point
            # Negative values = more anomalous (we'll flip this)
            raw_score = self.lof_model.decision_function(current_point)[0]

            # Flip sign: positive = anomalous, negative = normal
            lof_score = -raw_score

            # Normalize for fair comparison
            anomaly_score = self._normalize_lof_score(lof_score)

            end_compute = time.perf_counter()
            compute_ms = (end_compute - start_compute) * 1000.0

            # NOW add current point to history (after scoring)
            self.history_data.append(
                [float(lat), float(lon), float(speed_kmh)])

            # Maintain window size
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)

            # Periodic retraining (after scoring)
            samples_since_update = self.sample_count - self.last_update_count
            if samples_since_update >= self.update_interval:
                data = np.array(self.history_data)
                self._train_lof(data)
                self.last_update_count = self.sample_count

            vehicle['lof_score'] = float(anomaly_score)
            vehicle['lof_compute_ms'] = compute_ms

            return json.dumps(vehicle)

        except Exception as e:
            # Fallback behavior
            vehicle = json.loads(json_str)
            vehicle['lof_score'] = None
            vehicle['lof_compute_ms'] = None
            print(f"Error in LOFAnomalyDetector: {e}")
            return json.dumps(vehicle)
