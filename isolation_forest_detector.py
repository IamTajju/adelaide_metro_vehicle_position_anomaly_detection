# isolation_forest_detector_sklearn.py (Updated with Sliding Window & Periodic Retraining)

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
from typing import Dict, Any, List
import time

from sklearn.ensemble import IsolationForest

from constants import (
    IFOREST_N_ESTIMATORS,
    IFOREST_MAX_SAMPLES,
    IFOREST_WINDOW_SIZE,
    IFOREST_RETRAIN_INTERVAL,
    RANDOM_STATE

)


class IsolationForestAnomalyDetector(MapFunction):
    """
    Isolation Forest anomaly detector with sliding window and periodic retraining.
    Uses: (latitude, longitude, speed_kmh)

    - Maintains a sliding window of recent samples
    - Periodically retrains the model to adapt to concept drift
    - Configurable retraining frequency
    """

    def __init__(self, window_size=IFOREST_WINDOW_SIZE, retrain_interval=IFOREST_RETRAIN_INTERVAL):
        """
        Args:
            window_size: Size of sliding window for training data (default: 500)
            retrain_interval: Number of samples between retraining (default: 100)
        """
        # Sliding window to maintain recent samples
        self.sliding_window: List[List[float]] = []
        self.window_size = window_size

        # Initial buffer for first model training
        self.init_buffer: List[List[float]] = []
        self.buffer_limit = IFOREST_WINDOW_SIZE

        # Retraining configuration
        self.retrain_interval = retrain_interval
        self.sample_count = 0
        self.last_retrain_count = 0

        # Model state
        self.iforest: IsolationForest = None
        self.ready: bool = False

        # Model hyperparameters
        self.n_estimators = IFOREST_N_ESTIMATORS
        self.random_state = RANDOM_STATE

        # Performance tracking
        self.total_retrains = 0

    # -----------------------
    # Train/Retrain Model
    # -----------------------
    def _train_model(self, training_data: List[List[float]]):
        """Train or retrain the Isolation Forest model."""
        if len(training_data) == 0:
            return

        X_train = np.array(training_data)

        # Use min of window_size and actual data size for max_samples
        max_samples_val = min(len(training_data), IFOREST_MAX_SAMPLES)

        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=max_samples_val,
            contamination="auto",
            random_state=self.random_state,
        )
        self.iforest.fit(X_train)
        self.ready = True
        self.total_retrains += 1

    def _add_to_sliding_window(self, sample: List[float]):
        """Add sample to sliding window, removing oldest if at capacity."""
        self.sliding_window.append(sample)

        # Remove oldest sample if window exceeds size
        if len(self.sliding_window) > self.window_size:
            self.sliding_window.pop(0)

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        # Check if enough new samples have arrived since last retrain
        samples_since_retrain = self.sample_count - self.last_retrain_count

        # Retrain if:
        # 1. Enough samples have accumulated
        # 2. Window has sufficient data
        return (
            samples_since_retrain >= self.retrain_interval and
            len(self.sliding_window) >= self.buffer_limit
        )

    # -----------------------
    # Main Map Function
    # -----------------------
    def map(self, json_str: str) -> str:
        try:
            vehicle: Dict[str, Any] = json.loads(json_str)
            lat = vehicle.get("latitude")
            lon = vehicle.get("longitude")
            speed_kmh = vehicle.get("speed_kmh")

            # Check valid coordinates
            if lat is None or lon is None:
                vehicle["iforest_score"] = None
                vehicle["iforest_compute_ms"] = None
                return json.dumps(vehicle)

            # Feature vector
            x_raw = np.array(
                [float(lat), float(lon), float(speed_kmh)],
                dtype=float
            )

            self.sample_count += 1

            # -----------------------------
            # 1. Initial Training Phase
            # -----------------------------
            if not self.ready:
                # Fill initial buffer
                if len(self.init_buffer) < self.buffer_limit:
                    self.init_buffer.append(x_raw.tolist())
                    # Also add to sliding window for future use
                    self._add_to_sliding_window(x_raw.tolist())

                # Train initial model when buffer full
                if len(self.init_buffer) >= self.buffer_limit and not self.ready:
                    self._train_model(self.init_buffer)
                    self.last_retrain_count = self.sample_count

                vehicle["iforest_score"] = None
                vehicle["iforest_compute_ms"] = None
                return json.dumps(vehicle)

            # -----------------------------
            # 2. Scoring Phase (model ready)
            # -----------------------------

            # Score current sample
            X_test = x_raw.reshape(1, -1)

            start_compute = time.perf_counter()

            # IsolationForest: higher decision_function = more normal.
            # We invert it so higher = more anomalous.
            anomaly_score = -self.iforest.decision_function(X_test)[0]

            end_compute = time.perf_counter()

            compute_ms = (end_compute - start_compute) * 1000.0

            vehicle["iforest_score"] = float(anomaly_score)
            vehicle["iforest_compute_ms"] = compute_ms

            # Add current sample to sliding window
            self._add_to_sliding_window(x_raw.tolist())

            # Check if retraining needed
            if self._should_retrain():
                self._train_model(self.sliding_window)
                self.last_retrain_count = self.sample_count

            return json.dumps(vehicle)

        except Exception as e:
            # Fallback behavior
            vehicle = json.loads(json_str)
            vehicle["iforest_score"] = None
            vehicle["iforest_compute_ms"] = None
            # Log exception for debugging
            print(f"Error in IsolationForestAnomalyDetector: {e}")
            return json.dumps(vehicle)
