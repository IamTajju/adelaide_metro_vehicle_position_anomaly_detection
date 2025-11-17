# lof_detector.py (Updated to use pre-enriched speed_kmh)

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
import time
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Dict, Any
import warnings

# Suppress early-sample warnings
warnings.filterwarnings(
    "ignore",
    message="n_neighbors (\\d+) is greater than the total number of samples (\\d+). n_neighbors will be set to (n_samples - 1) for estimation.",
    category=UserWarning
)


class LOFAnomalyDetector(MapFunction):
    """
    Detects anomalies using the Local Outlier Factor (LOF) algorithm
    using (latitude, longitude, speed_kmh).
    """

    def __init__(self):
        # Store recent history: [lat, lon, timestamp, speed_kmh]
        self.history_data: List[List[float]] = []

        self.window_size: int = 300
        self.counter: int = 0
        self.update_interval: int = 20

        # LOF Hyperparameters
        self.n_neighbors: int = 20
        self.min_stable_samples: int = max(self.n_neighbors + 1, 50)

    def _predict_lof(self, current_point: np.ndarray, data: np.ndarray) -> float:
        """Runs LOF on (lat, lon, speed)."""
        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=False)
        lof.fit(data)

        # LOF assigns scores to the TRAINING set, so last element = current point
        score = -lof.negative_outlier_factor_[-1]
        return score

    def map(self, json_str: str) -> str:
        start_total = time.perf_counter()
        lof_compute_ms = 0.0

        try:
            vehicle: Dict[str, Any] = json.loads(json_str)

            lat = vehicle.get('latitude')
            lon = vehicle.get('longitude')
            timestamp = vehicle.get('timestamp', 0)
            speed_kmh = vehicle.get('speed_kmh', 0.0)  # <-- use enriched speed

            # --- Initial State Handling ---
            if lat is None or lon is None:
                vehicle['lof_score'] = 1.0
                vehicle['lof_compute_ms'] = 0.0
                return json.dumps(vehicle)

            # If no previous data, store this and skip LOF
            if len(self.history_data) == 0:
                self.history_data.append([lat, lon, timestamp, speed_kmh])
                vehicle['lof_score'] = 1.0
                vehicle['lof_compute_ms'] = 0.0
                return json.dumps(vehicle)

            # --- Store new record ---
            current_record = [float(lat), float(lon), float(timestamp), float(speed_kmh)]
            self.history_data.append(current_record)

            # Maintain window size
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)

            self.counter += 1
            anomaly_score = 1.0  # Default neutral score

            # --- Conditional LOF Computation ---
            if (
                self.counter % self.update_interval == 0 and 
                len(self.history_data) >= self.min_stable_samples
            ):
                data = np.array([[p[0], p[1], p[3]] for p in self.history_data])

                start_compute = time.perf_counter()
                anomaly_score = self._predict_lof(data[-1], data)
                lof_compute_ms = (time.perf_counter() - start_compute) * 1000.0

            # Save results
            vehicle['lof_score'] = float(anomaly_score)
            vehicle['lof_compute_ms'] = float(lof_compute_ms)
            vehicle['lof_total_map_ms'] = (
                time.perf_counter() - start_total
            ) * 1000.0

            return json.dumps(vehicle)

        except Exception:
            # Return safe fallback
            try:
                vehicle = json.loads(json_str)
                vehicle['lof_score'] = 1.0
                vehicle['lof_compute_ms'] = 0.0
                return json.dumps(vehicle)
            except Exception:
                return json_str
