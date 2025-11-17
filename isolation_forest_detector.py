# isolation_forest_detector_sklearn.py

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
from typing import Dict, Any, List
import time

from sklearn.ensemble import IsolationForest

from constants import (
    HST_NUM_TREES,
    HST_INIT_WINDOW,
    HST_DEFAULT_SCORE,
    HST_RANDOM_SEED,
)


class IsolationForestAnomalyDetector(MapFunction):
    """
    Isolation Forest anomaly detector using:
       (latitude, longitude, speed_kmh)
    """

    def __init__(self):
        # For initial model training phase
        self.init_buffer: List[List[float]] = []
        self.buffer_limit = HST_INIT_WINDOW

        self.iforest: IsolationForest = None
        self.ready: bool = False

        # model hyperparameters
        self.n_estimators = HST_NUM_TREES
        self.random_state = HST_RANDOM_SEED

    # -----------------------
    # Train when buffer full
    # -----------------------
    def _train_model_from_buffer(self):
        if len(self.init_buffer) == 0:
            return

        X_train = np.array(self.init_buffer)

        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.buffer_limit,
            contamination="auto",
            random_state=self.random_state,
        )
        self.iforest.fit(X_train)
        self.ready = True

    # -----------------------
    # Main Map Function
    # -----------------------
    def map(self, json_str: str) -> str:
        start_total = time.perf_counter()

        try:
            vehicle: Dict[str, Any] = json.loads(json_str)
            lat = vehicle.get("latitude")
            lon = vehicle.get("longitude")
            speed_kmh = vehicle.get("speed_kmh")

            # Check valid coordinates
            if lat is None or lon is None:
                vehicle["iforest_score"] = HST_DEFAULT_SCORE
                vehicle["iforest_compute_ms"] = 0.0
                return json.dumps(vehicle)

            # Feature vector uses provided speed
            x_raw = np.array(
                [float(lat), float(lon), float(speed_kmh)],
                dtype=float
            )

            # -----------------------------
            # 1. Initialization (buffering)
            # -----------------------------
            if not self.ready:
                # Fill the buffer
                if len(self.init_buffer) < self.buffer_limit:
                    self.init_buffer.append(x_raw.tolist())

                # Train when buffer full
                if len(self.init_buffer) >= self.buffer_limit and not self.ready:
                    self._train_model_from_buffer()

                compute_ms = (time.perf_counter() - start_total) * 1000.0
                vehicle["iforest_score"] = HST_DEFAULT_SCORE
                vehicle["iforest_compute_ms"] = compute_ms
                return json.dumps(vehicle)

            # -----------------------------
            # 2. Scoring (model ready)
            # -----------------------------
            X_test = x_raw.reshape(1, -1)

            start_compute = time.perf_counter()

            # IsolationForest: higher decision_function = more normal.
            # We invert it so higher = more anomalous.
            anomaly_score = -self.iforest.decision_function(X_test)[0]

            end_compute = time.perf_counter()

            compute_ms = (end_compute - start_compute) * 1000.0

            vehicle["iforest_score"] = float(anomaly_score)
            vehicle["iforest_compute_ms"] = compute_ms
            vehicle["iforest_total_map_ms"] = (
                time.perf_counter() - start_total
            ) * 1000.0

            return json.dumps(vehicle)

        except Exception:
            # Fallback behavior
            try:
                vehicle = json.loads(json_str)
                vehicle["iforest_score"] = HST_DEFAULT_SCORE
                vehicle["iforest_compute_ms"] = 0.0
                return json.dumps(vehicle)
            except Exception:
                return json_str
