# dbscan_detector_with_speed.py

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
from collections import deque

from utils import haversine_distance_km


class DBSCANAnomalyDetector(MapFunction):
    """
    Detects anomalies by identifying points labeled as 'noise' by the DBSCAN 
    clustering algorithm using lat, lon, and speed features.
    Includes timing for the clustering step.
    """

    def __init__(self):
        # Store recent (vehicle_id, lat, lon, speed, timestamp) tuples
        self.history_data: List[Tuple[str, float, float, float, float]] = []
        self.window_size = 500  # Number of points to cluster
        self.update_interval = 100  # Recalculate clusters every 100 points
        self.counter = 0

        # Track last points for speed calculation (per vehicle)
        self.vehicle_last_points: Dict[str, deque] = {}

        # DBSCAN parameters
        # Epsilon radius (adjusted for standardized 3D space)
        self.eps = 0.5
        self.min_samples = 5    # Minimum points for a core point

        # Initialize the label for the current point
        self.last_computed_label = 0

        # StandardScaler for feature normalization (critical for 3D clustering)
        self.scaler = StandardScaler()
        self.scaler_fitted = False

    def _calc_speed_kmh(self, vehicle_id: str, lat: float, lon: float, ts: float) -> float:
        """
        Calculate speed for a specific vehicle using its last known position.
        """
        if vehicle_id not in self.vehicle_last_points:
            self.vehicle_last_points[vehicle_id] = deque(maxlen=2)

        last_point = self.vehicle_last_points[vehicle_id]

        if len(last_point) < 1:
            return 0.0

        prev = last_point[-1]
        prev_lat, prev_lon, prev_ts = prev

        if prev_ts is None or ts is None or (ts == prev_ts):
            return 0.0

        dist_km = haversine_distance_km(prev_lat, prev_lon, lat, lon)
        time_s = abs(ts - prev_ts)

        if time_s == 0:
            return 0.0

        return (dist_km / time_s) * 3600.0

    def _cluster_and_label(self, data: np.ndarray) -> np.ndarray:
        """
        Runs DBSCAN on standardized 3D features (lat, lon, speed).
        Returns the cluster labels for all points in the window.
        """
        # Standardize features to ensure all dimensions contribute equally
        if not self.scaler_fitted:
            data_scaled = self.scaler.fit_transform(data)
            self.scaler_fitted = True
        else:
            data_scaled = self.scaler.transform(data)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(data_scaled)
        return labels

    def map(self, json_str: str) -> str:
        start_total = time.perf_counter()
        compute_ms = 0.0

        try:
            vehicle: Dict[str, Any] = json.loads(json_str)
            vehicle_id = vehicle.get('vehicle_id', 'unknown')
            lat = vehicle.get('latitude')
            lon = vehicle.get('longitude')
            ts = vehicle.get('timestamp', 0)

            # --- Pre-check ---
            if lat is None or lon is None:
                vehicle['dbscan_is_anomaly'] = False
                vehicle['dbscan_label'] = self.last_computed_label
                vehicle['dbscan_compute_ms'] = 0.0
                return json.dumps(vehicle)

            # --- Calculate Speed ---
            speed_kmh = self._calc_speed_kmh(vehicle_id, lat, lon, ts)

            # Update vehicle's last point for future speed calculations
            if vehicle_id not in self.vehicle_last_points:
                self.vehicle_last_points[vehicle_id] = deque(maxlen=2)
            self.vehicle_last_points[vehicle_id].append(
                [float(lat), float(lon), float(ts)])

            # --- Data Maintenance ---
            # Store the data point: (vehicle_id, lat, lon, speed, timestamp)
            self.history_data.append(
                (vehicle_id, float(lat), float(lon), float(speed_kmh), float(ts)))

            # Maintain window size
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)

            self.counter += 1
            is_anomaly = False
            current_label = self.last_computed_label

            # --- Clustering and Anomaly Check ---
            if self.counter % self.update_interval == 0 and len(self.history_data) >= self.min_samples:

                # Extract features: [lat, lon, speed]
                # Note: Using same order as HST for consistency
                coords = np.array([[item[1], item[2], item[3]]
                                  for item in self.history_data])

                # --- Start Compute Timing ---
                start_compute = time.perf_counter()

                labels = self._cluster_and_label(coords)

                end_compute = time.perf_counter()
                # --- End Compute Timing ---

                compute_ms = (end_compute - start_compute) * 1000.0

                # Check the label for the last point added (the current vehicle)
                current_label = int(labels[-1])
                self.last_computed_label = current_label

                # Label -1 means noise (anomaly) in DBSCAN
                if current_label == -1:
                    is_anomaly = True

            # --- Output Formatting ---
            vehicle['dbscan_is_anomaly'] = is_anomaly
            vehicle['dbscan_label'] = current_label
            vehicle['dbscan_compute_ms'] = float(compute_ms)
            vehicle['dbscan_total_map_ms'] = (
                time.perf_counter() - start_total) * 1000.0

            return json.dumps(vehicle)

        except Exception as e:
            # On error, return input unchanged with default values
            try:
                if 'vehicle' in locals():
                    vehicle['dbscan_is_anomaly'] = False
                    vehicle['dbscan_label'] = self.last_computed_label
                    vehicle['dbscan_compute_ms'] = 0.0
                    return json.dumps(vehicle)
                return json_str
            except Exception:
                return json_str
