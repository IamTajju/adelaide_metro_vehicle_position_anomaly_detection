# half_space_detector.py

from pyflink.datastream.functions import MapFunction
import numpy as np
import json
from typing import Dict, Any
import time
from collections import deque

from river import anomaly
from river import preprocessing
from river import compose

from constants import (
    HST_NUM_TREES,
    HST_TREE_HEIGHT,
    HST_INIT_WINDOW,
    RANDOM_STATE,
)


class HalfSpaceAnomalyDetector(MapFunction):
    """
    Streaming Half-Space Trees ensemble using River ML implementation.
    Uses MinMaxScaler preprocessing as recommended by River documentation.

    Follows score-then-learn paradigm to avoid train-test leakage.
    """

    def __init__(self):
        # Track last points for speed calculation
        self.last_point = deque(maxlen=2)  # stores [lat, lon, timestamp]

        # Initialize River ML pipeline with MinMaxScaler + HalfSpaceTrees
        self.model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(
                n_trees=HST_NUM_TREES,
                height=HST_TREE_HEIGHT,
                window_size=HST_INIT_WINDOW,
                seed=RANDOM_STATE
            )
        )

        # Warm-up counter to ensure initial learning
        self.warmup_count = 0
        self.warmup_threshold = HST_INIT_WINDOW
        self.ready = False

    def map(self, json_str: str) -> str:
        start_total = time.perf_counter()

        try:
            vehicle: Dict[str, Any] = json.loads(json_str)
            lat = vehicle.get("latitude")
            lon = vehicle.get("longitude")
            speed_kmh = vehicle.get("speed_kmh")

            # Pre-check & Default Return
            # Use None for invalid data to exclude from evaluation
            if lat is None or lon is None:
                vehicle["half_space_score"] = None
                vehicle["half_space_compute_ms"] = 0.0
                return json.dumps(vehicle)

            # Create feature dictionary for River ML
            # River expects dict with feature names
            features = {
                'longitude': float(lon),
                'latitude': float(lat),
                'speed_kmh': float(speed_kmh)
            }

            # -----------------------------
            # 1. Warm-up Phase
            # -----------------------------
            # Just learn without scoring during warm-up
            if not self.ready:
                self.model.learn_one(features)
                self.warmup_count += 1

                if self.warmup_count >= self.warmup_threshold:
                    self.ready = True

                # Use None during warm-up to exclude from evaluation
                vehicle["half_space_score"] = None
                vehicle["half_space_compute_ms"] = None
                vehicle["half_space_warmup"] = True
                return json.dumps(vehicle)

            # -----------------------------
            # 2. Scoring Phase (after warm-up)
            # -----------------------------
            start_compute = time.perf_counter()

            # Score FIRST (before learning) to avoid train-test leakage
            score = self.model.score_one(features)

            # THEN update model with new observation
            self.model.learn_one(features)

            end_compute = time.perf_counter()

            compute_ms = (end_compute - start_compute) * 1000.0
            total_ms = (time.perf_counter() - start_total) * 1000.0

            vehicle["half_space_score"] = float(score)
            vehicle["half_space_compute_ms"] = float(compute_ms)
            vehicle["half_space_total_map_ms"] = float(total_ms)
            vehicle["half_space_warmup"] = False

            return json.dumps(vehicle)

        except Exception as e:
            # On error, return input unchanged with None score to exclude from evaluation
            vehicle = json.loads(json_str)
            vehicle["half_space_score"] = None
            vehicle["half_space_compute_ms"] = None
            # Log exception for debugging
            print(f"Error in HalfSpaceAnomalyDetector: {e}")
            return json.dumps(vehicle)
