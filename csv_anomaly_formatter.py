# csv_anomaly_formatter.py (IMPROVED)

from pyflink.datastream.functions import MapFunction
import json
from typing import Dict, Any


class CSVAnomalyFormatter(MapFunction):
    """
    Formats enriched vehicle data into CSV format for output.

    Options for handling warm-up period:
    1. SKIP_WARMUP: Only output samples where all algorithms have valid scores
    2. INCLUDE_ALL: Output all samples, marking warm-up status

    Default: SKIP_WARMUP for clean comparison data
    """

    def __init__(self, mode='SKIP_WARMUP'):
        """
        Args:
            mode: 'SKIP_WARMUP' or 'INCLUDE_ALL'
        """
        self.mode = mode
        self.sample_count = 0

    def map(self, json_str: str) -> str:
        try:
            vehicle: Dict[str, Any] = json.loads(json_str)
            self.sample_count += 1

            # Extract common fields
            timestamp = vehicle.get('timestamp', '')
            vehicle_id = vehicle.get('vehicle_id', 'unknown')
            route_id = vehicle.get('route_id', 'unknown')
            lat = vehicle.get('latitude', 0.0)
            lon = vehicle.get('longitude', 0.0)
            speed_kmh = vehicle.get('speed_kmh', 0.0)

            # Extract rule-based detection
            rule_anomaly = vehicle.get('is_anomaly', False)
            rule_types = ','.join(vehicle.get('anomalies', [])) if vehicle.get(
                'anomalies') else 'none'

            # Extract ML detector scores
            hst_score = vehicle.get('half_space_score')
            hst_ms = vehicle.get('half_space_compute_ms')

            iforest_score = vehicle.get('iforest_score')
            iforest_ms = vehicle.get('iforest_compute_ms')

            lof_score = vehicle.get('lof_score')
            lof_ms = vehicle.get('lof_compute_ms')

            dbscan_score = vehicle.get('dbscan_score')
            dbscan_ms = vehicle.get('dbscan_compute_ms')

            # Check if all ML scores are valid (not None)
            all_scores_valid = all([
                hst_score is not None,
                iforest_score is not None,
                lof_score is not None,
                dbscan_score is not None
            ])

            # Handle based on mode
            if self.mode == 'SKIP_WARMUP':
                # Skip samples during warm-up
                if not all_scores_valid:
                    return ""  # Empty string = filtered out

            # Format values (handle None gracefully)
            def fmt(val, default='NA'):
                if val is None:
                    return default
                if isinstance(val, bool):
                    return str(val)
                if isinstance(val, float):
                    return f"{val:.6f}"
                return str(val)

            # Build CSV row
            csv_row = (
                f"{timestamp},"
                f"{vehicle_id},"
                f"{route_id},"
                f"{fmt(lat)},"
                f"{fmt(lon)},"
                f"{fmt(speed_kmh)},"
                f"{rule_anomaly},"
                f"{rule_types},"
                f"{fmt(hst_score)},"
                f"{fmt(hst_ms)},"
                f"{fmt(iforest_score)},"
                f"{fmt(iforest_ms)},"
                f"{fmt(lof_score)},"
                f"{fmt(lof_ms)},"
                f"{fmt(dbscan_score)},"
                f"{fmt(dbscan_ms)},"
            )

            return csv_row

        except Exception as e:
            print(f"Error in CSVAnomalyFormatter: {e}")
            return ""  # Return empty string on error


class CSVAnomalyFormatterDebug(MapFunction):
    """
    Debug version that outputs statistics about warm-up periods.
    Use this to verify all algorithms are warming up correctly.
    """

    def __init__(self):
        self.total_samples = 0
        self.hst_ready = 0
        self.iforest_ready = 0
        self.lof_ready = 0
        self.dbscan_ready = 0
        self.all_ready = 0

    def map(self, json_str: str) -> str:
        vehicle = json.loads(json_str)
        self.total_samples += 1

        # Count ready algorithms
        if vehicle.get('half_space_score') is not None:
            self.hst_ready += 1
        if vehicle.get('iforest_score') is not None:
            self.iforest_ready += 1
        if vehicle.get('lof_score') is not None:
            self.lof_ready += 1
        if vehicle.get('dbscan_score') is not None:
            self.dbscan_ready += 1

        # Count when all are ready
        if all([
            vehicle.get('half_space_score') is not None,
            vehicle.get('iforest_score') is not None,
            vehicle.get('lof_score') is not None,
            vehicle.get('dbscan_score') is not None
        ]):
            self.all_ready += 1

        # Print statistics every 100 samples
        if self.total_samples % 100 == 0:
            print(f"\n📊 Warm-up Statistics (Sample {self.total_samples}):")
            print(
                f"   HST ready:     {self.hst_ready}/{self.total_samples} ({self.hst_ready/self.total_samples*100:.1f}%)")
            print(
                f"   IForest ready: {self.iforest_ready}/{self.total_samples} ({self.iforest_ready/self.total_samples*100:.1f}%)")
            print(
                f"   LOF ready:     {self.lof_ready}/{self.total_samples} ({self.lof_ready/self.total_samples*100:.1f}%)")
            print(
                f"   DBSCAN ready:  {self.dbscan_ready}/{self.total_samples} ({self.dbscan_ready/self.total_samples*100:.1f}%)")
            print(
                f"   All ready:     {self.all_ready}/{self.total_samples} ({self.all_ready/self.total_samples*100:.1f}%)")

        return json_str  # Pass through unchanged
