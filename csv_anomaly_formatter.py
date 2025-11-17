# anomaly_logger.py (CSVAnomalyFormatter) - REVISED

import json
from pyflink.datastream.functions import MapFunction
from typing import Dict, Any


class CSVAnomalyFormatter(MapFunction):
    """
    Formats the final enriched JSON string into a single CSV row, 
    including scores, compute times, and the specific rule-based anomaly types.
    """

    # Define simple anomaly thresholds for filtering the output
    HST_SCORE_ANOMALY_THRESHOLD = 0.5
    IFOREST_SCORE_ANOMALY_THRESHOLD = 0.5
    LOF_SCORE_ANOMALY_THRESHOLD = 1.05

    def map(self, json_str: str) -> str:
        try:
            vehicle: Dict[str, Any] = json.loads(json_str)

            # --- 1. Extract Scores, Flags, and Types ---

            # Rule-Based (Ground Labeller)
            is_rule_anomaly = vehicle.get('is_anomaly', False)
            # The list of anomaly types (e.g., ['SPEED', 'STATIONARY']) joined by ','
            rule_anomaly_types = ','.join(vehicle.get('anomalies', []))

            # Half-Space Tree (HST)
            hs_score = vehicle.get('half_space_score', 0.0)
            hs_compute = vehicle.get('half_space_compute_ms', 0.0)
            is_hs_anomaly = hs_score > self.HST_SCORE_ANOMALY_THRESHOLD

            # Isolation Forest (iForest)
            if_score = vehicle.get('iforest_score', 0.0)
            if_compute = vehicle.get('iforest_compute_ms', 0.0)
            is_if_anomaly = if_score > self.IFOREST_SCORE_ANOMALY_THRESHOLD

            # LOF (Local Outlier Factor)
            lof_score = vehicle.get('lof_score', 1.0)
            lof_compute = vehicle.get('lof_compute_ms', 0.0)
            is_lof_anomaly = lof_score > self.LOF_SCORE_ANOMALY_THRESHOLD

            # DBSCAN
            db_label = vehicle.get('dbscan_label', 0)
            db_compute = vehicle.get('dbscan_compute_ms', 0.0)
            is_db_anomaly = db_label == -1

            # --- 2. Filter Output ---

            # Output only if at least one detector flags it as an anomaly
            if not (is_rule_anomaly or is_hs_anomaly or is_if_anomaly or is_lof_anomaly or is_db_anomaly):
                return ""

            # --- 3. Format CSV Record ---

            # NEW CSV Header Structure:
            # timestamp,vehicle_id,latitude,longitude,rule_anomaly,rule_anomaly_types,
            # half_space_score,half_space_compute_ms,
            # iforest_score,iforest_compute_ms,
            # lof_score,lof_compute_ms,
            # dbscan_label,dbscan_compute_ms

            csv_row = (
                f"{vehicle.get('timestamp', '')},"
                f"{vehicle.get('vehicle_id', 'UNKNOWN')},"
                f"{vehicle.get('route_id', 'UNKNOWN')},"
                f"{vehicle.get('latitude', 0.0):.6f},"
                f"{vehicle.get('longitude', 0.0):.6f},"
                f"{1 if is_rule_anomaly else 0},"
                f"{rule_anomaly_types},"  # <-- NEW FIELD HERE

                # HST Fields
                f"{hs_score:.6f},"
                f"{hs_compute:.3f},"

                # iForest Fields
                f"{if_score:.6f},"
                f"{if_compute:.3f},"

                # LOF Fields
                f"{lof_score:.6f},"
                f"{lof_compute:.3f},"

                # DBSCAN Fields
                f"{db_label},"
                f"{db_compute:.3f}"
            )

            return csv_row

        except Exception:
            return ""
