from pyflink.datastream.functions import MapFunction
from collections import defaultdict
import json
from math import radians, cos, sin, asin, sqrt

from constants import (
    AnomalyType,
    MAX_SPEED_KMH,
    DEFAULT_EARTH_RADIUS_KM,
    MIN_GLOBAL_HISTORY,
    GEO_OUTLIER_THRESHOLD_KM,
    GEO_NEIGHBOR_SAMPLE,
    STATIONARY_WINDOW,
    STATIONARY_DISTANCE_KM,
    VEHICLE_HISTORY_WINDOW,
    GLOBAL_POSITION_HISTORY,
)


class RuleBasedAnomalyDetector(MapFunction):
    def __init__(self):
        self.vehicle_history = defaultdict(list)
        self.all_positions = []
        self.window_size = VEHICLE_HISTORY_WINDOW

    def map(self, json_str: str) -> str:
        try:
            vehicle = json.loads(json_str)

            if not all([
                vehicle.get('vehicle_id'),
                vehicle.get('latitude'),
                vehicle.get('longitude')
            ]):
                return json_str

            vehicle_id = vehicle['vehicle_id']
            lat = vehicle['latitude']
            lon = vehicle['longitude']
            timestamp = vehicle.get('timestamp', 0)

            # Maintain per-vehicle history
            self.vehicle_history[vehicle_id].append({
                'lat': lat,
                'lon': lon,
                'timestamp': timestamp
            })
            if len(self.vehicle_history[vehicle_id]) > self.window_size:
                self.vehicle_history[vehicle_id].pop(0)

            # Global history
            self.all_positions.append({'lat': lat, 'lon': lon})
            if len(self.all_positions) > GLOBAL_POSITION_HISTORY:
                self.all_positions.pop(0)

            anomalies = []

            if self._is_speed_anomaly(vehicle_id):
                anomalies.append(AnomalyType.SPEED.value)

            if self._is_geographic_outlier(lat, lon):
                anomalies.append(AnomalyType.GEO_OUTLIER.value)

            if self._is_stationary_too_long(vehicle_id):
                anomalies.append(AnomalyType.STATIONARY.value)

            vehicle['anomalies'] = anomalies
            vehicle['is_anomaly'] = bool(anomalies)

            return json.dumps(vehicle)

        except Exception:
            return json_str

    # ---- Detection Rules ---- #

    def _is_speed_anomaly(self, vehicle_id: str) -> bool:
        history = self.vehicle_history[vehicle_id]
        if len(history) < 2:
            return False

        pos1, pos2 = history[-2], history[-1]

        distance = self._haversine_distance(
            pos1['lat'], pos1['lon'],
            pos2['lat'], pos2['lon']
        )

        time_diff = abs(pos2['timestamp'] - pos1['timestamp'])
        if time_diff == 0:
            return False

        speed_kmh = (distance / time_diff) * 3600
        return speed_kmh > MAX_SPEED_KMH

    def _is_geographic_outlier(self, lat: float, lon: float) -> bool:
        if len(self.all_positions) < MIN_GLOBAL_HISTORY:
            return False

        distances = [
            self._haversine_distance(lat, lon, pos['lat'], pos['lon'])
            for pos in self.all_positions[-GEO_NEIGHBOR_SAMPLE:]
        ]

        return min(distances) > GEO_OUTLIER_THRESHOLD_KM

    def _is_stationary_too_long(self, vehicle_id: str) -> bool:
        history = self.vehicle_history[vehicle_id]
        if len(history) < STATIONARY_WINDOW:
            return False

        first_pos = history[0]
        for pos in history[1:]:
            distance = self._haversine_distance(
                first_pos['lat'], first_pos['lon'],
                pos['lat'], pos['lon']
            )
            if distance > STATIONARY_DISTANCE_KM:
                return False

        return True

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * DEFAULT_EARTH_RADIUS_KM
